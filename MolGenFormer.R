#!/usr/bin/env Rscript



suppressPackageStartupMessages({
  library(torch)
  library(coro)
  library(data.table)
  library(dplyr)
  library(tibble)
  library(stringr)
  library(readr)
  library(rcdk)
  library(rcdklibs)
  library(fingerprint)
  library(pbapply)
  library(progress)
  library(tools)
})

## =========================
## CPU / Apple Silicon friendly
## =========================
n_cores <- parallel::detectCores()
use_cores <- max(1L, min(as.integer(n_cores - 1L), 8L))

Sys.setenv(
  OMP_NUM_THREADS = use_cores,
  MKL_NUM_THREADS = use_cores
)
torch_set_num_threads(as.integer(use_cores))
if (exists("torch_set_num_interop_threads")) {
  try(torch_set_num_interop_threads(as.integer(max(1L, min(4L, floor(use_cores / 2L))))), silent = TRUE)
}

## =========================
## FAST MODE KNOBS
## =========================
max_train_rows <- 2000000L
len_pct        <- 0.95
max_len_cap    <- 64L
batch_size     <- 768L
epochs_lm      <- 2L
epochs_vae     <- 0L
d_model        <- 192L
nhead          <- 6L
nlayers        <- 3L
d_ff           <- 768L
dropout        <- 0.10
latent_dim     <- 96L

## Optional stages
rl_iters       <- 0L
rl_batch       <- 512L
top_k_tf       <- 64L
vae_lat_steps  <- 6L
vae_step_size  <- 0.75
temperature    <- 1.0

## Base scoring weights
w_props        <- 0.35
w_thermo       <- 0.35
bonus_novel    <- 0.25
penal_seen     <- 0.50
pen_pains      <- 2.00

## Sampling/export
N_from_LM        <- 50000L
N_from_VAE       <- 0L
top_hits         <- 1000L
fp_bits          <- 2048L
do_diversity     <- TRUE

## Diversity controls
sample_temps     <- c(0.80, 0.95, 1.10, 1.25)
prefilter_keep   <- max(15000L, top_hits * 25L)
div_lambda       <- 0.70
max_sim_allowed  <- 0.72

## Scaffold controls
murcko_min_size     <- 6L
max_per_scaffold    <- 3L
first_pass_one_each <- TRUE
scaffold_bonus      <- 0.15
unknown_scaffold_id <- "NO_MURCKO"

## Secondary bonuses for final greedy selection
novelty_bonus2   <- 0.15
admet_bonus2     <- 0.05
soft_bonus2      <- 0.05

## I/O
train_smi  <- "canonical.smi"   # SMILES \t ID
out_dir    <- "hybrid_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
cache_path <- file.path(out_dir, "train_can_cache.rds")

## 3D export
use_obabel <- TRUE
obabel_bin <- "/opt/homebrew/bin/obabel"
pH         <- 7.4

set.seed(42)
device <- torch_device("cpu")
options(warn = 1)

## =========================
## Load training SMILES
## =========================
chem <- fread(
  train_smi,
  sep = "\t",
  header = FALSE,
  fill = TRUE,
  quote = "",
  col.names = c("smiles", "id")
)
chem <- chem[!is.na(smiles) & nchar(smiles) > 0]
cat("Loaded training:", nrow(chem), "\n")

if (!is.na(max_train_rows) && nrow(chem) > max_train_rows) {
  set.seed(42)
  chem <- chem[sample(.N, max_train_rows)]
  cat("Capped training rows to:", nrow(chem), "\n")
}

## =========================
## Chemistry helpers
## =========================
mol_of <- function(s) {
  tryCatch(parse.smiles(s)[[1]], error = function(e) NULL)
}

can_of <- function(m) {
  tryCatch(
    get.smiles(m, smiles.flavors(c("Canonical"))),
    error = function(e) NA_character_
  )
}

get_mass_safe <- function(m) {
  for (fn in c("get.exact.mass", "get.mass", "get.total.mass")) {
    if (exists(fn, mode = "function")) {
      val <- suppressWarnings(
        tryCatch(do.call(get(fn), list(m)), error = function(e) NA_real_)
      )
      if (is.numeric(val) && length(val) == 1L && is.finite(val)) {
        return(as.numeric(val))
      }
    }
  }
  NA_real_
}

props_of <- function(m) {
  if (is.null(m)) return(rep(NA_real_, 6))
  MW    <- suppressWarnings(tryCatch(get_mass_safe(m),              error = function(e) NA_real_))
  XlogP <- suppressWarnings(tryCatch(get.xlogp(m),                  error = function(e) NA_real_))
  RotB  <- suppressWarnings(tryCatch(get.rotor.count(m),            error = function(e) NA_real_))
  HBD   <- suppressWarnings(tryCatch(get.hbond.donor.count(m),      error = function(e) NA_real_))
  HBA   <- suppressWarnings(tryCatch(get.hbond.acceptor.count(m),   error = function(e) NA_real_))
  TPSA  <- suppressWarnings(tryCatch(get.tpsa(m),                   error = function(e) NA_real_))
  c(MW, XlogP, RotB, HBD, HBA, TPSA)
}

## =========================
## Standardization helpers
## =========================
get_atom_count_safe <- function(m) {
  tryCatch(
    {
      atoms <- m$getAtomCount()
      as.integer(atoms)
    },
    error = function(e) NA_integer_
  )
}

configure_mol_safe <- function(m) {
  if (is.null(m)) return(NULL)
  tryCatch({
    do.aromaticity(m)
    convert.implicit.to.explicit(m)
    m
  }, error = function(e) m)
}

split_components_via_smiles <- function(smiles) {
  if (is.na(smiles) || !nzchar(smiles)) return(character(0))
  comps <- unlist(strsplit(smiles, "\\.", perl = TRUE))
  comps <- trimws(comps)
  comps[nzchar(comps)]
}

largest_fragment_smiles <- function(smiles) {
  comps <- split_components_via_smiles(smiles)
  if (!length(comps)) return(NA_character_)
  
  comp_stats <- lapply(comps, function(x) {
    m <- mol_of(x)
    if (is.null(m)) {
      return(list(smiles = x, natoms = -1L, mw = -Inf))
    }
    m <- configure_mol_safe(m)
    list(
      smiles = x,
      natoms = get_atom_count_safe(m),
      mw = get_mass_safe(m)
    )
  })
  
  natoms <- vapply(comp_stats, function(z) ifelse(is.na(z$natoms), -1L, z$natoms), integer(1))
  mw     <- vapply(comp_stats, function(z) ifelse(is.na(z$mw), -Inf, z$mw), numeric(1))
  
  ord <- order(natoms, mw, decreasing = TRUE, na.last = TRUE)
  comp_stats[[ord[1]]]$smiles
}

standardize_parent_smiles <- function(smiles) {
  if (is.na(smiles) || !nzchar(smiles)) return(NA_character_)
  
  ## 1. split disconnected components and keep largest fragment
  parent0 <- largest_fragment_smiles(smiles)
  if (is.na(parent0) || !nzchar(parent0)) return(NA_character_)
  
  ## 2. parse and configure consistently
  m <- mol_of(parent0)
  if (is.null(m)) return(NA_character_)
  m <- configure_mol_safe(m)
  
  ## 3. canonical parent smiles
  parent_can <- can_of(m)
  if (is.na(parent_can) || !nzchar(parent_can)) return(NA_character_)
  
  ## 4. second-pass canonicalization for stability
  m2 <- mol_of(parent_can)
  if (is.null(m2)) return(parent_can)
  m2 <- configure_mol_safe(m2)
  parent_can2 <- can_of(m2)
  if (is.na(parent_can2) || !nzchar(parent_can2)) parent_can else parent_can2
}

run_obabel_capture <- function(args, timeout_sec = 120L) {
  tryCatch(
    system2(
      obabel_bin,
      args = args,
      stdout = TRUE,
      stderr = TRUE,
      timeout = timeout_sec
    ),
    error = function(e) structure(conditionMessage(e), class = "try-error")
  )
}

standardized_inchikey_from_smiles <- function(smiles, obabel_bin) {
  if (is.na(smiles) || !nzchar(smiles)) return(NA_character_)
  
  smi_tmp <- tempfile(fileext = ".smi")
  out_tmp <- tempfile(fileext = ".txt")
  on.exit(unlink(c(smi_tmp, out_tmp), force = TRUE), add = TRUE)
  
  writeLines(smiles, smi_tmp)
  
  res <- run_obabel_capture(
    c("-ismi", smi_tmp, "-oinchikey", "-O", out_tmp),
    timeout_sec = 120L
  )
  
  if (!file.exists(out_tmp) || file.size(out_tmp) <= 0) return(NA_character_)
  x <- tryCatch(readLines(out_tmp, warn = FALSE), error = function(e) character(0))
  x <- trimws(x)
  x <- x[nzchar(x)]
  if (!length(x)) return(NA_character_)
  x[1]
}

## Optional salt / solvent blacklist for tiny fragments
is_trivial_fragment_smiles <- function(smi) {
  if (is.na(smi) || !nzchar(smi)) return(TRUE)
  trivial <- c(
    "[Na+]", "[K+]", "[Li+]", "[Cl-]", "[Br-]", "[I-]",
    "O", "[OH-]", "[H+]", "[NH4+]", "CO", "CCO"
  )
  smi %in% trivial
}

standardize_parent_smiles_strict <- function(smiles) {
  comps <- split_components_via_smiles(smiles)
  if (!length(comps)) return(NA_character_)
  
  ## remove obviously trivial fragments if multiple components exist
  if (length(comps) > 1L) {
    keep <- !vapply(comps, is_trivial_fragment_smiles, logical(1))
    if (any(keep)) comps <- comps[keep]
  }
  if (!length(comps)) return(NA_character_)
  
  ## choose largest meaningful fragment
  parent0 <- largest_fragment_smiles(paste(comps, collapse = "."))
  if (is.na(parent0) || !nzchar(parent0)) return(NA_character_)
  
  m <- mol_of(parent0)
  if (is.null(m)) return(NA_character_)
  m <- configure_mol_safe(m)
  
  parent_can <- can_of(m)
  if (is.na(parent_can) || !nzchar(parent_can)) return(NA_character_)
  
  ## stabilize by reparsing/canonicalizing again
  m2 <- mol_of(parent_can)
  if (is.null(m2)) return(parent_can)
  m2 <- configure_mol_safe(m2)
  parent_can2 <- can_of(m2)
  if (is.na(parent_can2) || !nzchar(parent_can2)) parent_can else parent_can2
}

## =========================
## Tokenizer
## =========================
bos <- "<bos>"
eos <- "<eos>"
pad <- "<pad>"

smiles_charset <- c(
  letters, LETTERS,
  "#", "%", "(", ")", "+", "-", ".", "/", "=", "@", "[", "]", "\\",
  "0","1","2","3","4","5","6","7","8","9"
)
smiles_charset <- unique(smiles_charset)

vocab <- c(pad, bos, eos, smiles_charset)
stoi  <- setNames(seq_along(vocab), vocab)
itos  <- setNames(vocab, seq_along(vocab))

map_chars <- function(s) {
  ch <- unlist(strsplit(s, ""))
  ch[!ch %in% smiles_charset] <- pad
  ch
}

encode <- function(s) {
  c(stoi[[bos]], unname(stoi[map_chars(s)]), stoi[[eos]])
}

decode <- function(ix) {
  toks <- itos[as.character(ix)]
  toks <- toks[!toks %in% c(bos, eos, pad)]
  paste0(toks, collapse = "")
}

pad_to <- function(v, L) {
  v <- c(v, rep(stoi[[pad]], max(0L, L - length(v))))
  v[1:L]
}

## =========================
## Max sequence length
## =========================
sm_len  <- nchar(chem$smiles)
max_len <- min(
  max_len_cap,
  as.integer(quantile(sm_len, probs = len_pct, names = FALSE)) + 2L
)
cat("Using max_len:", max_len, "\n")

## =========================
## Canonical cache
## =========================
get_train_can_vec <- function(smiles_vec, path, assume_canonical = TRUE) {
  if (file.exists(path)) {
    vec <- readRDS(path)
    cat("Loaded canonical cache:", path, " (", length(vec), " entries)\n", sep = "")
    return(vec)
  }
  if (assume_canonical) {
    cat("Assuming input SMILES are already canonical; building cache from unique strings...\n")
    vec <- unique(smiles_vec[!is.na(smiles_vec) & nzchar(smiles_vec)])
    saveRDS(vec, path)
    cat("Saved canonical cache:", path, " (", length(vec), " entries)\n", sep = "")
    return(vec)
  }
  stop("Non-fast-path canonicalization disabled in FAST MODE.")
}
train_can_vec <- get_train_can_vec(unique(chem$smiles), cache_path, TRUE)

## =========================
## Fast input pipeline
## =========================
lens <- nchar(chem$smiles) + 2L
nbuckets <- 16L
qs <- quantile(lens, probs = seq(0, 1, length.out = nbuckets + 1), names = FALSE)
bucket_of <- cut(lens, breaks = unique(qs), include.lowest = TRUE, labels = FALSE)

make_bucket_dl <- function(bucket_id, batch_size) {
  idx <- which(bucket_of == bucket_id)
  if (!length(idx)) return(NULL)
  
  idx <- sample(idx, length(idx))
  smiles_subset <- chem$smiles[idx]
  
  ds <- dataset(
    initialize = function(smiles_vec, max_len) {
      self$smiles <- smiles_vec
      self$max_len <- max_len
    },
    .getitem = function(i) {
      xi <- pad_to(encode(self$smiles[i]), self$max_len)
      list(
        inp = torch_tensor(head(xi, -1), dtype = torch_long()),
        tgt = torch_tensor(tail(xi, -1), dtype = torch_long())
      )
    },
    .length = function() length(self$smiles)
  )(smiles_subset, max_len)
  
  dataloader(
    ds,
    batch_size = as.integer(batch_size),
    shuffle = FALSE,
    num_workers = 0L,
    pin_memory = FALSE
  )
}

bucket_dls <- lapply(seq_len(nbuckets), make_bucket_dl, batch_size = batch_size)
bucket_dls <- bucket_dls[!vapply(bucket_dls, is.null, logical(1))]
nbatches_est <- sum(vapply(bucket_dls, length, integer(1)))
if (nbatches_est <= 0L) stop("No batches to train.")

## =========================
## Cached causal masks
## =========================
mask_cache <- new.env(parent = emptyenv())

get_causal_mask <- function(Tt) {
  key <- as.character(Tt)
  m <- mask_cache[[key]]
  if (is.null(m)) {
    m <- torch_triu(torch_ones(Tt, Tt), diagonal = 1)$to(dtype = torch_bool())
    mask_cache[[key]] <- m
  }
  m
}

## =========================
## Models
## =========================
tf_lm <- nn_module(
  initialize = function(vocab_size, d_model, nhead, nlayers, d_ff, dropout) {
    self$emb  <- nn_embedding(vocab_size, d_model, padding_idx = stoi[[pad]])
    self$posE <- nn_parameter(torch_randn(1, max_len - 1, d_model) * 0.01)
    
    enc_layer <- nn_transformer_encoder_layer(
      d_model = d_model,
      nhead = nhead,
      dim_feedforward = d_ff,
      dropout = dropout,
      batch_first = TRUE
    )
    self$enc <- nn_transformer_encoder(enc_layer, num_layers = nlayers)
    self$out <- nn_linear(d_model, vocab_size)
  },
  
  forward = function(y_inp) {
    Tt <- y_inp$size(2)
    e <- self$emb(y_inp) + self$posE[, 1:Tt, ]
    h <- self$enc(e, mask = get_causal_mask(Tt))
    self$out(h)
  },
  
  sample = function(n = 256L, temperature = 1.0) {
    self$eval()
    with_no_grad({
      y <- torch_full(c(n, 1), stoi[[bos]], dtype = torch_long())
      pb <- progress_bar$new(
        total = max_len - 1,
        clear = FALSE,
        format = "LM sample [:bar] :current/:total | eta=:eta"
      )
      
      for (t in 1:(max_len - 1)) {
        logits <- self$forward(y)[, ncol(y), ]
        logits <- logits / temperature
        probs  <- nnf_softmax(logits, dim = 2)
        tok    <- torch_multinomial(probs, num_samples = 1)
        y <- torch_cat(list(y, tok), dim = 2)
        pb$tick()
      }
      
      seqs <- as_array(y)
      smi <- character(n)
      for (i in seq_len(n)) {
        s <- seqs[i, ]
        eos_pos <- which(s == stoi[[eos]])[1]
        if (!is.na(eos_pos) && eos_pos > 2L) {
          s <- s[2:(eos_pos - 1)]
        } else {
          s <- s[2:length(s)]
        }
        smi[i] <- decode(s)
      }
      smi
    })
  }
)

tf_vae <- nn_module(
  initialize = function(vocab_size, d_model = d_model, nhead = nhead,
                        nlayers = nlayers, d_ff = d_ff,
                        latent = latent_dim, dropout = dropout) {
    self$emb   <- nn_embedding(vocab_size, d_model, padding_idx = stoi[[pad]])
    self$posE  <- nn_parameter(torch_randn(1, max_len - 1, d_model) * 0.01)
    
    enc_layer  <- nn_transformer_encoder_layer(
      d_model = d_model,
      nhead = nhead,
      dim_feedforward = d_ff,
      dropout = dropout,
      batch_first = TRUE
    )
    self$enc    <- nn_transformer_encoder(enc_layer, num_layers = nlayers)
    self$dec    <- nn_transformer_encoder(enc_layer, num_layers = nlayers)
    self$to_mu  <- nn_linear(d_model, latent)
    self$to_lv  <- nn_linear(d_model, latent)
    self$from_z <- nn_linear(latent, d_model)
    self$out    <- nn_linear(d_model, vocab_size)
  },
  
  encode = function(x) {
    e <- self$emb(x) + self$posE[, 1:ncol(x), ]
    h <- self$enc(e)
    h_cls <- h[, 1, ]
    list(mu = self$to_mu(h_cls), lv = self$to_lv(h_cls))
  },
  
  reparam = function(mu, lv) {
    mu + torch_randn_like(mu) * (0.5 * lv)$exp()
  },
  
  decode = function(y_inp, z) {
    B  <- y_inp$size(1)
    Tt <- y_inp$size(2)
    z_proj <- self$from_z(z)$unsqueeze(2)$expand(c(B, Tt, -1))
    de <- self$emb(y_inp) + z_proj + self$posE[, 1:Tt, ]
    h <- self$dec(de, mask = get_causal_mask(Tt))
    self$out(h)
  },
  
  forward = function(x_inp, y_inp) {
    enc <- self$encode(x_inp)
    z   <- self$reparam(enc$mu, enc$lv)
    list(logits = self$decode(y_inp, z), mu = enc$mu, lv = enc$lv)
  }
)

model_lm  <- tf_lm(length(vocab), d_model, nhead, nlayers, d_ff, dropout)$to(device = device)
model_vae <- tf_vae(length(vocab), d_model, nhead, nlayers, d_ff, latent_dim, dropout)$to(device = device)

## =========================
## Optimizers / schedulers / loss
## =========================
make_adamw <- function(...) {
  if (exists("optim_adamw")) optim_adamw(...) else optim_adam(...)
}

opt_lm  <- make_adamw(model_lm$parameters,  lr = 3e-3, weight_decay = 0.01)
opt_vae <- make_adamw(model_vae$parameters, lr = 3e-3, weight_decay = 0.01)

get_base_lr <- function(opt) opt$param_groups[[1]]$lr
set_lr <- function(opt, lr) {
  for (i in seq_along(opt$param_groups)) opt$param_groups[[i]]$lr <- lr
  invisible(NULL)
}

make_cosine_scheduler <- function(opt, T_max, eta_min = 0, base_lr = NULL) {
  if (is.null(base_lr)) base_lr <- get_base_lr(opt)
  t <- 0L
  list(
    step = function() {
      t <<- t + 1L
      tt <- min(t, T_max)
      lr <- eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * tt / T_max))
      set_lr(opt, lr)
      lr
    },
    lr = function() get_base_lr(opt)
  )
}

sched_lm  <- make_cosine_scheduler(opt_lm,  T_max = max(1L, epochs_lm * nbatches_est), eta_min = 3e-4)
sched_vae <- make_cosine_scheduler(opt_vae, T_max = max(1L, max(1L, epochs_vae) * nbatches_est), eta_min = 3e-4)

ce_tokens <- function(logits, y, ignore_index) {
  B  <- logits$size(1)
  Tt <- logits$size(2)
  V  <- logits$size(3)
  logits2 <- logits$reshape(c(B * Tt, V))
  y2      <- y$reshape(c(B * Tt))
  nnf_cross_entropy(logits2, y2, ignore_index = ignore_index, reduction = "mean")
}

trim_to_batch <- function(x, y, pad_id) {
  xa <- as_array(x)
  last <- apply(xa, 1, function(r) {
    nz <- which(r != pad_id)
    if (length(nz)) max(nz) else 1L
  })
  Tt <- max(last)
  list(x = x[, 1:Tt], y = y[, 1:Tt], Tt = as.integer(Tt))
}

## =========================
## Warm-up
## =========================
with_no_grad({
  B  <- min(8L, batch_size)
  Tt <- min(64L, max_len - 1L)
  xw <- torch_full(c(B, Tt), stoi[[bos]], dtype = torch_long())
  invisible(get_causal_mask(Tt))
  invisible(model_lm$out(model_lm$enc(model_lm$emb(xw) + model_lm$posE[, 1:Tt, ], mask = get_causal_mask(Tt))))
})

## =========================
## LM training
## =========================
cat("\n== FAST LM training ==\n")
for (ep in 1:epochs_lm) {
  model_lm$train()
  tot <- 0
  nb  <- 0L
  
  pb <- progress_bar$new(
    total  = nbatches_est,
    clear  = FALSE,
    format = sprintf("LM ep %d/%d [:bar] :current/:total | NLL=:val | eta=:eta", ep, epochs_lm)
  )
  
  for (dlb in bucket_dls) {
    coro::loop(for (b in dlb) {
      x <- b$inp$to(device = device)
      y <- b$tgt$to(device = device)
      tb <- trim_to_batch(x, y, pad_id = stoi[[pad]])
      x <- tb$x
      y <- tb$y
      Tt <- tb$Tt
      
      opt_lm$zero_grad()
      e <- model_lm$emb(x) + model_lm$posE[, 1:Tt, ]
      h <- model_lm$enc(e, mask = get_causal_mask(Tt))
      logits <- model_lm$out(h)
      loss <- ce_tokens(logits, y, ignore_index = stoi[[pad]])
      loss$backward()
      opt_lm$step()
      sched_lm$step()
      
      tot <- tot + as.numeric(loss$item())
      nb  <- nb + 1L
      pb$tick(tokens = list(val = sprintf("%.4f", tot / nb)))
    })
  }
  
  try(pb$terminate(), silent = TRUE)
  cat(sprintf("LM epoch %d/%d NLL=%.4f\n", ep, epochs_lm, tot / max(1L, nb)))
}
torch_save(model_lm, file.path(out_dir, "lm_mle.pt"))

## =========================
## Drug-likeness + thermodynamics-inspired proxy
## =========================
druglike_score <- function(smiles) {
  m <- mol_of(smiles)
  if (is.null(m)) return(0)
  p <- props_of(m)
  names(p) <- c("MW","XlogP","RotB","HBD","HBA","TPSA")
  
  s <- 1.0
  if (!is.na(p["MW"])    && (p["MW"] < 200 || p["MW"] > 520))     s <- s - 0.20
  if (!is.na(p["XlogP"]) && (p["XlogP"] < -1 || p["XlogP"] > 5))  s <- s - 0.10
  if (!is.na(p["RotB"])  && p["RotB"] > 10)                       s <- s - 0.10
  if (!is.na(p["TPSA"])  && p["TPSA"] > 120)                      s <- s - 0.10
  max(0, as.numeric(s))
}

thermo_proxy_score_from_props <- function(MW, XlogP, RotB, HBD, HBA, TPSA) {
  score <- 1.0
  
  if (!is.na(RotB)) {
    if (RotB <= 6) score <- score + 0.10
    if (RotB > 8)  score <- score - 0.03 * (RotB - 8)
  }
  
  if (!is.na(XlogP) && !is.na(TPSA)) {
    if (XlogP >= 1.0 && XlogP <= 3.5 && TPSA >= 40 && TPSA <= 100) {
      score <- score + 0.20
    } else {
      if (XlogP > 5.0) score <- score - 0.10
      if (XlogP < 0.0) score <- score - 0.05
      if (TPSA > 120)  score <- score - 0.10
      if (TPSA < 20)   score <- score - 0.05
    }
  }
  
  if (!is.na(HBD) && HBD > 3) score <- score - 0.05 * (HBD - 3)
  if (!is.na(HBA) && HBA > 8) score <- score - 0.03 * (HBA - 8)
  
  if (!is.na(MW)) {
    if (MW >= 250 && MW <= 450) {
      score <- score + 0.10
    } else if (MW > 520) {
      score <- score - 0.15
    } else if (MW < 180) {
      score <- score - 0.08
    }
  }
  
  max(0, min(1.5, as.numeric(score)))
}

## =========================
## ADMET / PAINS
## =========================
passes_lipinski <- function(P) {
  (is.na(P["MW"])    || P["MW"]    <= 500) &&
    (is.na(P["XlogP"]) || P["XlogP"] <= 5) &&
    (is.na(P["HBD"])   || P["HBD"]   <= 5) &&
    (is.na(P["HBA"])   || P["HBA"]   <= 10)
}
passes_veber <- function(P) {
  (is.na(P["RotB"]) || P["RotB"] <= 10) &&
    (is.na(P["TPSA"]) || P["TPSA"] <= 140)
}
passes_egan <- function(P) {
  (is.na(P["TPSA"]) || P["TPSA"] <= 131) &&
    (is.na(P["XlogP"]) || P["XlogP"] <= 5.88)
}

admet_ok_vec <- function(smiles_vec) {
  pb <- progress_bar$new(total = length(smiles_vec), clear = FALSE,
                         format = "ADMET [:bar] :current/:total | eta=:eta")
  mols <- lapply(smiles_vec, function(s) { pb$tick(); mol_of(s) })
  props <- lapply(mols, props_of)
  vapply(props, function(p) {
    names(p) <- c("MW","XlogP","RotB","HBD","HBA","TPSA")
    passes_lipinski(p) && passes_veber(p) && passes_egan(p)
  }, logical(1))
}

pains_mode    <- "soft"
pains_penalty <- pen_pains

load_pains_smarts <- function(csv = "pains_smarts.csv") {
  if (file.exists(csv)) {
    df <- tryCatch(data.table::fread(csv), error = function(e) NULL)
    if (!is.null(df) && "smarts" %in% names(df)) {
      sm <- unique(stats::na.omit(df$smarts))
      message("Loaded ", length(sm), " PAINS SMARTS from ", csv)
      return(sm)
    }
  }
  builtin <- c(
    "c1ccc(cc1)N=Nc2ccccc2",
    "O=c1ccc(O)cc1=O",
    "Oc1ccc(O)cc1",
    "O=C1NC(=S)SC1=O",
    "O=CC=C",
    "O=CC=CC=O",
    "c1ccc(NN=O)cc1",
    "O=C(C=C)C=C"
  )
  message("pains_smarts.csv not found — using built-in mini-panel (", length(builtin), " patterns).")
  builtin
}
pains_smarts <- load_pains_smarts()

is_pains <- function(smiles, patterns = pains_smarts) {
  if (!length(patterns)) return(FALSE)
  m <- mol_of(smiles)
  if (is.null(m)) return(FALSE)
  for (smt in patterns) {
    hit <- tryCatch({
      res <- matches(m, smt)
      length(res) > 0
    }, error = function(e) FALSE)
    if (isTRUE(hit)) return(TRUE)
  }
  FALSE
}

## =========================
## Reward for optional RL / latent search
## =========================
reward_of <- function(smiles) {
  m <- mol_of(smiles)
  if (is.null(m)) return(-5)
  
  p <- props_of(m)
  names(p) <- c("MW","XlogP","RotB","HBD","HBA","TPSA")
  
  r <- 0
  r <- r + w_props  * druglike_score(smiles)
  r <- r + w_thermo * thermo_proxy_score_from_props(p["MW"], p["XlogP"], p["RotB"], p["HBD"], p["HBA"], p["TPSA"])
  
  can <- can_of(m)
  if (!is.na(can) && nzchar(can)) {
    if (!(can %chin% train_can_vec)) r <- r + bonus_novel else r <- r - penal_seen
  }
  if (is_pains(smiles)) r <- r - pen_pains
  as.numeric(r)
}

## =========================
## Optional VAE utilities
## =========================
decode_from_z <- function(z, temp = 1.0) {
  n_batch <- z$size(1)
  y <- torch_full(c(n_batch, 1), stoi[[bos]], dtype = torch_long())
  for (t in 1:(max_len - 1)) {
    logits <- model_vae$decode(y, z)[, ncol(y), , drop = FALSE]$squeeze(2) / temp
    probs  <- nnf_softmax(logits, dim = 2)
    tok    <- torch_multinomial(probs, num_samples = 1)
    y <- torch_cat(list(y, tok), dim = 2)
  }
  seqs <- as_array(y)
  res <- character(n_batch)
  for (i in seq_len(n_batch)) {
    s <- seqs[i, ]
    eos_pos <- which(s == stoi[[eos]])[1]
    if (!is.na(eos_pos) && eos_pos > 2L) s <- s[2:(eos_pos - 1)] else s <- s[2:length(s)]
    res[i] <- decode(s)
  }
  res
}

latent_opt_sample <- function(n_batch = 512L, steps = vae_lat_steps, step_size = vae_step_size) {
  z <- torch_randn(c(n_batch, latent_dim))
  for (st in 1:steps) {
    smi <- decode_from_z(z, temp = 1.0)
    r <- vapply(smi, reward_of, numeric(1))
    ord <- order(r, decreasing = TRUE)
    top <- ceiling(length(r) / 2)
    smi_top <- smi[ord][seq_len(top)]
    to_mu <- function(s) {
      xi <- encode(s)
      x  <- torch_tensor(head(pad_to(xi, max_len), -1), dtype = torch_long())$unsqueeze(1)
      enc <- model_vae$encode(x$squeeze(1))
      enc$mu$to(device = torch_device("cpu"))
    }
    mu_list <- pblapply(smi_top, to_mu)
    mu_mean <- do.call(torch_stack, c(mu_list, list(dim = 1)))$mean(dim = 1)
    mu_mean <- mu_mean$unsqueeze(1)$expand_as(z)
    z <- (1 - step_size) * z + step_size * mu_mean
  }
  decode_from_z(z, temp = 1.0)
}

## =========================
## Multi-temperature sampling
## =========================
collect_lm_samples <- function(target_n, temps = c(0.80, 0.95, 1.10, 1.25), chunk_n = 2000L) {
  out <- character(0)
  iter <- 1L
  pb <- progress_bar$new(
    total = target_n,
    clear = FALSE,
    format = "Collect LM [:bar] :current/:total | temp=:temp | eta=:eta"
  )
  
  while (length(out) < target_n) {
    temp <- temps[((iter - 1L) %% length(temps)) + 1L]
    chunk <- model_lm$sample(n = chunk_n, temperature = temp)
    chunk <- chunk[!is.na(chunk) & nzchar(chunk)]
    out <- unique(c(out, chunk))
    pb$update(min(1, length(out) / target_n), tokens = list(temp = sprintf("%.2f", temp)))
    iter <- iter + 1L
  }
  out[seq_len(min(length(out), target_n))]
}

cat("\n== Sampling from LM", if (N_from_VAE > 0L) " and VAE" else "", " ==\n", sep = "")
gen_lm <- collect_lm_samples(
  target_n = N_from_LM,
  temps    = sample_temps,
  chunk_n  = 2000L
)

gen_vae <- character(0)
if (N_from_VAE > 0L) {
  while (length(gen_vae) < N_from_VAE) {
    chunk <- latent_opt_sample(n_batch = 1500L)
    chunk <- chunk[!is.na(chunk) & nzchar(chunk)]
    gen_vae <- unique(c(gen_vae, chunk))
  }
  gen_vae <- gen_vae[seq_len(min(length(gen_vae), N_from_VAE))]
}

all_gen <- unique(c(gen_lm, gen_vae))
write_lines(all_gen, file.path(out_dir, "generated_raw.smi"))
cat("Generated total:", length(all_gen), "\n")

## =========================
## Validate + canonicalize + properties
## =========================
cat("Validating and computing properties...\n")
val_mols <- pblapply(all_gen, mol_of)
valid <- !vapply(val_mols, is.null, logical(1))
cat(sprintf("Parsed valid: %d / %d\n", sum(valid), length(all_gen)))

all_gen  <- all_gen[valid]
val_mols <- val_mols[valid]

cans <- vapply(val_mols, can_of, character(1))
props <- t(vapply(val_mols, props_of, numeric(6)))
colnames(props) <- c("MW","XlogP","RotB","HBD","HBA","TPSA")

dt <- as.data.table(props)
dt[, smiles := all_gen]
dt[, can := cans]
dt <- dt[!is.na(can) & nzchar(can)]
dt <- unique(dt, by = "can")
cat(sprintf("After canonical dedup: %d\n", nrow(dt)))

## Novelty
dt[, is_train := can %chin% train_can_vec]
cat(sprintf("Seen vs training: %d seen, %d novel\n", sum(dt$is_train), sum(!dt$is_train)))

## ADMET / PAINS / soft property window
dt[, admet_ok := admet_ok_vec(smiles)]
dt[, pains := vapply(smiles, is_pains, logical(1))]

if (identical(pains_mode, "hard")) {
  n_bad <- sum(dt$pains, na.rm = TRUE)
  if (n_bad > 0) message("PAINS hard filter removed: ", n_bad)
  dt <- dt[!pains]
}

in_soft_window <- function(MW, XlogP, RotB, TPSA) {
  ok_mw   <- is.na(MW)    | (MW >= 180 & MW <= 560)
  ok_logp <- is.na(XlogP) | (XlogP > -1.5 & XlogP < 6.0)
  ok_rotb <- is.na(RotB)  | (RotB <= 12)
  ok_tpsa <- is.na(TPSA)  | (TPSA <= 150)
  ok_mw & ok_logp & ok_rotb & ok_tpsa
}
dt[, soft_ok := in_soft_window(MW, XlogP, RotB, TPSA)]

## Primary scores
dt[, prop_s := vapply(smiles, druglike_score, numeric(1))]
dt[, thermo_s := mapply(
  thermo_proxy_score_from_props,
  MW, XlogP, RotB, HBD, HBA, TPSA
)]

score_base <- w_props * dt$prop_s + w_thermo * dt$thermo_s
score_novl <- ifelse(!dt$is_train, bonus_novel, -penal_seen * 0.5)
score_pain <- if (identical(pains_mode, "hard")) 0 else ifelse(dt$pains, -pains_penalty, 0)
score_soft <- ifelse(dt$soft_ok, 0.10, 0) + ifelse(dt$admet_ok, 0.05, 0)

dt[, score := score_base + score_novl + score_pain + score_soft]
setorder(dt, -score)

## =========================
## Candidate pool before scaffold/diversity
## =========================
pick_candidates <- function(D, target_n, keep_factor = 3L) {
  stopifnot(is.data.table(D))
  stages <- list(
    function(X) X,
    function(X) X[soft_ok == TRUE],
    function(X) X[admet_ok == TRUE],
    function(X) X[
      (is.na(MW)    | between(MW, 200, 520)) &
        (is.na(XlogP) | (XlogP > -1 & XlogP < 5)) &
        (is.na(RotB)  | RotB <= 10) &
        (is.na(TPSA)  | TPSA <= 120)
    ]
  )
  
  last_nonempty <- D
  for (i in seq_along(stages)) {
    Di <- stages[[i]](copy(D))
    cat(sprintf("Stage %d candidates: %d\n", i - 1L, nrow(Di)))
    if (nrow(Di) > 0) last_nonempty <- Di
    if (nrow(Di) >= target_n) {
      setorder(Di, -score)
      return(Di[seq_len(min(nrow(Di), target_n * keep_factor))])
    }
  }
  setorder(last_nonempty, -score)
  last_nonempty[seq_len(min(nrow(last_nonempty), target_n * keep_factor))]
}

cand_pool <- pick_candidates(copy(dt), target_n = prefilter_keep)

## =========================
## Fingerprints
## =========================
fp_bits_of <- function(s, fp_bits_local = 2048L) {
  tryCatch({
    m <- mol_of(s)
    if (is.null(m)) return(integer(0))
    fp <- get.fingerprint(m, type = "circular", depth = 6, size = fp_bits_local)
    bits <- fp@bits
    if (length(bits) == 0L) integer(0) else sort(unique(as.integer(bits)))
  }, error = function(e) integer(0))
}

tanimoto_bits <- function(a, b) {
  if (length(a) == 0L && length(b) == 0L) return(0)
  inter <- length(intersect(a, b))
  uni   <- length(unique(c(a, b)))
  if (uni == 0L) 0 else inter / uni
}

## =========================
## Bemis–Murcko scaffold extraction
## =========================
murcko_of_mol <- function(m, min_frag_size = 6L) {
  if (is.null(m)) return(unknown_scaffold_id)
  
  out <- tryCatch(
    get.murcko.fragments(
      m,
      min.frag.size = min_frag_size,
      as.smiles = TRUE,
      single.framework = TRUE
    ),
    error = function(e) NULL
  )
  if (is.null(out)) return(unknown_scaffold_id)
  
  fw <- NULL
  if (is.list(out) && !is.null(out$frameworks)) {
    fw <- out$frameworks
  } else if (is.list(out) && length(out) >= 1L && is.list(out[[1]]) && !is.null(out[[1]]$frameworks)) {
    fw <- out[[1]]$frameworks
  } else if (is.character(out)) {
    fw <- out
  }
  
  if (is.null(fw) || !length(fw)) return(unknown_scaffold_id)
  
  scaf <- fw[1]
  if (is.na(scaf) || !nzchar(scaf)) return(unknown_scaffold_id)
  
  sm <- can_of(mol_of(scaf))
  if (is.na(sm) || !nzchar(sm)) scaf else sm
}

murcko_of_smiles <- function(s, min_frag_size = 6L) {
  murcko_of_mol(mol_of(s), min_frag_size = min_frag_size)
}

cat("Computing Bemis–Murcko scaffolds on candidate pool...\n")
cand_pool[, murcko := vapply(smiles, murcko_of_smiles, character(1), min_frag_size = murcko_min_size)]
cand_pool[, scaffold_size := .N, by = murcko]

## =========================
## Scaffold-aware diversity pick
## =========================
scaffold_diverse_pick <- function(CAND,
                                  top_k,
                                  fp_bits_local = 2048L,
                                  lambda = 0.70,
                                  max_sim_allowed = 0.72,
                                  max_per_scaffold = 3L,
                                  first_pass_one_each = TRUE) {
  stopifnot(is.data.table(CAND))
  if (nrow(CAND) == 0L) return(CAND)
  
  CAND <- copy(CAND)
  setorder(CAND, -score)
  
  message("Building fingerprints for scaffold-aware selection...")
  fps <- lapply(CAND$smiles, fp_bits_of, fp_bits_local = fp_bits_local)
  ok <- lengths(fps) > 0L
  CAND <- CAND[ok]
  fps  <- fps[ok]
  if (nrow(CAND) == 0L) return(CAND)
  
  score_z <- as.numeric(scale(CAND$score))
  score_z[is.na(score_z)] <- 0
  
  CAND[, base_obj :=
         score_z +
         ifelse(!is_train, novelty_bonus2, 0) +
         ifelse(admet_ok, admet_bonus2, 0) +
         ifelse(soft_ok,  soft_bonus2, 0) -
         ifelse(pains,    0.10, 0)
  ]
  
  CAND[, base_obj := base_obj + scaffold_bonus / sqrt(pmax(1, scaffold_size))]
  
  n <- nrow(CAND)
  k <- min(as.integer(top_k), n)
  
  selected <- integer(0)
  remaining <- seq_len(n)
  scaffold_counts <- integer(0)
  names(scaffold_counts) <- character(0)
  max_sim_to_sel <- rep(0, n)
  
  first_idx <- which.max(CAND$base_obj)
  selected <- c(selected, first_idx)
  remaining <- setdiff(remaining, first_idx)
  scaffold_counts[CAND$murcko[first_idx]] <- 1L
  
  pb <- progress::progress_bar$new(
    total = max(1L, k - 1L),
    clear = FALSE,
    format = "Scaffold pick [:bar] :current/:total | selected=:sel scaffolds=:scaf | eta=:eta"
  )
  
  while (length(selected) < k && length(remaining) > 0L) {
    last_idx <- selected[length(selected)]
    sims <- vapply(
      remaining,
      function(j) tanimoto_bits(fps[[last_idx]], fps[[j]]),
      numeric(1)
    )
    max_sim_to_sel[remaining] <- pmax(max_sim_to_sel[remaining], sims)
    
    scaf_vec <- CAND$murcko[remaining]
    scaf_used <- ifelse(scaf_vec %in% names(scaffold_counts), scaffold_counts[scaf_vec], 0L)
    
    if (isTRUE(first_pass_one_each)) {
      unused_scaf <- remaining[scaf_used == 0L]
      allowed_scaf <- if (length(unused_scaf)) unused_scaf else remaining
    } else {
      allowed_scaf <- remaining
    }
    
    scaf_vec2 <- CAND$murcko[allowed_scaf]
    scaf_used2 <- ifelse(scaf_vec2 %in% names(scaffold_counts), scaffold_counts[scaf_vec2], 0L)
    allowed_scaf <- allowed_scaf[scaf_used2 < max_per_scaffold]
    if (!length(allowed_scaf)) break
    
    allowed_sim <- allowed_scaf[max_sim_to_sel[allowed_scaf] <= max_sim_allowed]
    if (!length(allowed_sim)) {
      allowed_sim <- allowed_scaf
    }
    
    obj <- CAND$base_obj[allowed_sim] - lambda * max_sim_to_sel[allowed_sim]
    next_idx <- allowed_sim[which.max(obj)]
    
    selected <- c(selected, next_idx)
    remaining <- setdiff(remaining, next_idx)
    
    sc <- CAND$murcko[next_idx]
    if (!(sc %in% names(scaffold_counts))) scaffold_counts[sc] <- 0L
    scaffold_counts[sc] <- scaffold_counts[sc] + 1L
    
    pb$tick(tokens = list(
      sel = length(selected),
      scaf = length(scaffold_counts)
    ))
  }
  
  out <- CAND[selected]
  setorder(out, -score)
  out
}

if (isTRUE(do_diversity)) {
  cand <- scaffold_diverse_pick(
    CAND                 = cand_pool,
    top_k                = top_hits,
    fp_bits_local        = fp_bits,
    lambda               = div_lambda,
    max_sim_allowed      = max_sim_allowed,
    max_per_scaffold     = max_per_scaffold,
    first_pass_one_each  = first_pass_one_each
  )
} else {
  cand <- head(cand_pool[order(-score)], min(top_hits, nrow(cand_pool)))
}

## =========================
## QC metrics
## =========================
calc_mean_sim_to_set <- function(smiles_vec, fp_bits_local = 2048L) {
  fps <- lapply(smiles_vec, fp_bits_of, fp_bits_local = fp_bits_local)
  n <- length(fps)
  out <- numeric(n)
  for (i in seq_len(n)) {
    sims <- vapply(seq_len(n), function(j) {
      if (i == j) return(NA_real_)
      tanimoto_bits(fps[[i]], fps[[j]])
    }, numeric(1))
    out[i] <- mean(sims, na.rm = TRUE)
  }
  out
}

cand[, mean_sim_selected := calc_mean_sim_to_set(smiles, fp_bits_local = fp_bits)]
cand[, scaffold_count_selected := .N, by = murcko]

## =========================
## STRICT STANDARDIZED PARENT IDENTITY
## =========================
cat("Computing standardized parent identities...\n")
cand[, parent_can := vapply(smiles, standardize_parent_smiles_strict, character(1))]
cand[, parent_inchikey := vapply(parent_can, standardized_inchikey_from_smiles, character(1), obabel_bin = obabel_bin)]

make_identity_key <- function(parent_inchikey, parent_can) {
  if (!is.na(parent_inchikey) && nzchar(parent_inchikey)) {
    return(paste0("INCHIKEY::", parent_inchikey))
  }
  if (!is.na(parent_can) && nzchar(parent_can)) {
    return(paste0("CAN::", parent_can))
  }
  NA_character_
}

cand[, identity_key := mapply(make_identity_key, parent_inchikey, parent_can, USE.NAMES = FALSE)]
cand <- cand[!is.na(identity_key) & nzchar(identity_key)]

## Deduplicate BEFORE 3D export
setorder(cand, -score)
cand_unique <- cand[!duplicated(identity_key)]
cand_dups   <- cand[duplicated(identity_key)]

cat("Candidates after strict standardized dedup:", nrow(cand_unique), "\n")
cat("Removed duplicates by standardized identity:", nrow(cand_dups), "\n")

cand <- cand_unique
cand[, id := paste0("GEN_", seq_len(.N))]

## Save duplicate map
dup_parent_tsv <- file.path(out_dir, "standardized_parent_duplicates.tsv")
if (nrow(cand_dups) > 0L) {
  kept_map <- cand_unique[, .(identity_key, kept_id = id, kept_smiles = smiles)]
  cand_dups2 <- merge(
    cand_dups[, .(smiles, parent_can, parent_inchikey, identity_key, score)],
    kept_map,
    by = "identity_key",
    all.x = TRUE,
    sort = FALSE
  )
  fwrite(cand_dups2, dup_parent_tsv, sep = "\t")
}

## =========================
## Export SMILES / score tables
## =========================
out_smi <- file.path(out_dir, "hits_for_docking.smi")
fwrite(cand[, .(smiles, id)], out_smi, sep = "\t", col.names = FALSE, quote = FALSE)

out_csv <- file.path(out_dir, "hits_for_docking_scores.csv")
fwrite(
  cand[, .(
    id, smiles, murcko, score, prop_s, thermo_s,
    is_train, admet_ok, pains, soft_ok,
    MW, XlogP, RotB, HBD, HBA, TPSA,
    mean_sim_selected, scaffold_count_selected,
    parent_can, parent_inchikey, identity_key
  )],
  out_csv
)

out_scaf <- file.path(out_dir, "scaffold_summary.csv")
fwrite(
  cand[, .(
    n_selected = .N,
    best_score = max(score, na.rm = TRUE),
    mean_score = mean(score, na.rm = TRUE)
  ), by = murcko][order(-n_selected, -best_score)],
  out_scaf
)

cat("\n=== Hybrid Results (FAST MODE, strict standardized unique) ===\n")
cat("Final molecules          :", nrow(cand), "\n")
cat("Unique Murcko scaffolds  :", uniqueN(cand$murcko), "\n")
cat("Unique parent canonical  :", uniqueN(cand$parent_can), "\n")
cat("Unique parent InChIKey   :", uniqueN(cand$parent_inchikey[!is.na(cand$parent_inchikey) & nzchar(cand$parent_inchikey)]), "\n")
cat("SMILES output            :", out_smi, "\n")
cat("Scores CSV               :", out_csv, "\n")
cat("Scaffold summary         :", out_scaf, "\n")
if (file.exists(dup_parent_tsv)) cat("Parent duplicate map      :", dup_parent_tsv, "\n")

## =========================
## Optional: 3D export (Open Babel)
## Now exports ONLY one representative per standardized parent identity
## =========================
if (is.null(obabel_bin) || !nzchar(obabel_bin) || !file.exists(obabel_bin)) {
  obabel_bin <- Sys.which("obabel")
}
if (use_obabel && !nzchar(obabel_bin)) {
  stop("Open Babel not found. Install with `brew install open-babel` and ensure `obabel` is on PATH.")
}

safe_md5 <- function(path) {
  if (!file.exists(path) || file.size(path) <= 0) return(NA_character_)
  as.character(tools::md5sum(path)[[1]])
}

prepared_can_from_sdf <- function(sdf_file) {
  if (!file.exists(sdf_file) || file.size(sdf_file) <= 0) return(NA_character_)
  mols <- tryCatch(rcdk::load.molecules(sdf_file), error = function(e) NULL)
  if (is.null(mols) || length(mols) < 1L || is.null(mols[[1]])) return(NA_character_)
  m <- configure_mol_safe(mols[[1]])
  can_of(m)
}

prepared_inchikey_from_sdf <- function(sdf_file, obabel_bin) {
  if (!file.exists(sdf_file) || file.size(sdf_file) <= 0) return(NA_character_)
  
  tmp_out <- tempfile(fileext = ".txt")
  on.exit(unlink(tmp_out, force = TRUE), add = TRUE)
  
  res <- tryCatch(
    system2(
      obabel_bin,
      args = c("-isdf", sdf_file, "-oinchikey", "-O", tmp_out),
      stdout = TRUE,
      stderr = TRUE,
      timeout = 120
    ),
    error = function(e) NULL
  )
  
  if (!file.exists(tmp_out) || file.size(tmp_out) <= 0) return(NA_character_)
  x <- tryCatch(readLines(tmp_out, warn = FALSE), error = function(e) character(0))
  x <- trimws(x)
  x <- x[nzchar(x)]
  if (!length(x)) return(NA_character_)
  x[1]
}

run_obabel_cmd <- function(args, timeout_sec = 180L) {
  out <- tryCatch(
    system2(
      obabel_bin,
      args = args,
      stdout = TRUE,
      stderr = TRUE,
      timeout = timeout_sec
    ),
    error = function(e) structure(conditionMessage(e), class = "try-error")
  )
  out
}

if (use_obabel) {
  final_molecules  <- nrow(cand)
  unique_scaffolds <- if ("murcko" %in% colnames(cand)) uniqueN(cand$murcko) else NA_integer_
  novel_molecules  <- if ("is_train" %in% colnames(cand)) sum(!cand$is_train, na.rm = TRUE) else NA_integer_
  
  cat("\n=== Export Summary ===\n")
  cat("Final molecules          :", final_molecules, "\n")
  cat("Unique Murcko scaffolds  :", unique_scaffolds, "\n")
  cat("Novel molecules          :", novel_molecules, "\n")
  
  sdf_dir        <- file.path(out_dir, "sdf")
  pdbqt_dir      <- file.path(out_dir, "pdbqt")
  uniq_pdbqt_dir <- file.path(out_dir, "prepared_unique_pdbqt")
  
  dir.create(sdf_dir,        showWarnings = FALSE, recursive = TRUE)
  dir.create(pdbqt_dir,      showWarnings = FALSE, recursive = TRUE)
  dir.create(uniq_pdbqt_dir, showWarnings = FALSE, recursive = TRUE)
  
  old_unique <- list.files(uniq_pdbqt_dir, full.names = TRUE)
  if (length(old_unique)) unlink(old_unique, force = TRUE)
  
  fast_3d        <- TRUE
  minimize_steps <- 100
  obabel_timeout <- 180L
  
  cat("Candidates before OBabel: ", final_molecules, " | novel=", novel_molecules, "\n")
  
  pb <- progress::progress_bar$new(
    total = nrow(cand),
    clear = FALSE,
    format = "OBabel export [:bar] :current/:total | :percent | eta=:eta"
  )
  
  res_list <- vector("list", nrow(cand))
  
  for (i in seq_len(nrow(cand))) {
    s  <- cand$smiles[i]
    id <- cand$id[i]
    
    smi_tmp <- tempfile(fileext = ".smi")
    writeLines(s, smi_tmp)
    
    sdf_out   <- file.path(sdf_dir,   paste0(id, ".sdf"))
    pdbqt_out <- file.path(pdbqt_dir, paste0(id, ".pdbqt"))
    
    flags <- c("-ismi", smi_tmp, "-osdf", "-p", as.character(pH), "--gen3d")
    if (!fast_3d) {
      flags <- c(flags, "--minimize")
    }
    
    ## 1) Generate SDF
    res1 <- run_obabel_cmd(c(flags, "-O", sdf_out), timeout_sec = obabel_timeout)
    ok1  <- file.exists(sdf_out) && file.size(sdf_out) > 0
    
    if (!ok1) {
      res_list[[i]] <- list(
        id = id,
        sdf = FALSE,
        pdbqt = FALSE,
        prepared_can = NA_character_,
        prepared_inchikey = NA_character_,
        sdf_md5 = NA_character_,
        pdbqt_md5 = NA_character_,
        err = if (inherits(res1, "try-error")) as.character(res1) else paste(res1, collapse = "\n")
      )
      unlink(smi_tmp, force = TRUE)
      pb$tick()
      next
    }
    
    ## 2) Diagnostics on prepared state
    prepared_can      <- prepared_can_from_sdf(sdf_out)
    prepared_inchikey <- prepared_inchikey_from_sdf(sdf_out, obabel_bin)
    
    ## 3) Convert to PDBQT
    res2 <- run_obabel_cmd(c("-isdf", sdf_out, "-opdbqt", "-O", pdbqt_out), timeout_sec = obabel_timeout)
    ok2  <- file.exists(pdbqt_out) && file.size(pdbqt_out) > 0
    
    res_list[[i]] <- list(
      id = id,
      sdf = ok1,
      pdbqt = ok2,
      prepared_can = prepared_can,
      prepared_inchikey = prepared_inchikey,
      sdf_md5 = safe_md5(sdf_out),
      pdbqt_md5 = if (ok2) safe_md5(pdbqt_out) else NA_character_,
      err = if (!ok2) {
        if (inherits(res2, "try-error")) as.character(res2) else paste(res2, collapse = "\n")
      } else {
        ""
      }
    )
    
    unlink(smi_tmp, force = TRUE)
    pb$tick()
  }
  
  res_dt <- data.table::rbindlist(lapply(res_list, as.data.table), fill = TRUE)
  
  cat(sprintf(
    "OBabel finished: SDF ok=%d fail=%d | PDBQT ok=%d fail=%d\n",
    sum(res_dt$sdf, na.rm = TRUE),
    sum(!res_dt$sdf, na.rm = TRUE),
    sum(res_dt$pdbqt, na.rm = TRUE),
    sum(!res_dt$pdbqt, na.rm = TRUE)
  ))
  
  prep_ok <- res_dt[sdf == TRUE & pdbqt == TRUE]
  
  if (nrow(prep_ok) > 0L) {
    ## Merge prepared diagnostics with pre-standardized parent identity
    prep_ok <- merge(
      prep_ok,
      cand[, .(id, smiles, parent_can, parent_inchikey, identity_key, murcko, score,
               prop_s, thermo_s, is_train, admet_ok, pains, soft_ok,
               MW, XlogP, RotB, HBD, HBA, TPSA)],
      by = "id",
      all.x = TRUE,
      sort = FALSE
    )
    
    ## Since cand was already deduplicated by strict parent identity, this should be unique.
    ## We keep a defensive second check here in case preparation collapses states unexpectedly.
    prep_ok[, prepared_identity_key := fifelse(
      !is.na(prepared_inchikey) & nzchar(prepared_inchikey),
      paste0("PREP_INCHIKEY::", prepared_inchikey),
      fifelse(
        !is.na(prepared_can) & nzchar(prepared_can),
        paste0("PREP_CAN::", prepared_can),
        paste0("PARENT::", identity_key)
      )
    )]
    
    setorder(prep_ok, -score)
    prep_ok[, is_prepared_duplicate := duplicated(prepared_identity_key)]
    prep_ok[, prepared_group := match(prepared_identity_key, unique(prepared_identity_key))]
    
    prepared_unique <- prep_ok[is_prepared_duplicate == FALSE]
    prepared_dups   <- prep_ok[is_prepared_duplicate == TRUE]
    
    prepared_unique_ligands  <- nrow(prepared_unique)
    unique_prepared_can      <- prepared_unique[!is.na(prepared_can) & nzchar(prepared_can), uniqueN(prepared_can)]
    unique_prepared_inchikey <- prepared_unique[!is.na(prepared_inchikey) & nzchar(prepared_inchikey), uniqueN(prepared_inchikey)]
    
    cat("Prepared unique ligands   :", prepared_unique_ligands, "\n")
    cat("Prepared duplicates       :", nrow(prepared_dups), "\n")
    cat("Unique prepared canonical :", unique_prepared_can, "\n")
    cat("Unique prepared InChIKey  :", unique_prepared_inchikey, "\n")
    
    dup_tsv <- file.path(out_dir, "prepared_ligand_duplicates.tsv")
    if (nrow(prepared_dups) > 0L) {
      first_ids <- prepared_unique[, .(prepared_identity_key, kept_id = id)]
      prepared_dups <- merge(
        prepared_dups,
        first_ids,
        by = "prepared_identity_key",
        all.x = TRUE,
        sort = FALSE
      )
      
      fwrite(
        prepared_dups[, .(
          id, kept_id, prepared_group,
          identity_key, prepared_identity_key,
          parent_inchikey, parent_can,
          prepared_inchikey, prepared_can,
          pdbqt_md5
        )],
        dup_tsv,
        sep = "\t"
      )
      cat("Prepared duplicate map    :", dup_tsv, "\n")
    }
    
    ## Copy unique PDBQT files
    for (ii in seq_len(nrow(prepared_unique))) {
      uid <- prepared_unique$id[ii]
      src_pdbqt <- file.path(pdbqt_dir, paste0(uid, ".pdbqt"))
      dst_pdbqt <- file.path(uniq_pdbqt_dir, paste0(uid, ".pdbqt"))
      if (file.exists(src_pdbqt)) file.copy(src_pdbqt, dst_pdbqt, overwrite = TRUE)
    }
    
    prep_manifest <- file.path(out_dir, "prepared_unique_for_docking.csv")
    fwrite(
      prepared_unique[, .(
        id, smiles, murcko, score, prop_s, thermo_s,
        is_train, admet_ok, pains, soft_ok,
        MW, XlogP, RotB, HBD, HBA, TPSA,
        parent_can, parent_inchikey, identity_key,
        prepared_can, prepared_inchikey, prepared_identity_key,
        pdbqt_md5
      )][order(-score)],
      prep_manifest
    )
    
    prep_ids_txt <- file.path(out_dir, "prepared_unique_ids.txt")
    writeLines(prepared_unique$id, prep_ids_txt)
    
  } else {
    prepared_unique_ligands  <- 0L
    unique_prepared_can      <- 0L
    unique_prepared_inchikey <- 0L
    
    prep_manifest <- file.path(out_dir, "prepared_unique_for_docking.csv")
    fwrite(data.table(), prep_manifest)
    
    prep_ids_txt <- file.path(out_dir, "prepared_unique_ids.txt")
    writeLines(character(0), prep_ids_txt)
  }
  
  if ("murcko" %in% colnames(cand)) {
    murcko_csv <- file.path(out_dir, "murcko_scaffold_counts.csv")
    fwrite(
      cand[, .N, by = murcko][order(-N)],
      murcko_csv
    )
    cat("Murcko scaffold counts saved:", murcko_csv, "\n")
  }
  
  summary_txt <- file.path(out_dir, "export_summary.txt")
  writeLines(
    c(
      "=== Export Summary ===",
      paste("Final molecules               :", final_molecules),
      paste("Unique Murcko scaffolds       :", unique_scaffolds),
      paste("Novel molecules               :", novel_molecules),
      paste("Prepared unique ligands       :", prepared_unique_ligands),
      paste("Unique prepared canonical     :", unique_prepared_can),
      paste("Unique prepared InChIKey      :", unique_prepared_inchikey),
      paste("SDF directory                 :", sdf_dir),
      paste("PDBQT directory               :", pdbqt_dir),
      paste("Unique PDBQT directory        :", uniq_pdbqt_dir)
    ),
    con = summary_txt
  )
  
  summary_csv <- file.path(out_dir, "export_summary.csv")
  fwrite(
    data.table(
      final_molecules = final_molecules,
      unique_murcko_scaffolds = unique_scaffolds,
      novel_molecules = novel_molecules,
      prepared_unique_ligands = prepared_unique_ligands,
      unique_prepared_canonical = unique_prepared_can,
      unique_prepared_inchikey = unique_prepared_inchikey
    ),
    summary_csv
  )
  
  if (any(!res_dt$sdf, na.rm = TRUE) || any(!res_dt$pdbqt, na.rm = TRUE)) {
    fwrite(
      res_dt[!(sdf & pdbqt)][, .(
        id, sdf, pdbqt, prepared_can, prepared_inchikey, pdbqt_md5, err
      )],
      file.path(out_dir, "obabel_errors.tsv"),
      sep = "\t"
    )
    message("Wrote failures to ", file.path(out_dir, "obabel_errors.tsv"))
  }
  
  cat("Export summary text saved :", summary_txt, "\n")
  cat("Export summary csv saved  :", summary_csv, "\n")
  cat("Prepared unique manifest  :", prep_manifest, "\n")
  cat("Prepared unique IDs       :", prep_ids_txt, "\n")
  cat("Unique PDBQT dir          :", uniq_pdbqt_dir, "\n")
}