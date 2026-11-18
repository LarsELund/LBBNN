#!/usr/bin/env Rscript
# =============================================================================
# Benchmark 5 (Section 3.6): Multivariate 3D Dynamical System Discovery (SINDy)
# 3D Rössler-Lorenz Hybrid ODE System (extending Hubin, 2025, arXiv:2507.06776):
#   dx1/dt = -x2 - x3 + 1.5 * sin(x1) + eps1
#   dx2/dt =  x1 + 0.2 * x2 + eps2
#   dx3/dt =  0.4 + x1 * x3 - 5.7 * x3 + eps3
# Embedded in p = 6 state/noise covariates (x4, x5, x6 pure noise) across M = 100 seeds.
# =============================================================================

suppressPackageStartupMessages({
  pkgload::load_all(".", quiet = TRUE)
  library(FBMS)
  library(glmnet)
  library(earth)
  library(mgcv)
  library(ranger)
})

if (!exists("fit_sr_pareto_r", mode = "function")) {
  source("inst/benchmarks/bench_baselines.R")
}

generate_sindy3d_data <- function(n = 500L, n_test = 250L, p = 6L, sigma = 0.05, seed = 1L) {
  set.seed(seed)
  n_tot <- n + n_test
  X_all <- matrix(stats::runif(n_tot * p, min = -pi, max = pi), nrow = n_tot, ncol = p)
  var_names <- paste0("x", seq_len(p))
  colnames(X_all) <- var_names

  y1_clean <- -X_all[, 2] - X_all[, 3] + 1.5 * sin(X_all[, 1])
  y2_clean <- X_all[, 1] + 0.2 * X_all[, 2]
  y3_clean <- 0.4 + X_all[, 1] * X_all[, 3] - 5.7 * X_all[, 3]
  Y_clean <- cbind(dx1 = y1_clean, dx2 = y2_clean, dx3 = y3_clean)

  Y_obs <- Y_clean + matrix(stats::rnorm(n_tot * 3L, mean = 0.0, sd = sigma), nrow = n_tot, ncol = 3L)

  idx_tr <- seq_len(n)
  idx_te <- seq.int(n + 1L, n_tot)
  list(
    x_train = X_all[idx_tr, , drop = FALSE],
    y_train = Y_obs[idx_tr, , drop = FALSE],
    y_clean_train = Y_clean[idx_tr, , drop = FALSE],
    x_test = X_all[idx_te, , drop = FALSE],
    y_test = Y_obs[idx_te, , drop = FALSE],
    y_clean_test = Y_clean[idx_te, , drop = FALSE],
    var_names = var_names,
    true_support = list(c(1L, 2L, 3L), c(1L, 2L), c(1L, 3L))
  )
}

extract_vars_from_formula <- function(f_str, p = 6L) {
  if (is.na(f_str) || !nzchar(f_str)) return(integer(0))
  hits <- integer(0)
  for (j in seq_len(p)) {
    pat <- paste0("(^|[^a-zA-Z0-9_])x", j, "([^0-9]|$)")
    if (grepl(pat, f_str)) hits <- c(hits, j)
  }
  sort(unique(hits))
}

score_sindy3d_method <- function(seed, method, S_hat, pred_test, dat, n_terms, formula_str,
                                 is_symbolic = TRUE) {
  p <- ncol(dat$x_train)
  true_S <- dat$true_support
  tp <- 0L
  fp <- 0L
  for (r in 1:3) {
    s_r <- sort(unique(as.integer(S_hat[[r]])))
    tp <- tp + sum(s_r %in% true_S[[r]])
    fp <- fp + sum(!(s_r %in% true_S[[r]]))
  }
  var_tpr <- tp / 7.0
  var_fpr <- fp / 11.0
  struct_rec <- as.integer(tp == 7L && fp == 0L)

  # Clean noiseless R^2 across the 3 ODEs to verify functional exactness (not linear surrogate)
  r2_clean_vec <- vapply(1:3, function(r) {
    c_val <- suppressWarnings(stats::cor(pred_test[, r], dat$y_clean_test[, r]))
    if (!is.finite(c_val)) 0.0 else c_val^2
  }, numeric(1))
  exact_rec <- as.integer(isTRUE(is_symbolic) && struct_rec == 1L && all(r2_clean_vec > 0.9985))

  test_rmse <- sqrt(mean((pred_test - dat$y_test)^2))
  r2_obs_vec <- vapply(1:3, function(r) {
    c_val <- suppressWarnings(stats::cor(pred_test[, r], dat$y_test[, r]))
    if (!is.finite(c_val)) 1e-6 else max(1e-6, min(1 - 1e-6, c_val^2))
  }, numeric(1))

  data.frame(
    seed = seed,
    benchmark = "sindy_3d",
    method = method,
    structural_recovery = struct_rec,
    exact_recovery = exact_rec,
    var_tpr = var_tpr,
    var_fpr = var_fpr,
    n_terms = n_terms,
    test_rmse = test_rmse,
    test_r2 = mean(r2_obs_vec),
    rmse_eq1 = sqrt(mean((pred_test[, 1] - dat$y_test[, 1])^2)),
    rmse_eq2 = sqrt(mean((pred_test[, 2] - dat$y_test[, 2])^2)),
    rmse_eq3 = sqrt(mean((pred_test[, 3] - dat$y_test[, 3])^2)),
    formula = paste(formula_str, collapse = " || "),
    stringsAsFactors = FALSE
  )
}

run_single_sindy3d_seed <- function(seed = 1L,
                                    n = 500L,
                                    n_test = 250L,
                                    p = 6L,
                                    sigma = 0.05,
                                    epochs = 350L,
                                    debias_epochs = 250L,
                                    fbms_cores = getOption("lbbnn.fbms_cores", 1L),
                                    fbms_runs = getOption("lbbnn.fbms_runs", 24L),
                                    device = get_compute_device(),
                                    verbose = FALSE) {
  dat <- generate_sindy3d_data(n = n, n_test = n_test, p = p, sigma = sigma, seed = seed)
  X_tr <- dat$x_train; Y_tr <- dat$y_train
  X_te <- dat$x_test;  Y_te <- dat$y_test
  var_names <- dat$var_names

  rows <- list()

  # --- 1. Joint Multivariate Symbolic-LBBNN (q = 3 shared hidden + Input-Skip readout) ---
  set.seed(seed)
  torch::torch_manual_seed(seed)
  x_tr_t <- torch::torch_tensor(X_tr, dtype = torch::torch_float32(), device = device)
  y_tr_t <- torch::torch_tensor(Y_tr, dtype = torch::torch_float32(), device = device)
  x_te_t <- torch::torch_tensor(X_te, dtype = torch::torch_float32(), device = device)
  train_ds <- torch::tensor_dataset(x_tr_t, y_tr_t)
  train_dl <- torch::dataloader(train_ds, batch_size = 128L, shuffle = TRUE)

  acts <- list(
    list(clog = p, id = 4L, sin = 4L, square = 6L),
    list(exp = 3L, id = 4L, square = 6L, sin = 2L)
  )
  sizes <- c(p, 20L, 15L, 3L)
  priors <- compute_symbolic_priors(
    sizes = sizes, layer_activations_list = acts,
    base_prior = 0.20, depth_decay = 0.4, complexity_weight = 0.35, input_skip = TRUE
  )
  inc_init <- matrix(rep(c(-0.5, 1.5), 3), nrow = 2, ncol = 3)
  sym_mod <- lbbnn_symbolic_net(
    problem_type = "regression", sizes = sizes, prior = priors, std = c(1, 1, 1),
    inclusion_inits = inc_init, device = device, activations = acts
  )
  train_symbolic_lbbnn(
    epochs = epochs, LBBNN = sym_mod, lr = 0.025, train_dl = train_dl, device = device,
    debias_epochs = debias_epochs, debias_lr = 0.01,
    snap_rationals = TRUE, prune_bic = TRUE, prune_readout = TRUE, verbose = verbose
  )
  sym_ext <- extract_symbolic_mpm(sym_mod, var_names = var_names, digits = 3, simplify = TRUE)
  torch::with_no_grad({
    pred_sym <- as.matrix(sym_mod(x_te_t, MPM = TRUE)$cpu())
  })
  S_sym <- lapply(1:3, function(r) extract_vars_from_formula(sym_ext$formula_str[r], p = p))
  # Active terms across all 3 readout equations (edges on active paths)
  n_terms_sym <- nrow(sym_ext$active_paths_df)
  rows[[length(rows) + 1L]] <- score_sindy3d_method(
    seed = seed, method = "Symbolic-LBBNN", S_hat = S_sym,
    pred_test = pred_sym, dat = dat, n_terms = n_terms_sym,
    formula_str = sym_ext$formula_str, is_symbolic = TRUE
  )

  # --- 2. Joint Multivariate ReLU-LBBNN (q = 3, activations = NULL) ---
  set.seed(seed)
  torch::torch_manual_seed(seed)
  relu_mod <- lbbnn_net(
    problem_type = "regression", sizes = sizes, prior = c(0.25, 0.25, 0.25),
    std = c(1, 1, 1), inclusion_inits = inc_init, input_skip = TRUE,
    flow = FALSE, device = device, activations = NULL
  )
  suppressMessages(
    train_lbbnn(epochs = min(epochs, 80L), LBBNN = relu_mod, lr = 0.01, train_dl = train_dl, device = device)
  )
  relu_mod$compute_paths_input_skip()
  torch::with_no_grad({
    pred_relu <- as.matrix(relu_mod(x_te_t, MPM = TRUE)$cpu())
  })
  # Extract active input variables feeding into each output node via active-path matrices
  a1 <- as.matrix(relu_mod$layers$children[[1]]$alpha_active_path$cpu()) > 0.5
  a2 <- as.matrix(relu_mod$layers$children[[2]]$alpha_active_path$cpu()) > 0.5
  ao <- as.matrix(relu_mod$out_layer$alpha_active_path$cpu()) > 0.5
  reach1 <- a1
  reach2 <- (a2[, 1:20, drop = FALSE] %*% reach1 + a2[, 21:(20 + p), drop = FALSE]) > 0
  S_relu <- lapply(1:3, function(r) {
    v_reach <- as.numeric(ao[r, 1:15, drop = FALSE] %*% reach2 + ao[r, 16:(15 + p), drop = FALSE]) > 0
    which(v_reach)
  })
  n_terms_relu <- sum(a1) + sum(a2) + sum(ao)
  rows[[length(rows) + 1L]] <- score_sindy3d_method(
    seed = seed, method = "ReLU-LBBNN", S_hat = S_relu,
    pred_test = pred_relu, dat = dat, n_terms = n_terms_relu,
    formula_str = rep("ReLU-LBBNN", 3), is_symbolic = FALSE
  )

  # --- 3. FBMS-BGNLM (parallel GMJMCMC per ODE under top-performing ZS-null prior) ---
  register_fbms_custom_transforms()
  tr_sindy <- c("sin_rad", "cos_rad", "p2", "p3")
  probs_sindy <- FBMS::gen.probs.gmjmcmc(tr_sindy)
  probs_sindy$gen <- c(0.40, 0.45, 0.05, 0.10)
  params_sindy <- FBMS::gen.params.gmjmcmc(p)
  params_sindy$feat$pop.max <- 25L

  pred_fbms <- matrix(0.0, nrow = nrow(X_te), ncol = 3L)
  S_fbms <- vector("list", 3L)
  f_fbms <- character(3L)
  n_terms_fbms <- 0L
  fbms_ok <- TRUE
  for (r in 1:3) {
    df_r <- data.frame(y = Y_tr[, r], X_tr)
    res_r <- run_fbms_jcs_parallel(
      df_tr = df_r, X_tr = X_tr, X_te = X_te, y_tr = Y_tr[, r], y_te = Y_te[, r],
      var_names = var_names, transforms = tr_sindy, beta_prior_type = "ZS-null",
      beta_prior = list(type = "ZS-null"),
      probs = probs_sindy, params = params_sindy,
      seed = seed * 10L + r, cores = fbms_cores, runs = fbms_runs, P = 15L, N = 120L
    )
    if (is.null(res_r)) {
      fbms_ok <- FALSE; break
    }
    pred_fbms[, r] <- res_r$pred
    S_fbms[[r]] <- if (!is.null(res_r$v_used)) sort(unique(as.integer(res_r$v_used))) else integer(0)
    f_fbms[r] <- res_r$formula
    n_terms_fbms <- n_terms_fbms + length(res_r$f_strs)
  }
  if (fbms_ok) {
    rows[[length(rows) + 1L]] <- score_sindy3d_method(
      seed = seed, method = "FBMS-BGNLM", S_hat = S_fbms,
      pred_test = pred_fbms, dat = dat, n_terms = n_terms_fbms,
      formula_str = f_fbms, is_symbolic = TRUE
    )
  }

  # --- 4 & 5. SR-PySR and SR-Operon (Pareto / MDL symbolic regression per ODE) ---
  pred_pysr <- matrix(0.0, nrow = nrow(X_te), ncol = 3L)
  pred_oper <- matrix(0.0, nrow = nrow(X_te), ncol = 3L)
  S_pysr <- vector("list", 3L); S_oper <- vector("list", 3L)
  f_pysr <- character(3L); f_oper <- character(3L)
  nt_pysr <- 0L; nt_oper <- 0L
  for (r in 1:3) {
    sr_r <- run_python_sr_pair(X_tr, Y_tr[, r], X_te, Y_te[, r], var_names, "sindy_3d", seed = seed * 10L + r)
    # Re-evaluate Pareto selected formula or recover prediction directly from fit_sr_pareto_r?
    # Note: fit_sr_pareto_r computes pred_te internally; let's evaluate formula string cleanly on X_te:
    eval_sr_f <- function(f_s, X_m) {
      env_s <- list2env(as.list(as.data.frame(X_m)), parent = .GlobalEnv)
      out <- tryCatch(as.numeric(eval(parse(text = f_s), envir = env_s)), error = function(e) rep(0.0, nrow(X_m)))
      if (length(out) == 1L) out <- rep(out, nrow(X_m))
      out[!is.finite(out)] <- 0.0
      out
    }
    i_py <- which(sr_r$method == "SR_PySR")[1]
    i_op <- which(sr_r$method == "SR_Operon")[1]
    f_pysr[r] <- sr_r$formula[i_py]; f_oper[r] <- sr_r$formula[i_op]
    S_pysr[[r]] <- sr_r$vars_idx[[i_py]]; S_oper[[r]] <- sr_r$vars_idx[[i_op]]
    nt_pysr <- nt_pysr + as.integer(sr_r$complexity[i_py])
    nt_oper <- nt_oper + as.integer(sr_r$complexity[i_op])
    pred_pysr[, r] <- eval_sr_f(f_pysr[r], X_te)
    pred_oper[, r] <- eval_sr_f(f_oper[r], X_te)
  }
  rows[[length(rows) + 1L]] <- score_sindy3d_method(
    seed = seed, method = "SR-PySR", S_hat = S_pysr, pred_test = pred_pysr,
    dat = dat, n_terms = nt_pysr, formula_str = f_pysr, is_symbolic = TRUE
  )
  rows[[length(rows) + 1L]] <- score_sindy3d_method(
    seed = seed, method = "SR-Operon", S_hat = S_oper, pred_test = pred_oper,
    dat = dat, n_terms = nt_oper, formula_str = f_oper, is_symbolic = TRUE
  )

  # --- 6. MARS (earth degree = 2 per ODE) ---
  pred_mars <- matrix(0.0, nrow = nrow(X_te), ncol = 3L)
  S_mars <- vector("list", 3L)
  nt_mars <- 0L
  for (r in 1:3) {
    set.seed(seed * 10L + r)
    mf <- earth::earth(x = X_tr, y = Y_tr[, r], degree = 2L)
    pred_mars[, r] <- as.numeric(predict(mf, newdata = X_te))
    ev <- earth::evimp(mf)
    S_mars[[r]] <- if (!is.null(ev) && nrow(ev) > 0L) as.integer(ev[, "col"]) else integer(0)
    nt_mars <- nt_mars + nrow(mf$dirs)
  }
  rows[[length(rows) + 1L]] <- score_sindy3d_method(
    seed = seed, method = "MARS", S_hat = S_mars, pred_test = pred_mars,
    dat = dat, n_terms = nt_mars, formula_str = rep("MARS(deg=2)", 3), is_symbolic = FALSE
  )

  # --- 7. GAM (mgcv::gam select=TRUE per ODE) ---
  pred_gam <- matrix(0.0, nrow = nrow(X_te), ncol = 3L)
  S_gam <- vector("list", 3L)
  nt_gam <- 0L
  gam_form <- as.formula(paste("y ~", paste(sprintf("s(%s, k = 6)", var_names), collapse = " + ")))
  for (r in 1:3) {
    gf <- mgcv::gam(gam_form, data = data.frame(y = Y_tr[, r], X_tr), select = TRUE, method = "REML")
    pred_gam[, r] <- as.numeric(predict(gf, newdata = as.data.frame(X_te)))
    edfs <- summary(gf)$edf
    S_gam[[r]] <- which(edfs > 0.5)
    nt_gam <- nt_gam + round(sum(edfs))
  }
  rows[[length(rows) + 1L]] <- score_sindy3d_method(
    seed = seed, method = "GAM", S_hat = S_gam, pred_test = pred_gam,
    dat = dat, n_terms = nt_gam, formula_str = rep("GAM(thin-plate)", 3), is_symbolic = FALSE
  )

  # --- 8. Lasso (cv.glmnet per ODE) ---
  pred_las <- matrix(0.0, nrow = nrow(X_te), ncol = 3L)
  S_las <- vector("list", 3L)
  nt_las <- 0L
  for (r in 1:3) {
    set.seed(seed * 10L + r)
    cl <- glmnet::cv.glmnet(X_tr, Y_tr[, r], alpha = 1.0, nfolds = 5L)
    pred_las[, r] <- as.numeric(predict(cl, newx = X_te, s = "lambda.1se"))
    wl <- as.numeric(coef(cl, s = "lambda.1se"))[-1]
    S_las[[r]] <- which(abs(wl) > 1e-3)
    nt_las <- nt_las + length(S_las[[r]])
  }
  rows[[length(rows) + 1L]] <- score_sindy3d_method(
    seed = seed, method = "Lasso", S_hat = S_las, pred_test = pred_las,
    dat = dat, n_terms = nt_las, formula_str = rep("Lasso(linear)", 3), is_symbolic = TRUE
  )

  # --- 9. RandomForest (ranger per ODE) ---
  pred_rf <- matrix(0.0, nrow = nrow(X_te), ncol = 3L)
  S_rf <- vector("list", 3L)
  for (r in 1:3) {
    rf <- ranger::ranger(y ~ ., data = data.frame(y = Y_tr[, r], X_tr), num.trees = 300L,
                         importance = "permutation", seed = seed * 10L + r)
    pred_rf[, r] <- ranger::predictions(predict(rf, data = as.data.frame(X_te)))
    imp <- rf$variable.importance
    S_rf[[r]] <- which(imp > 0.10 * max(imp))
  }
  rows[[length(rows) + 1L]] <- score_sindy3d_method(
    seed = seed, method = "RandomForest", S_hat = S_rf, pred_test = pred_rf,
    dat = dat, n_terms = 900L, formula_str = rep("RandomForest(300x3)", 3), is_symbolic = FALSE
  )

  do.call(rbind, rows)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  n_seeds <- if (length(args) >= 1L) as.integer(args[1]) else 100L
  out_dir <- "results_baselines/sindy_3d"
  ckpt_dir <- file.path(out_dir, "checkpoints")
  dir.create(ckpt_dir, recursive = TRUE, showWarnings = FALSE)

  n_workers <- if (n_seeds <= 2L) 1L else max(1L, min(8L, parallel::detectCores(logical = FALSE) - 1L))
  fbms_c <- if (n_workers == 1L) max(1L, min(8L, parallel::detectCores(logical = FALSE) - 1L)) else 1L
  message(sprintf("Running Benchmark 5 (3D SINDy) across M = %d seeds (%d parallel workers, fbms_cores = %d)...",
                  n_seeds, n_workers, fbms_c))

  run_worker <- function(s) {
    torch::torch_set_num_threads(1L)
    ckpt_f <- file.path(ckpt_dir, sprintf("seed_%04d.rds", s))
    if (file.exists(ckpt_f)) return(readRDS(ckpt_f))
    t0 <- proc.time()[3]
    res_s <- run_single_sindy3d_seed(
      seed = s, epochs = 120L, debias_epochs = 60L,
      fbms_cores = fbms_c, fbms_runs = 8L, device = "cpu", verbose = FALSE
    )
    saveRDS(res_s, ckpt_f)
    message(sprintf("  [SINDy 3D] Seed %d/%d finished in %.1fs | Sym-LBBNN RMSE=%.4f Exact=%d | FBMS RMSE=%.4f Exact=%d",
                    s, n_seeds, proc.time()[3] - t0,
                    res_s$test_rmse[res_s$method == "Symbolic-LBBNN"],
                    res_s$exact_recovery[res_s$method == "Symbolic-LBBNN"],
                    res_s$test_rmse[res_s$method == "FBMS-BGNLM"],
                    res_s$exact_recovery[res_s$method == "FBMS-BGNLM"]))
    res_s
  }

  if (n_workers == 1L) {
    all_rows <- lapply(seq_len(n_seeds), run_worker)
  } else {
    cl <- parallel::makePSOCKcluster(n_workers)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterExport(cl, c("ckpt_dir", "n_seeds", "fbms_c"), envir = environment())
    parallel::clusterEvalQ(cl, {
      suppressPackageStartupMessages({
        pkgload::load_all(".", quiet = TRUE)
        library(FBMS); library(glmnet); library(earth); library(mgcv); library(ranger)
      })
      torch::torch_set_num_threads(1L)
      source("inst/benchmarks/bench_baselines.R")
      source("inst/benchmarks/bench_sindy_3d.R")
      NULL
    })
    all_rows <- parallel::parLapplyLB(cl, seq_len(n_seeds), run_worker)
  }
  raw_df <- do.call(rbind, all_rows)
  utils::write.csv(raw_df, file.path(out_dir, "raw_sindy_3d_baselines.csv"), row.names = FALSE)

  methods <- unique(raw_df$method)
  sum_df <- do.call(rbind, lapply(methods, function(m) {
    sub <- raw_df[raw_df$method == m, , drop = FALSE]
    data.frame(
      benchmark = "sindy_3d",
      method = m,
      n_seeds = nrow(sub),
      structural_recovery_mean = mean(sub$structural_recovery),
      structural_recovery_sd = stats::sd(sub$structural_recovery),
      exact_recovery_mean = mean(sub$exact_recovery),
      exact_recovery_sd = stats::sd(sub$exact_recovery),
      var_tpr_mean = mean(sub$var_tpr),
      var_tpr_sd = stats::sd(sub$var_tpr),
      var_fpr_mean = mean(sub$var_fpr),
      var_fpr_sd = stats::sd(sub$var_fpr),
      n_terms_mean = mean(sub$n_terms),
      n_terms_sd = stats::sd(sub$n_terms),
      test_rmse_mean = mean(sub$test_rmse),
      test_rmse_sd = stats::sd(sub$test_rmse),
      test_r2_mean = mean(sub$test_r2),
      test_r2_sd = stats::sd(sub$test_r2),
      stringsAsFactors = FALSE
    )
  }))
  utils::write.csv(sum_df, file.path(out_dir, "summary_sindy_3d_baselines.csv"), row.names = FALSE)
  print(sum_df[, c("method", "structural_recovery_mean", "exact_recovery_mean", "var_tpr_mean", "var_fpr_mean", "n_terms_mean", "test_rmse_mean", "test_r2_mean")])
}
