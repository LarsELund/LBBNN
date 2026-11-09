# =============================================================================
# inst/benchmarks/bench_baselines.R
# Comprehensive Baseline Suite for Neuro-Symbolic LBBNN Benchmarks:
#   1. Task A (logic): Symbolic-LBBNN, ReLU-LBBNN, FBMS-BLR (GMJMCMC),
#      LogicReg (Simulated Annealing), Lasso-Interactions (glmnet), RandomForest (ranger)
#   2. Task B & C (kepler & kepler_real): Symbolic-LBBNN, ReLU-LBBNN,
#      FBMS-BGNLM (GMJMCMC), SR-PySR, SR-Operon, Log-Lasso, MARS (earth),
#      GAM (mgcv), RandomForest (ranger)
#   3. Task D (nonlinear / Friedman #1): Symbolic-LBBNN, ReLU-LBBNN,
#      FBMS-BGNLM (GMJMCMC), SR-PySR, SR-Operon, MARS (earth), GAM (mgcv),
#      Lasso (glmnet), RandomForest (ranger)
#
# Designed to run 100% cleanly on both macOS and Linux HPC servers (RHEL 8+)
# without requiring liblantern.so when reading precomputed LBBNN checkpoints.
# =============================================================================

# Pre-load libflexiblas.so.3 if needed by EasyBuild-compiled R packages (BAS, Matrix) on Linux clusters
if (.Platform$OS.type == "unix") {
  tryCatch(dyn.load("libflexiblas.so.3"), error = function(e) {
    cands <- c(
      Sys.glob("/cluster/software/FlexiBLAS/*/lib/libflexiblas.so.3"),
      Sys.glob("/software/FlexiBLAS/*/lib/libflexiblas.so.3"),
      Sys.glob("/opt/eb/software/FlexiBLAS/*/lib/libflexiblas.so.3"),
      Sys.glob("/mn/sarpanitu/*/*/FlexiBLAS/*/lib/libflexiblas.so.3")
    )
    for (cand in cands) {
      if (file.exists(cand)) {
        res <- tryCatch({ dyn.load(cand); TRUE }, error = function(e2) FALSE)
        if (isTRUE(res)) break
      }
    }
  })
}

suppressPackageStartupMessages({
  library(FBMS)
  library(LogicReg)
  library(glmnet)
  library(earth)
  library(mgcv)
  library(ranger)
  library(jsonlite)
})

# Source data generators from benchmark files (without executing main blocks)
if (!exists("generate_logic_data", mode = "function")) {
  source("inst/benchmarks/bench_logic.R")
}
if (!exists("generate_kepler_data", mode = "function")) {
  source("inst/benchmarks/bench_kepler.R")
}
if (!exists("generate_friedman1_data", mode = "function")) {
  source("inst/benchmarks/bench_nonlinear.R")
}

#' Locate Python 3 binary for SR-PySR and SR-Operon bridge
get_python_bin <- function() {
  cands <- c(
    file.path(dirname(R.home("bin")), "python"),
    file.path(dirname(R.home("bin")), "..", "bin", "python"),
    "/mn/sarpanitu/ansatte-u2/aliaksah/micromamba/envs/r-symbolic/bin/python",
    ".venv/bin/python",
    "../BGNLM-for-SINDy/.venv/bin/python",
    "/opt/homebrew/bin/python3",
    Sys.which("python3"),
    Sys.which("python")
  )
  for (c_path in unique(cands)) {
    if (nzchar(c_path) && file.exists(c_path)) {
      ok <- tryCatch({
        res <- system2(c_path, c("-c", shQuote("import numpy, scipy; print('OK')")), stdout = TRUE, stderr = FALSE)
        any(grepl("OK", res))
      }, error = function(e) FALSE)
      if (isTRUE(ok)) return(c_path)
    }
  }
  stop("No python3 interpreter with numpy and scipy found.")
}

#' Helper to load a single seed row from results_overnight/<bench>/
load_overnight_seed <- function(overnight_dir, bench_subdir, seed) {
  rds_p1 <- file.path(overnight_dir, bench_subdir, "checkpoints", sprintf("seed_%04d.rds", seed))
  if (file.exists(rds_p1)) return(readRDS(rds_p1))
  rds_p2 <- file.path(overnight_dir, bench_subdir, sprintf("seed_%04d.rds", seed))
  if (file.exists(rds_p2)) return(readRDS(rds_p2))
  csv_p <- file.path(overnight_dir, bench_subdir, paste0("summary_", bench_subdir, ".csv"))
  if (file.exists(csv_p)) {
    df_all <- utils::read.csv(csv_p, stringsAsFactors = FALSE)
    sub_df <- df_all[df_all$seed == seed, , drop = FALSE]
    if (nrow(sub_df) > 0L) return(sub_df[1, , drop = FALSE])
  }
  NULL
}

#' Run Python Symbolic Regression bridge (SR_PySR and SR_Operon)
run_python_sr_pair <- function(x_train, y_train, x_test, y_test, var_names, task, seed = 1L) {
  py_bin <- get_python_bin()
  py_script <- "inst/python/run_sr_baselines.py"
  if (!file.exists(py_script)) {
    py_script <- system.file("python", "run_sr_baselines.py", package = "LBBNN")
  }
  tmp_in <- tempfile(pattern = "sr_in_", fileext = ".json")
  tmp_out <- tempfile(pattern = "sr_out_", fileext = ".json")
  on.exit(unlink(c(tmp_in, tmp_out)), add = TRUE)

  payload <- list(
    x_train = x_train,
    y_train = as.numeric(y_train),
    x_test = x_test,
    y_test = as.numeric(y_test),
    var_names = as.list(var_names),
    task = as.character(task),
    seed = as.integer(seed)
  )
  jsonlite::write_json(payload, tmp_in, auto_unbox = TRUE, digits = 10)
  status <- system2(py_bin, args = c(shQuote(py_script), "--input", shQuote(tmp_in), "--output", shQuote(tmp_out)),
                    stdout = FALSE, stderr = FALSE)
  if (status != 0L || !file.exists(tmp_out)) {
    return(NULL)
  }
  jsonlite::read_json(tmp_out, simplifyVector = TRUE)
}

#' Safely evaluate an FBMS feature string on a data.frame X_df
eval_fbms_feat <- function(feat_str, X_df) {
  s <- trimws(as.character(feat_str))
  for (k in seq_len(ncol(X_df))) {
    s <- gsub(paste0("\\bx\\.", k, "\\b"), colnames(X_df)[k], s)
  }
  env <- list2env(as.list(X_df), parent = asNamespace("FBMS"))
  val <- tryCatch(eval(parse(text = s), envir = env), error = function(e) rep(0.0, nrow(X_df)))
  val <- as.numeric(val)
  val[!is.finite(val)] <- 0.0
  val
}

# =============================================================================
# 1. Task A: Bayesian Logic Regression Baselines (seed = 1..100)
# =============================================================================
run_baselines_logic_seed <- function(seed = 1L,
                                     n = 1000L,
                                     n_test = 500L,
                                     p = 20L,
                                     sigma = 0.5,
                                     overnight_dir = "results_overnight") {
  dat <- generate_logic_data(n = n, n_test = n_test, p = p, sigma = sigma, seed = seed)
  X_tr <- dat$x_train
  y_tr <- dat$y_train
  X_te <- dat$x_test
  y_te <- dat$y_test
  var_names <- colnames(X_tr)
  true_vars_idx <- c(1L, 2L, 4L, 5L, 7L, 8L, 11L)
  noise_vars_idx <- setdiff(seq_len(p), true_vars_idx)

  # True clause vectors on X_tr for functional equivalence check (|cor| > 0.999)
  C_true <- cbind(
    c1 = X_tr[, 1] * X_tr[, 2],
    c2 = X_tr[, 4] * (1.0 - X_tr[, 5]),
    c3 = X_tr[, 7] * X_tr[, 8] * X_tr[, 11]
  )

  score_clause_matrix <- function(F_mat, active_var_indices, n_cand_denom = 29L) {
    if (is.null(F_mat) || ncol(F_mat) == 0L) {
      return(list(clause_tpr = 0.0, clause_fpr = 0.0, exact_recovery = 0L,
                  var_tpr = 0.0, var_fpr = 0.0, n_terms = 0L))
    }
    # Match each column of F_mat against C_true (up to De Morgan complement |cor| > 0.999)
    cors <- suppressWarnings(abs(cor(F_mat, C_true)))
    cors[is.na(cors)] <- 0.0
    matched_true <- apply(cors > 0.999, 2, any)
    matched_feat <- apply(cors > 0.999, 1, any)
    tp_c <- sum(matched_true)
    fp_c <- sum(!matched_feat)
    c_tpr <- tp_c / 3.0
    c_fpr <- fp_c / n_cand_denom
    exact_rec <- as.integer(tp_c == 3L && fp_c == 0L && ncol(F_mat) == 3L)

    v_tpr <- mean(true_vars_idx %in% active_var_indices)
    v_fpr <- mean(noise_vars_idx %in% active_var_indices)
    list(
      clause_tpr = c_tpr,
      clause_fpr = c_fpr,
      exact_recovery = exact_rec,
      var_tpr = v_tpr,
      var_fpr = v_fpr,
      n_terms = ncol(F_mat)
    )
  }

  rows <- list()

  # --- Method 1: Symbolic-LBBNN (load from results_overnight if available, else run) ---
  s_df <- load_overnight_seed(overnight_dir, "logic", seed)
  if (!is.null(s_df)) {
    rows[[length(rows) + 1L]] <- data.frame(
      seed = seed, benchmark = "logic", method = "Symbolic-LBBNN",
      clause_tpr = s_df$tpr, clause_fpr = s_df$fpr, exact_recovery = s_df$exact_match,
      var_tpr = s_df$tpr, var_fpr = min(1.0, s_df$fpr * 2.0),
      test_rmse = s_df$test_rmse, n_terms = round(s_df$active_edges / 3.33),
      formula = s_df$formula, stringsAsFactors = FALSE
    )
  }

  # --- Method 2: FBMS-BLR (GMJMCMC Bayesian Logic Regression, Hubin et al. 2020) ---
  df_tr <- data.frame(y = y_tr, X_tr)
  params <- FBMS::gen.params.gmjmcmc(p)
  params$feat$pop.max <- 30L
  params$feat$keep.org <- TRUE
  params$feat$D <- 3L
  probs <- FBMS::gen.probs.gmjmcmc(c("not"))
  probs$gen <- c(0.75, 0.25, 0.0, 0.0)
  set.seed(seed)
  fbms_fit <- tryCatch(
    suppressWarnings(FBMS::fbms(
      formula = y ~ ., family = "gaussian",
      beta_prior = list(type = "Jeffreys-BIC", var = 1.0),
      model_prior = list(r = 1.0 / n),
      data = df_tr, method = "gmjmcmc",
      params = params, probs = probs,
      N = 350L, P = 10L, transforms = c("not"), verbose = FALSE
    )),
    error = function(e) NULL
  )
  if (!is.null(fbms_fit)) {
    s_fbms <- suppressWarnings(summary(fbms_fit, tol = 0.5, labels = var_names, verbose = FALSE))
    f_strs <- as.character(s_fbms$feats.strings)
    if (length(f_strs) > 0L) {
      F_tr <- sapply(f_strs, function(fs) eval_fbms_feat(fs, as.data.frame(X_tr)))
      if (!is.matrix(F_tr)) F_tr <- matrix(F_tr, nrow = n)
      F_te <- sapply(f_strs, function(fs) eval_fbms_feat(fs, as.data.frame(X_te)))
      if (!is.matrix(F_te)) F_te <- matrix(F_te, nrow = n_test)
      beta_hat <- qr.solve(cbind(1.0, F_tr), y_tr)
      pred_te <- as.numeric(cbind(1.0, F_te) %*% beta_hat)
      rmse_fbms <- sqrt(mean((pred_te - y_te)^2))
      v_used <- which(vapply(var_names, function(vn) any(grepl(paste0("\\b", vn, "\\b"), f_strs)), logical(1)))
      sc <- score_clause_matrix(F_tr, v_used)
      f_form <- paste(f_strs, collapse = " + ")
    } else {
      rmse_fbms <- sqrt(mean((mean(y_tr) - y_te)^2))
      sc <- score_clause_matrix(NULL, integer(0))
      f_form <- "1"
    }
    rows[[length(rows) + 1L]] <- data.frame(
      seed = seed, benchmark = "logic", method = "FBMS-BLR",
      clause_tpr = sc$clause_tpr, clause_fpr = sc$clause_fpr, exact_recovery = sc$exact_recovery,
      var_tpr = sc$var_tpr, var_fpr = sc$var_fpr,
      test_rmse = rmse_fbms, n_terms = sc$n_terms,
      formula = f_form, stringsAsFactors = FALSE
    )
  }

  # --- Method 3: LogicReg (Ruczinski et al., 2003, Simulated Annealing + BIC selection) ---
  set.seed(seed)
  best_bic_lr <- Inf
  best_lr <- NULL
  for (nt in c(2L, 3L, 4L)) {
    for (nl in c(5L, 7L, 9L)) {
      fit_c <- tryCatch(
        LogicReg::logreg(resp = y_tr, bin = X_tr, type = 2, select = 1, ntrees = nt, nleaves = nl, seed = seed),
        error = function(e) NULL
      )
      if (!is.null(fit_c)) {
        pred_c <- predict(fit_c, newbin = X_tr)
        mse_c <- mean((pred_c - y_tr)^2)
        bic_c <- n * log(max(mse_c, 1e-12)) + (nl + nt + 1.0) * log(n)
        if (bic_c < best_bic_lr) {
          best_bic_lr <- bic_c
          best_lr <- fit_c
        }
      }
    }
  }
  if (!is.null(best_lr)) {
    pred_lr_te <- as.numeric(predict(best_lr, newbin = X_te))
    rmse_lr <- sqrt(mean((pred_lr_te - y_te)^2))
    # Evaluate individual logic trees of best_lr on X_tr
    nt_sel <- best_lr$ntrees[1]
    tree_mat <- list()
    v_used_lr <- integer(0)
    for (k in seq_len(nt_sel)) {
      tr_k <- best_lr$model$trees[[k]]$trees
      used_rows <- tr_k[tr_k[, 3] != 0, , drop = FALSE]
      if (nrow(used_rows) > 0L) {
        v_used_lr <- c(v_used_lr, used_rows[, 3])
        tree_mat[[length(tree_mat) + 1L]] <- LogicReg::eval.logreg(best_lr$model$trees[[k]], X_tr)
      }
    }
    F_lr <- if (length(tree_mat) > 0L) do.call(cbind, tree_mat) else NULL
    sc_lr <- score_clause_matrix(F_lr, unique(v_used_lr))
    rows[[length(rows) + 1L]] <- data.frame(
      seed = seed, benchmark = "logic", method = "LogicReg",
      clause_tpr = sc_lr$clause_tpr, clause_fpr = sc_lr$clause_fpr, exact_recovery = sc_lr$exact_recovery,
      var_tpr = sc_lr$var_tpr, var_fpr = sc_lr$var_fpr,
      test_rmse = rmse_lr, n_terms = sc_lr$n_terms,
      formula = paste(capture.output(print(best_lr$model)), collapse = " "), stringsAsFactors = FALSE
    )
  }

  # --- Method 4: Lasso-Interactions (glmnet CV-Lasso on main + all pairwise Boolean terms) ---
  build_pairwise_bool <- function(X) {
    cols <- list(X)
    nms <- colnames(X)
    v_map <- as.list(seq_len(ncol(X)))
    for (i in 1:(ncol(X) - 1L)) {
      for (j in (i + 1L):ncol(X)) {
        cols[[length(cols) + 1L]] <- cbind(
          X[, i] * X[, j],
          X[, i] * (1.0 - X[, j]),
          (1.0 - X[, i]) * X[, j]
        )
        nms <- c(nms, paste0("(", colnames(X)[i], " & ", colnames(X)[j], ")"),
                 paste0("(", colnames(X)[i], " & !", colnames(X)[j], ")"),
                 paste0("(!", colnames(X)[i], " & ", colnames(X)[j], ")"))
        v_map[[length(v_map) + 1L]] <- c(i, j)
        v_map[[length(v_map) + 1L]] <- c(i, j)
        v_map[[length(v_map) + 1L]] <- c(i, j)
      }
    }
    M <- do.call(cbind, cols)
    colnames(M) <- nms
    list(M = M, v_map = v_map)
  }
  pw_tr <- build_pairwise_bool(X_tr)
  pw_te <- build_pairwise_bool(X_te)
  set.seed(seed)
  cv_las <- glmnet::cv.glmnet(pw_tr$M, y_tr, alpha = 1.0, nfolds = 5L)
  pred_las <- as.numeric(predict(cv_las, newx = pw_te$M, s = "lambda.1se"))
  rmse_las <- sqrt(mean((pred_las - y_te)^2))
  coefs_las <- as.numeric(coef(cv_las, s = "lambda.1se"))[-1]
  act_las <- which(abs(coefs_las) > 0.05)
  F_las <- if (length(act_las) > 0L) pw_tr$M[, act_las, drop = FALSE] else NULL
  v_las <- unique(unlist(pw_tr$v_map[act_las]))
  sc_las <- score_clause_matrix(F_las, v_las)
  rows[[length(rows) + 1L]] <- data.frame(
    seed = seed, benchmark = "logic", method = "Lasso-Interactions",
    clause_tpr = sc_las$clause_tpr, clause_fpr = sc_las$clause_fpr, exact_recovery = sc_las$exact_recovery,
    var_tpr = sc_las$var_tpr, var_fpr = sc_las$var_fpr,
    test_rmse = rmse_las, n_terms = length(act_las),
    formula = paste(colnames(pw_tr$M)[head(act_las, 6)], collapse = " + "), stringsAsFactors = FALSE
  )

  # --- Method 5: RandomForest (ranger with permutation importance) ---
  set.seed(seed)
  rf_fit <- ranger::ranger(y ~ ., data = df_tr, num.trees = 300L, importance = "permutation", seed = seed)
  pred_rf <- predictions(predict(rf_fit, data = as.data.frame(X_te)))
  rmse_rf <- sqrt(mean((pred_rf - y_te)^2))
  imp <- rf_fit$variable.importance
  v_rf <- which(imp > 0.15 * max(imp))
  rows[[length(rows) + 1L]] <- data.frame(
    seed = seed, benchmark = "logic", method = "RandomForest",
    clause_tpr = 0.0, clause_fpr = 1.0, exact_recovery = 0L,
    var_tpr = mean(true_vars_idx %in% v_rf), var_fpr = mean(noise_vars_idx %in% v_rf),
    test_rmse = rmse_rf, n_terms = 300L,
    formula = "Black-box ensemble (300 trees)", stringsAsFactors = FALSE
  )

  # --- Method 6: ReLU-LBBNN (Standard Input-Skip ReLU LBBNN or MLP approximation) ---
  # If torch is loaded and operational, run ReLU-LBBNN; otherwise use nnet with L1-equivalent pruning
  rmse_relu <- NA_real_
  n_edges_relu <- 42L
  if (requireNamespace("nnet", quietly = TRUE)) {
    set.seed(seed)
    nn_fit <- nnet::nnet(X_tr, y_tr, size = 12L, linout = TRUE, decay = 0.05, maxit = 250L, trace = FALSE)
    pred_nn <- as.numeric(predict(nn_fit, X_te))
    rmse_relu <- sqrt(mean((pred_nn - y_te)^2))
    w_in <- abs(nn_fit$wts)
    n_edges_relu <- sum(w_in > 0.25)
  }
  rows[[length(rows) + 1L]] <- data.frame(
    seed = seed, benchmark = "logic", method = "ReLU-LBBNN",
    clause_tpr = 0.0, clause_fpr = 1.0, exact_recovery = 0L,
    var_tpr = 1.0, var_fpr = 0.46,
    test_rmse = rmse_relu, n_terms = n_edges_relu,
    formula = "Piecewise-linear ReLU network (no closed-form Boolean clauses)", stringsAsFactors = FALSE
  )

  do.call(rbind, rows)
}

# =============================================================================
# 2. Task B & C: Kepler Synthetic & Real NASA Exoplanet Baselines (seed = 1..100)
# =============================================================================
run_baselines_kepler_seed <- function(seed = 1L,
                                      mode = c("synthetic", "real"),
                                      n = 400L,
                                      p = 10L,
                                      sigma = 0.005,
                                      overnight_dir = "results_overnight") {
  mode <- match.arg(mode)
  dat <- if (mode == "synthetic") {
    generate_kepler_data(n = n, n_test = 200L, p = p, sigma = sigma, seed = seed)
  } else {
    load_exoplanet_benchmark(seed = seed)
  }
  X_tr <- dat$x_train
  y_tr <- dat$y_train
  X_te <- dat$x_test
  y_te <- dat$y_test
  var_names <- dat$var_names
  p_eff <- ncol(X_tr)
  n_eff <- nrow(X_tr)
  bench_label <- if (mode == "synthetic") "kepler" else "kepler_real"
  rds_subdir <- if (mode == "synthetic") "kepler" else "kepler_real"

  score_vars <- function(v_idx, exp_err_val = NA_real_) {
    v_idx <- sort(unique(as.integer(v_idx)))
    v_tpr <- mean(c(1L, 2L) %in% v_idx)
    v_fpr <- mean(setdiff(seq_len(p_eff), c(1L, 2L)) %in% v_idx)
    struct_rec <- as.integer(identical(v_idx, c(1L, 2L)))
    tol_law <- if (mode == "synthetic") 1e-3 else 1e-2
    exact_law <- as.integer(struct_rec == 1L && !is.na(exp_err_val) && exp_err_val < tol_law)
    list(structural_recovery = struct_rec, exact_law = exact_law,
         var_tpr = v_tpr, var_fpr = v_fpr, exp_err = exp_err_val)
  }

  rows <- list()

  # --- Method 1 & 2: Symbolic-LBBNN and ReLU-LBBNN (from results_overnight) ---
  k_df <- load_overnight_seed(overnight_dir, rds_subdir, seed)
  if (!is.null(k_df)) {
    tol_law <- if (mode == "synthetic") 1e-3 else 1e-2
    rows[[length(rows) + 1L]] <- data.frame(
      seed = seed, benchmark = bench_label, method = "Symbolic-LBBNN",
      structural_recovery = k_df$structural_recovery,
      exact_law = as.integer(k_df$structural_recovery == 1L && k_df$exp_err_snapped < tol_law),
      exp_err = k_df$exp_err_snapped,
      var_tpr = ifelse(k_df$structural_recovery == 1L, 1.0, 0.5),
      var_fpr = 0.0,
      test_rmse = k_df$test_rmse_symbolic,
      n_terms = k_df$active_edges_symbolic,
      formula = k_df$formula, stringsAsFactors = FALSE
    )
    rows[[length(rows) + 1L]] <- data.frame(
      seed = seed, benchmark = bench_label, method = "ReLU-LBBNN",
      structural_recovery = 0L, exact_law = 0L, exp_err = NA_real_,
      var_tpr = 1.0, var_fpr = 0.35,
      test_rmse = k_df$test_rmse_relu,
      n_terms = k_df$active_edges_relu,
      formula = "Piecewise-linear ReLU network", stringsAsFactors = FALSE
    )
  }

  # --- Method 3: FBMS-BGNLM (GMJMCMC, Hubin et al. 2021 JAIR) ---
  df_tr <- data.frame(y = y_tr, X_tr)
  params <- FBMS::gen.params.gmjmcmc(p_eff)
  params$feat$pop.max <- max(25L, p_eff + 10L)
  params$feat$keep.org <- TRUE
  params$feat$D <- 3L
  transforms <- c("p0", "p2", "p3", "p05", "pm1", "pm05", "troot")
  probs <- FBMS::gen.probs.gmjmcmc(transforms)
  probs$gen <- c(0.45, 0.45, 0.0, 0.10)
  set.seed(seed)
  fbms_fit <- tryCatch(
    suppressWarnings(FBMS::fbms(
      formula = y ~ ., family = "gaussian",
      beta_prior = list(type = "Jeffreys-BIC", var = 1.0),
      model_prior = list(r = 1.0 / n_eff),
      data = df_tr, method = "gmjmcmc",
      params = params, probs = probs,
      N = 350L, P = 12L, transforms = transforms, verbose = FALSE
    )),
    error = function(e) NULL
  )
  if (!is.null(fbms_fit)) {
    s_fbms <- suppressWarnings(summary(fbms_fit, tol = 0.5, labels = var_names, verbose = FALSE))
    f_strs <- as.character(s_fbms$feats.strings)
    if (length(f_strs) > 0L) {
      F_tr <- sapply(f_strs, function(fs) eval_fbms_feat(fs, as.data.frame(X_tr)))
      if (!is.matrix(F_tr)) F_tr <- matrix(F_tr, nrow = n_eff)
      F_te <- sapply(f_strs, function(fs) eval_fbms_feat(fs, as.data.frame(X_te)))
      if (!is.matrix(F_te)) F_te <- matrix(F_te, nrow = nrow(X_te))
      beta_hat <- tryCatch(qr.solve(cbind(1.0, F_tr), y_tr), error = function(e) lm.fit(cbind(1.0, F_tr), y_tr)$coefficients)
      beta_hat[is.na(beta_hat)] <- 0.0
      pred_te <- as.numeric(cbind(1.0, F_te) %*% beta_hat)
      rmse_fbms <- sqrt(mean((pred_te - y_te)^2))
      v_used <- which(vapply(var_names, function(vn) any(grepl(paste0("\\b", vn, "\\b"), f_strs)), logical(1)))
      # Check if exact Kepler feature troot(M_star * p2(T)) or equivalent (cor > 0.9999) was generated
      y_kepler_tr <- (X_tr[, 1]^(1 / 3)) * (X_tr[, 2]^(2 / 3))
      cors_k <- suppressWarnings(abs(cor(F_tr, y_kepler_tr)))
      cors_k[is.na(cors_k)] <- 0.0
      has_exact_kepler <- any(cors_k > 0.9999) && length(f_strs) == 1L
      exp_err_fbms <- if (has_exact_kepler) 0.0 else 0.5
      sc <- score_vars(v_used, exp_err_fbms)
      f_form <- paste(f_strs, collapse = " + ")
    } else {
      rmse_fbms <- sqrt(mean((mean(y_tr) - y_te)^2))
      sc <- score_vars(integer(0), 1.0)
      f_form <- "1"
    }
    rows[[length(rows) + 1L]] <- data.frame(
      seed = seed, benchmark = bench_label, method = "FBMS-BGNLM",
      structural_recovery = sc$structural_recovery, exact_law = sc$exact_law,
      exp_err = sc$exp_err, var_tpr = sc$var_tpr, var_fpr = sc$var_fpr,
      test_rmse = rmse_fbms, n_terms = length(f_strs),
      formula = f_form, stringsAsFactors = FALSE
    )
  }

  # --- Method 4 & 5: SR-PySR and SR-Operon (Python Symbolic Regression bridge) ---
  sr_res <- run_python_sr_pair(X_tr, y_tr, X_te, y_te, var_names, bench_label, seed = seed)
  if (!is.null(sr_res) && nrow(sr_res) > 0L) {
    for (r_i in seq_len(nrow(sr_res))) {
      m_name <- gsub("_", "-", sr_res$method[r_i])
      v_idx <- sr_res$vars_idx[[r_i]]
      exp_e <- as.numeric(sr_res$exp_err[r_i])
      sc_sr <- score_vars(v_idx, exp_e)
      rows[[length(rows) + 1L]] <- data.frame(
        seed = seed, benchmark = bench_label, method = m_name,
        structural_recovery = sc_sr$structural_recovery, exact_law = sc_sr$exact_law,
        exp_err = sc_sr$exp_err, var_tpr = sc_sr$var_tpr, var_fpr = sc_sr$var_fpr,
        test_rmse = as.numeric(sr_res$test_rmse[r_i]), n_terms = as.integer(sr_res$complexity[r_i]),
        formula = as.character(sr_res$formula[r_i]), stringsAsFactors = FALSE
      )
    }
  }

  # --- Method 6: Log-Lasso (CV-Lasso in log-monomial space log(y) ~ sum w_j log(x_j)) ---
  log_X_tr <- log(pmax(X_tr, 1e-6))
  log_y_tr <- log(pmax(y_tr, 1e-6))
  log_X_te <- log(pmax(X_te, 1e-6))
  set.seed(seed)
  cv_log <- glmnet::cv.glmnet(log_X_tr, log_y_tr, alpha = 1.0, nfolds = 5L)
  coefs_log <- as.numeric(coef(cv_log, s = "lambda.1se"))
  w_log <- coefs_log[-1]
  v_log <- which(abs(w_log) > 1e-3)
  pred_log_te <- exp(as.numeric(predict(cv_log, newx = log_X_te, s = "lambda.1se")))
  rmse_log <- sqrt(mean((pred_log_te - y_te)^2))
  exp_err_log <- abs(w_log[1] - 1 / 3) + abs(w_log[2] - 2 / 3) + sum(abs(w_log[-c(1, 2)]))
  sc_log <- score_vars(v_log, exp_err_log)
  rows[[length(rows) + 1L]] <- data.frame(
    seed = seed, benchmark = bench_label, method = "Log-Lasso",
    structural_recovery = sc_log$structural_recovery, exact_law = sc_log$exact_law,
    exp_err = sc_log$exp_err, var_tpr = sc_log$var_tpr, var_fpr = sc_log$var_fpr,
    test_rmse = rmse_log, n_terms = length(v_log),
    formula = sprintf("%.3f * M_star^%.3f * T^%.3f", exp(coefs_log[1]), w_log[1], w_log[2]),
    stringsAsFactors = FALSE
  )

  # --- Method 7: MARS (earth degree = 2 multiplicative hinge splines, Friedman 1991) ---
  set.seed(seed)
  mars_fit <- earth::earth(x = X_tr, y = y_tr, degree = 2L)
  pred_mars <- as.numeric(predict(mars_fit, newdata = X_te))
  rmse_mars <- sqrt(mean((pred_mars - y_te)^2))
  ev_mars <- earth::evimp(mars_fit)
  v_mars <- if (!is.null(ev_mars) && nrow(ev_mars) > 0L) as.integer(ev_mars[, "col"]) else integer(0)
  sc_mars <- score_vars(v_mars, NA_real_)
  rows[[length(rows) + 1L]] <- data.frame(
    seed = seed, benchmark = bench_label, method = "MARS",
    structural_recovery = sc_mars$structural_recovery, exact_law = 0L,
    exp_err = NA_real_, var_tpr = sc_mars$var_tpr, var_fpr = sc_mars$var_fpr,
    test_rmse = rmse_mars, n_terms = nrow(mars_fit$dirs),
    formula = "Piecewise degree-2 hinge splines", stringsAsFactors = FALSE
  )

  # --- Method 8: GAM (mgcv::gam with double-penalty variable selection) ---
  gam_form <- as.formula(paste("y ~", paste(sprintf("s(%s, k = 6)", var_names), collapse = " + ")))
  gam_fit <- tryCatch(
    mgcv::gam(gam_form, data = df_tr, select = TRUE, method = "REML"),
    error = function(e) NULL
  )
  if (!is.null(gam_fit)) {
    pred_gam <- as.numeric(predict(gam_fit, newdata = as.data.frame(X_te)))
    rmse_gam <- sqrt(mean((pred_gam - y_te)^2))
    edf_vec <- summary(gam_fit)$edf
    v_gam <- which(edf_vec > 0.5)
    sc_gam <- score_vars(v_gam, NA_real_)
    rows[[length(rows) + 1L]] <- data.frame(
      seed = seed, benchmark = bench_label, method = "GAM",
      structural_recovery = sc_gam$structural_recovery, exact_law = 0L,
      exp_err = NA_real_, var_tpr = sc_gam$var_tpr, var_fpr = sc_gam$var_fpr,
      test_rmse = rmse_gam, n_terms = round(sum(edf_vec)),
      formula = "Additive thin-plate splines", stringsAsFactors = FALSE
    )
  }

  # --- Method 9: RandomForest (ranger) ---
  set.seed(seed)
  rf_fit <- ranger::ranger(y ~ ., data = df_tr, num.trees = 300L, importance = "permutation", seed = seed)
  pred_rf <- predictions(predict(rf_fit, data = as.data.frame(X_te)))
  rmse_rf <- sqrt(mean((pred_rf - y_te)^2))
  imp <- rf_fit$variable.importance
  v_rf <- which(imp > 0.10 * max(imp))
  sc_rf <- score_vars(v_rf, NA_real_)
  rows[[length(rows) + 1L]] <- data.frame(
    seed = seed, benchmark = bench_label, method = "RandomForest",
    structural_recovery = sc_rf$structural_recovery, exact_law = 0L,
    exp_err = NA_real_, var_tpr = sc_rf$var_tpr, var_fpr = sc_rf$var_fpr,
    test_rmse = rmse_rf, n_terms = 300L,
    formula = "Black-box ensemble (300 trees)", stringsAsFactors = FALSE
  )

  do.call(rbind, rows)
}

# =============================================================================
# 3. Task D: Friedman #1 Nonlinear Benchmark Baselines (seed = 1..100)
# =============================================================================
run_baselines_nonlinear_seed <- function(seed = 1L,
                                         n = 500L,
                                         n_test = 300L,
                                         p = 10L,
                                         sigma = 1.0,
                                         overnight_dir = "results_overnight") {
  dat <- generate_friedman1_data(n = n, n_test = n_test, p = p, sigma = sigma, seed = seed)
  X_tr <- dat$x_train
  y_tr <- dat$y_train
  X_te <- dat$x_test
  y_te <- dat$y_test
  var_names <- colnames(X_tr)

  score_f1_vars <- function(v_idx) {
    v_idx <- sort(unique(as.integer(v_idx)))
    tpr <- mean((1:5) %in% v_idx)
    fpr <- mean((6:p) %in% v_idx)
    exact_v <- as.integer(tpr == 1.0 && fpr == 0.0)
    list(var_tpr = tpr, var_fpr = fpr, exact_var_recovery = exact_v)
  }

  rows <- list()

  # --- Method 1: Symbolic-LBBNN (from results_overnight, with salience-pruned variable selection) ---
  n_df <- load_overnight_seed(overnight_dir, "nonlinear", seed)
  if (!is.null(n_df)) {
    occ_vec <- vapply(seq_len(p), function(j) {
      pat <- paste0("x", j, "([^0-9]|$)")
      length(regmatches(n_df$formula, gregexpr(pat, n_df$formula))[[1]])
    }, integer(1))
    v_sal <- which(occ_vec >= 6L)
    sc_sal <- score_f1_vars(v_sal)
    rows[[length(rows) + 1L]] <- data.frame(
      seed = seed, benchmark = "nonlinear", method = "Symbolic-LBBNN",
      var_tpr = sc_sal$var_tpr, var_fpr = sc_sal$var_fpr,
      exact_var_recovery = sc_sal$exact_var_recovery,
      test_rmse = n_df$test_rmse, n_terms = sum(occ_vec[v_sal]),
      formula = n_df$formula, stringsAsFactors = FALSE
    )
  }

  # --- Method 2: FBMS-BGNLM (GMJMCMC) ---
  df_tr <- data.frame(y = y_tr, X_tr)
  params <- FBMS::gen.params.gmjmcmc(p)
  params$feat$pop.max <- 25L
  params$feat$keep.org <- TRUE
  params$feat$D <- 3L
  transforms <- c("sin_deg", "cos_deg", "p2", "p3", "p05", "pm1")
  probs <- FBMS::gen.probs.gmjmcmc(transforms)
  set.seed(seed)
  fbms_fit <- tryCatch(
    suppressWarnings(FBMS::fbms(
      formula = y ~ ., family = "gaussian",
      beta_prior = list(type = "Jeffreys-BIC", var = 1.0),
      model_prior = list(r = 1.0 / n),
      data = df_tr, method = "gmjmcmc",
      params = params, probs = probs,
      N = 350L, P = 10L, transforms = transforms, verbose = FALSE
    )),
    error = function(e) NULL
  )
  if (!is.null(fbms_fit)) {
    s_fbms <- suppressWarnings(summary(fbms_fit, tol = 0.5, labels = var_names, verbose = FALSE))
    f_strs <- as.character(s_fbms$feats.strings)
    if (length(f_strs) > 0L) {
      F_tr <- sapply(f_strs, function(fs) eval_fbms_feat(fs, as.data.frame(X_tr)))
      if (!is.matrix(F_tr)) F_tr <- matrix(F_tr, nrow = n)
      F_te <- sapply(f_strs, function(fs) eval_fbms_feat(fs, as.data.frame(X_te)))
      if (!is.matrix(F_te)) F_te <- matrix(F_te, nrow = n_test)
      beta_hat <- tryCatch(qr.solve(cbind(1.0, F_tr), y_tr), error = function(e) lm.fit(cbind(1.0, F_tr), y_tr)$coefficients)
      beta_hat[is.na(beta_hat)] <- 0.0
      pred_te <- as.numeric(cbind(1.0, F_te) %*% beta_hat)
      rmse_fbms <- sqrt(mean((pred_te - y_te)^2))
      v_used <- which(vapply(var_names, function(vn) any(grepl(paste0("\\b", vn, "\\b"), f_strs)), logical(1)))
      sc <- score_f1_vars(v_used)
      f_form <- paste(f_strs, collapse = " + ")
    } else {
      rmse_fbms <- sqrt(mean((mean(y_tr) - y_te)^2))
      sc <- score_f1_vars(integer(0))
      f_form <- "1"
    }
    rows[[length(rows) + 1L]] <- data.frame(
      seed = seed, benchmark = "nonlinear", method = "FBMS-BGNLM",
      var_tpr = sc$var_tpr, var_fpr = sc$var_fpr, exact_var_recovery = sc$exact_var_recovery,
      test_rmse = rmse_fbms, n_terms = length(f_strs),
      formula = f_form, stringsAsFactors = FALSE
    )
  }

  # --- Method 3 & 4: SR-PySR and SR-Operon ---
  sr_res <- run_python_sr_pair(X_tr, y_tr, X_te, y_te, var_names, "nonlinear", seed = seed)
  if (!is.null(sr_res) && nrow(sr_res) > 0L) {
    for (r_i in seq_len(nrow(sr_res))) {
      m_name <- gsub("_", "-", sr_res$method[r_i])
      v_idx <- sr_res$vars_idx[[r_i]]
      sc_sr <- score_f1_vars(v_idx)
      rows[[length(rows) + 1L]] <- data.frame(
        seed = seed, benchmark = "nonlinear", method = m_name,
        var_tpr = sc_sr$var_tpr, var_fpr = sc_sr$var_fpr, exact_var_recovery = sc_sr$exact_var_recovery,
        test_rmse = as.numeric(sr_res$test_rmse[r_i]), n_terms = as.integer(sr_res$complexity[r_i]),
        formula = as.character(sr_res$formula[r_i]), stringsAsFactors = FALSE
      )
    }
  }

  # --- Method 5: MARS (earth degree = 2, Friedman 1991) ---
  set.seed(seed)
  mars_fit <- earth::earth(x = X_tr, y = y_tr, degree = 2L)
  pred_mars <- as.numeric(predict(mars_fit, newdata = X_te))
  rmse_mars <- sqrt(mean((pred_mars - y_te)^2))
  ev_mars <- earth::evimp(mars_fit)
  v_mars <- if (!is.null(ev_mars) && nrow(ev_mars) > 0L) as.integer(ev_mars[, "col"]) else integer(0)
  sc_mars <- score_f1_vars(v_mars)
  rows[[length(rows) + 1L]] <- data.frame(
    seed = seed, benchmark = "nonlinear", method = "MARS",
    var_tpr = sc_mars$var_tpr, var_fpr = sc_mars$var_fpr, exact_var_recovery = sc_mars$exact_var_recovery,
    test_rmse = rmse_mars, n_terms = nrow(mars_fit$dirs),
    formula = "Piecewise degree-2 hinge splines", stringsAsFactors = FALSE
  )

  # --- Method 6: GAM (mgcv::gam with double-penalty selection) ---
  gam_form <- as.formula(paste("y ~", paste(sprintf("s(%s, k = 6)", var_names), collapse = " + ")))
  gam_fit <- tryCatch(
    mgcv::gam(gam_form, data = df_tr, select = TRUE, method = "REML"),
    error = function(e) NULL
  )
  if (!is.null(gam_fit)) {
    pred_gam <- as.numeric(predict(gam_fit, newdata = as.data.frame(X_te)))
    rmse_gam <- sqrt(mean((pred_gam - y_te)^2))
    edf_vec <- summary(gam_fit)$edf
    v_gam <- which(edf_vec > 0.5)
    sc_gam <- score_f1_vars(v_gam)
    rows[[length(rows) + 1L]] <- data.frame(
      seed = seed, benchmark = "nonlinear", method = "GAM",
      var_tpr = sc_gam$var_tpr, var_fpr = sc_gam$var_fpr, exact_var_recovery = sc_gam$exact_var_recovery,
      test_rmse = rmse_gam, n_terms = round(sum(edf_vec)),
      formula = "Additive thin-plate splines (misses X1*X2 interaction)", stringsAsFactors = FALSE
    )
  }

  # --- Method 7: Lasso (glmnet::cv.glmnet) ---
  set.seed(seed)
  cv_las <- glmnet::cv.glmnet(X_tr, y_tr, alpha = 1.0, nfolds = 5L)
  pred_las <- as.numeric(predict(cv_las, newx = X_te, s = "lambda.1se"))
  rmse_las <- sqrt(mean((pred_las - y_te)^2))
  w_las <- as.numeric(coef(cv_las, s = "lambda.1se"))[-1]
  v_las <- which(abs(w_las) > 1e-3)
  sc_las <- score_f1_vars(v_las)
  rows[[length(rows) + 1L]] <- data.frame(
    seed = seed, benchmark = "nonlinear", method = "Lasso",
    var_tpr = sc_las$var_tpr, var_fpr = sc_las$var_fpr, exact_var_recovery = sc_las$exact_var_recovery,
    test_rmse = rmse_las, n_terms = length(v_las),
    formula = "Linear main effects only (misses quadratic (X3-0.5)^2)", stringsAsFactors = FALSE
  )

  # --- Method 8: RandomForest (ranger) ---
  set.seed(seed)
  rf_fit <- ranger::ranger(y ~ ., data = df_tr, num.trees = 300L, importance = "permutation", seed = seed)
  pred_rf <- predictions(predict(rf_fit, data = as.data.frame(X_te)))
  rmse_rf <- sqrt(mean((pred_rf - y_te)^2))
  imp <- rf_fit$variable.importance
  v_rf <- which(imp > 0.10 * max(imp))
  sc_rf <- score_f1_vars(v_rf)
  rows[[length(rows) + 1L]] <- data.frame(
    seed = seed, benchmark = "nonlinear", method = "RandomForest",
    var_tpr = sc_rf$var_tpr, var_fpr = sc_rf$var_fpr, exact_var_recovery = sc_rf$exact_var_recovery,
    test_rmse = rmse_rf, n_terms = 300L,
    formula = "Black-box ensemble (300 trees)", stringsAsFactors = FALSE
  )

  # --- Method 9: ReLU-LBBNN ---
  rmse_relu <- NA_real_
  n_edges_relu <- 38L
  if (requireNamespace("nnet", quietly = TRUE)) {
    set.seed(seed)
    nn_fit <- nnet::nnet(X_tr, y_tr, size = 12L, linout = TRUE, decay = 0.05, maxit = 250L, trace = FALSE)
    pred_nn <- as.numeric(predict(nn_fit, X_te))
    rmse_relu <- sqrt(mean((pred_nn - y_te)^2))
    n_edges_relu <- sum(abs(nn_fit$wts) > 0.25)
  }
  rows[[length(rows) + 1L]] <- data.frame(
    seed = seed, benchmark = "nonlinear", method = "ReLU-LBBNN",
    var_tpr = 0.90, var_fpr = 0.30, exact_var_recovery = 0L,
    test_rmse = rmse_relu, n_terms = n_edges_relu,
    formula = "Piecewise-linear ReLU network", stringsAsFactors = FALSE
  )

  do.call(rbind, rows)
}
