#!/usr/bin/env Rscript
# ==============================================================================
# inst/scripts/test_variance_ablation.R
# Phase 0 Diagnostic Ablation (M = 5 seeds x 5 Benchmarks x 4 Modes):
#   Mode A  : var_mode = "oracle"       + post_mode = "variational_only"
#   Mode B  : var_mode = "jeffreys_ema" + post_mode = "variational_only"
#   Mode C1 : var_mode = "jeffreys_ema" + post_mode = "variational_readout_prune"
#   Mode C2 : var_mode = "jeffreys_ema" + post_mode = "laplace_nig_subspace"
#
# Tests strictly within-MPM post-training (use_bgnlm_screen = FALSE, no external
# raw-data candidate dictionary cheating) across all 5 benchmarks.
# ==============================================================================

suppressPackageStartupMessages({
  pkgload::load_all(".", quiet = TRUE)
})

source("inst/benchmarks/bench_logic.R")
source("inst/benchmarks/bench_kepler.R")
source("inst/benchmarks/bench_nonlinear.R")

# Pure algebraic clause & term helpers without loading heavy baseline packages
extract_terms_vars <- function(term_strs, var_names) {
  lapply(term_strs, function(ts) {
    if (is.na(ts) || !nzchar(trimws(ts))) return(integer(0))
    idx <- which(vapply(seq_along(var_names), function(k) {
      pat <- paste0("(^|[^a-zA-Z0-9_])", var_names[k], "([^a-zA-Z0-9_]|$)")
      grepl(pat, ts)
    }, logical(1)))
    sort(unique(as.integer(idx)))
  })
}

count_readout_terms <- function(sym_obj) {
  if (is.list(sym_obj$basis_terms)) {
    sum(vapply(sym_obj$basis_terms, function(bt) length(setdiff(names(bt), "(Intercept)")), integer(1)))
  } else {
    length(setdiff(names(sym_obj$basis_terms), "(Intercept)"))
  }
}

eval_logic_terms <- function(term_strs, X_mat, var_names) {
  if (length(term_strs) == 0L) return(NULL)
  df_ev <- as.data.frame(X_mat)
  cols <- lapply(term_strs, function(ts) {
    s <- tolower(ts)
    for (vi in rev(seq_len(ncol(X_mat)))) {
      s <- gsub(paste0("!", var_names[vi]), paste0("(1 - ", var_names[vi], ")"), s, fixed = TRUE)
    }
    s <- gsub("&", "*", s, fixed = TRUE)
    s <- gsub("|", "+", s, fixed = TRUE)
    tryCatch(as.numeric(eval(parse(text = s), envir = df_ev)), error = function(e) rep(0.0, nrow(X_mat)))
  })
  do.call(cbind, cols)
}

score_logic_sym <- function(sym, dat) {
  bt_names <- setdiff(names(sym$basis_terms), "(Intercept)")
  n_terms <- length(bt_names)
  if (n_terms == 0L) return(0L)
  X_tr <- dat$x_train
  F_mat <- eval_logic_terms(bt_names, X_tr, colnames(X_tr))
  f_vars <- extract_terms_vars(bt_names, colnames(X_tr))
  C_true <- list(
    list(vec = X_tr[, 1] * X_tr[, 2], vars = c(1L, 2L)),
    list(vec = X_tr[, 4] * (1.0 - X_tr[, 5]), vars = c(4L, 5L)),
    list(vec = X_tr[, 7] * X_tr[, 8] * X_tr[, 11], vars = c(7L, 8L, 11L))
  )
  used <- rep(FALSE, ncol(F_mat))
  tp_c <- 0L
  for (cm in C_true) {
    cand_j <- which(vapply(f_vars, function(vv) length(vv) > 0L && all(vv %in% cm$vars), logical(1)))
    if (length(cand_j) == 0L) next
    tss <- sum((cm$vec - mean(cm$vec))^2)
    fit_all <- stats::lm.fit(cbind(1.0, F_mat[, cand_j, drop = FALSE]), cm$vec)
    if (tss <= 0 || sum(fit_all$residuals^2) / tss >= 1e-6) next
    found_sub <- NULL
    for (sz in seq_len(min(length(cand_j), 4L))) {
      for (sub_j in utils::combn(cand_j, sz, simplify = FALSE)) {
        fit_sub <- stats::lm.fit(cbind(1.0, F_mat[, sub_j, drop = FALSE]), cm$vec)
        if (sum(fit_sub$residuals^2) / tss < 1e-6) { found_sub <- sub_j; break }
      }
      if (!is.null(found_sub)) break
    }
    if (!is.null(found_sub)) { tp_c <- tp_c + 1L; used[found_sub] <- TRUE }
  }
  as.integer(tp_c == 3L && sum(!used) == 0L)
}

run_one_ablation_case <- function(bench, seed, mode_name, var_mode, post_mode, device = "cpu") {
  set.seed(seed)
  torch::torch_manual_seed(seed)
  torch::torch_set_num_threads(1L)

  if (bench == "logic") {
    dat <- generate_logic_data(n = 1000L, n_test = 500L, p = 20L, sigma = 0.5, seed = seed)
    oracle_sig <- 0.5
    p <- 20L; sizes <- c(p, 12L, 1L)
    layer_acts <- list(list(logic_and = 10L, id = 2L))
    priors <- compute_symbolic_priors(sizes, layer_acts, base_prior = 0.15, depth_decay = 0.4, complexity_weight = 0.3, input_skip = TRUE)
    stds <- c(1.0, 1.0); inc <- matrix(rep(c(-1.0, 1.0), 2), nrow = 2L)
    var_names <- paste0("x", seq_len(p))
    epochs <- 260L; debias_epochs <- 80L; snap_rat <- FALSE; lr <- 0.025
    bs <- 256L

  } else if (bench == "kepler") {
    dat <- generate_kepler_data(n = 400L, n_test = 200L, p = 10L, sigma = 0.005, seed = seed)
    oracle_sig <- 0.005
    p <- 10L; sizes <- c(p, p + 6L, 12L, 1L)
    layer_acts <- list(list(clog = p, id = 4L, sin = 2L), list(exp = 6L, id = 4L, square = 2L))
    priors <- compute_symbolic_priors(sizes, layer_acts, base_prior = 0.25, depth_decay = 0.4, complexity_weight = 0.3, input_skip = TRUE)
    stds <- c(1.0, 1.0, 1.0); inc <- matrix(rep(c(-0.5, 1.5), 3), nrow = 2L)
    var_names <- dat$var_names
    epochs <- 300L; debias_epochs <- 120L; snap_rat <- TRUE; lr <- 0.02
    bs <- 128L

  } else if (bench == "kepler_real") {
    dat <- load_exoplanet_benchmark(seed = seed)
    oracle_sig <- 0.017
    p <- ncol(dat$x_train); sizes <- c(p, p + 6L, 12L, 1L)
    layer_acts <- list(list(clog = p, id = 4L, sin = 2L), list(exp = 6L, id = 4L, square = 2L))
    priors <- compute_symbolic_priors(sizes, layer_acts, base_prior = 0.25, depth_decay = 0.4, complexity_weight = 0.3, input_skip = TRUE)
    stds <- c(1.0, 1.0, 1.0); inc <- matrix(rep(c(-0.5, 1.5), 3), nrow = 2L)
    var_names <- dat$var_names
    epochs <- 300L; debias_epochs <- 120L; snap_rat <- TRUE; lr <- 0.02
    bs <- 128L

  } else if (bench == "friedman_3layer") {
    dat <- generate_friedman1_data(n = 500L, n_test = 300L, p = 10L, sigma = 1.0, seed = seed)
    oracle_sig <- 1.0
    p <- 10L; sizes <- c(p, 12L, 8L, 8L, 1L)
    layer_acts <- list(list(clog = p, id = 2L), list(exp = 3L, square = 2L, id = 3L), list(sin = 3L, id = 5L))
    priors <- compute_symbolic_priors(sizes, layer_acts, base_prior = 0.25, depth_decay = 0.3, complexity_weight = 0.25, input_skip = TRUE)
    stds <- c(1.0, 1.0, 1.0, 1.0); inc <- matrix(rep(c(-0.5, 1.5), 4), nrow = 2L)
    var_names <- paste0("x", seq_len(p))
    epochs <- 300L; debias_epochs <- 150L; snap_rat <- TRUE; lr <- 0.02
    bs <- 128L

  } else if (bench == "sindy_3d") {
    set.seed(seed)
    X_all <- matrix(stats::runif(750L * 6L, min = -pi, max = pi), nrow = 750L, ncol = 6L)
    colnames(X_all) <- paste0("x", 1:6)
    Y_clean <- cbind(
      dx1 = -X_all[, 2] - X_all[, 3] + 1.5 * sin(X_all[, 1]),
      dx2 = X_all[, 1] + 0.2 * X_all[, 2],
      dx3 = 0.4 + X_all[, 1] * X_all[, 3] - 5.7 * X_all[, 3]
    )
    Y_obs <- Y_clean + matrix(stats::rnorm(750L * 3L, 0.0, 0.05), nrow = 750L, ncol = 3L)
    dat <- list(
      x_train = X_all[1:500, , drop = FALSE], y_train = Y_obs[1:500, , drop = FALSE],
      x_test = X_all[501:750, , drop = FALSE], y_test = Y_obs[501:750, , drop = FALSE],
      y_clean_test = Y_clean[501:750, , drop = FALSE], var_names = paste0("x", 1:6)
    )
    oracle_sig <- rep(0.05, 3L)
    p <- 6L; sizes <- c(p, 20L, 15L, 3L)
    layer_acts <- list(list(clog = p, id = 4L, sin = 4L, square = 6L), list(exp = 3L, id = 4L, square = 6L, sin = 2L))
    priors <- compute_symbolic_priors(sizes, layer_acts, base_prior = 0.20, depth_decay = 0.4, complexity_weight = 0.35, input_skip = TRUE)
    stds <- c(1.0, 1.0, 1.0); inc <- matrix(rep(c(-0.5, 1.5), 3), nrow = 2L)
    var_names <- dat$var_names
    epochs <- 220L; debias_epochs <- 100L; snap_rat <- TRUE; lr <- 0.025
    bs <- 128L
  }

  x_tr <- torch::torch_tensor(dat$x_train, dtype = torch::torch_float32(), device = device)
  y_tr <- torch::torch_tensor(dat$y_train, dtype = torch::torch_float32(), device = device)
  x_te <- torch::torch_tensor(dat$x_test, dtype = torch::torch_float32(), device = device)
  train_dl <- torch::dataloader(torch::tensor_dataset(x_tr, y_tr), batch_size = bs, shuffle = TRUE)

  model <- lbbnn_symbolic_net(
    problem_type = "regression", sizes = sizes, prior = priors, std = stds,
    inclusion_inits = inc, input_skip = TRUE, flow = FALSE, device = device,
    activations = layer_acts
  )

  train_symbolic_lbbnn(
    epochs = epochs, LBBNN = model, lr = lr, train_dl = train_dl, device = device,
    debias_epochs = debias_epochs, snap_rationals = snap_rat,
    prune_bic = TRUE, prune_readout = (post_mode == "laplace_nig_subspace"),
    prior_sd_post = 10.0,
    var_mode = var_mode, oracle_sigma = oracle_sig,
    post_mode = post_mode, use_bgnlm_screen = FALSE,
    verbose = FALSE
  )

  sym <- extract_symbolic_mpm(model, var_names = var_names, digits = 3, simplify = TRUE)
  n_edges <- nrow(sym$active_paths_df)
  n_terms <- count_readout_terms(sym)

  torch::with_no_grad({
    pred_te <- as.matrix(model(x_te, MPM = TRUE)$cpu())
  })
  if (bench == "sindy_3d") {
    test_rmse <- sqrt(mean((pred_te - dat$y_test)^2))
    r2_vec <- vapply(1:3, function(r) stats::cor(pred_te[, r], dat$y_test[, r])^2, numeric(1))
    test_r2 <- mean(pmin(pmax(r2_vec, 0.0), 1.0))
    r2_clean <- vapply(1:3, function(r) stats::cor(pred_te[, r], dat$y_clean_test[, r])^2, numeric(1))
    true_S <- list(c(1L, 2L, 3L), c(1L, 2L), c(1L, 3L))
    S_sym <- lapply(1:3, function(r) extract_terms_vars(sym$formula_str[r], var_names)[[1]])
    ok_s <- sum(vapply(1:3, function(r) identical(S_sym[[r]], true_S[[r]]), logical(1))) == 3L
    exact_rec <- as.integer(ok_s && all(r2_clean > 0.9985))
    f_out <- paste(sym$formula_str, collapse = " || ")
  } else {
    pv <- as.numeric(pred_te)
    test_rmse <- sqrt(mean((pv - dat$y_test)^2))
    test_r2 <- max(0.0, min(1.0, 1.0 - (test_rmse^2) / mean((dat$y_test - mean(dat$y_test))^2)))
    f_out <- sym$formula_str
    if (bench == "logic") {
      exact_rec <- score_logic_sym(sym, dat)
    } else if (bench %in% c("kepler", "kepler_real")) {
      used_v <- extract_terms_vars(setdiff(names(sym$basis_terms), "(Intercept)"), var_names)
      uniq_v <- sort(unique(unlist(used_v)))
      l2_act <- as.matrix(model$layers$children[[2]]$alpha_active_path$cpu())
      exp_rows <- which(model$layers$children[[2]]$act_vec == "exp")
      w_mat <- as.matrix(model$layers$children[[2]]$weight_mean$cpu())
      co_inc <- any(vapply(exp_rows, function(r) l2_act[r, 1] > 0.5 && l2_act[r, 2] > 0.5 && sum(l2_act[r, -c(1, 2)] > 0.5) == 0L, logical(1)))
      if (co_inc) {
        r_ok <- exp_rows[which(vapply(exp_rows, function(r) l2_act[r, 1] > 0.5 && l2_act[r, 2] > 0.5 && sum(l2_act[r, -c(1, 2)] > 0.5) == 0L, logical(1)))[1]]
        e_err <- abs(w_mat[r_ok, 1] - 1 / 3) + abs(w_mat[r_ok, 2] - 2 / 3)
      } else {
        e_err <- 1.0
      }
      tol <- if (bench == "kepler") 1e-3 else 1e-2
      exact_rec <- as.integer(identical(uniq_v, c(1L, 2L)) && n_terms == 1L && e_err < tol)
    } else {
      used_v <- sort(unique(unlist(extract_terms_vars(f_out, var_names))))
      y_c <- 10.0 * sin(pi * dat$x_test[, 1] * dat$x_test[, 2]) + 20.0 * (dat$x_test[, 3] - 0.5)^2 + 10.0 * dat$x_test[, 4] + 5.0 * dat$x_test[, 5]
      r2_cl <- stats::cor(pv, y_c)^2
      exact_rec <- as.integer(identical(used_v, 1:5) && is.finite(r2_cl) && r2_cl > 0.995 && n_terms <= 6L)
    }
  }

  data.frame(
    benchmark = bench,
    seed = seed,
    mode = mode_name,
    var_mode = var_mode,
    post_mode = post_mode,
    sigma_hat = if (!is.null(model$stage1_sigma_hat)) mean(model$stage1_sigma_hat) else NA_real_,
    active_edges = n_edges,
    active_terms = n_terms,
    exact_recovery = exact_rec,
    test_rmse = test_rmse,
    test_r2 = test_r2,
    formula = f_out,
    stringsAsFactors = FALSE
  )
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  n_seeds <- if (length(args) >= 1L) as.integer(args[1]) else 5L
  benches <- c("logic", "kepler", "kepler_real", "friedman_3layer", "sindy_3d")
  modes <- list(
    list(name = "A_Oracle_VIonly",       vm = "oracle",       pm = "variational_only"),
    list(name = "A2_Oracle_NIG",         vm = "oracle",       pm = "laplace_nig_subspace"),
    list(name = "B_JeffreysEMA_VIonly",  vm = "jeffreys_ema", pm = "variational_only"),
    list(name = "C1_JeffreysEMA_RO_VI",  vm = "jeffreys_ema", pm = "variational_readout_prune"),
    list(name = "C2_JeffreysEMA_NIG",    vm = "jeffreys_ema", pm = "laplace_nig_subspace")
  )
  grid <- expand.grid(bench = benches, seed = seq_len(n_seeds), mode_i = seq_along(modes), stringsAsFactors = FALSE)

  n_workers <- min(8L, max(1L, parallel::detectCores(logical = FALSE) - 1L))
  message(sprintf("Running Phase 0 Diagnostic Ablation: %d tasks (%d workers)...", nrow(grid), n_workers))
  cl <- parallel::makePSOCKcluster(n_workers)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterExport(cl, c("grid", "modes"), envir = environment())
  parallel::clusterEvalQ(cl, {
    suppressPackageStartupMessages({ pkgload::load_all(".", quiet = TRUE) })
    source("inst/scripts/test_variance_ablation.R")
    NULL
  })
  res_list <- parallel::parLapplyLB(cl, seq_len(nrow(grid)), function(idx) {
    g <- grid[idx, ]
    m <- modes[[g$mode_i]]
    tryCatch(
      run_one_ablation_case(g$bench, g$seed, m$name, m$vm, m$pm, device = "cpu"),
      error = function(e) {
        data.frame(
          benchmark = g$bench, seed = g$seed, mode = m$name, var_mode = m$vm, post_mode = m$pm,
          sigma_hat = NA_real_, active_edges = NA_integer_, active_terms = NA_integer_,
          exact_recovery = 0L, test_rmse = NA_real_, test_r2 = NA_real_,
          formula = paste("ERR:", conditionMessage(e)), stringsAsFactors = FALSE
        )
      }
    )
  })
  df_all <- do.call(rbind, res_list)
  dir.create("results_baselines", recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(df_all, "results_baselines/variance_posttraining_ablation_5seeds.csv", row.names = FALSE)

  # Print clean summary table
  cat("\n=== PHASE 0 VARIANCE & POST-TRAINING ABLATION SUMMARY (M =", n_seeds, "SEEDS, NO EXTERNAL CHEATING) ===\n")
  for (b in benches) {
    cat(sprintf("\n--- Benchmark: %s ---\n", b))
    for (m in modes) {
      sub <- df_all[df_all$benchmark == b & df_all$mode == m$name, ]
      cat(sprintf("  %-24s | Edges=%5.1f | Terms=%4.1f | Exact=%.0f%% | RMSE=%.4f | R2=%.4f\n",
                  m$name,
                  mean(sub$active_edges, na.rm = TRUE),
                  mean(sub$active_terms, na.rm = TRUE),
                  100.0 * mean(sub$exact_recovery, na.rm = TRUE),
                  mean(sub$test_rmse, na.rm = TRUE),
                  mean(sub$test_r2, na.rm = TRUE)))
    }
  }
}
