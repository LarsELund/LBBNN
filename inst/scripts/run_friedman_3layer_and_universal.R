#!/usr/bin/env Rscript
# =============================================================================
# Step 3: Friedman #1 (2-Layer Backward BIC Readout Pruning & 3-Layer
# clog -> exp -> sin Exact sin(pi*X1*X2) Discovery across M = 100 seeds) +
# Universal Activation Bank Verification across all 5 Benchmarks.
# =============================================================================

suppressPackageStartupMessages({
  pkgload::load_all(".", quiet = TRUE)
  library(parallel)
})
source("inst/benchmarks/bench_logic.R")
source("inst/benchmarks/bench_kepler.R")
source("inst/benchmarks/bench_nonlinear.R")
source("inst/benchmarks/bench_sindy_3d.R")

n_cores <- max(1L, min(8L, parallel::detectCores(logical = FALSE) - 1L))

run_psock_map <- function(items, worker_fn, cores = n_cores) {
  cl <- parallel::makePSOCKcluster(cores)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterEvalQ(cl, {
    suppressPackageStartupMessages(pkgload::load_all(".", quiet = TRUE))
    torch::torch_set_num_threads(1L)
    source("inst/benchmarks/bench_logic.R")
    source("inst/benchmarks/bench_kepler.R")
    source("inst/benchmarks/bench_nonlinear.R")
    source("inst/benchmarks/bench_sindy_3d.R")
    NULL
  })
  parallel::parLapplyLB(cl, items, worker_fn)
}

run_friedman_variant <- function(arch = c("2layer_pruned", "3layer"), n_seeds = 100L) {
  arch <- match.arg(arch)
  ckpt_dir <- file.path("results_baselines/nonlinear", paste0("checkpoints_", arch))
  dir.create(ckpt_dir, recursive = TRUE, showWarnings = FALSE)
  message(sprintf("Running Friedman #1 variant '%s' across M = %d seeds (%d PSOCK workers)...",
                  arch, n_seeds, n_cores))

  worker <- function(s) {
    ckpt_f <- file.path(ckpt_dir, sprintf("seed_%04d.rds", s))
    if (file.exists(ckpt_f)) return(readRDS(ckpt_f))
    set.seed(s); torch::torch_manual_seed(s)
    res <- run_single_nonlinear_seed(
      seed = s, n = 500, n_test = 300, p = 10, sigma = 1.0,
      epochs = 120, debias_epochs = 60, lr = 0.02,
      arch = arch, device = "cpu", verbose = FALSE
    )
    saveRDS(res, ckpt_f)
    res
  }

  cl <- parallel::makePSOCKcluster(n_cores)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterExport(cl, c("arch", "ckpt_dir"), envir = environment())
  parallel::clusterEvalQ(cl, {
    suppressPackageStartupMessages(pkgload::load_all(".", quiet = TRUE))
    torch::torch_set_num_threads(1L)
    source("inst/benchmarks/bench_nonlinear.R")
    NULL
  })
  res_list <- parallel::parLapplyLB(cl, seq_len(n_seeds), worker)
  raw_df <- do.call(rbind, res_list)
  utils::write.csv(raw_df, file.path("results_baselines/nonlinear", paste0("raw_friedman_", arch, ".csv")), row.names = FALSE)
  raw_df
}

run_universal_bank_check <- function(seeds = 1:10) {
  message(sprintf("Running Universal Activation Bank Verification across %d seeds x 5 tasks...", length(seeds)))
  # 1. Binary Logic Regression with continuous + logic operators active in Layer 1
  # 2. Continuous tasks (Kepler, NASA Exoplanets, Friedman, 3D SINDy) with logic_and/logic_or + continuous operators
  worker_univ <- function(task_seed_pair) {
    torch::torch_set_num_threads(1L)
    task <- task_seed_pair$task
    s <- task_seed_pair$seed
    set.seed(s); torch::torch_manual_seed(s)

    if (task == "logic") {
      dat <- generate_logic_data(n = 1000, n_test = 500, p = 20, sigma = 0.5, seed = s)
      p <- ncol(dat$x_train); q <- 1L
      sizes <- c(p, 20L, 1L)
      # Universal Layer 1: logic gates + continuous algebraic/trigonometric units
      acts <- list(list(logic_and = 8L, logic_or = 4L, clog = 2L, square = 2L, sin = 2L, id = 2L))
      inc <- matrix(c(-1.0, 1.0, -1.0, 1.0), 2, 2)
      comp_w <- 0.35; base_p <- 0.15
    } else if (task %in% c("kepler", "kepler_real")) {
      dat <- if (task == "kepler") generate_kepler_data(seed = s) else load_exoplanet_benchmark(seed = s)
      p <- ncol(dat$x_train); q <- 1L
      sizes <- c(p, p + 8L, 14L, 1L)
      # Universal bank: includes mismatched logic_and/logic_or gates in Layer 1 & Layer 2
      acts <- list(
        list(clog = p, id = 4L, sin = 2L, logic_and = 2L),
        list(exp = 4L, id = 4L, square = 2L, sin = 2L, logic_and = 2L)
      )
      inc <- matrix(rep(c(-0.5, 1.5), 3), 2, 3)
      comp_w <- 0.30; base_p <- 0.25
    } else if (task == "friedman") {
      dat <- generate_friedman1_data(seed = s)
      p <- ncol(dat$x_train); q <- 1L
      sizes <- c(p, 14L, 10L, 10L, 1L)
      acts <- list(
        list(clog = p, id = 2L, logic_and = 2L),
        list(exp = 3L, square = 2L, id = 3L, logic_and = 2L),
        list(sin = 3L, id = 5L, logic_or = 2L)
      )
      inc <- matrix(rep(c(-0.5, 1.5), 4), 2, 4)
      comp_w <- 0.25; base_p <- 0.25
    } else {
      dat <- generate_sindy3d_data(seed = s)
      p <- ncol(dat$x_train); q <- 3L
      sizes <- c(p, 20L, 16L, 3L)
      acts <- list(
        list(clog = p, id = 4L, sin = 4L, square = 4L, logic_and = 2L),
        list(exp = 2L, id = 4L, square = 6L, sin = 2L, logic_and = 2L)
      )
      inc <- matrix(rep(c(-0.5, 1.5), 3), 2, 3)
      comp_w <- 0.35; base_p <- 0.20
    }

    x_tr <- torch::torch_tensor(dat$x_train, dtype = torch::torch_float32())
    y_tr <- torch::torch_tensor(dat$y_train, dtype = torch::torch_float32())
    x_te <- torch::torch_tensor(dat$x_test, dtype = torch::torch_float32())
    dl <- torch::dataloader(torch::tensor_dataset(x_tr, y_tr), batch_size = min(128L, nrow(dat$x_train)), shuffle = TRUE)

    priors <- compute_symbolic_priors(sizes, acts, base_prior = base_p, depth_decay = 0.4, complexity_weight = comp_w, input_skip = TRUE)
    m <- lbbnn_symbolic_net("regression", sizes, priors, rep(1.0, length(sizes) - 1L), inc, acts)
    train_symbolic_lbbnn(epochs = 120, LBBNN = m, lr = 0.02, train_dl = dl, debias_epochs = 60,
                         snap_rationals = TRUE, prune_bic = TRUE, prune_readout = TRUE, verbose = FALSE)
    sym <- extract_symbolic_mpm(m, var_names = colnames(dat$x_train), digits = 3)
    df_e <- sym$active_paths_df

    # Count mismatched operators surviving Stage 1 MPM vs Post-Training MPM
    mismatch_ops <- if (task == "logic") c("clog", "log", "exp", "square", "sin", "cos", "inv", "sqrt") else c("logic_and", "logic_or")
    surv_post <- sum(df_e$operator %in% mismatch_ops)

    # Also check raw Stage 1 inclusion probability on mismatched rows
    raw_mismatch_max_alpha <- 0.0
    for (l_i in seq_along(m$layers$children)) {
      l_obj <- m$layers$children[[l_i]]
      bad_r <- which(l_obj$act_vec %in% mismatch_ops)
      if (length(bad_r) > 0L && !is.null(m$raw_state)) {
        raw_a <- m$raw_state$layers[[l_i]]$alpha[bad_r, , drop = FALSE]
        raw_mismatch_max_alpha <- max(raw_mismatch_max_alpha, max(raw_a))
      }
    }

    torch::with_no_grad({
      pr <- m(x_te, MPM = TRUE)$cpu()
      pr_m <- if (q == 1L) as.numeric(pr$squeeze(dim = 2L)) else as.matrix(pr)
    })
    rmse <- sqrt(mean((pr_m - dat$y_test)^2))

    data.frame(
      task = task, seed = s, test_rmse = rmse,
      n_active_edges = nrow(df_e),
      mismatched_edges_post = surv_post,
      any_mismatched_post = as.integer(surv_post > 0L),
      raw_mismatch_max_alpha = raw_mismatch_max_alpha,
      stringsAsFactors = FALSE
    )
  }

  tasks <- c("logic", "kepler", "kepler_real", "friedman", "sindy_3d")
  grid <- unlist(lapply(tasks, function(t) lapply(seeds, function(s) list(task = t, seed = s))), recursive = FALSE)
  rows <- run_psock_map(grid, worker_univ, cores = n_cores)
  raw_u <- do.call(rbind, rows)
  utils::write.csv(raw_u, "results_baselines/raw_universal_bank_verification.csv", row.names = FALSE)

  sum_u <- do.call(rbind, lapply(tasks, function(t) {
    sub <- raw_u[raw_u$task == t, ]
    data.frame(
      task = t,
      n_seeds = nrow(sub),
      mismatched_pruned_pct = 100.0 * (1.0 - mean(sub$any_mismatched_post)),
      mean_mismatched_edges_post = mean(sub$mismatched_edges_post),
      mean_raw_mismatch_max_alpha = mean(sub$raw_mismatch_max_alpha),
      mean_test_rmse = mean(sub$test_rmse),
      sd_test_rmse = stats::sd(sub$test_rmse),
      stringsAsFactors = FALSE
    )
  }))
  utils::write.csv(sum_u, "results_baselines/universal_bank_verification.csv", row.names = FALSE)
  print(sum_u)
  sum_u
}

if (sys.nframe() == 0L) {
  df_2p <- run_friedman_variant("2layer_pruned", n_seeds = 100L)
  df_3l <- run_friedman_variant("3layer", n_seeds = 100L)

  sum_nl <- rbind(
    data.frame(
      variant = "Symbolic-LBBNN (2-Layer, Unpruned MPM baseline)",
      n_seeds = 100L,
      tpr_mean = NA_real_, fpr_mean = NA_real_, exact_var_mean = NA_real_, exact_eq_mean = 0.0,
      n_edges_mean = 58.99, test_rmse_mean = 1.1612, test_r2_mean = NA_real_
    ),
    data.frame(
      variant = "Symbolic-LBBNN (2-Layer + Readout Pruning)",
      n_seeds = nrow(df_2p),
      tpr_mean = mean(df_2p$tpr), fpr_mean = mean(df_2p$fpr),
      exact_var_mean = mean(df_2p$exact_var_recovery), exact_eq_mean = mean(df_2p$exact_eq_recovery),
      n_edges_mean = mean(df_2p$active_edges), test_rmse_mean = mean(df_2p$test_rmse),
      test_r2_mean = mean(df_2p$test_r2)
    ),
    data.frame(
      variant = "Symbolic-LBBNN (3-Layer clog->exp->sin)",
      n_seeds = nrow(df_3l),
      tpr_mean = mean(df_3l$tpr), fpr_mean = mean(df_3l$fpr),
      exact_var_mean = mean(df_3l$exact_var_recovery), exact_eq_mean = mean(df_3l$exact_eq_recovery),
      n_edges_mean = mean(df_3l$active_edges), test_rmse_mean = mean(df_3l$test_rmse),
      test_r2_mean = mean(df_3l$test_r2)
    )
  )
  utils::write.csv(sum_nl, "results_baselines/nonlinear/friedman_3layer_summary.csv", row.names = FALSE)
  print(sum_nl)
  run_universal_bank_check(seeds = 1:10)
}
