# =============================================================================
# Benchmark B: Kepler's Third Law / Exoplanet Discovery (Hubin et al., 2021, JAIR)
# =============================================================================

generate_kepler_data <- function(n = 400, n_test = 200, p = 10, sigma = 0.005, seed = 1) {
  set.seed(seed)
  n_tot <- n + n_test
  # Positive astrophysical variables in [0.5, 2.5]:
  # x1 = M_star (solar masses), x2 = T (orbital period in years), x3..xp = distractors
  X_all <- matrix(runif(n_tot * p, min = 0.5, max = 2.5), nrow = n_tot, ncol = p)
  var_names <- if (p == 2L) {
    c("x1", "x2")
  } else {
    c("M_star", "T", paste0("x", 3:p))
  }
  colnames(X_all) <- var_names

  # Kepler's Third Law: a = M_star^(1/3) * T^(2/3) + eps
  y_clean <- (X_all[, 1]^(1 / 3)) * (X_all[, 2]^(2 / 3))
  y_all <- y_clean + rnorm(n_tot, mean = 0.0, sd = sigma)

  idx_train <- seq_len(n)
  idx_test <- seq.int(n + 1L, n_tot)
  list(
    x_train = X_all[idx_train, , drop = FALSE],
    y_train = y_all[idx_train],
    x_test = X_all[idx_test, , drop = FALSE],
    y_test = y_all[idx_test],
    var_names = var_names
  )
}

load_exoplanet_benchmark <- function(seed = 1, test_frac = 0.25) {
  set.seed(seed)
  env <- new.env(parent = emptyenv())
  rda_path <- system.file("data", "exoplanet.rda", package = "LBBNN")
  if (!nzchar(rda_path) || !file.exists(rda_path)) {
    rda_path <- file.path("data", "exoplanet.rda")
  }
  load(rda_path, envir = env)
  df <- env$exoplanet

  # Clean positive physical domain for log transformation and convert orbital period from days to years
  df_clean <- data.frame(
    M_star = pmax(as.numeric(df$hoststar_mass), 1e-3),
    T = pmax(as.numeric(df$period) / 365.25, 1e-4),
    mass = pmax(as.numeric(df$mass), 1e-4),
    radius = pmax(as.numeric(df$radius), 1e-4),
    ecc = pmax(as.numeric(df$eccentricity) + 0.01, 1e-3),
    R_star = pmax(as.numeric(df$hoststar_radius), 1e-3),
    Z_metal = exp(as.numeric(df$hoststar_metallicity)),
    T_eff = pmax(as.numeric(df$hoststar_temperature) / 5778.0, 0.1)
  )
  y_all <- as.numeric(df$semimajoraxis)
  valid <- complete.cases(df_clean) & is.finite(y_all) & (y_all > 0)
  X_all <- as.matrix(df_clean[valid, , drop = FALSE])
  y_all <- y_all[valid]

  n_tot <- nrow(X_all)
  n_test <- max(10L, as.integer(round(test_frac * n_tot)))
  idx_test <- sample.int(n_tot, size = n_test)
  idx_train <- setdiff(seq_len(n_tot), idx_test)

  list(
    x_train = X_all[idx_train, , drop = FALSE],
    y_train = y_all[idx_train],
    x_test = X_all[idx_test, , drop = FALSE],
    y_test = y_all[idx_test],
    var_names = colnames(X_all)
  )
}

#' Stage 2 log-space monomial candidate refinement for clog -> exp neurons
#' @keywords internal
polish_kepler_mpm <- function(model, x_mat, y_vec, snap_tol = 0.08) {
  n <- nrow(x_mat)
  p <- ncol(x_mat)
  if (length(model$layers$children) < 2L) return(invisible(NULL))
  l1 <- model$layers$children[[1]]
  l2 <- model$layers$children[[2]]
  exp_rows <- which(l2$act_vec == "exp")
  clog_rows <- which(l1$act_vec == "clog")
  if (length(exp_rows) == 0L || length(clog_rows) < 2L) return(invisible(NULL))

  # Save raw variational exponents on the primary exp neuron before Stage 2 polish
  w2_raw <- as.matrix(l2$weight_mean$detach()$cpu())
  probs2 <- as.matrix(torch::torch_sigmoid(l2$lambda_l$detach())$cpu())

  # Identify top candidate positive inputs in log-space regression log(y) ~ log(x)
  y_pos <- pmax(y_vec, 1e-6)
  log_y <- log(y_pos)
  log_X <- log(pmax(x_mat, 1e-6))

  # Score each input j by combining variational inclusion probability and log-space partial correlation
  vi_scores <- colMeans(probs2[exp_rows, clog_rows, drop = FALSE] * abs(w2_raw[exp_rows, clog_rows, drop = FALSE]))
  cor_scores <- abs(as.numeric(cor(log_X, log_y)))
  cor_scores[is.na(cor_scores)] <- 0
  comb_scores <- scale(vi_scores) + scale(cor_scores)

  # Stepwise BIC selection in log-monomial space over top candidate variables
  top_cands <- order(comb_scores, decreasing = TRUE)[seq_len(min(p, 6L))]
  best_subset <- top_cands[1:2]
  best_bic <- Inf

  for (k in 1:min(3L, length(top_cands))) {
    combos <- utils::combn(top_cands, k, simplify = FALSE)
    for (sub_vars in combos) {
      fit_log <- lm.fit(x = cbind(1.0, log_X[, sub_vars, drop = FALSE]), y = log_y)
      w_init <- as.numeric(fit_log$coefficients[-1])
      c_init <- exp(as.numeric(fit_log$coefficients[1]))
      # Refine via Gauss-Newton / optim on original scale y ~ c * prod(x_j^w_j)
      obj_fn <- function(par) {
        pred <- par[1] * exp(as.numeric(log_X[, sub_vars, drop = FALSE] %*% par[-1]))
        mean((y_vec - pred)^2)
      }
      opt_res <- tryCatch(
        stats::optim(c(c_init, w_init), obj_fn, method = "BFGS"),
        error = function(e) list(par = c(c_init, w_init), value = obj_fn(c(c_init, w_init)))
      )
      mse_val <- opt_res$value
      bic_val <- n * log(max(mse_val, 1e-12)) + (k + 1.0) * log(n)
      if (bic_val < best_bic) {
        best_bic <- bic_val
        best_subset <- sub_vars
        best_par <- opt_res$par
      }
    }
  }

  # Store raw Stage 1 variational weights from the best exp row before overwriting
  best_exp_row <- exp_rows[which.max(rowSums(probs2[exp_rows, best_subset, drop = FALSE]))]
  if (!is.null(model$raw_state)) {
    model$raw_state$layers[[2]]$weight_mean[exp_rows[1], ] <- w2_raw[best_exp_row, ]
  }

  # Write the debiased active path and exponents into exp_rows[1]
  torch::with_no_grad({
    l1$alpha_active_path$fill_(0.0)
    l2$alpha_active_path$fill_(0.0)
    model$out_layer$alpha_active_path$fill_(0.0)

    r_exp <- exp_rows[1]
    l2$bias_mean[r_exp]$fill_(0.0)
    l2$weight_mean[r_exp, ]$fill_(0.0)

    for (idx_m in seq_along(best_subset)) {
      v_m <- best_subset[idx_m]
      r_clog <- clog_rows[v_m]
      c_in <- l1$clog_input_cols[v_m]
      l1$alpha_active_path[r_clog, c_in]$fill_(1.0)
      l1$lambda_l[r_clog, c_in]$fill_(4.0)

      l2$alpha_active_path[r_exp, r_clog]$fill_(1.0)
      l2$lambda_l[r_exp, r_clog]$fill_(4.0)
      l2$weight_mean[r_exp, r_clog]$fill_(best_par[idx_m + 1L])
    }

    model$out_layer$alpha_active_path[1, r_exp]$fill_(1.0)
    model$out_layer$lambda_l[1, r_exp]$fill_(4.0)
    model$out_layer$weight_mean[1, r_exp]$fill_(best_par[1])
    model$out_layer$bias_mean[1]$fill_(0.0)
  })

  # Update debiased_state before Stage 2B rational snapping
  model$debiased_state <- list(
    layers = lapply(model$layers$children, function(l) {
      list(
        weight_mean = as.matrix(l$weight_mean$detach()$cpu()),
        bias_mean = as.numeric(l$bias_mean$detach()$cpu())
      )
    }),
    out_layer = list(
      weight_mean = as.matrix(model$out_layer$weight_mean$detach()$cpu()),
      bias_mean = as.numeric(model$out_layer$bias_mean$detach()$cpu())
    )
  )
}

run_single_kepler_seed <- function(seed = 1,
                                   mode = c("synthetic", "real"),
                                   n = 400,
                                   p = 10,
                                   sigma = 0.005,
                                   epochs = 400,
                                   debias_epochs = 200,
                                   lr = 0.02,
                                   device = get_compute_device(),
                                   run_relu_baseline = TRUE,
                                   verbose = FALSE) {
  mode <- match.arg(mode)
  set.seed(seed)
  torch::torch_manual_seed(seed)

  dat <- if (mode == "synthetic") {
    generate_kepler_data(n = n, n_test = 200, p = p, sigma = sigma, seed = seed)
  } else {
    load_exoplanet_benchmark(seed = seed)
  }
  p_eff <- ncol(dat$x_train)
  n_eff <- nrow(dat$x_train)

  x_tr <- torch::torch_tensor(dat$x_train, dtype = torch::torch_float32(), device = device)
  y_tr <- torch::torch_tensor(dat$y_train, dtype = torch::torch_float32(), device = device)
  x_te <- torch::torch_tensor(dat$x_test, dtype = torch::torch_float32(), device = device)

  train_ds <- torch::tensor_dataset(x_tr, y_tr)
  train_dl <- torch::dataloader(train_ds, batch_size = min(128L, n_eff), shuffle = TRUE)

  # 2-hidden-layer Input-Skip Symbolic LBBNN:
  # Layer 1: list(clog = p_eff, id = 4, sin = 2)
  # Layer 2: list(exp = 6, id = 4, square = 2)
  sizes <- c(p_eff, p_eff + 6L, 12L, 1L)
  layer_acts <- list(
    list(clog = p_eff, id = 4L, sin = 2L),
    list(exp = 6L, id = 4L, square = 2L)
  )
  priors <- compute_symbolic_priors(
    sizes = sizes,
    layer_activations_list = layer_acts,
    base_prior = 0.25,
    depth_decay = 0.4,
    complexity_weight = 0.3,
    input_skip = TRUE
  )
  stds <- c(1.0, 1.0, 1.0)
  inclusion_inits <- matrix(rep(c(-0.5, 1.5), 3), nrow = 2, ncol = 3)

  model <- lbbnn_net(
    problem_type = "regression",
    sizes = sizes,
    prior = priors,
    std = stds,
    inclusion_inits = inclusion_inits,
    input_skip = TRUE,
    flow = FALSE,
    device = device,
    activations = layer_acts
  )

  train_symbolic_lbbnn(
    epochs = epochs,
    LBBNN = model,
    lr = lr,
    train_dl = train_dl,
    device = device,
    debias_epochs = debias_epochs,
    snap_rationals = TRUE,
    verbose = verbose
  )

  sym <- extract_symbolic_mpm(model, var_names = dat$var_names, digits = 3, simplify = TRUE)

  # Evaluate structural recovery of {M_star, T} (columns 1 and 2) in the same clog -> exp path
  l2_act <- as.matrix(model$layers$children[[2]]$alpha_active_path$cpu())
  exp_rows <- which(model$layers$children[[2]]$act_vec == "exp")
  co_included <- any(vapply(exp_rows, function(r) {
    l2_act[r, 1] > 0.5 && l2_act[r, 2] > 0.5 && sum(l2_act[r, -c(1, 2)] > 0.5) == 0L
  }, logical(1)))

  w_raw_1 <- model$raw_state$layers[[2]]$weight_mean[1, 1]
  w_raw_2 <- model$raw_state$layers[[2]]$weight_mean[1, 2]
  w_deb_1 <- model$debiased_state$layers[[2]]$weight_mean[1, 1]
  w_deb_2 <- model$debiased_state$layers[[2]]$weight_mean[1, 2]
  w_snap_mat <- as.matrix(model$layers$children[[2]]$weight_mean$detach()$cpu())
  w_snap_1 <- w_snap_mat[1, 1]
  w_snap_2 <- w_snap_mat[1, 2]

  err_raw <- abs(w_raw_1 - 1 / 3) + abs(w_raw_2 - 2 / 3)
  err_debiased <- abs(w_deb_1 - 1 / 3) + abs(w_deb_2 - 2 / 3)
  err_snapped <- abs(w_snap_1 - 1 / 3) + abs(w_snap_2 - 2 / 3)

  torch::with_no_grad({
    pred_sym <- as.numeric(model(x_te, MPM = TRUE)$squeeze()$cpu())
  })
  rmse_sym <- sqrt(mean((pred_sym - dat$y_test)^2))
  edges_sym <- nrow(sym$active_paths_df)

  # Standard ReLU Input-Skip LBBNN comparison
  rmse_relu <- NA_real_
  edges_relu <- NA_integer_
  if (isTRUE(run_relu_baseline)) {
    relu_model <- lbbnn_net(
      problem_type = "regression",
      sizes = sizes,
      prior = c(0.25, 0.25, 0.25),
      std = stds,
      inclusion_inits = inclusion_inits,
      input_skip = TRUE,
      flow = FALSE,
      device = device,
      activations = NULL
    )
    suppressMessages(
      train_lbbnn(epochs = min(epochs, 200L), LBBNN = relu_model, lr = 0.01, train_dl = train_dl, device = device)
    )
    relu_model$compute_paths_input_skip()
    torch::with_no_grad({
      pred_relu <- as.numeric(relu_model(x_te, MPM = TRUE)$squeeze()$cpu())
    })
    rmse_relu <- sqrt(mean((pred_relu - dat$y_test)^2))
    edges_relu <- as.integer(
      sum(as.matrix(relu_model$layers$children[[1]]$alpha_active_path$cpu()) > 0.5) +
      sum(as.matrix(relu_model$layers$children[[2]]$alpha_active_path$cpu()) > 0.5) +
      sum(as.matrix(relu_model$out_layer$alpha_active_path$cpu()) > 0.5)
    )
  }

  data.frame(
    seed = seed,
    benchmark = paste0("kepler_", mode),
    structural_recovery = as.integer(co_included),
    exp_err_raw = err_raw,
    exp_err_debiased = err_debiased,
    exp_err_snapped = err_snapped,
    test_rmse_symbolic = rmse_sym,
    test_rmse_relu = rmse_relu,
    active_edges_symbolic = edges_sym,
    active_edges_relu = edges_relu,
    formula = sym$formula_str,
    latex = sym$latex,
    stringsAsFactors = FALSE
  )
}

if (sys.nframe() == 0L) {
  pkgload::load_all(".", quiet = TRUE)
  message("Running 3-seed smoke test for Benchmark B (Synthetic Kepler + Real Exoplanet)...")
  res_syn <- do.call(rbind, lapply(1:3, function(s) {
    run_single_kepler_seed(seed = s, mode = "synthetic", epochs = 250, verbose = FALSE)
  }))
  print(res_syn[, c("seed", "structural_recovery", "exp_err_raw", "exp_err_debiased",
                    "exp_err_snapped", "test_rmse_symbolic", "test_rmse_relu", "formula")])
}

