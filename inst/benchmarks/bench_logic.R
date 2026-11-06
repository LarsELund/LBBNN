# =============================================================================
# Benchmark A: Bayesian Logic Regression (Hubin et al., 2020, Bayesian Analysis)
# =============================================================================

generate_logic_data <- function(n = 1000, n_test = 500, p = 20, sigma = 0.5, seed = 1) {
  set.seed(seed)
  n_tot <- n + n_test
  X_all <- matrix(rbinom(n_tot * p, size = 1, prob = 0.5), nrow = n_tot, ncol = p)
  colnames(X_all) <- paste0("x", seq_len(p))

  # True data-generating mechanism:
  # Y = 1.0 + 1.5*(X1 & X2) + 3.0*(X4 & !X5) + 2.0*(X7 & X8 & X11) + eps
  c1 <- X_all[, 1] * X_all[, 2]
  c2 <- X_all[, 4] * (1 - X_all[, 5])
  c3 <- X_all[, 7] * X_all[, 8] * X_all[, 11]
  y_true <- 1.0 + 1.5 * c1 + 3.0 * c2 + 2.0 * c3
  y_all <- y_true + rnorm(n_tot, mean = 0.0, sd = sigma)

  idx_train <- seq_len(n)
  idx_test <- seq.int(n + 1L, n_tot)

  list(
    x_train = X_all[idx_train, , drop = FALSE],
    y_train = y_all[idx_train],
    x_test = X_all[idx_test, , drop = FALSE],
    y_test = y_all[idx_test],
    true_clauses = c("(x1 & x2)", "(x4 & !x5)", "(x7 & x8 & x11)")
  )
}

#' Candidate-assisted Stage 2 logic clause refinement (BGNLM hybrid MPM polish)
#' @keywords internal
polish_logic_mpm <- function(model, x_mat, y_vec, max_clauses = 5L) {
  n <- nrow(x_mat)
  p <- ncol(x_mat)
  l1 <- model$layers$children[[1]]
  if (is.null(l1$logic_rows) || length(l1$logic_rows) == 0L) return(invisible(NULL))

  # Gather candidate conjunctions discovered across all logic_and neurons (using top alpha/weight signals)
  probs_mat <- as.matrix(torch::torch_sigmoid(l1$lambda_l$detach())$cpu())
  w_mat <- as.matrix(l1$weight_mean$detach()$cpu())
  act_mat <- as.matrix(l1$alpha_active_path$detach()$cpu())

  cand_list <- list()
  add_cand <- function(cols, signs) {
    if (length(cols) == 0L || length(cols) > 4L) return()
    ord <- order(cols)
    cols <- cols[ord]
    signs <- signs[ord]
    key <- paste0(cols, ":", ifelse(signs >= 0, "+", "-"), collapse = "&")
    cand_list[[key]] <<- list(cols = cols, signs = ifelse(signs >= 0, 1.0, -1.0))
  }

  for (r in l1$logic_rows) {
    c_act <- which(act_mat[r, ] > 0.5)
    if (length(c_act) > 0L) add_cand(c_act, w_mat[r, c_act])
    # Also inspect top 2 and top 3 highest-alpha * |w| literals in neuron r
    score_r <- probs_mat[r, ] * abs(w_mat[r, ])
    top_k <- order(score_r, decreasing = TRUE)
    for (k in 2:min(3L, p)) {
      cols_k <- top_k[seq_len(k)]
      add_cand(cols_k, w_mat[r, cols_k])
    }
  }

  # Also screen residual-correlated 2-way and 3-way conjunctions among top marginal/interaction features
  eval_clause <- function(cols, signs, X) {
    out <- rep(1.0, nrow(X))
    for (i in seq_along(cols)) {
      v <- X[, cols[i]]
      out <- out * (if (signs[i] >= 0) v else (1.0 - v))
    }
    out
  }

  # Compute marginal + pairwise correlations with y_vec to seed any missed interaction among active features
  top_vars <- order(vapply(seq_len(p), function(j) {
    max(abs(cor(x_mat[, j], y_vec)), max(probs_mat[l1$logic_rows, j]))
  }, numeric(1)), decreasing = TRUE)[seq_len(min(p, 12L))]

  for (i1 in seq_along(top_vars)) {
    for (i2 in seq_along(top_vars)) {
      if (i2 <= i1) next
      v1 <- top_vars[i1]
      v2 <- top_vars[i2]
      for (s1 in c(1, -1)) {
        for (s2 in c(1, -1)) {
          add_cand(c(v1, v2), c(s1, s2))
        }
      }
    }
  }

  # Evaluate candidate clauses via forward stepwise BIC selection
  cand_keys <- names(cand_list)
  if (length(cand_keys) == 0L) return(invisible(NULL))
  C_mat <- vapply(cand_list, function(cl) eval_clause(cl$cols, cl$signs, x_mat), numeric(n))
  if (!is.matrix(C_mat)) C_mat <- matrix(C_mat, nrow = n)

  # Also check 3-way extensions of the top 5 2-way candidates with remaining variables
  cors2 <- abs(as.numeric(cor(C_mat, y_vec)))
  cors2[is.na(cors2)] <- 0
  top2_idx <- order(cors2, decreasing = TRUE)[seq_len(min(6L, ncol(C_mat)))]
  for (idx2 in top2_idx) {
    cl2 <- cand_list[[idx2]]
    if (length(cl2$cols) == 2L && all(cl2$signs > 0)) {
      for (v3 in top_vars) {
        if (!v3 %in% cl2$cols) {
          add_cand(c(cl2$cols, v3), c(cl2$signs, 1))
        }
      }
    }
  }
  C_mat <- vapply(cand_list, function(cl) eval_clause(cl$cols, cl$signs, x_mat), numeric(n))
  if (!is.matrix(C_mat)) C_mat <- matrix(C_mat, nrow = n)

  selected_idx <- integer(0)
  curr_mse <- mean((y_vec - mean(y_vec))^2)
  curr_bic <- n * log(max(curr_mse, 1e-12)) + 1.0 * log(n)

  for (step in seq_len(min(max_clauses, length(l1$logic_rows)))) {
    best_bic <- curr_bic
    best_j <- NA_integer_
    for (j in seq_along(cand_list)) {
      if (j %in% selected_idx) next
      try_idx <- c(selected_idx, j)
      X_try <- cbind(1.0, C_mat[, try_idx, drop = FALSE])
      fit_b <- tryCatch(qr.solve(X_try, y_vec), error = function(e) NULL)
      if (is.null(fit_b)) next
      res <- y_vec - as.numeric(X_try %*% fit_b)
      mse_try <- mean(res^2)
      # Each clause uses length(cols) literals + 1 output weight
      n_lits <- sum(vapply(cand_list[try_idx], function(cl) length(cl$cols), integer(1)))
      k_try <- 1.0 + length(try_idx) + 0.5 * n_lits
      bic_try <- n * log(max(mse_try, 1e-12)) + k_try * log(n)
      if (bic_try < best_bic - 4.0) {
        best_bic <- bic_try
        best_j <- j
      }
    }
    if (is.na(best_j)) break
    selected_idx <- c(selected_idx, best_j)
    curr_bic <- best_bic
  }

  if (length(selected_idx) > 0L) {
    torch::with_no_grad({
      l1$alpha_active_path$fill_(0.0)
      model$out_layer$alpha_active_path$fill_(0.0)
      for (k in seq_along(selected_idx)) {
        r_row <- l1$logic_rows[k]
        cl <- cand_list[[selected_idx[k]]]
        for (m in seq_along(cl$cols)) {
          col_m <- cl$cols[m]
          l1$alpha_active_path[r_row, col_m]$fill_(1.0)
          l1$lambda_l[r_row, col_m]$fill_(4.0)
          l1$weight_mean[r_row, col_m]$fill_(cl$signs[m])
        }
        model$out_layer$alpha_active_path[1, r_row]$fill_(1.0)
        model$out_layer$lambda_l[1, r_row]$fill_(4.0)
      }
    })
  }
}

run_single_logic_seed <- function(seed = 1,
                                  n = 1000,
                                  n_test = 500,
                                  p = 20,
                                  sigma = 0.5,
                                  epochs = 350,
                                  debias_epochs = 100,
                                  lr = 0.025,
                                  device = get_compute_device(),
                                  verbose = FALSE) {
  set.seed(seed)
  torch::torch_manual_seed(seed)

  dat <- generate_logic_data(n = n, n_test = n_test, p = p, sigma = sigma, seed = seed)
  x_tr <- torch::torch_tensor(dat$x_train, dtype = torch::torch_float32(), device = device)
  y_tr <- torch::torch_tensor(dat$y_train, dtype = torch::torch_float32(), device = device)
  x_te <- torch::torch_tensor(dat$x_test, dtype = torch::torch_float32(), device = device)

  train_ds <- torch::tensor_dataset(x_tr, y_tr)
  train_dl <- torch::dataloader(train_ds, batch_size = min(256L, n), shuffle = TRUE)

  sizes <- c(p, 12L, 1L)
  layer_acts <- list(list(logic_and = 10L, id = 2L))
  priors <- compute_symbolic_priors(
    sizes = sizes,
    layer_activations_list = layer_acts,
    base_prior = 0.15,
    depth_decay = 0.4,
    complexity_weight = 0.3,
    input_skip = TRUE
  )
  stds <- c(1.0, 1.0)
  inclusion_inits <- matrix(rep(c(-1.0, 1.0), 2), nrow = 2, ncol = 2)

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
    snap_rationals = FALSE,
    verbose = verbose
  )

  # Polish Stage 2 MPM logic clauses and refit output OLS weights
  polish_logic_mpm(model, dat$x_train, dat$y_train)
  refit_output_ols(model, list(x = x_tr, y = y_tr, n = n))

  sym <- extract_symbolic_mpm(model, digits = 3, simplify = TRUE)
  discovered_terms <- setdiff(names(sym$basis_terms), "(Intercept)")

  tp <- sum(dat$true_clauses %in% discovered_terms)
  fp <- sum(!discovered_terms %in% dat$true_clauses)
  tpr <- tp / length(dat$true_clauses)
  # Total possible non-true candidate terms in the 12-unit + 20-skip architecture
  fpr <- fp / (12 + p - length(dat$true_clauses))
  exact_match <- as.integer(tp == length(dat$true_clauses) && fp == 0L)

  torch::with_no_grad({
    pred_test <- as.numeric(model(x_te, MPM = TRUE)$squeeze()$cpu())
  })
  test_rmse <- sqrt(mean((pred_test - dat$y_test)^2))

  data.frame(
    seed = seed,
    benchmark = "logic",
    tpr = tpr,
    fpr = fpr,
    exact_match = exact_match,
    test_rmse = test_rmse,
    active_density = model$density_active_path(),
    active_edges = nrow(sym$active_paths_df),
    formula = sym$formula_str,
    latex = sym$latex,
    stringsAsFactors = FALSE
  )
}

if (sys.nframe() == 0L) {
  pkgload::load_all(".", quiet = TRUE)
  message("Running 3-seed smoke test for Benchmark A (Bayesian Logic Regression)...")
  res <- do.call(rbind, lapply(1:3, function(s) run_single_logic_seed(seed = s, epochs = 250, verbose = FALSE)))
  print(res[, c("seed", "tpr", "fpr", "exact_match", "test_rmse", "active_density", "formula")])
}

