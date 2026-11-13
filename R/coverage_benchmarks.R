#' Monte Carlo Frequentist Coverage Benchmark Under Repeated Informative Sampling (P_nu)
#'
#' Validates both uncertainty quantification frameworks implemented in \code{LBBNN}
#' across \code{M_MC} repeated informative survey samples drawn from a finite population
#' under \eqn{P_\nu}:
#' \enumerate{
#'   \item \strong{Target Class A — Domain Summary Estimators} (\code{predict_domains_lbbnn}):
#'     Compares nominal \code{level} credible interval coverage between the unadjusted
#'     pseudo-posterior (\code{dirichlet_uq = FALSE}) and the Dirichlet Bayesian
#'     bootstrap (\code{dirichlet_uq = TRUE}, Savitsky & Toth, 2016).
#'   \item \strong{Target Class B — Active MPM Slab Parameters} (\code{adjust_slab_sandwich_lbbnn}):
#'     Conditioned on the sparse Median Probability Model (MPM) active slab subspace
#'     discovered in Stage 1 (where Bernstein-von Mises regularity holds), compares
#'     unadjusted mean-field variational draws against the Williams & Savitsky (2021,
#'     \code{doi:10.1111/insr.12376}) Cholesky sandwich projection \eqn{R_2^\top R_1^{-\top}}.
#' }
#'
#' @param M_MC Integer number of Monte Carlo survey sample replicates (default 50).
#' @param N_pop Finite population size \eqn{N} (default 2000).
#' @param epochs Number of training epochs for Stage 1 sparse LBBNN structure discovery (default 200).
#' @param lr Learning rate for Stage 1 training (default 0.02).
#' @param hidden_units Number of hidden units in the discovery LBBNN (default 12).
#' @param prior Numeric length-2 prior inclusion probabilities (default \code{c(0.1, 0.3)}).
#' @param std Numeric length-2 prior standard deviations (default \code{c(1.0, 5.0)}).
#' @param sigma Observation noise standard deviation (default 0.40).
#' @param draws_domain Number of posterior draws per replicate for domain estimation (default 200).
#' @param draws_sandwich Number of posterior draws per replicate for sandwich projection (default 400).
#' @param level Nominal credible interval coverage level (default 0.95).
#' @param seed Random seed (default 2026).
#' @param verbose Logical; print progress if TRUE (default FALSE).
#' @return A list containing \code{domain_coverage}, \code{slab_coverage},
#'   \code{overall_summary}, \code{mpm_model}, and \code{sample_info}.
#' @export
run_coverage_benchmark_lbbnn <- function(M_MC = 50,
                                         N_pop = 2000,
                                         epochs = 200,
                                         lr = 0.02,
                                         hidden_units = 12,
                                         prior = c(0.1, 0.3),
                                         std = c(1.0, 5.0),
                                         sigma = 0.40,
                                         draws_domain = 200,
                                         draws_sandwich = 400,
                                         level = 0.95,
                                         seed = 2026,
                                         verbose = FALSE) {
  set.seed(seed)
  torch::torch_manual_seed(seed)

  # 1. Construct finite population U (size N_pop) with active linear/non-linear + noise covariates
  x1 <- stats::rnorm(N_pop, 0, 1)
  x2 <- stats::runif(N_pop, -1, 1)
  x3 <- stats::rnorm(N_pop, 0, 1)
  x4 <- stats::rnorm(N_pop, 0, 1)
  y_pop <- 0.0 + 2.0 * x1 + 1.5 * sin(pi * x2) + stats::rnorm(N_pop, 0, sigma)
  X_pop <- cbind(x1 = x1, x2 = x2, x3 = x3, x4 = x4)

  dom_pop <- cut(
    x1,
    breaks = c(-Inf, -0.4, 0.4, Inf),
    labels = c("Region_North", "Region_Central", "Region_South")
  )
  true_dom_means <- c(tapply(y_pop, dom_pop, mean), Population = mean(y_pop))

  # Informative inclusion probabilities
  pi_pop <- pmax(stats::plogis(-1.2 + 0.8 * y_pop), 0.06)

  # 2. Stage 1: LBBNN sparse structure discovery (pruning noise X3, X4; isolating active MPM slab)
  df_pop <- data.frame(X_pop, y = y_pop)
  dl_base <- get_dataloaders(
    df_pop,
    train_proportion = 1.0,
    train_batch_size = min(128L, N_pop),
    test_batch_size  = min(128L, N_pop),
    standardize      = FALSE,
    seed             = 42
  )
  net_base <- lbbnn_net(
    problem_type    = "regression",
    sizes           = c(4L, as.integer(hidden_units), 1L),
    prior           = prior,
    std             = std,
    inclusion_inits = "balanced",
    input_skip      = TRUE,
    sigma           = sigma,
    device          = "cpu"
  )
  train_lbbnn(
    epochs   = epochs,
    LBBNN    = net_base,
    lr       = lr,
    train_dl = dl_base$train_loader,
    verbose  = verbose
  )

  # Identify active MPM slab basis on U and compute exact census truth theta_0_slab
  sw_census <- adjust_slab_sandwich_lbbnn(
    net_base,
    X       = X_pop,
    y       = y_pop,
    weights = rep(1.0, N_pop),
    draws   = 200,
    level   = level,
    seed    = seed
  )
  theta_names <- sw_census$summary_table$Parameter

  x_t_pop <- torch::torch_tensor(X_pop, dtype = torch::torch_float32(), device = net_base$device)
  torch::with_no_grad({
    h_pop <- net_base$act(.forward_layer_mpm_mean(net_base$layers$children[[1]], x_t_pop, 0.5))
    H_full_pop <- as.matrix(torch::torch_cat(list(h_pop, x_t_pop), dim = 2)$cpu())
  })
  n_hid <- ncol(H_full_pop) - ncol(X_pop)
  col_labels <- c(paste0("Hidden_h", seq_len(n_hid)), paste0("Skip_", colnames(X_pop)))
  keep_cols <- match(setdiff(theta_names, "Intercept"), col_labels)

  H_pop_act <- cbind(Intercept = 1.0, H_full_pop[, keep_cols, drop = FALSE])
  d_slab <- ncol(H_pop_act)
  prior_sd_slab <- if (!is.null(net_base$prior_std) && length(net_base$prior_std) > 0) {
    as.numeric(net_base$prior_std[length(net_base$prior_std)])
  } else {
    1.0
  }
  prior_prec <- diag(1.0 / (prior_sd_slab^2), d_slab)
  sigma2 <- sigma^2
  theta_0_slab <- as.numeric(solve(
    crossprod(H_pop_act) / sigma2 + prior_prec,
    crossprod(H_pop_act, y_pop) / sigma2
  ))
  names(theta_0_slab) <- theta_names

  # Lock Stage 1 hidden MPM feature map cleanly so predict_domains_lbbnn & adjust_slab_sandwich_lbbnn
  # operate identically on the conditioned MPM slab subspace across replicates
  torch::with_no_grad({
    l1 <- net_base$layers$children[[1]]
    m1 <- (torch::torch_sigmoid(l1$lambda_l) > 0.5) * 1.0
    l1$weight_mean$mul_(m1)
    l1$weight_rho$fill_(-20.0)
    l1$bias_rho$fill_(-20.0)
    l1$lambda_l$copy_(torch::torch_where(m1 > 0.5, torch::torch_tensor(20.0), torch::torch_tensor(-20.0)))

    m_out <- rep(-20.0, length(col_labels))
    m_out[keep_cols] <- 20.0
    net_base$out_layer$lambda_l$copy_(torch::torch_tensor(matrix(m_out, nrow = 1)))
  })
  net_base$compute_paths_input_skip()

  softplus_inv <- function(s) log(pmax(exp(s) - 1.0, 1e-12))

  # 3. Repeated Sampling Evaluation Across M_MC Informative Survey Samples Under P_nu
  cov_unadj   <- matrix(NA, M_MC, d_slab)
  cov_sand    <- matrix(NA, M_MC, d_slab)
  wid_unadj   <- matrix(NA_real_, M_MC, d_slab)
  wid_sand    <- matrix(NA_real_, M_MC, d_slab)
  deff_slab   <- matrix(NA_real_, M_MC, d_slab)

  dom_cov_un  <- matrix(NA, M_MC, 4L)
  dom_cov_dir <- matrix(NA, M_MC, 4L)
  dom_wid_un  <- matrix(NA_real_, M_MC, 4L)
  dom_wid_dir <- matrix(NA_real_, M_MC, 4L)

  n_vec    <- numeric(M_MC)
  kish_vec <- numeric(M_MC)

  for (m in seq_len(M_MC)) {
    set.seed(seed + 1000L + m)
    inc <- stats::rbinom(N_pop, 1, pi_pop) == 1
    w_m <- 1.0 / pi_pop[inc]
    n_m <- length(w_m)
    n_vec[m] <- n_m
    kish_vec[m] <- n_m * sum(w_m^2) / (sum(w_m)^2)

    w_tilde <- w_m * (n_m / sum(w_m))
    y_m     <- y_pop[inc]
    X_m     <- X_pop[inc, , drop = FALSE]
    H_m     <- H_pop_act[inc, , drop = FALSE]

    G_m <- cbind(stats::model.matrix(~ dom_pop[inc] - 1), Population = 1.0)
    colnames(G_m) <- c(levels(dom_pop), "Population")

    # Weighted pseudo-posterior variational optimum in active MPM slab
    H1_m  <- crossprod(H_m, H_m * (w_tilde / sigma2)) + prior_prec
    mu_m  <- as.numeric(solve(H1_m, crossprod(H_m, y_m * w_tilde / sigma2)))
    sd_mf <- 1.0 / sqrt(diag(H1_m))

    torch::with_no_grad({
      w_full   <- numeric(length(col_labels))
      rho_full <- rep(-20.0, length(col_labels))
      w_full[keep_cols]   <- mu_m[-1]
      rho_full[keep_cols] <- softplus_inv(sd_mf[-1])
      net_base$out_layer$weight_mean$copy_(torch::torch_tensor(matrix(w_full, nrow = 1)))
      net_base$out_layer$weight_rho$copy_(torch::torch_tensor(matrix(rho_full, nrow = 1)))
      net_base$out_layer$bias_mean$copy_(torch::torch_tensor(mu_m[1]))
      net_base$out_layer$bias_rho$copy_(torch::torch_tensor(softplus_inv(sd_mf[1])))
    })

    # Target Class B: Active MPM Slab Parameters (Unadjusted vs Williams-Savitsky Sandwich)
    sw_m <- adjust_slab_sandwich_lbbnn(
      net_base,
      X       = X_m,
      y       = y_m,
      weights = w_m,
      draws   = draws_sandwich,
      level   = level,
      sigma   = sigma,
      seed    = seed + 1000L + m
    )
    tbl <- sw_m$summary_table
    idx <- match(theta_names, tbl$Parameter)
    cov_unadj[m, ] <- (tbl$Unadj_Low[idx] <= theta_0_slab & theta_0_slab <= tbl$Unadj_High[idx])
    cov_sand[m, ]  <- (tbl$Sand_Low[idx]  <= theta_0_slab & theta_0_slab <= tbl$Sand_High[idx])
    wid_unadj[m, ] <- tbl$Unadj_High[idx] - tbl$Unadj_Low[idx]
    wid_sand[m, ]  <- tbl$Sand_High[idx]  - tbl$Sand_Low[idx]
    deff_slab[m, ] <- tbl$DEFF_Ratio[idx]

    # Target Class A: Domain Summary Estimators (Unadjusted vs Dirichlet Bootstrap)
    d_un <- predict_domains_lbbnn(
      net_base,
      newdata       = X_m,
      domain_matrix = G_m,
      weights       = w_m,
      draws         = draws_domain,
      type          = "conditional_mean",
      dirichlet_uq  = FALSE,
      level         = level
    )
    d_dr <- predict_domains_lbbnn(
      net_base,
      newdata       = X_m,
      domain_matrix = G_m,
      weights       = w_m,
      draws         = draws_domain,
      type          = "conditional_mean",
      dirichlet_uq  = TRUE,
      level         = level
    )

    dom_cov_un[m, ]  <- (d_un$ci_lower <= true_dom_means & true_dom_means <= d_un$ci_upper)
    dom_cov_dir[m, ] <- (d_dr$ci_lower <= true_dom_means & true_dom_means <= d_dr$ci_upper)
    dom_wid_un[m, ]  <- d_un$ci_upper - d_un$ci_lower
    dom_wid_dir[m, ] <- d_dr$ci_upper - d_dr$ci_lower
  }

  domain_df <- data.frame(
    Domain               = names(true_dom_means),
    True_Mean            = as.numeric(true_dom_means),
    Unadj_Coverage       = colMeans(dom_cov_un),
    Dirichlet_Coverage   = colMeans(dom_cov_dir),
    Unadj_Mean_Width     = colMeans(dom_wid_un),
    Dirichlet_Mean_Width = colMeans(dom_wid_dir),
    row.names            = NULL,
    stringsAsFactors     = FALSE
  )

  slab_df <- data.frame(
    Parameter          = theta_names,
    Census_Truth       = as.numeric(theta_0_slab),
    Unadj_Coverage     = colMeans(cov_unadj),
    Sandwich_Coverage  = colMeans(cov_sand),
    Unadj_Mean_Width   = colMeans(wid_unadj),
    Sandwich_Mean_Width= colMeans(wid_sand),
    Mean_DEFF_Ratio    = colMeans(deff_slab),
    row.names          = NULL,
    stringsAsFactors   = FALSE
  )

  skip_x1_idx <- which(theta_names == "Skip_x1")
  overall_df <- data.frame(
    Target_Class = c(
      "Domain Summary Means (All 4 Domains/Pop)",
      "Active MPM Slab Weights (All Slab Params)",
      "Direct Input-Skip Slope (Skip_x1)"
    ),
    UQ_Method_Unadjusted = c(
      "Pseudo-Posterior (dirichlet_uq = FALSE)",
      "Raw Variational Slab Draws (Sigma_1)",
      "Raw Variational Slab Draws (Sigma_1)"
    ),
    Coverage_Unadjusted = c(
      mean(dom_cov_un),
      mean(cov_unadj),
      if (length(skip_x1_idx) == 1L) mean(cov_unadj[, skip_x1_idx]) else NA_real_
    ),
    UQ_Method_Corrected = c(
      "Dirichlet Bayesian Bootstrap (dirichlet_uq = TRUE)",
      "Williams-Savitsky (2021) Sandwich (R2^T R1^{-T})",
      "Williams-Savitsky (2021) Sandwich (R2^T R1^{-T})"
    ),
    Coverage_Corrected = c(
      mean(dom_cov_dir),
      mean(cov_sand),
      if (length(skip_x1_idx) == 1L) mean(cov_sand[, skip_x1_idx]) else NA_real_
    ),
    stringsAsFactors = FALSE
  )

  list(
    domain_coverage = domain_df,
    slab_coverage   = slab_df,
    overall_summary = overall_df,
    mpm_model       = net_base,
    sample_info     = list(
      N_pop     = N_pop,
      M_MC      = M_MC,
      mean_n    = mean(n_vec),
      mean_deff = mean(kish_vec),
      level     = level
    )
  )
}
