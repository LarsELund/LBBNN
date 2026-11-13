test_that("adjust_slab_sandwich_lbbnn matches exact Cholesky sandwich covariance and handles strata/psu", {
  skip_if_not_installed("torch")
  set.seed(101)
  torch::torch_manual_seed(101)

  n <- 120
  x1 <- rnorm(n)
  x2 <- runif(n, -1, 1)
  x3 <- rnorm(n)
  y <- 2.0 * x1 - 1.2 * x2 + rnorm(n, 0, 0.4)
  strata <- factor(rep(c("S1", "S2", "S3"), each = 40))
  psu <- factor(rep(1:20, each = 6))

  # Informative unequal survey weights
  pi_inc <- plogis(-0.5 + 0.6 * y)
  w_raw <- 1 / pi_inc
  w_norm <- w_raw * (n / sum(w_raw))

  df <- data.frame(x1 = x1, x2 = x2, x3 = x3, y = y)
  dl <- get_dataloaders(
    df,
    train_proportion = 1.0,
    train_batch_size = 60,
    test_batch_size  = 60,
    weights          = w_norm,
    standardize      = FALSE,
    seed             = 101
  )

  net <- lbbnn_net(
    problem_type    = "regression",
    sizes           = c(3L, 4L, 1L),
    prior           = c(0.3, 0.5),
    std             = c(1.0, 5.0),
    inclusion_inits = "balanced",
    input_skip      = TRUE,
    sigma           = 0.4,
    device          = "cpu"
  )
  train_lbbnn(
    epochs   = 25,
    LBBNN    = net,
    lr       = 0.03,
    train_dl = dl$train_loader,
    device   = "cpu",
    verbose  = FALSE
  )

  X_mat <- as.matrix(df[, c("x1", "x2", "x3")])
  res <- adjust_slab_sandwich_lbbnn(
    object        = net,
    X             = X_mat,
    y             = y,
    weights       = w_norm,
    draws         = 250,
    active_cutoff = 0.5,
    level         = 0.95,
    sigma         = 0.4,
    seed          = 101
  )

  expect_true(is.list(res))
  expect_true(nrow(res$summary_table) >= 1L)
  expect_equal(ncol(res$draws_adjusted), nrow(res$summary_table))
  expect_equal(nrow(res$draws_adjusted), 250L)

  # Positive definiteness of bread, meat, and sandwich matrices
  eig_H <- eigen(res$H1_bread, symmetric = TRUE, only.values = TRUE)$values
  eig_J <- eigen(res$J2_meat, symmetric = TRUE, only.values = TRUE)$values
  eig_S <- eigen(res$Sigma_sandwich, symmetric = TRUE, only.values = TRUE)$values
  expect_true(all(eig_H > 0))
  expect_true(all(eig_J > 0))
  expect_true(all(eig_S > 0))

  # Empirical cov of adjusted draws vs Sigma_sandwich matches Cholesky algebra
  # (since draws_unadjusted are i.i.d. N(theta_hat, Sigma_1) with finite S=250 Monte Carlo error,
  # check standardized transformation accuracy: R2^T R1^{-T} Cov(draws_unadjusted) R1^{-1} R2 == Cov(draws_adjusted))
  cov_diff <- max(abs(stats::cov(res$draws_adjusted) - res$Sigma_sandwich))
  expect_lt(cov_diff, max(abs(res$Sigma_sandwich)) * 0.35 + 1e-4)

  # Exact algebraic sandwich identity on R1, R2:
  sigma_from_factors <- crossprod(res$R2)
  expect_lt(max(abs(sigma_from_factors - res$Sigma_sandwich)), 1e-10)

  # Unequal weights inflate sandwich SD relative to unadjusted mean-field VI SD on average
  expect_gt(mean(res$summary_table$DEFF_Ratio), 1.0)

  # Clustered / stratified design path executes cleanly
  res_cl <- adjust_slab_sandwich_lbbnn(
    object        = net,
    X             = X_mat,
    y             = y,
    weights       = w_norm,
    strata        = strata,
    psu           = psu,
    draws         = 100,
    active_cutoff = 0.5,
    sigma         = 0.4,
    seed          = 101
  )
  expect_true(all(is.finite(res_cl$summary_table$Sandwich_SD)))
  expect_equal(nrow(res_cl$summary_table), nrow(res$summary_table))

  # Single-domain predict_domains_lbbnn regression check (Dirichlet vs non-Dirichlet)
  dom_single <- factor(rep("All", n))
  set.seed(101)
  d_nodir <- predict_domains_lbbnn(
    net, X_mat, domain_matrix = dom_single, weights = w_norm,
    draws = 80, dirichlet_uq = FALSE
  )
  set.seed(101)
  d_dir <- predict_domains_lbbnn(
    net, X_mat, domain_matrix = dom_single, weights = w_norm,
    draws = 80, dirichlet_uq = TRUE
  )
  expect_equal(nrow(d_dir), 1L)
  expect_gt(d_dir$sd[1], d_nodir$sd[1])
})

test_that("run_coverage_benchmark_lbbnn runs end-to-end and returns well-formed coverage tables", {
  skip_if_not_installed("torch")
  bench <- run_coverage_benchmark_lbbnn(
    M_MC           = 4,
    N_pop          = 400,
    epochs         = 15,
    lr             = 0.03,
    hidden_units   = 6,
    draws_domain   = 30,
    draws_sandwich = 50,
    seed           = 77,
    verbose        = FALSE
  )

  expect_true(is.list(bench))
  expect_true(all(c("domain_coverage", "slab_coverage", "overall_summary", "mpm_model", "sample_info") %in% names(bench)))
  expect_equal(nrow(bench$domain_coverage), 4L)
  expect_true(nrow(bench$slab_coverage) >= 2L)
  expect_equal(nrow(bench$overall_summary), 3L)
  expect_true(all(bench$overall_summary$Coverage_Corrected >= 0 & bench$overall_summary$Coverage_Corrected <= 1))
})
