test_that("extract_lbbnn_support and eval_support_abs_loglik work for regression and binary models", {
  skip_if_not_installed("torch")
  set.seed(123)
  torch::torch_manual_seed(123)

  n <- 60
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  y_reg <- 1.0 + 2.0 * x1 - 0.5 * x2 + rnorm(n, 0, 0.5)
  df_reg <- data.frame(x1 = x1, x2 = x2, y = y_reg)

  dl_reg <- get_dataloaders(
    df_reg,
    train_proportion = 1.0,
    train_batch_size = 30,
    test_batch_size  = 30,
    standardize      = FALSE,
    seed             = 123
  )
  net_reg <- lbbnn_net(
    problem_type    = "regression",
    sizes           = c(2L, 6L, 1L),
    prior           = c(0.5, 0.5),
    std             = c(1.0, 2.0),
    inclusion_inits = "balanced",
    input_skip      = TRUE,
    sigma           = 0.5,
    device          = "cpu"
  )
  train_lbbnn(
    epochs   = 10,
    LBBNN    = net_reg,
    lr       = 0.02,
    train_dl = dl_reg$train_loader,
    device   = "cpu",
    verbose  = FALSE
  )

  X_df <- df_reg[, c("x1", "x2")]
  mu_sup <- extract_lbbnn_support(net_reg, X_df, B = 25, mpm = FALSE)
  expect_true(is.matrix(mu_sup))
  expect_equal(dim(mu_sup), c(25L, n))
  expect_true(all(is.finite(mu_sup)))

  ll_reg <- eval_support_abs_loglik(mu_sup, y_reg, problem_type = "regression", sigma = 0.5)
  expect_true(is.matrix(ll_reg))
  expect_equal(dim(ll_reg), c(25L, n))
  expect_true(all(is.finite(ll_reg)))
  expect_true(all(ll_reg >= 0))

  # Binary classification support check
  p_prob <- stats::plogis(0.8 * x1 - 0.6 * x2)
  y_bin <- rbinom(n, 1, p_prob)
  ll_bin <- eval_support_abs_loglik(matrix(stats::plogis(mu_sup), nrow = 25, ncol = n),
                                    y_bin, problem_type = "binary")
  expect_equal(dim(ll_bin), c(25L, n))
  expect_true(all(is.finite(ll_bin)))
  expect_true(all(ll_bin >= 0))
})

test_that("calibrate_dp_finite_sample returns valid finite-sample bounds and monotone nugget", {
  skip_if_not_installed("torch")
  set.seed(42)
  torch::torch_manual_seed(42)

  n <- 80
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  y <- 1.5 + 0.8 * x1 - 0.5 * x2 + rnorm(n, 0, 0.5)
  # Add two mild outliers to induce right-tail replicate exceedance
  y[c(1, 2)] <- y[c(1, 2)] + 3.0
  X_mat <- cbind(x1 = x1, x2 = x2)

  spec <- list(
    problem_type    = "regression",
    sizes           = c(2L, 8L, 1L),
    prior           = c(0.5, 0.5),
    std             = c(1.0, 2.0),
    inclusion_inits = "balanced",
    input_skip      = TRUE,
    sigma           = 0.6
  )

  dp_fit <- fit_dp_lbbnn(
    model_spec    = spec,
    X             = X_mat,
    y             = y,
    epochs        = 15,
    lr            = 0.02,
    batch_size    = 40,
    S             = 30,
    stage1_method = "linear",
    stage2_method = "vector_upscale",
    M_thresh      = 3.5,
    sigma         = 0.6,
    verbose       = FALSE
  )

  expect_true(!is.null(dp_fit$M_thresh))
  expect_equal(dp_fit$M_thresh, 3.5)
  expect_true(is.matrix(dp_fit$support_pass1))
  expect_true(is.matrix(dp_fit$support_pass2))
  expect_equal(dim(dp_fit$support_pass1), c(30L, n))
  expect_equal(dim(dp_fit$support_pass2), c(30L, n))
  L2_direct <- eval_support_abs_loglik(dp_fit$support_pass2, y, problem_type = "regression", sigma = dp_fit$sigma_p1)
  expect_equal(dp_fit$delta_alpha, max(sweep(L2_direct, 2, dp_fit$alpha, `*`)), tolerance = 1e-10)
  expect_lte(dp_fit$delta_alpha, 3.5 + 1e-9)

  deltas <- c(0.10, 0.05, 0.01)
  cal <- calibrate_dp_finite_sample(
    dp_fit       = dp_fit,
    X            = X_mat,
    y            = y,
    R            = 100,
    B            = 30,
    delta_target = deltas,
    seed         = 2026
  )

  expect_equal(cal$delta_alpha_obs, dp_fit$delta_alpha, tolerance = 1e-10)

  expect_true(is.finite(cal$delta_alpha_obs) && cal$delta_alpha_obs > 0)
  expect_equal(cal$epsilon_obs, 2 * cal$delta_alpha_obs, tolerance = 1e-10)
  expect_equal(length(cal$delta_rep_vec), 100L)
  expect_true(all(is.finite(cal$delta_rep_vec) & cal$delta_rep_vec > 0))
  expect_true(cal$delta_hat_0 >= 0 && cal$delta_hat_0 <= 1)

  tab <- cal$calibration_table
  expect_equal(nrow(tab), 3L)
  expect_equal(tab$delta_target, deltas)
  expect_true(all(tab$eta_nugget >= 0))
  # Smaller delta_target requires at least as large an additive nugget eta
  expect_true(tab$eta_nugget[1] <= tab$eta_nugget[2] + 1e-12)
  expect_true(tab$eta_nugget[2] <= tab$eta_nugget[3] + 1e-12)
  # Epsilon increases monotonically by 2 * eta
  expect_equal(tab$epsilon_eta, 2 * tab$delta_alpha_eta, tolerance = 1e-10)
  # Empirical exceedance rate on calibrated replicate distribution <= delta_target
  expect_true(all(tab$empirical_exceedance <= tab$delta_target + 1e-12))
})
