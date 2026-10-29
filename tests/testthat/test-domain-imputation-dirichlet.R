testthat::test_that("hajek_estimate supports domain_matrix and dirichlet_uq", {
  set.seed(42)
  n <- 120
  y <- rnorm(n, mean = 5, sd = 1)
  pi_hat <- runif(n, 0.2, 0.8)
  domain_vec <- sample(c("RegionA", "RegionB", "RegionC"), size = n, replace = TRUE)

  # 1. Standard overall scalar
  est_scalar <- hajek_estimate(y = y, pi_hat = pi_hat)
  testthat::expect_type(est_scalar, "double")
  testthat::expect_length(est_scalar, 1)

  # 2. Overall with Dirichlet UQ
  est_overall_uq <- hajek_estimate(y = y, pi_hat = pi_hat, dirichlet_uq = TRUE, n_draws = 50)
  testthat::expect_s3_class(est_overall_uq, "data.frame")
  testthat::expect_equal(nrow(est_overall_uq), 1)
  testthat::expect_true(est_overall_uq$sd > 0)
  testthat::expect_true(est_overall_uq$ci_lower < est_overall_uq$ci_upper)

  # 3. Domain breakdown without Dirichlet UQ
  est_dom <- hajek_estimate(y = y, pi_hat = pi_hat, domain_matrix = domain_vec)
  testthat::expect_s3_class(est_dom, "data.frame")
  testthat::expect_equal(nrow(est_dom), 3)
  testthat::expect_true(all(c("domain", "n_units", "mean", "total_est") %in% names(est_dom)))

  # 4. Domain breakdown with Dirichlet UQ
  est_dom_uq <- hajek_estimate(y = y, pi_hat = pi_hat, domain_matrix = domain_vec,
                               dirichlet_uq = TRUE, n_draws = 50)
  testthat::expect_s3_class(est_dom_uq, "data.frame")
  testthat::expect_equal(nrow(est_dom_uq), 3)
  testthat::expect_true(all(est_dom_uq$sd > 0))
  testthat::expect_true(all(est_dom_uq$ci_lower < est_dom_uq$ci_upper))
  testthat::expect_true(all(est_dom_uq$total_ci_lower < est_dom_uq$total_ci_upper))
})

testthat::test_that("predict_domains_lbbnn evaluates domain-level posterior metrics", {
  set.seed(123)
  torch::torch_manual_seed(123)

  n <- 80
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  colnames(X) <- paste0("x", 1:p)
  y <- 2 * X[, 1] - 1.5 * X[, 2] + rnorm(n, sd = 0.5)
  df <- data.frame(X, y = y)
  domains <- sample(c("Metro", "Rural"), size = n, replace = TRUE)

  dl <- get_dataloaders(df, train_proportion = 1, train_batch_size = 40,
                        test_batch_size = 40, standardize = FALSE)

  net <- lbbnn_net(problem_type = "regression", sizes = c(p, 4, 1),
                   sigma = 0.5, input_skip = TRUE)
  train_lbbnn(LBBNN = net, train_dl = dl$train_loader, epochs = 10, lr = 0.05)

  # Latent mean without Dirichlet UQ
  res_mean <- predict_domains_lbbnn(
    object = net, newdata = df[, 1:p], domain_matrix = domains,
    draws = 20, type = "latent_mean", dirichlet_uq = FALSE
  )
  testthat::expect_s3_class(res_mean, "data.frame")
  testthat::expect_equal(nrow(res_mean), 2)
  testthat::expect_true(all(c("domain", "n_units", "mean", "sd", "ci_lower", "ci_upper") %in% names(res_mean)))

  # Predictive with Dirichlet UQ
  res_pred_uq <- predict_domains_lbbnn(
    object = net, newdata = df[, 1:p], domain_matrix = domains,
    draws = 20, type = "predictive", dirichlet_uq = TRUE
  )
  testthat::expect_s3_class(res_pred_uq, "data.frame")
  testthat::expect_equal(nrow(res_pred_uq), 2)
  testthat::expect_true(all(res_pred_uq$sd > 0))
  testthat::expect_true(all(res_pred_uq$ci_lower < res_pred_uq$ci_upper))
})

testthat::test_that("impute_response_lbbnn and get_dataloaders handle missing values", {
  set.seed(456)
  n <- 60
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  colnames(X) <- paste0("x", 1:p)
  y <- 3 * X[, 1] + rnorm(n, sd = 0.4)
  y[c(5, 12, 25)] <- NA  # 3 missing responses

  df_mis <- data.frame(X, y = y)

  # get_dataloaders filters missing units from train_loader
  dl_mis <- get_dataloaders(df_mis, train_proportion = 1, train_batch_size = 20,
                            test_batch_size = 20, allow_missing = TRUE)
  testthat::expect_equal(length(dl_mis$train_loader$dataset), 57)

  # Fit model on observed units
  net <- lbbnn_net(problem_type = "regression", sizes = c(p, 4, 1), sigma = 0.4)
  train_lbbnn(LBBNN = net, train_dl = dl_mis$train_loader, epochs = 8, lr = 0.05)

  # Impute missing values
  imp <- impute_response_lbbnn(object = net, newdata = df_mis, draws = 25, type = "latent_mean")
  testthat::expect_equal(length(imp$missing_indices), 3)
  testthat::expect_equal(nrow(imp$imputed_draws), 3)
  testthat::expect_equal(ncol(imp$imputed_draws), 25)
  testthat::expect_false(anyNA(imp$completed_data$y))
  testthat::expect_equal(nrow(imp$imputed_summary), 3)
  testthat::expect_true(all(imp$imputed_summary$ci_lower < imp$imputed_summary$ci_upper))
})
