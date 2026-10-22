testthat::test_that("weighted training with rep(1, n) matches unweighted training exactly", {
  testthat::skip_on_cran()
  testthat::skip_if_not(torch_available())

  set.seed(42)
  torch::torch_manual_seed(42)

  n <- 50
  p <- 4
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- rnorm(n)

  # Train unweighted model
  torch::torch_manual_seed(123)
  m_unweighted <- lbbnn_net(
    problem_type = "regression",
    sizes = c(p, 8, 1),
    prior = c(0.5, 0.5),
    std = c(1.0, 1.0),
    inclusion_inits = "balanced",
    flow = FALSE,
    device = "cpu"
  )

  t_x <- torch::torch_tensor(x, dtype = torch::torch_float32())
  t_y <- torch::torch_tensor(y, dtype = torch::torch_float32())
  ds_unweighted <- torch::tensor_dataset(t_x, t_y)
  dl_unweighted <- torch::dataloader(ds_unweighted, batch_size = n, shuffle = FALSE)

  train_res_unweighted <- train_lbbnn(
    epochs = 3,
    LBBNN = m_unweighted,
    lr = 0.01,
    train_dl = dl_unweighted,
    device = "cpu",
    verbose = FALSE
  )

  # Train weighted model with rep(1, n)
  torch::torch_manual_seed(123)
  m_weighted <- lbbnn_net(
    problem_type = "regression",
    sizes = c(p, 8, 1),
    prior = c(0.5, 0.5),
    std = c(1.0, 1.0),
    inclusion_inits = "balanced",
    flow = FALSE,
    device = "cpu"
  )

  t_w <- torch::torch_ones(n, dtype = torch::torch_float32())
  ds_weighted <- torch::tensor_dataset(t_x, t_y, t_w)
  dl_weighted <- torch::dataloader(ds_weighted, batch_size = n, shuffle = FALSE)

  train_res_weighted <- train_lbbnn(
    epochs = 3,
    LBBNN = m_weighted,
    lr = 0.01,
    train_dl = dl_weighted,
    device = "cpu",
    verbose = FALSE
  )

  testthat::expect_equal(train_res_unweighted$loss, train_res_weighted$loss, tolerance = 1e-4)
})

testthat::test_that("get_dataloaders supports weights and target_cols", {
  testthat::skip_on_cran()
  testthat::skip_if_not(torch_available())

  n <- 60
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y1 = rbinom(n, 1, 0.5),
    y2 = rnorm(n)
  )
  weights <- runif(n, 0.1, 1.0)

  # Single target column with weights
  dls <- get_dataloaders(
    dataset = df[, c("x1", "x2", "y1")],
    train_proportion = 0.8,
    train_batch_size = 16,
    test_batch_size = 12,
    weights = weights,
    seed = 123
  )

  b <- coro::collect(dls$train_loader, n = 1)[[1]]
  testthat::expect_equal(length(b), 3)
  testthat::expect_equal(b[[1]]$shape[2], 2) # 2 features
  testthat::expect_equal(b[[3]]$shape[1], 16) # batch size

  # Multiple target columns
  dls_multi <- get_dataloaders(
    dataset = df,
    train_proportion = 0.8,
    train_batch_size = 16,
    test_batch_size = 12,
    target_cols = 2,
    weights = weights,
    seed = 123
  )

  b_multi <- coro::collect(dls_multi$train_loader, n = 1)[[1]]
  testthat::expect_equal(b_multi[[1]]$shape[2], 2) # 2 features
  testthat::expect_equal(b_multi[[2]]$shape[2], 2) # 2 targets
})

testthat::test_that("loglik_contributions returns valid S x n matrix for binary and regression", {
  testthat::skip_on_cran()
  testthat::skip_if_not(torch_available())

  set.seed(1)
  n <- 40
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  y_reg <- rnorm(n)
  y_bin <- rbinom(n, 1, 0.5)

  # Regression
  m_reg <- lbbnn_net(
    problem_type = "regression",
    sizes = c(p, 4, 1),
    prior = c(0.5, 0.5),
    std = c(1, 1),
    inclusion_inits = "balanced",
    flow = FALSE,
    device = "cpu"
  )

  eval_dl_reg <- torch::dataloader(
    torch::tensor_dataset(
      torch::torch_tensor(X, dtype = torch::torch_float32()),
      torch::torch_tensor(y_reg, dtype = torch::torch_float32())
    ),
    batch_size = 20,
    shuffle = FALSE
  )

  L_reg <- loglik_contributions(m_reg, eval_dl_reg, draws = 15, sigma = 1.0)
  testthat::expect_equal(dim(L_reg), c(15, n))
  testthat::expect_true(all(is.finite(L_reg)))

  # Binary classification
  m_bin <- lbbnn_net(
    problem_type = "binary classification",
    sizes = c(p, 4, 1),
    prior = c(0.5, 0.5),
    std = c(1, 1),
    inclusion_inits = "balanced",
    flow = FALSE,
    device = "cpu"
  )

  eval_dl_bin <- torch::dataloader(
    torch::tensor_dataset(
      torch::torch_tensor(X, dtype = torch::torch_float32()),
      torch::torch_tensor(y_bin, dtype = torch::torch_float32())
    ),
    batch_size = 20,
    shuffle = FALSE
  )

  L_bin <- loglik_contributions(m_bin, eval_dl_bin, draws = 10)
  testthat::expect_equal(dim(L_bin), c(10, n))
  testthat::expect_true(all(L_bin <= 0)) # Log-probabilities must be <= 0
  testthat::expect_true(all(is.finite(L_bin)))
})

testthat::test_that("custom problem_type works with custom link, nll and loglik_contributions", {
  testthat::skip_on_cran()
  testthat::skip_if_not(torch_available())

  n <- 30
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  # Define custom Bernoulli link and loss
  custom_link <- function(x) torch::torch_sigmoid(x)
  custom_nll <- function(input, target, reduction = "sum") {
    eps <- 1e-7
    p_hat <- torch::torch_clamp(input, eps, 1 - eps)
    t_flt <- target$squeeze()$to(dtype = torch::torch_float32())
    p_flt <- p_hat$squeeze()
    loss_i <- -(t_flt * torch::torch_log(p_flt) + (1 - t_flt) * torch::torch_log(1 - p_flt))
    if (reduction == "none") loss_i else torch::torch_sum(loss_i)
  }

  m_custom <- lbbnn_net(
    problem_type = "custom",
    sizes = c(p, 4, 1),
    prior = c(0.5, 0.5),
    std = c(1, 1),
    inclusion_inits = "balanced",
    flow = FALSE,
    link = custom_link,
    nll = custom_nll,
    device = "cpu"
  )

  ds <- torch::tensor_dataset(
    torch::torch_tensor(X, dtype = torch::torch_float32()),
    torch::torch_tensor(y, dtype = torch::torch_float32())
  )
  dl <- torch::dataloader(ds, batch_size = 15, shuffle = FALSE)

  train_res <- train_lbbnn(epochs = 2, LBBNN = m_custom, lr = 0.01, train_dl = dl, device = "cpu", verbose = FALSE)
  testthat::expect_true(length(train_res$loss) == 2)
  testthat::expect_true(all(is.finite(train_res$loss)))

  # loglik_contributions with custom problem
  L_cust <- loglik_contributions(m_custom, dl, draws = 5)
  testthat::expect_equal(dim(L_cust), c(5, n))
  testthat::expect_true(all(is.finite(L_cust)))
})

testthat::test_that("fit_dp_lbbnn runs two-pass synthesizer and calculates local DP bound", {
  testthat::skip_on_cran()
  testthat::skip_if_not(torch_available())

  set.seed(99)
  n <- 35
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  model_spec <- list(
    problem_type = "regression",
    sizes = c(p, 4, 1),
    prior = c(0.5, 0.5),
    std = c(1, 1),
    inclusion_inits = "balanced",
    flow = FALSE,
    device = "cpu"
  )

  dp_res <- fit_dp_lbbnn(
    model_spec = model_spec,
    X = X,
    y = y,
    epochs = 2,
    lr = 0.01,
    batch_size = 20,
    S = 10,
    c_scale = 0.8,
    g_shift = 0.1,
    device = "cpu",
    verbose = FALSE
  )

  testthat::expect_type(dp_res, "list")
  testthat::expect_equal(length(dp_res$alpha), n)
  testthat::expect_true(all(dp_res$alpha >= 0 & dp_res$alpha <= 1))
  testthat::expect_equal(length(dp_res$f_vec), n)
  testthat::expect_true(is.numeric(dp_res$delta_alpha))
  testthat::expect_true(dp_res$delta_alpha >= 0)
  testthat::expect_equal(dp_res$epsilon_local, 2 * dp_res$delta_alpha)
  testthat::expect_equal(dim(dp_res$L_pass1), c(10, n))
  testthat::expect_equal(dim(dp_res$L_pass2), c(10, n))
})
