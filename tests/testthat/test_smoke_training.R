test_that("Smoke: tiny model trains one epoch", {
  testthat::skip_on_cran()
  if (!requireNamespace("torch", quietly = TRUE)) {
    testthat::skip("torch not available")
  }
  library(torch)
  set.seed(1)
  n <- 64
  p <- 5
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta <- rnorm(p)
  y_cont <- as.numeric(x %*% beta + rnorm(n))
  y <- as.numeric(y_cont > median(y_cont))

  ds <- torch::tensor_dataset(torch::torch_tensor(x), torch::torch_tensor(y))
  dl <- torch::dataloader(ds, batch_size = 32, shuffle = TRUE)

  sizes <- c(p, 4, 1)
  prior <- c(0.5, 0.5)
  stds <- c(1.0, 1.0)
  inits <- matrix(rep(c(-10, 10), length(prior)), nrow = 2)

  net <- LBBNN_Net(
    problem_type = 'binary classification',
    sizes = sizes,
    prior = prior,
    std = stds,
    inclusion_inits = inits,
    input_skip = FALSE,
    flow = FALSE,
    device = 'cpu'
  )

  res <- train_LBBNN(epochs = 1, LBBNN = net, lr = 0.01, train_dl = dl, device = 'cpu')
  expect_true(length(res$loss) == 1)
  expect_true(is.numeric(res$density[1]))
})



