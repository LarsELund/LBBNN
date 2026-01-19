test_that("Smoke: tiny model trains for one epoch", {
  testthat::skip_on_cran()
  if (!requireNamespace("torch", quietly = TRUE)) {
    testthat::skip("torch not available")
  }
  i <- 5000
  j <- 15
  set.seed(2)
  torch::torch_manual_seed(2)
  #generate some data
  X <- matrix(rnorm(i * j, mean = - 0.1, sd = 0.1), ncol = j)
  #make some X relevant for prediction
  y_base <-  (0.1 * log(abs(X[, 1])) + 3 * cos(X[, 2]) + 2 * X[, 3] * X[, 4]
              + 2 * X[, 5] - 2 * X[, 6] ** 2 + rnorm(i, sd = 0.01))
  y <- c()
  # change y to 0 and 1
  y[y_base > median(y_base)] <- 1
  y[y_base <= median(y_base)] <- 0
  sim_data <- as.data.frame(X)
  sim_data <- cbind(sim_data, y)
  loaders <- get_dataloaders(sim_data, train_proportion = 0.9,
                             train_batch_size = 1500, test_batch_size = 500,
                             standardize = FALSE)
  train_loader <- loaders$train_loader
  test_loader  <- loaders$test_loader
  problem <- "binary classification"
  sizes <- c(j, 5, 5, 1) # 2 hidden layers, 5 neurons in each
  incl_priors <- c(0.5, 0.5, 0.5) #prior inclusion probs for each weight matrix
  stds <- c(100, 100, 100) #prior distribution for the standard deviation of the weights
  incl_inits <- matrix(rep(c(-15, 10), 3), nrow = 2, ncol = 3) #initializations for inclusion params
  device <- "cpu" #can also be 'gpu' or 'mps'
  model_input_skip <- lbbnn_net(problem_type = problem, sizes = sizes,
                                prior = incl_priors,
                                inclusion_inits = incl_inits, input_skip = TRUE,
                                std = stds, flow = TRUE, device = device)

  res <- train_lbbnn(epochs = 1, LBBNN = model_input_skip,
              lr = 0.005, train_dl = train_loader, device = device)
  validate_lbbnn(LBBNN = model_input_skip, num_samples = 1,
                 test_dl = test_loader, device)

  expect_true(length(res$loss) == 1)
  expect_true(is.numeric(res$density[1]))
})
