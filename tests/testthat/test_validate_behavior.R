test_that("validate_lbbnn does not modify model state", {
  
  library(LBBNN)
  testthat::skip_on_cran()
  if (!requireNamespace("torch", quietly = TRUE)) {
    testthat::skip("torch not available")
  }
  
  
  i <- 10
  j <- 2
  
  set.seed(42)
  torch::torch_manual_seed(42)
  X <- matrix(rnorm(i * j, mean = 0, sd = 1), ncol = j)
  #make some X relevant for prediction
  y_base <- c()
  y_base <-  0.6 * X[, 1] - 0.4 * X[, 2]
  sim_data <- as.data.frame(X)
  sim_data <- cbind(sim_data, y_base)
  loaders <- get_dataloaders(sim_data, train_proportion = 0.9,
                             train_batch_size = 9, test_batch_size = 1,
                             standardize = FALSE)
  train_loader <- loaders$train_loader
  test_loader  <- loaders$test_loader
  problem <- "regression"
  sizes <- c(j, 2, 1) 
  incl_priors <- c(0.5, 0.5) 
  stds <- c(1, 1) 
  incl_inits <- 'balanced'
  device <- "cpu"
  
  model_test <- lbbnn_net(problem_type = problem, sizes = sizes,
                          prior = incl_priors, inclusion_inits = incl_inits,
                          std = stds, input_skip = TRUE, flow = FALSE,
                          num_transforms = 2, dims = c(10, 10, 10),
                          raw_output = FALSE, custom_act = NULL,
                          link = NULL, nll = NULL,
                          bias_inclusion_prob = FALSE, device = device)
  
  train_lbbnn(epochs = 2, LBBNN = model_test,
              lr = 0.05, train_dl = train_loader, device = device, 
              verbose = FALSE)
  
  
  
  
  
  state_after_train <- model_test$clone(deep = TRUE)
  
  validate_lbbnn(model_test,num_samples = 1, test_dl = test_loader)
  
  state_after_validate <- model_test$clone(deep = TRUE)
  
  expect_equal(state_after_train$training, state_after_validate$training)
  expect_equal(state_after_train$raw_output, state_after_validate$raw_output)
  
  
  
})