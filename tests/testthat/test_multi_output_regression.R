test_that("train + validate works for multi-output regression", {
  
  torch::torch_manual_seed(1)
  
  # synthetic dataset: multi-output regression (3 targets)
  x <- torch::torch_randn(40, 5)
  y <- torch::torch_randn(40, 3)
  
  ds <- torch::tensor_dataset(x, y)
  dl <- torch::dataloader(ds, batch_size = 8, shuffle = TRUE)
  
  model <- lbbnn_net(
    problem_type = "regression",
    sizes = c(5, 1, 3),
    prior = c(0.5, 0.5),
    inclusion_inits = "balanced",
    std = c(1, 1),
    input_skip = FALSE,
    flow = FALSE,
    num_transforms = 2,
    dims = c(10, 10, 10),
    raw_output = FALSE,
    device = "cpu"
  )
 
  train_lbbnn(epochs = 1, LBBNN = model, lr = 0.01, train_dl = dl,
              device = "cpu", verbose = FALSE)

  
  res <- validate_lbbnn(LBBNN = model, num_samples = 1, test_dl = dl,
                        device = "cpu")
  
  expect_true(is.list(res))
  expect_true(!is.null(res$validation_error))
  
  out <- model(x)
  expect_equal(dim(out), dim(y))
})