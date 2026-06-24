test_that("forward pass produces valid and different output with MPM", {
  testthat::skip_on_cran()
  if (! torch_available()){
    testthat::skip("torch or LibTorch is unavailable")
  }
  torch::torch_manual_seed(42)
  
  # synthetic dataset
  x <- torch::torch_randn(40, 5)
  y <- torch::torch_randn(40, 1)
  
  ds <- torch::tensor_dataset(x, y)
  dl <- torch::dataloader(ds, batch_size = 8, shuffle = TRUE)
  
  model <- lbbnn_net(
    problem_type = "regression",
    sizes = c(5, 1, 1),
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
  out1 <- model(x, MPM = FALSE)
  out2 <- model(x, MPM = TRUE)
  
  #check output shapes
  expect_equal(out1$size()[1], 40)
  expect_equal(out2$size()[1], 40)
  
  # Check outputs are not identical
  expect_false(torch::torch_allclose(out1, out2))
})