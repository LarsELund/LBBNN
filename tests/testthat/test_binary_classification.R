test_that("binary classification output is 1D", {
  
  testthat::skip_on_cran()
  if (! torch_available()){
    testthat::skip("torch or LibTorch is unavailable")
  }
  torch::torch_manual_seed(42)
  x <- torch::torch_randn(10, 3)
  
  model <- lbbnn_net(
    problem_type = "binary classification",
    sizes = c(3, 5, 1),
    prior = c(0.5, 0.5),
    inclusion_inits = "balanced",
    std = c(1, 1),
    device = "cpu"
  )
  
  out <- model(x)
  expect_true(length(dim(out)) <= 2)
})