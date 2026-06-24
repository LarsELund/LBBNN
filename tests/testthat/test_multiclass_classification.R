test_that("multi-class output has correct shape", {
  
  testthat::skip_on_cran()
  if (! torch_available()){
    testthat::skip("torch or LibTorch is unavailable")
  }
  torch::torch_manual_seed(42)
  
  x <- torch::torch_randn(8, 5)
  model <- lbbnn_net(
    problem_type = "multiclass classification",
    sizes = c(5, 6, 4),
    prior = c(0.5, 0.5),
    inclusion_inits = "balanced",
    std = c(1, 1),
    device = "cpu"
  )
  
  out <- model(x)
  
  expect_equal(dim(out)[1], 8)
  expect_equal(dim(out)[2], 4)
})