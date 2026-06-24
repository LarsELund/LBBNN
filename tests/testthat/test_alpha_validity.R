test_that("alpha is always in [0,1]", {
  
  testthat::skip_on_cran()
  if (! torch_available()){
    testthat::skip("torch or LibTorch is unavailable")
  }
  torch::torch_manual_seed(42)
  layer <- lbbnn_linear(
    5, 3, 0.5, 1,
    density_init = c(-10, 10),
    flow = FALSE,
    device = "cpu"
  )
  
  x <- torch::torch_randn(4, 5)
  layer(x)
  a <- layer$alpha
  expect_true(a$all()$item() >= 0)
  expect_true(a$all()$item() <= 1)
})