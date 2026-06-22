test_that("conv layer alpha and sparsity behave consistently", {
  library(LBBNN)
  testthat::skip_on_cran()
  if (!requireNamespace("torch", quietly = TRUE)) {
    testthat::skip("torch not available")
  }
  torch::torch_manual_seed(1)
  
  x <- torch::torch_randn(2, 1, 8, 8)
  
  layer <- lbbnn_conv2d(
    1, 2, 3,
    prior_inclusion = 0.5,
    standard_prior = 1,
    density_init = c(-2, 2),
    flow = FALSE
  )
  
  layer$eval()
  
  layer(x, MPM = FALSE)
  
  expect_true(layer$alpha$min()$item() >= 0)
  expect_true(layer$alpha$max()$item() <= 1)
})