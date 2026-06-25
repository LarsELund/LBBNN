test_that("conv layer alpha and sparsity behave consistently", {
  testthat::skip_on_cran()
  if (! torch_available()){
    testthat::skip("torch or LibTorch is unavailable")
  }
  
  
  x <- torch::torch_randn(2, 1, 8, 8)
  
  layer <- lbbnn_conv2d(
    1, 2, 3,
    prior_inclusion = 0.5,
    standard_prior = 1,
    density_init = c(-2, 2),
    flow = TRUE
  )
  
  layer$eval()
  
  set.seed(42)
  torch::torch_manual_seed(42)
  out1 <- layer(x, MPM = FALSE)
  set.seed(42)
  torch::torch_manual_seed(42)
  out2 <- layer(x, MPM = FALSE)
  
  print(out1)
  print(out2)
  #check for deterministic output
  expect_true(torch::torch_allclose(out1, out2, atol = 1e-10)) 
  

  expect_equal(out1$shape, c(2, 2, 6, 6))  
  expect_true(layer$alpha$min()$item() >= 0)
  expect_true(layer$alpha$max()$item() <= 1)

  expect_true(torch::torch_isfinite(out1)$all()$item())
  expect_false(torch::torch_isnan(out1)$any()$item())
  
  
  
})