library(testthat)
library(torch)
library(torchvision)
library(coro)
library(LBBNN)

test_that("KMNIST batch has correct shape", {
  
  batch_size <- 100
  dir <- "./dataset/kmnist"
  
  kmnist_transform <- function(x) {
    print("before transform:")
    print(dim(x))
    d <- dim(x)
    if (length(d) == 3 && d[3] > 1 && d[1] == d[2]) {
      x <- torchvision::transform_to_tensor(x)
      x <- x$unsqueeze(2)
    } else {
      x <- torchvision::transform_to_tensor(x)
    }
    print("after transform:")
    print(dim(x))
    x
  }
  
  train_ds <- torchvision::kmnist_dataset(
    dir,
    download = TRUE,
    transform = kmnist_transform
  )
  
  train_loader <- torch::dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
  
  it <- train_loader$.iter()
  batch <- it$.next()[[1]]
  
  print("Batch shape:")
  print(dim(batch))
  
  expect_equal(dim(batch)[1], batch_size)
  expect_equal(dim(batch)[2], 1)
  expect_equal(dim(batch)[3], 28)
  expect_equal(dim(batch)[4], 28)
})


# ---------------------------------------------------------
# 🔴 NEW TEST: Reproduce prediction accumulation issue
# ---------------------------------------------------------

test_that("LBBNN ConvNet prediction accumulation is stable", {
  
  skip_on_cran()  # avoid CRAN timeouts
  
  torch::torch_manual_seed(1)
  
  batch_size <- 100
  draws <- 3
  out_dim <- 10
  dir <- "./dataset/kmnist"
  device <- "cpu"
  
  kmnist_transform <- function(x) {
    x <- torchvision::transform_to_tensor(x)
    if (length(dim(x)) == 3) {
      x <- x$unsqueeze(1)
    }
    x
  }
  
  test_ds <- torchvision::kmnist_dataset(
    dir,
    train = FALSE,
    download = TRUE,
    transform = kmnist_transform
  )
  
  test_loader <- torch::dataloader(test_ds, batch_size = batch_size)
  
  # --- YOUR MODEL (no training) ---
  
  conv_layer_1 <- lbbnn_conv2d(1, 32, 5, 0.5, 1, c(-10, 10), 2, FALSE, c(200,200), device)
  conv_layer_2 <- lbbnn_conv2d(32, 64, 5, 0.5, 1, c(-10, 15), 2, FALSE, c(200,200), device)
  
  linear_layer_1 <- lbbnn_linear(1024, 300, 0.5, 1, c(-10, 10), 2, FALSE, c(200,200), device,
                                 bias_inclusion_prob = FALSE, conv_net = TRUE)
  
  linear_layer_2 <- lbbnn_linear(300, 10, 0.5, 1, c(-5, 15), 2, FALSE, c(200,200), device,
                                 bias_inclusion_prob = FALSE, conv_net = TRUE)
  
  model <- LBBNN_ConvNet(conv_layer_1, conv_layer_2,
                         linear_layer_1, linear_layer_2, device)
  
  model$eval()
  
  # --- ORIGINAL BUG-PRONE LOGIC ---
  
  predictions <- NULL
  
  torch::with_no_grad({ 
    coro::loop(for (b in test_loader) { 
      
      outputs <- torch::torch_zeros(draws, dim(b[[1]])[1], out_dim)
      
      for (i in 1:draws) {
        data <- b[[1]]
        outputs[i] <- model(data, MPM = TRUE, predict = TRUE)
      }
      
      predictions <- torch::torch_cat(c(predictions, outputs), dim = 2)
    })  
  })
  
  print("Final prediction shape:")
  print(dim(predictions))
  
  # --- ASSERTIONS ---
  
  expect_true(length(dim(predictions)) == 3)
  expect_equal(dim(predictions)[1], draws)
  expect_equal(dim(predictions)[3], out_dim)
  expect_equal(dim(predictions)[2], length(test_ds))  # should be 10000
  
})