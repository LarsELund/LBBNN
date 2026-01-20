library(testthat)
library(torch)
library(torchvision)
library(LBBNN)

test_that("KMNIST batch has correct shape", {
  

  
  batch_size <- 100
  dir <- "./dataset/kmnist"
  
  # Transform function to ensure channel dimension exists
  kmnist_transform <- function(x) {
    print("before transform:")
    print(dim(x))
    d <- dim(x)
    if (length(d) == 3 && d[3] > 1 && d[1] == d[2]) {#if shape [28,28,batch] as on windows and linux(?)
      x <- x[, , 1, drop = TRUE] #get it to be the same shape as Mac OS
    }
    x <- torchvision::transform_to_tensor(x)
    print("after transform:")
    print(dim(x))
    print("after fix:")
    print(dim(x))
    x
  }
  
  # Load datasets
  train_ds <- torchvision::kmnist_dataset(
    dir,
    download = TRUE,
    transform = kmnist_transform
  )
  
  train_loader <- torch::dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
  
  # Grab first batch
  it <- train_loader$.iter()
  batch <- it$.next()[[1]]
  
  print("Batch shape:")
  print(dim(batch))
  
  # Check that batch has shape [batch_size, 1, 28, 28]
  expect_equal(dim(batch)[1], batch_size)
  expect_equal(dim(batch)[2], 1)
  expect_equal(dim(batch)[3], 28)
  expect_equal(dim(batch)[4], 28)
})
