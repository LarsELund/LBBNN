# tests/testthat/test_windows_batch_shape.R

test_that("KMNIST convnet forward pass shape (Windows check)", {
  
  
  batch_size <- 100
  out_dim <- 10
  device <- "cpu"
  mpm <- TRUE
  
  # Download KMNIST dataset (small batch for speed)
  dir <- "./dataset/kmnist"
  train_ds <- torchvision::kmnist_dataset(
    dir,
    download = TRUE,
    transform = torchvision::transform_to_tensor
  )
  
  train_loader <- torch::dataloader(train_ds, batch_size = 100, shuffle = TRUE)
  
  ### create the convolutional network for MNIST
  
  device <- "cpu"
  conv_layer_1 <- lbbnn_conv2d(in_channels = 1, out_channels = 32, kernel_size = 5,
                               prior_inclusion = 0.5, standard_prior = 1,
                               density_init = c(-10, 10), num_transforms = 2,
                               flow = FALSE, hidden_dims = c(200, 200),
                               device = device)
  conv_layer_2 <- lbbnn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 5,
                               prior_inclusion = 0.5, standard_prior = 1,
                               density_init = c(-10, 15), num_transforms = 2,
                               flow = FALSE, hidden_dims = c(200, 200),
                               device = device)
  
  linear_layer_1 <- lbbnn_linear(in_features = 1024, out_features = 300,
                                 prior_inclusion = 0.5, standard_prior = 1,
                                 density_init = c(-10, 10), num_transforms = 2,
                                 flow = FALSE, hidden_dims = c(200, 200), device = device,
                                 bias_inclusion_prob = FALSE, conv_net = TRUE)
  
  linear_layer_2 <- lbbnn_linear(in_features = 300, out_features = 10,
                                 prior_inclusion = 0.5, standard_prior = 1,
                                 density_init = c(-5, 15),num_transforms = 2,
                                 flow = FALSE, hidden_dims = c(200, 200), device = device,
                                 bias_inclusion_prob = FALSE, conv_net = TRUE)
  
  LBBNN_ConvNet <- torch::nn_module(
    "LBBNN_ConvNet",
    
    initialize = function(conv1, conv2, fc1 ,fc2 ,device = device) {
      self$problem_type <- "multiclass classification"
      self$input_skip <- FALSE
      self$conv1 <- conv1
      self$conv2 <- conv2
      self$fc1 <- fc1
      self$fc2 <- fc2
      
      
      self$pool <- torch::nn_max_pool2d(2)
      self$act <- torch::nn_leaky_relu()
      self$out <- torch::nn_log_softmax(dim = 2)
      self$pout <- torch::nn_softmax(dim = 2)
      self$loss_fn <- torch::nn_nll_loss(reduction = "sum")
    },
    
    forward = function(x, MPM = FALSE, predict = FALSE) {
      x = self$act(self$conv1(x, MPM))
      x = self$pool(x)
      x = self$act(self$conv2(x, MPM))
      x = self$pool(x)
      x = torch::torch_flatten(x,start_dim = 2)
      x = self$act(self$fc1(x, MPM))
      if(!predict)
        x = self$out(self$fc2(x ,MPM))
      else
        x = self$pout(self$fc2(x ,MPM))
    },
    kl_div = function(){
      kl <- self$conv1$kl_div() + self$conv2$kl_div() +
        self$fc1$kl_div() + self$fc2$kl_div()
      return(kl)
    },
    density = function(){
      alphas <- NULL
      alphas <- c(as.numeric(self$conv1$alpha), as.numeric(self$conv2$alpha)
                  ,as.numeric(self$fc1$alpha), as.numeric(self$fc2$alpha))
      return(mean(alphas > 0.5))
      
      
    },
    compute_paths = function(){
      NULL
    },
    density_active_path = function(){
      NA
    }
  )
  
  model <- LBBNN_ConvNet(conv_layer_1, conv_layer_2, linear_layer_1,
                         linear_layer_2, device)
  
  
  # Grab first batch from dataloader
  it <- train_loader$.iter()
  batch <- it$.next()[[1]]  # KMNIST images
  
  # Forward pass
  model$eval()
  out <- model(batch, MPM = mpm)
  print(out)
  
  # Check output shape: [batch_size, out_dim]
  expect_equal(dim(out)[1], batch_size)
  expect_equal(dim(out)[2], out_dim)
})
