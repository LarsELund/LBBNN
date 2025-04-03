
test_that("Simple network created", {
  library(torch)
  layers <- c(20,200,200,5) #Two hidden layers 
  alpha <- c(0.3,0.5,0.9)  # One prior inclusion probability for each weight matrix 
  inclusions <- matrix(rep(c(-2,2),3),nrow = 2,ncol = 3)
  std <- c(1,1,1)
  prob <- 'multiclass classification'
  device = 'cpu' #can try 'mps' to check if all tensors and paramters can be moved to a different device
  net <- LBBNN_Net(problem_type = prob,sizes =layers,prior = alpha,std = std,
                   inclusion_inits = inclusions,input_skip = FALSE,flow = FALSE,
                   num_transforms = 2, dims = c(200,200),
                   device = 'cpu')
  net$to(device = device)
  print(net)
  x <- torch_rand(100,20,requires_grad = FALSE,device = device) #generate some dummy data
  print(net)
  output <- net(x) #forward pass
 
  expect_equal(dim(output)[2],5)
  expect_equal(length(net$kl_div()$item()),1) #get KL-divergence
  expect_equal((net$density()<1),TRUE) #get the density of the network
  
  
})