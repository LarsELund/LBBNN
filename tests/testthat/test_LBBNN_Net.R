
test_that("Simple network created", {
  library(torch)
  layers <- c(20,200,200,5) #Two hidden layers 
  alpha <- c(0.3,0.5,0.9)  # One prior inclusion probability for each weight matrix 
  prob <- 'multiclass classification'
  net <- LBBNN_Net(problem_type = prob, sizes = layers,
                   prior = alpha,device = 'cpu')
  print(net)
  
  x <- torch_rand(100,20,requires_grad = FALSE) #generate some dummy data
  output <- net(x) #forward pass
  expect_equal(dim(output)[2],5)
  expect_equal(length(net$kl_div()$item()),1) #get KL-divergence
  expect_equal((net$density()<1),TRUE) #get the density of the network
  
  
})