library(LBBNN)

### Tutorial 4: Convolutional architecture on the KMNIST dataset
### this example needs the torchvision package to download data
### install.packages("torchvision") 
library(torchvision)

dir <- "./dataset/kmnist"   #directory to install dataset
train_ds <- torchvision::kmnist_dataset(
  dir,
  download = TRUE,
  transform = torchvision::transform_to_tensor)

test_ds <- torchvision::kmnist_dataset(
  dir,
  train = FALSE,
  transform = torchvision::transform_to_tensor)

train_loader <- torch::dataloader(train_ds, batch_size = 100, shuffle = TRUE)
test_loader <- torch::dataloader(test_ds, batch_size = 100)





### create the convolutional network for MNIST 

device <- 'mps'
torch::torch_manual_seed(42)

conv_layer_1 <- LBBNN_Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5,
                           prior_inclusion = 0.5,standard_prior = 1,density_init = c(-10,10),
                           num_transforms = 2,flow = FALSE,hidden_dims = c(200,200),device = device)
conv_layer_2 <- LBBNN_Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5,
                           prior_inclusion = 0.5,standard_prior = 1,density_init = c(-10,15),
                           num_transforms = 2,flow = FALSE,hidden_dims = c(200,200),device = device)

linear_layer_1 <- LBBNN_Linear(in_features = 1024, out_features = 300,
                         prior_inclusion = 0.5,standard_prior = 1,
                         density_init = c(-10,10),num_transforms = 2,
                         flow = FALSE,hidden_dims = c(200,200),device = device,
                         bias_inclusion_prob = FALSE,conv_net = TRUE)

linear_layer_2 <- LBBNN_Linear(in_features = 300,out_features = 10,
                         prior_inclusion = 0.5,standard_prior = 1,
                         density_init = c(-5,15),num_transforms = 2,
                         flow = FALSE,hidden_dims = c(200,200),device = device,
                         bias_inclusion_prob = FALSE,conv_net = TRUE)





LBBNN_ConvNet <- torch::nn_module(
  "LBBNN_ConvNet",
  
  initialize = function(conv1,conv2,fc1,fc2,device = device) {
    self$problem_type <- 'multiclass classification'
    self$input_skip <- FALSE
    self$conv1 <- conv1
    self$conv2 <- conv2
    self$fc1 <- fc1
    self$fc2 <- fc2
    
    
    self$pool <- torch::nn_max_pool2d(2)
    self$act <- torch::nn_leaky_relu()
    self$out <- torch::nn_log_softmax(dim = 2)
    self$pout <- torch::nn_softmax(dim = 2)
    self$loss_fn <- torch::nn_nll_loss(reduction='sum')
  },
  
  forward = function(x,MPM=FALSE, predict = FALSE) {
    x = self$act(self$conv1(x,MPM))
    x = self$pool(x)
    x = self$act(self$conv2(x,MPM))
    x = self$pool(x)
    x = torch::torch_flatten(x,start_dim = 2)
    x = self$act(self$fc1(x,MPM))
    if(!predict)
      x = self$out(self$fc2(x,MPM))
    else
      x = self$pout(self$fc2(x,MPM))
  },
  kl_div = function(){
    kl <- self$conv1$kl_div() + self$conv2$kl_div() +
          self$fc1$kl_div() + self$fc2$kl_div()
    return(kl)
  },
  density = function(){
    alphas <- NULL
    alphas <- c(as.numeric(self$conv1$alpha),as.numeric(self$conv2$alpha)
                ,as.numeric(self$fc1$alpha),as.numeric(self$fc2$alpha))
    return(mean(alphas > 0.5))
    
    
  },
  compute_paths = function(){
    NULL
  },
  density_active_path = function(){
    NA
  }
)

model <- LBBNN_ConvNet(conv_layer_1,conv_layer_2,
                       linear_layer_1,linear_layer_2,device)
model$to(device = device)

train_LBBNN(epochs = 1,LBBNN = model, lr = 0.001,train_dl = train_loader,
            device = device)

validate_LBBNN(model,num_samples = 10,test_dl = test_loader,device = device)
print(model)


draws <- 20 # how many samples from posterior to use
out_dim <- 10 # dimensionality of the output
mpm <- TRUE # if to use the MPM
model$eval() # to avoid gradient computations
predictions <- NULL
torch::with_no_grad({ 
  coro::loop(for (b in test_loader)# go through all data
    { 
    outputs <- torch::torch_zeros(draws,dim(b[[1]])[1],out_dim)$to(device=device)
    for(i in 1:draws)# go through all draws 
    {
      data <- b[[1]]$to(device = device)
      outputs[i]<- model(data,MPM=mpm,predict = TRUE)
    }
    predictions <- torch::torch_cat(c(predictions,outputs),dim = 2) #combine all
    
  })  
})

dim(predictions)

print(predictions[,1,])

