library(torch)
library(torchvision)

dir <- "./dataset/mnist"

train_ds <- mnist_dataset(
  dir,
  download = TRUE,
  transform = transform_to_tensor
)

test_ds <- mnist_dataset(
  dir,
  train = FALSE,
  transform = transform_to_tensor
)

train_loader <- dataloader(train_ds, batch_size = 100, shuffle = TRUE)
test_loader <- dataloader(test_ds, batch_size = 1000)





### create the convolutional network for MNIST 

LBBNN_ConvNet <- nn_module(
  "LBBNN_ConvNet",
  
  initialize = function(problem_type,flow = FALSE,
                        num_transforms = 2, dims = c(200,200),device) {
    self$problem_type = problem_type
    self$flow = flow
    self$num_transforms = num_transforms
    self$dims = dims
    self$conv1 <- LBBNN_Conv2d(in_channels = 1, out_channels =32, kernel_size = 5,
                               prior_inclusion = 0.25,standard_prior = 1,density_init = c(-10,10),
                               num_transforms = self$num_transforms,hidden_dims = self$dims,device = device)
    self$conv2 <- LBBNN_Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5,
                               prior_inclusion = 0.25,standard_prior = 1,density_init = c(-10,15),
                               num_transforms = self$num_transforms,hidden_dims = self$dims,device = device)
    self$fc1 <- LBBNN_Linear(in_features = 1024, out_features = 300,
                             prior_inclusion = 0.25,standard_prior = 1,density_init = c(-10,10),
                             num_transforms = self$num_transforms,hidden_dims = self$dims,device = device)
    self$fc2 <- LBBNN_Linear(in_features = 300,out_features = 10,
                             prior_inclusion = 0.25,standard_prior = 1,density_init = c(-5,15),
                             num_transforms = self$num_transforms,hidden_dims = self$dims,device = device)

    self$pool <- torch::nn_max_pool2d(2)
    self$out <- torch::nn_log_softmax(dim = 2)
    self$loss_fn <- torch::nn_nll_loss(reduction='sum')
  },
  
  forward = function(x,MPM=FALSE) {
    x = torch::nnf_leaky_relu(self$conv1(x,MPM))
    x = self$pool(x)
    x = torch::nnf_leaky_relu(self$conv2(x,MPM))
    x = self$pool(x)
    x = torch::torch_flatten(x,start_dim = 2)
    x = torch::nnf_leaky_relu(self$fc1(x,MPM))
    x = self$out(self$fc2(x,MPM))
    
  },
  kl_div = function(){
    kl <- self$conv1$kl_div() + self$conv2$kl_div()+ self$fc1$kl_div() + self$fc2$kl_div()
    return(kl)
  },
  density = function(){
    alphas <- NULL
    alphas <- c(as.numeric(self$conv1$alpha),as.numeric(self$conv2$alpha)
                ,as.numeric(self$fc1$alpha),as.numeric(self$fc2$alpha))
    return(mean(alphas > 0.5))
    
    
  }
)





problem <- 'MNIST'
sizes <- c(28*28,400,600,10) #7 input variables, one hidden layer of 100 neurons, 1 output neuron.
inclusion_priors <-c(0.1,0.1,0.1) #one prior probability per weight matrix.
std_priors <-c(1.0,1.0,1.0) #one prior probability per weight matrix.
inclusion_inits <- matrix(rep(c(-15,10),3),nrow = 2,ncol = 3)
device <- 'mps'
torch_manual_seed(0)
model <- LBBNN_Net(problem_type = problem,sizes = sizes,
                   prior = inclusion_priors,inclusion_inits =inclusion_inits ,input_skip = FALSE,
                   std = std_priors,flow = TRUE,num_transforms = 2,dims = c(200,200),device = device)
model$to(device = device)
results <- train_LBBNN(epochs = 1,LBBNN = model, lr = 0.0001,train_dl = train_loader,device = device)
validate <-validate_LBBNN(model,num_samples = 100,test_dl = test_loader,device = device)







