################# Install packages if needed ###################################

if(!requireNamespace("torch"))
  install.packages("torch")
if(!requireNamespace("torchvision"))
  install.packages("torchvision")
if(!requireNamespace("devtools"))
  install.packages("devtools")

# use git by default as CRAN version will be updated while on git we have
# a static version of the package
use_cran = FALSE
if(use_cran) 
{  
  install.packages("LBBNN")
}else
{
  devtools::install_github("LarsELund/LBBNN@rjournal-0.1.2",
                 force = TRUE, build_vignettes = FALSE)
}


################# Tutorial 1:  simulated data with linear effects ##############
rm(list = ls())
library(LBBNN)

i = 1000
j = 15

set.seed(42)
torch::torch_manual_seed(42)
#generate some data
X <- matrix(rnorm(i*j,mean = 0,sd = 1), ncol = j)
#make some X relevant for prediction
y_base <- c()
y_base <-  0.6* X[,1] - 0.4*X[,2] + 0.5 * X[,3] +rnorm(n = i,sd = 0.1)
sim_data <- as.data.frame(X)
sim_data <-cbind(sim_data,y_base)

loaders <- get_dataloaders(sim_data,train_proportion = 0.9,
                           train_batch_size = 450,test_batch_size = 100,
                           standardize = FALSE)
train_loader <- loaders$train_loader
test_loader  <- loaders$test_loader

problem <- 'regression'
sizes <- c(j,5,5,1) # 2 hidden layers, 5 neurons in each 
incl_priors <-c(0.5,0.5,0.5) #prior inclusion probability
stds <- c(1,1,1) #prior for the standard deviation of the weights
incl_inits <- matrix(rep(c(-10,10),3),nrow = 2,ncol = 3) #inclusion inits
device <- 'cpu' #can also be 'gpu' or 'mps'


model_input_skip <- LBBNN_Net(problem_type = problem,sizes = sizes,
                              prior = incl_priors,inclusion_inits = incl_inits,
                              std = stds, input_skip = TRUE,flow = FALSE,
                              num_transforms = 2,dims = c(2,2),
                              raw_output = FALSE,custom_act = NULL,
                              link = NULL,nll = NULL,
                              bias_inclusion_prob = FALSE,device = device)



train_LBBNN(epochs = 2000,LBBNN = model_input_skip,
            lr = 0.01,train_dl = train_loader,device = device)
validate_LBBNN(LBBNN = model_input_skip,num_samples = 10,test_dl = test_loader,
               device = device)

coef(model_input_skip,dataset = train_loader,inds = c(1,2,5,10,20),
     output_neuron = 1, num_data = 5, num_samples = 10)



x <- train_loader$dataset$tensors[[1]] #grab the dataset
y <- train_loader$dataset$tensors[[2]] 
ind <- 42
data <- x[42,] #plot this specific data-point
output <- y[ind]
print(output$item())
plot(model_input_skip,type = 'local',data = data)

summary(model_input_skip)

################# Tutorial 2 : simulated data with non-linear effects ##########
rm(list = ls())
library(LBBNN)


i = 1000
j = 15

set.seed(42)
torch::torch_manual_seed(42)
#generate some data
X <- matrix(runif(i*j,0,0.5), ncol = j)

#make some X relevant for prediction
y_base <- -3 +  0.1 * log(abs(X[,1])) + 3 * cos(X[,2]) + 2* X[,3] * X[,4] +   X[,5] -  X[,6] **2 + rnorm(i,sd = 0.1) 
hist(y_base)
y <- c()
# change y to 0 and 1
y[y_base > median(y_base)] = 1
y[y_base <= median(y_base)] = 0


sim_data <- as.data.frame(X)
sim_data <-cbind(sim_data,y)




loaders <- get_dataloaders(sim_data,train_proportion = 0.9,
                           train_batch_size =450 ,test_batch_size = 100,
                           standardize = FALSE)
train_loader <- loaders$train_loader
test_loader  <- loaders$test_loader

problem <- 'binary classification'
sizes <- c(j,5,5,1) # 2 hidden layers, 5 neurons in each 
incl_priors <-c(0.5,0.5,0.5) #prior inclusion probs for each weight matrix
stds <- c(1,1,1) #prior distribution for the standard deviation of the weights
incl_inits <- matrix(rep(c(-10,10),3),nrow = 2,ncol = 3) #initializations for inclusion params
device <- 'cpu' #can also be 'gpu' or 'mps'


model_input_skip <- LBBNN_Net(problem_type = problem,sizes = sizes,prior = incl_priors,
                              inclusion_inits = incl_inits,input_skip = TRUE,std = stds,
                              flow = TRUE,device = device,bias_inclusion_prob = F)



train_LBBNN(epochs = 1500,LBBNN = model_input_skip,
            lr = 0.005,train_dl = train_loader,device = device)

validate_LBBNN(LBBNN = model_input_skip,num_samples = 100,test_dl = test_loader,device)


plot(model_input_skip,type = 'global',vertex_size = 10,
     edge_width = 0.4,label_size = 0.3)

################# Tutorial 3: classification on gallstone dataset ##############
rm(list = ls())
library(LBBNN)


seed <- 42
torch::torch_manual_seed(seed)
loaders <- get_dataloaders(Gallstone_Dataset,train_proportion = 0.70,
                           train_batch_size = 223,test_batch_size = 96,standardize = TRUE,seed = seed)
train_loader <- loaders$train_loader
test_loader <- loaders$test_loader

#the paper reports approx 85% accuracy
#https://pmc.ncbi.nlm.nih.gov/articles/PMC11309733/#T2


problem <- 'binary classification'
sizes <- c(40,3,3,1) 
inclusion_priors <-c(0.5,0.5,0.5) #one prior probability per weight matrix.
stds <- c(1,1,1) #prior standard deviation for each layer.


inclusion_inits <- matrix(rep(c(-5,10),3),nrow = 2,ncol = 3) #one low and high for each layer
device <- "cpu"





model_input_skip <- LBBNN_Net(problem_type = problem,sizes = sizes,prior = inclusion_priors,
                              inclusion_inits = inclusion_inits,input_skip = TRUE,std = stds,
                              flow = TRUE,device = device)






results_input_skip <- train_LBBNN(epochs = 1000,LBBNN = model_input_skip,
                                  lr = 0.005,train_dl = train_loader,device = device,
                                  scheduler = 'step',sch_step_size = 1000)

validate_LBBNN(LBBNN = model_input_skip,num_samples = 100,test_dl = test_loader,device)


x <- train_loader$dataset$tensors[[1]] #grab the dataset
y <- train_loader$dataset$tensors[[2]] 
ind <- 42
data <- x[ind,] #plot this specific data-point
output <- y[ind]
print(output$item())
plot(model_input_skip,type = 'local',data = data)

plot(model_input_skip,type = 'global',vertex_size = 5,edge_width = 0.1,label_size = 0.2)

summary(model_input_skip)
coef(model_input_skip,train_loader)

predictions <- predict(model_input_skip, newdata = test_loader,
                       draws = 100,mpm = TRUE)

dim(predictions)

print(predictions)

################# Tutorial 4: : Convolutional architecture #####################
rm(list = ls())
library(LBBNN)

### This example needs the torchvision package to download data
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

device <- 'cpu'
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

train_LBBNN(epochs = 20,LBBNN = model, lr = 0.001,train_dl = train_loader,
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

print(torch::torch_round(predictions[1:5,258,],4))

print(torch::torch_round(predictions[1:5,254,],4))

################# The End ######################################################
