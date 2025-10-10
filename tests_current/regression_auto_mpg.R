library(ggplot2)
library(torch)
library(gbm)
library(mltools)

## performance of our method seems highly dependent on seed, 
## with seed = 42 it is on par with GBM, but much worse with others, e.g. 4
## so we have some convergence issues here, or in general some problems with
## regression? need to check this further 
seed <- 42
torch::torch_manual_seed(seed)
loaders <- get_dataloaders(mgp_dataset,train_proportion = 0.80,
                           train_batch_size = 318,test_batch_size = 80,standardize = TRUE,seed = seed)
train_loader <- loaders$train_loader
test_loader <- loaders$test_loader



set.seed(seed)
sample <- sample.int(n = nrow(mgp_dataset), size = floor(0.8*nrow(mgp_dataset)), replace = FALSE)
train  <- mgp_dataset[sample,]
test   <- mgp_dataset[-sample,]
gbm_model <- gbm(outcome ~ ., data = train, 
                 distribution = "gaussian", 
                 n.trees = 10000, 
                 interaction.depth = 3, 
                 shrinkage = 0.01,
                 cv.folds = 5) 

predictions <- predict(gbm_model, newdata = test) 
ground_truth <- test$outcome




problem <- 'regression'
sizes <- c(23,10,1) #7 input variables, one hidden layer of 100 neurons, 1 output neuron.
inclusion_priors <-c(0.5,0.5 ) #one prior probability per weight matrix.
stds <- c(0.1,1) #prior standard deviation for each layer.


inclusion_inits <- matrix(rep(c(-20,1),2),nrow = 2,ncol = 2) #one low and high for each layer
device <- 'cpu' #can also be mps or gpu.


#works when inclusion is activated, breaks otherwise
model_input_skip <- LBBNN_Net(problem_type = problem,sizes = sizes,prior = inclusion_priors,
                              inclusion_inits = inclusion_inits,input_skip = TRUE,std = stds,
                              flow = TRUE,device = device)



results_input_skip <- train_LBBNN(epochs = 5000,LBBNN = model_input_skip,
                                  lr = 0.005,train_dl = train_loader,device = device,
                                  scheduler = 'step',sch_step_size = 10000)

#need to run validate before plotting
validate_LBBNN(LBBNN = model_input_skip,num_samples = 100,test_dl = test_loader,device)

LBBNN_plot(model_input_skip,layer_spacing = 1,neuron_spacing = 1,vertex_size = 8,edge_width = 0.5)


#get a random sample from the dataloader
x <- torch::dataloader_next(torch::dataloader_make_iter(train_loader))[[1]]
inds <- sample.int(dim(x)[1],2)

d1 <- x[inds[1],]
d2 <- x[inds[2],]


plot_local_explanations_gradient(model_input_skip,d1,num_samples = 100)
plot_local_explanations_gradient(model_input_skip,d2,num_samples = 100)


b <- posterior_predict.LBBNN(model_input_skip,mpm = TRUE,test_loader,draws = 100)
b<-b$squeeze()
b <- torch::torch_mean(b,dim = 1)
b <- as.numeric(b)

print(paste('GBM MSE =',mltools::mse(predictions,ground_truth)))
print(paste('GMB R2 = ',cor(predictions,ground_truth)^2))

print(paste('LBBNN MSE =',mltools::mse(b,ground_truth)))
print(paste('LBBNN R2 = ',cor(b,ground_truth)^2))


