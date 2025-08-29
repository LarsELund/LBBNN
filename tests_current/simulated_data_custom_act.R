library(torch)


N = 5000
p = 10 #this gives the correct solution, but using p = 5 gives a solution using relu + linear.

set.seed(2)
torch::torch_manual_seed(2)
#generate some data
X <- matrix(runif(N*p,min = -5 ,max = 0), ncol = p)

#make some X relevant for prediction
y_base <- exp(X[,1])
                
hist(y_base)
y <- c()
# change y to 0 and 1
y[y_base > median(y_base)] = 1
y[y_base <= median(y_base)] = 0


sim_dat <- as.data.frame(X)
sim_dat <-cbind(sim_dat,y_base)




loaders <- get_dataloaders(sim_dat,train_proportion = 0.9,train_batch_size = 450,
                           test_batch_size = 100,standardize = FALSE)
train_loader <- loaders$train_loader
test_loader <- loaders$test_loader

problem <- 'regression'
sizes <- c(p,10,1) #p input variables
inclusion_priors <-c(0.5,0.5) #one prior probability per weight matrix.
stds <- c(100,100) #prior standard deviation for each layer.
inclusion_inits <- matrix(rep(c(-5,10),2),nrow = 2,ncol = 2) #one low and high for each layer
device <- 'cpu' #can also be mps or gpu.


#use the function Custom_activation() to get an activation function
#it lives in the file custom_activation_function.R
model_input_skip <- LBBNN_Net(problem_type = problem,sizes = sizes,prior = inclusion_priors,
                              inclusion_inits = inclusion_inits,input_skip = TRUE,std = stds,
                              flow = TRUE,device = device,custom_act = Custom_activation())



results_input_skip <- train_LBBNN(epochs = 500,LBBNN = model_input_skip,
                                  lr = 0.01,train_dl = train_loader,device = device)

#run validate before plotting
validate_LBBNN(LBBNN = model_input_skip,num_samples = 100,test_dl = test_loader,device)

LBBNN_plot(model_input_skip,layer_spacing = 1,neuron_spacing = 1,vertex_size = 10,edge_width = 0.5)


#get a random sample from the dataloader
x <- torch::dataloader_next(torch::dataloader_make_iter(train_loader))[[1]]
inds <- sample.int(dim(x)[1],2)

d1 <- x[inds[1],]
d2 <- x[inds[2],]


plot_local_explanations_gradient(model_input_skip,d1,num_samples = 100)
plot_local_explanations_gradient(model_input_skip,d2,num_samples = 100)

#check what the predicted values look like
b <- posterior_predict.LBBNN(model_input_skip,mpm = TRUE,test_loader,draws = 100)
b<-b$squeeze()
b <- torch::torch_mean(b,dim = 1)
b <- as.numeric(b)


set.seed(2)
sample <- sample.int(n = nrow(X), size = floor(0.9*nrow(X)), replace = FALSE)
train  <- sim_dat[sample,]
test   <- sim_dat[-sample,]
gbm_model <- gbm(y_base ~ ., data = train, 
                 distribution = "gaussian", 
                 n.trees = 10000, 
                 interaction.depth = 3, 
                 shrinkage = 0.01,
                 cv.folds = 5) 

predictions <- predict(gbm_model, newdata = test) 

mse_lbbnn <- mean((b - y_base)^2)
mse_gbm <- mean((predictions - y_base)^2)

print(mse_lbbnn)
print(mse_gbm)


