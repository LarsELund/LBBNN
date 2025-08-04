library(torch)


N = 5000
p = 15

set.seed(2)
torch::torch_manual_seed(2)
#generate some data
X <- matrix(rnorm(N*p,mean =-0.1 ,sd = 0.1), ncol = p)

#make some X relevant for prediction
y_base <-  0.3 * log(abs(X[,1])) + 0.2* cos(X[,2] * 2 * pi) + 3* X[,3] * X[,4] + 2.4 * X[,5] - 2* X[,6] **2 + rnorm(N,sd = 0.01) 
hist(y_base)
y <- c()
# change y to 0 and 1
y[y_base > median(y_base)] = 1
y[y_base <= median(y_base)] = 0


sim_dat <- as.data.frame(X)
sim_dat <-cbind(sim_dat,y)




loaders <- get_dataloaders(sim_dat,train_proportion = 0.9,train_batch_size = 1500,test_batch_size = 500)
train_loader <- loaders$train_loader
test_loader <- loaders$test_loader

problem <- 'binary classification'
sizes <- c(p,5,5,1) #p input variables
inclusion_priors <-c(0.5,0.5,0.5) #one prior probability per weight matrix.
stds <- c(100,100,100) #prior standard deviation for each layer.
inclusion_inits <- matrix(rep(c(-10,10),3),nrow = 2,ncol = 3) #one low and high for each layer
device <- 'cpu' #can also be mps or gpu.


model_input_skip <- LBBNN_Net(problem_type = problem,sizes = sizes,prior = inclusion_priors,
                              inclusion_inits = inclusion_inits,input_skip = TRUE,std = stds,
                              flow = TRUE,device = device)



results_input_skip <- train_LBBNN(epochs = 600,LBBNN = model_input_skip,
                                  lr = 0.005,train_dl = train_loader,device = device)

#run validate before plotting
validate_LBBNN(LBBNN = model_input_skip,num_samples = 10,test_dl = test_loader,device)

LBBNN_plot(model_input_skip,layer_spacing = 1,neuron_spacing = 1,vertex_size = 10,edge_width = 0.5)


#get a random sample from the dataloader
x <- torch::dataloader_next(torch::dataloader_make_iter(train_loader))[[1]]
set.seed(1)
inds <- sample.int(dim(x)[1],2)

d1 <- x[inds[1],]
d2 <- x[inds[2],]


plot_local_explanations_gradient(model_input_skip,d1,num_samples = 10)
plot_local_explanations_gradient(model_input_skip,d2,num_samples = 10)




