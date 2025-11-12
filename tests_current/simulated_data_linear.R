library(torch)
#the first tutorial in the article

i = 1000
j = 15



set.seed(42)
torch::torch_manual_seed(42)
#generate some data
X <- matrix(rnorm(i*j,mean = 0,sd = 1), ncol = j)
bias <- 0
rho <- 0.0
X[,3] <- rho * X[,1] + (1 - rho) * X[,3]
#make some X relevant for prediction
y_base <- c()
y_base <-  bias + 0.6* X[,1] - 0.4*X[,2] + 0.5 * X[,3] +rnorm(n = i,sd = 0.1)
hist(y_base)
sim_data <- as.data.frame(X)
sim_data <-cbind(sim_data,y_base)




loaders <- get_dataloaders(sim_data,train_proportion = 0.9,
                           train_batch_size = 450,test_batch_size = 100,
                           standardize = FALSE)
train_loader <- loaders$train_loader
test_loader  <- loaders$test_loader

problem <- 'regression'
sizes <- c(j,5,5,1) # 2 hidden layers, 5 neurons in each 
incl_priors <-c(0.5,0.5,0.5) #prior inclusion probs for each weight matrix
stds <- c(1,1,1) #prior distribution for the standard deviation of the weights
incl_inits <- matrix(rep(c(-10,10),3),nrow = 2,ncol = 3) #initializations for inclusion params
device <- 'cpu' #can also be 'gpu' or 'mps'


model_input_skip <- LBBNN_Net(problem_type = problem,sizes = sizes,prior = incl_priors,
                              inclusion_inits = incl_inits,input_skip = TRUE,std = stds,
                              flow = FALSE,device = device)



train_LBBNN(epochs = 2000,LBBNN = model_input_skip,
            lr = 0.01,train_dl = train_loader,device = device)

validate_LBBNN(LBBNN = model_input_skip,num_samples = 10,test_dl = test_loader,device)

coef(model_input_skip,dataset = train_loader,inds = c(1,5,60,100,250))

#LBBNN_plot(model_input_skip,layer_spacing = 1,neuron_spacing = 1,vertex_size = 8,edge_width = 0.5)

x <- train_loader$dataset$tensors[[1]] #grab the dataset
y <- train_loader$dataset$tensors[[2]] 
ind <- 42
data <- x[ind,] #plot this specific data-point
output <- y[ind]
print(output$item())
plot(model_input_skip,type = 'local',data = data)

plot(model_input_skip,type = 'global',vertex_size = 10,edge_width = 0.4,label_size = 0.3)

#plot_local_explanations_gradient(model_input_skip,d1,num_samples = 100)
#plot_local_explanations_gradient(model_input_skip,d2,num_samples = 100)
#plot_local_explanations_gradient(model_input_skip,rand_data,num_samples = 100)






log_reg <- glm(y_base ~ ., family = "gaussian", data = sim_data)
summary(log_reg)
