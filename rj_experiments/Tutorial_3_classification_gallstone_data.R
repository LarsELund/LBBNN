library(LBBNN)

#### Tutorial 3: classification on gallstone dataset

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
