library(ggplot2)
library(torch)
library(gbm)

#code for example two in the overleaf article


seed <- 42
torch::torch_manual_seed(seed)
loaders <- get_dataloaders(Gallstone_Dataset,train_proportion = 0.70,
                  train_batch_size = 223,test_batch_size = 96,standardize = TRUE,seed = seed)
train_loader <- loaders$train_loader
test_loader <- loaders$test_loader

#the paper reports approx 85% accuracy
#https://pmc.ncbi.nlm.nih.gov/articles/PMC11309733/#T2



set.seed(seed)
sample <- sample.int(n = nrow(Gallstone_Dataset), size = floor(0.7*nrow(Gallstone_Dataset)), replace = FALSE)
train  <- Gallstone_Dataset[sample,]
test   <- Gallstone_Dataset[-sample,]
gbm_model <- gbm(outcome ~ ., data = train, 
                 distribution = "bernoulli", 
                 n.trees = 10000, 
                 interaction.depth = 3, 
                 shrinkage = 0.01,
                 cv.folds = 5) 

predictions <- predict(gbm_model, newdata = test,type = 'response') 
ground_truth <- test$outcome
acc <- mean(((predictions > 0.5) == ground_truth))
print(paste('GBM accuracy =',acc))



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

