#use this to abstract away the torch dataloader objects so the user only needs the function defined here
library(torch)

#'@export
get_dataloaders <- function(dataset,train_proportion,train_batch_size,test_batch_size,
                            standardize = TRUE, shuffle_train = TRUE,shuffle_test = FALSE){
  if(! inherits(dataset, "data.frame"))stop(paste('dataset must be a data.frame object'))
  if(! inherits(train_proportion, "numeric"))stop(paste(train_proportion,'must be a numeric value'))
  if(train_proportion > 1 | train_proportion < 0)stop(paste(train_proportion,'must be a numeric between 0 and 1'))
  sample <- sample.int(n = nrow(dataset), size = floor(train_proportion*nrow(dataset)), replace = FALSE)
  train  <- dataset[sample,]
  test   <- dataset[-sample,]
  p <- dim(train)[2] - 1
  y_train <- as.numeric(train[,dim(train)[2]])
  y_test <- as.numeric(test[,dim(test)[2]])
  x_train <- as.matrix(train[,1:p])
  x_test <- as.matrix(test[,1:p])

  if(standardize){
    x_train <- scale(x_train)
    x_test <- scale(x_test, center=attr(x_train, "scaled:center"), scale=attr(x_train, "scaled:scale"))
  }
  train_data <- torch::tensor_dataset(torch::torch_tensor(x_train),torch::torch_tensor(y_train)) 
  test_data <- torch::tensor_dataset(torch::torch_tensor(x_test),torch::torch_tensor(y_test))
  
  if(train_batch_size > length(train_data))(stop('Can not have larger batch size than the amount of training data'))
  if(test_batch_size > length(test_data))(stop('Can not have larger test batch size than the amount of test data'))
  
  train_loader <- torch::dataloader(train_data, batch_size = train_batch_size , shuffle = shuffle_train)
  test_loader <- torch::dataloader(test_data, batch_size = test_batch_size,shuffle = shuffle_test)
  l = list('train_loader' = train_loader,'test_loader'=test_loader)
  return(l)
  
}




