#use this to abstract away the torch dataloader objects so the user only needs the function defined here
library(torch)

#' Wrapper around torch dataloader
#' @description  Takes a dataset and returns both torch train and test dataloaders,
#' needed for subsequent optimization. Avoids the user having to define these dataloaders
#' themselves. 
#' @param dataset Some dataset. Must be a data.frame, where the last column is y, the dependent variable.
#' @param train_proportion A number between 0 and 1, giving the proportion of data to be used for training.
#' Usually a large number e.g. 0.8, but could also be smaller than 0.5 even though this is uncommon. 
#' @param train_batch_size How many samples in each batch in the train dataloader. 
#' @param test_batch_size How many samples in each batch in the test dataloader. 
#' @param standardize default is TRUE. Usually an advantage for gradient based optimization.
#' @param shuffle_train default is TRUE. Ensures data is randomly shuffled before each iteration.
#' @param shuffle_test  default is FALSE, as there is no need to shuffle the test data, as the order of the data is irrelevant.
#' @param seed The seed used in splitting train/test. Important if one wants to compare to other algorithms that
#' do not need to use torch dataloaders.
#' @return A list containing a train_loader and a test_loader object. 
#'@export 
get_dataloaders <- function(dataset,train_proportion,train_batch_size,test_batch_size,
                            standardize = TRUE, shuffle_train = TRUE,shuffle_test = FALSE,seed = 1){
  if(! inherits(dataset, "data.frame"))stop(paste('dataset must be a data.frame object'))
  if(! inherits(train_proportion, "numeric"))stop(paste(train_proportion,'must be a numeric value'))
  if(train_proportion > 1 | train_proportion < 0)stop(paste(train_proportion,'must be a numeric between 0 and 1'))
  set.seed(seed)
  sample <- sample.int(n = nrow(dataset), size = floor(train_proportion*nrow(dataset)), replace = FALSE)
  train  <- dataset[sample,]
  test   <- dataset[-sample,]
  p <- dim(train)[2] - 1
  y_train <- as.numeric(train[,dim(train)[2]])
  y_test <- as.numeric(test[,dim(test)[2]])
  x_train <- as.matrix(train[,1:p])
  x_test <- as.matrix(test[,1:p])
  if(min(y_train) == 0 & max(y_train) > 1){y_train <- y_train + 1  #indexing needs to go from 1 <- C in multiclass case
                                           y_test<- y_test + 1}
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




