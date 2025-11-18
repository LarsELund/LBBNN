#use this to abstract away the torch dataloader objects so the user only needs the function defined here
library(torch)

#' @title Wrapper around \code{torch::dataloader}
#' @description  Avoids users having to manually define their own dataloaders.
#' @param dataset A \code{data.frame}. The last column is assumed to be the dependent variable.
#' @param train_proportion numeric, between 0 and 1. Proportion of data to be used for training. 
#' @param train_batch_size integer, number of samples per batch in the training dataloader. 
#' @param test_batch_size integer, number of sampels per batch in the testing dataloader.
#' @param standardize logical, whether to standardize input-features, default is TRUE.
#' @param shuffle_train logical, whether to shuffle the training data each epoch. default is TRUE
#' @param shuffle_test  logical, shuffle test data, default is FALSE. Usually not needed.
#' @param seed integer. Used for reproducibility purposes in the train/test split.
#' @return A list containing:
#'   \describe{
#'     \item{train_loader}{A \code{torch::dataloader} for the training data.}
#'     \item{test_loader}{A \code{torch::dataloader} for the test data.}
#'   }
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




