library(Matrix)
require(graphics)

#' Function that checks how many times inputs are included, and from which layer
#' @description Useful when the number of inputs and/or hidden neurons are very
#' large, and direct visualization of the network is difficult. 
#' @param model A trained LBBNN model with input_skip. 
#' @return A matrix of shape (p, L-1) where p is the number of input variables
#' and L the total number of layers (including input and output), with each element being 1 if the variable is included
#' or 0 if not included. 
#' @export
get_input_inclusions <- function(model){
  if(model$input_skip == FALSE)(stop('This function is currently only implemented for input-skip'))
  x_names <- c()
  layer_names <- c()
  for(k in 1:model$sizes[1]){
    x_names<- c(x_names,paste('x',k-1,sep = ''))
  }
  for(l in 1:(length(model$sizes)-1)){
    layer_names <- c(layer_names,paste('L',l-1,sep = ''))
  }
  
  
  inclusion_matrix <- matrix(0,nrow = model$sizes[1],ncol = length(model$sizes) - 1)
  #add the names
  colnames(inclusion_matrix) <- layer_names
  rownames(inclusion_matrix) <- x_names
  
  
  inp_size <- model$sizes[1]
  i <- 1
  for(l in model$layers$children){
    alp <- l$alpha_active_path
    incl<- as.numeric(torch::torch_sum(alp[,-inp_size:dim(alp)[2]],dim = 1))
    inclusion_matrix[,i] <- incl 
    i <- i + 1
  }
  alp_out <- model$out_layer$alpha_active_path
  incl<- as.numeric(torch::torch_sum(alp_out[,-inp_size:dim(alp_out)[2]],dim = 1))
  inclusion_matrix[,i] <- incl 
  i <- i + 1
  
  return(inclusion_matrix) 
}

#' @export
summary.LBBNN_Net <- function(object, ...) {

  if(object$input_skip == FALSE)(stop('Summary only applies to objects with input-skip = TRUE'))
  if(object$computed_paths == FALSE){object$compute_paths_input_skip()} 
  inclusions <- get_input_inclusions(object) # a matrix of size (p,L), with p # inputs, L # layers 
   
  #now get the average inclusion probabilities of each layer
  p <- object$sizes[1] # number of inputs
  L <- length(object$sizes) - 1 # number of layers
  alpha_means <- matrix(nrow = p, ncol = L)
  i <- 1
  all_alphas <- c()
  col_names <- c()
  for(l in object$layers$children){
    alpha_l <- l$alpha$clone()$detach()$cpu()
    aa <- alpha_l[, (dim(alpha_l)[2] - p + 1):dim(alpha_l)[2]] #only need last p corresponding to input, ignoring hidden layer alphas
    alpha_means[,i] <- round(as.numeric(aa$mean(dim = 1)),3)
    all_alphas <- rbind(all_alphas, as.matrix(aa))
    col_names <- c(col_names,paste('a',i - 1,sep = ''))
    i <- i + 1
  }
  #now do the output layer
  alpha_out <- object$out_layer$alpha$clone()$detach()
  a_out <- alpha_out[, (dim(alpha_out)[2] - p + 1):dim(alpha_out)[2]]
  all_alphas <- rbind(all_alphas, as.matrix(a_out))
  col_names <- c(col_names,paste('a',i - 1,sep = ''))
  alpha_means[,i] <- round(as.numeric(a_out$mean(dim = 1)),3)
  a_avg <- round(colMeans(all_alphas),3)
  colnames(alpha_means) <- col_names
  alpha_means <- cbind(alpha_means,a_avg)
  summary_out <- cbind(inclusions,alpha_means)
  cat("Summary of LBBNN_Net object:\n")
  cat("-----------------------------------\n")
  cat("Shows the number of times each variable was included from each layer\n")
  cat("-----------------------------------\n")
  cat("Then the average inclusion probability for each input from each layer\n")
  cat("-----------------------------------\n")
  cat("The final column shows the average inclusion probability across all layers\n")
  cat("-----------------------------------\n")
  print(summary_out)

}

#' @export
residuals.LBBNN_Net <- function(object,type = c('response'), ...) {
  y_true <- object$y
  y_predicted <- object$r
  if(type == 'response'){
    return(y_true - y_predicted)
  }
  else(stop('only y - y_pred residuals are currently implemented'))
  
  
  
}



#' @export
coef.LBBNN_Net <- function(object,data = c('train','test','other'),dataset = NULL,num_data = 1,num_samples = 10, ...) {
  d <- match.arg(data)
  all_means <- matrix(nrow = object$sizes[1],ncol = num_data)
  row_names <- c()
  for (i in 1:object$sizes[1]){
    row_names <- c(row_names,paste('x',i-1,sep = ''))
  }
  
  
  
  if(d == 'train'){
    X <- train_loader$dataset$tensors[[1]]$clone()$detach()$cpu()}
  else if(d == 'test'){
    X <- test_loader$dataset$tensors[[1]]$clone()$detach()$cpu()}
  
  else{X <- dataset         # should be a tensor with shape (num_data,p), but need to make sure it accepts MNIST or other img data
  if(class(X)[1] != 'torch_tensor')stop('the dataset must be a torch_tensor')   
  if(length(dim(X)) == 1){X <- X$unsqueeze(dim = 1)} #reshape (p) to shape (1,p)
  if(dim(X)[length(dim(X))] != object$sizes[1])stop('the last index must have shape equal to p')
  
  } 
  
  X_explain <- X[1:num_data,]
  
  if(dim(X)[1] < num_data)stop(paste('num_data =',num_data, 'can not be greater than the number of total data points,' ,dim(X)[1]))
  
  if(num_data == 1){ #here we get the CI from the uncertainty around the one sample
    expl <- get_local_explanations_gradient(object,X_explain,num_samples = num_samples)
    e <- expl$explanations
    mean_explanation <- as.matrix(e$mean(dim = -1))
    qs <- t(apply(mean_explanation,2,quants))
    rownames(qs) <- row_names
    return(qs)
    
    
    
    
  }
  
  
  for( i in 1:num_data){ #loop over data points, here the CI is across the means for each sample
    data <- X_explain[i,]
    expl <- get_local_explanations_gradient(object,data,num_samples = num_samples)
    e <- expl$explanations
    mean_explanation <- as.numeric(e$mean(dim = -1)$mean(dim = 1)) #first avg over classes, then over num_samples
    all_means[,i] <- mean_explanation
  }
  
  
  
  rownames(all_means) <- row_names
  qs <- t(apply(all_means,1,quants))
  
  
  return(qs)
  
}









