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


#' Summary of LBBNN fit
#' @description Summary method for objects of the LBBNN_Net class. Note that this function only applies to objects trained with input-skip.
#' @param object An object of class LBBNN_Net.
#' @param ... further arguments passed to or from other methods.
#' @details For each layer (aside from the output), the summary contains a column showing the number of connections (paths) for each variable from that layer.
#' In addition, each layer has a column showing the average inclusion probability associated with each variable. The final column shows the average inclusion probability 
#' across all layers.
#' @return The table of the summary values.
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
  summary_out <- as.data.frame(cbind(inclusions,alpha_means))
  cat("Summary of LBBNN_Net object:\n")
  cat("-----------------------------------\n")
  cat("Shows the number of times each variable was included from each layer\n")
  cat("-----------------------------------\n")
  cat("Then the average inclusion probability for each input from each layer\n")
  cat("-----------------------------------\n")
  cat("The final column shows the average inclusion probability across all layers\n")
  cat("-----------------------------------\n")
  print(summary_out)
  invisible(summary_out)

}


#' Residuals from LBBNN fit
#' @description Residuals from an object of the LBBNN_Net class.
#' @param object An object of class LBBNN_Net.
#' @param type Currently only 'response' is implemented i.e. y_true - y_predicted.
#' @param ... further arguments passed to or from other methods.
#' @return The response residuals.
#' @export
residuals.LBBNN_Net <- function(object,type = c('response'), ...) {
  y_true <- object$y
  y_predicted <- object$r
  if(type == 'response'){
    return(y_true - y_predicted)
  }
  else(stop('only y - y_pred residuals are currently implemented'))
  
  
  
}


#'  Get model coefficients (local explanations) of an LBBNN_Net object
#' @description Given an input sample x_1,... x_j (with j the number of variables), the local explanation is found by 
#' considering active paths. If relu activations are assumed, each path is a piecewise
#' linear function, so the contribution for x_j is just the sum of the weights associated with the paths connecting x_j to the output. 
#' The contributions are found by taking the gradient wrt x.   
#' @param object an object of class LBBNN_Net.
#' @param dataset Either a train_loader object or a torch_tensor object. In the latter case, the user can supply their own data.  
#' @param inds a numeric vector of indicies indicating which samples in the dataset to be used for local explanations.
#' @param output_neuron Which class to explain. In the case where we have more than one output neuron, each one has to be explained separately.
#' @param num_data If no inds are chosen, the first num_data of the dataset are automatically used for explanations.
#' @param num_samples how many samples to use for model averaging when sampling the weights in the active paths. 
#' @param ... further arguments passed to or from other methods.
#' @details If num_data = 1 (or the user only supplies one sample), the confidence interval is taken around the mean explanation of that sample, using model averaging
#' of the weights. If num_data > 1, then the confidence interval is obtained wrt all the mean explanations of each individual sample.
#' @return The mean coefficients and 95% CI.
#' @export
coef.LBBNN_Net <- function(object,dataset,inds = NULL,output_neuron = 1,num_data = 1,num_samples = 10, ...) {
  if(output_neuron > object$sizes[length(object$sizes)])stop(paste('output_neuron =',output_neuron, 'can not be greater than' ,object$sizes[length(object$sizes)]))
  if(is.null(inds)){
    all_means <- matrix(nrow = object$sizes[1],ncol = num_data)
  }
  else{
    all_means <- matrix(nrow = object$sizes[1],ncol = length(inds))
  }
  
  
  
  row_names <- c()
  for (i in 1:object$sizes[1]){
    row_names <- c(row_names,paste('x',i-1,sep = ''))
  }
  
  

  if(class(dataset)[1] == 'dataloader'){
    X <- dataset$dataset$tensors[[1]]$clone()$detach()$cpu()}
  else if(class(dataset)[1] == 'torch_tensor'){
  X <- dataset         # should be a tensor with shape (num_data,p), but need to make sure it accepts MNIST or other img data
  if(length(dim(X)) == 1){X <- X$unsqueeze(dim = 1)} #reshape (p) to shape (1,p)
  if(dim(X)[length(dim(X))] != object$sizes[1])stop('the last index must have shape equal to p')
  }
  else{stop('dataset must be either a torch_tensor or a dataloader object')}
  
  if(is.null(inds)){
    if(dim(X)[1] < num_data)stop(paste('num_data =',num_data, 'can not be greater than the number of total data points,' ,dim(X)[1]))
    X_explain <- X[1:num_data,]
 
  }
  else{
    inds <- as.integer(inds) #in case user sends a numeric vector
    inds <- unique(inds) #remove any duplicates
    if(dim(X)[1] < length(inds))stop(paste('number of indecies =',length(inds), 'can not be greater than the number of total data points,' ,dim(X)[1]))
    if(dim(X)[1] < max(inds))stop(paste('the largest index =',max(inds), 'can not be greater than the number of total data points,' ,dim(X)[1]))
    
    num_data <- length(inds)
    X_explain <- X[inds,]
   
    
  }
  
  
  
  if(num_data == 1){ #here we get the CI from the uncertainty around the one sample
    expl <- get_local_explanations_gradient(object,X_explain,num_samples = num_samples)
    e <- expl$explanations
    e <- e[,,output_neuron] #get the explanation based on which output neuron is chosen
    
    mean_explanation <- as.matrix(e)
    qs <- t(apply(mean_explanation,2,quants))
    rownames(qs) <- row_names
    return(as.data.frame(qs))
    
    
    
    
  }
  
  
  for( i in 1:num_data){ #loop over data points, here the CI is across the means for each sample
    data <- X_explain[i,]
    expl <- get_local_explanations_gradient(object,data,num_samples = num_samples)
    e <- expl$explanations
    ee <- e[,,output_neuron]
    mean_explanation <- as.numeric(ee$mean(dim = 1)) 
    all_means[,i] <- mean_explanation
  }
  
  
  
  rownames(all_means) <- row_names
  qs <- t(apply(all_means,1,quants))
  
  
  return(as.data.frame(qs))
  
}


#'Obtain predictions based on the variaiontal posterior distribution of an LBBNN object.
#'@description Draw from the (variational) posterior predictive distribution.
#'@param object An instance of a trained LBBNN_net.
#'@param mpm To use the median probability model or not. 
#'@param newdata A torch dataloader containing the variables with which to predict.
#'@param draws The number of times to sample from the variational posterior. 
#'@param device The device to perform the operations on. Default is cpu. 
#'@param link Link function to apply to the output of the network. 
#'@return A matrix of size (draws,N,C), where N is the number of data points in the test_loader,
#'and C the number of classes. (1 for regression).
#' @export
predict.LBBNN_Net <- function(object,mpm,newdata,draws,device = 'cpu',link = NULL,...){#should newdata be a dataloader or a dataset?
  object$eval()
  object$raw_output = TRUE #skip final sigmoid/softmax
  if(! object$computed_paths){
    if(object$input_skip){object$compute_paths_input_skip()} #need this to get active paths to compute mpm
    else(object$compute_paths)
  }
  out_shape <- object$sizes[length(object$sizes)] #number of output neurons
  all_outs <- NULL
  torch::with_no_grad({ 
    coro::loop(for (b in newdata){
      outputs <- torch::torch_zeros(draws,dim(b[[1]])[1],out_shape)$to(device=device)
      for(i in 1:draws){
        data <- b[[1]]$to(device = device)
        outputs[i]<- object(data,MPM=mpm)
      }
      all_outs <- torch::torch_cat(c(all_outs,outputs),dim = 2) #add all the mini-batches together
      
    })  
  })
  return(all_outs)
}



#' @export
print.LBBNN_Net <- function(x, ...) {
  
  module_info <- x$modules[[1]]
  
  # Model description
  model_name <- if (isTRUE(x$input_skip)) {
    "LBBNN with input-skip"
  } else {
    "LBBNN without input-skip"
  }
  
  flow <- if (isTRUE(x$flow)) {
    "with normalizing flows"
  } else {
    "without normalizing flows"
  }
  
  # Header
  cat("\n========================================\n")
  cat("          LBBNN Model Summary           \n")
  cat("========================================\n\n")
  
  # Module info
  total_params <- sum(sapply(module_info$parameters, length))
  cat("Module Overview:\n")
  cat("  - An `nn_module` containing", total_params, "parameters.\n\n")
  
  # Submodules
  cat("---------------- Submodules ----------------\n")
  submodules <- module_info$modules
  if (length(submodules) == 0) {
    cat("  No submodules detected.\n")
  } else {
    for (name in names(submodules)) {
      mod <- submodules[[name]]
      if (is.null(mod)) next
      n_params <- if (!is.null(mod$parameters)) sum(sapply(mod$parameters, length)) else 0
      cat(sprintf("  - %-20s : %-15s # %d parameters\n", name, class(mod)[1], n_params))
    }
  }
  
  # Model details
  cat("\nModel Configuration:\n")
  cat("  -", model_name, "\n")
  cat("  - Optimized using variational inference", flow, "\n\n")
  
  # Priors
  cat("Priors:\n")
  cat("  - Prior inclusion probabilities per layer: ",
      paste(x$prior_inclusion, collapse = ", "), "\n")
  cat("  - Prior std dev for weights per layer:    ",
      paste(x$prior_std, collapse = ", "), "\n")
  
  cat("\n=================================================================\n\n")
}


#'Plot LBBNN objects
#'@description Plot either the global structure or local explanations.
#'@param x An instance of LBBNN_net.
#'@param type 'global' or 'local'.
#'@param data If local is chosen, one sample must be provided to obtain the explanation. Must be a torch_tensor of shape(1,p).
#'@param num_samples How many samples to use for model averaging over the weights in case of local explanations.
#'@param ... further arguments passed to or from other methods.
#' @export
plot.LBBNN_Net <- function(x,type = c('global','local'),data = NULL,num_samples = 100, ...) {
  d <- match.arg(type)
  if(d == 'global'){
    LBBNN_plot(x,...)
  }
  else{
    if(is.null(data))stop('data must contain a sample to explain')
    plot_local_explanations_gradient(x,input_data = data,num_samples = num_samples,device = x$device)
    
  }
  
  
}






