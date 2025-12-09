library(Matrix)
require(graphics)

#' @title Function that checks how many times inputs are included, and from which layer. Used in summary function.
#' @description Useful when the number of inputs and/or hidden neurons are very
#' large, and direct visualization of the network is difficult. 
#' @param model An instance of \code{LBBNN_Net} where \code{input_skip = TRUE}.
#' @return A matrix of shape (p, L-1) where p is the number of input variables
#' and L the total number of layers (including input and output), with each element being 1 if the variable is included
#' or 0 if not included. 
#' @keywords internal
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


#' @title Summary of LBBNN fit
#' @description Summary method for objects of the \code{LBBNN_Net} class. 
#' Only applies to objects trained with \code{input_skip = TRUE}.
#' @param object An object of class \code{LBBNN_Net}.
#' @param ... further arguments passed to or from other methods.
#' @details
#' The returned table combines two types of information:
#' \itemize{
#'   \item Number of times each input variable is included in the active paths from each layer 
#'         (obtained from \code{get_input_inclusions()}).
#'   \item Average inclusion probabilities for each input variable from each layer, including a final 
#'         column showing the average across all layers.
#' }
#' @return A \code{data.frame} containing the above information. The function prints a formatted summary to the console. 
#'   The returned \code{data.frame} is invisible.
#' @examples
#' \donttest{ 
#'x<-torch::torch_randn(3,2) 
#'b <- torch::torch_rand(2)
#'y <- torch::torch_matmul(x,b)
#'train_data <- torch::tensor_dataset(x,y)
#'train_loader <- torch::dataloader(train_data,batch_size = 3,shuffle=FALSE)
#'problem<-'regression'
#'sizes <- c(2,1,1) 
#'inclusion_priors <-c(0.9,0.2) 
#'inclusion_inits <- matrix(rep(c(-10,10),2),nrow = 2,ncol = 2)
#'stds <- c(1.0,1.0)
#'model <- LBBNN_Net(problem,sizes,inclusion_priors,stds,inclusion_inits,flow = FALSE,
#'input_skip = TRUE)
#'train_LBBNN(epochs = 1,LBBNN = model, lr = 0.01,train_dl = train_loader)
#'summary(model)
#'}
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
  cat(paste('The model took',object$elapsed_time,'seconds to train, using',object$device)) #should return this as well in a list with summary_out? and update docs..
  invisible(summary_out)

}


#' @title Residuals from LBBNN fit
#' @description Residuals from an object of the \code{LBBNN_Net} class.
#' @param object An object of class \code{LBBNN_Net}.
#' @param type Currently only 'response' is implemented i.e. y_true - y_predicted.
#' @param ... further arguments passed to or from other methods.
#' @return A numeric vector of residuals (\code{y_true - y_predicted})
#' @examples
#' \donttest{ 
#'x<-torch::torch_randn(3,2) 
#'b <- torch::torch_rand(2)
#'y <- torch::torch_matmul(x,b)
#'train_data <- torch::tensor_dataset(x,y)
#'train_loader <- torch::dataloader(train_data,batch_size = 3,shuffle=FALSE)
#'problem<-'regression'
#'sizes <- c(2,1,1) 
#'inclusion_priors <-c(0.9,0.2) 
#'inclusion_inits <- matrix(rep(c(-10,10),2),nrow = 2,ncol = 2)
#'stds <- c(1.0,1.0)
#'model <- LBBNN_Net(problem,sizes,inclusion_priors,stds,inclusion_inits,flow = FALSE,
#'input_skip = TRUE)
#'train_LBBNN(epochs = 1,LBBNN = model, lr = 0.01,train_dl = train_loader)
#'residuals(model)
#'}
#' @export
residuals.LBBNN_Net <- function(object,type = c('response'), ...) {
  y_true <- object$y
  y_predicted <- object$r
  if(type == 'response'){
    return(y_true - y_predicted)
  }
  else(stop('only y - y_pred residuals are currently implemented'))
  
  
  
}


#' @title Get model coefficients (local explanations) of an \code{LBBNN_Net} object
#' @description Given an input sample x_1,... x_j (with j the number of variables), the local explanation is found by 
#' considering active paths. If relu activation functions are assumed, each path is a piecewise
#' linear function, so the contribution for x_j is just the sum of the weights associated with the paths connecting x_j to the output. 
#' The contributions are found by taking the gradient wrt x.   
#' @param object an object of class \code{LBBNN_Net}.
#' @param dataset Either a \code{torch::dataloader} object, or a \code{torch::torch_tensor} object. 
#' The former is assumed to be the same \code{torch::dataloader} used for training or testing.
#' The latter can be any user-defined data.
#' @param inds Optional integer vector of row indices in the dataset to compute explanations for.
#' @param output_neuron integer, which output neuron to explain (default = 1). 
#' @param num_data integer, if no indices are chosen, the first \code{num_data} of \code{dataset} are automatically used for explanations.
#' @param num_samples integer, how many samples to use for model averaging when sampling the weights in the active paths. 
#' @param ... further arguments passed to or from other methods.
#' @details
#' \itemize{
#'   \item If \code{num_data = 1}, confidence intervals are computed using model averaging over \code{num_samples} weight samples.
#'   \item If \code{num_data > 1}, confidence intervals are computed across the mean explanations for each sample.
#'   \item The output is a data frame with row names as input variables 
#'         (\code{x0}, \code{x1}, \code{x2}, ...) and columns giving mean and 95% confidence intervals for each variable.
#' }
#' @return A data frame with rows corresponding to input variables and the following columns:
#' \itemize{
#'   \item \code{lower}: lower bound of the 95% confidence interval
#'   \item \code{mean}: mean contribution of the variable
#'   \item \code{upper}: upper bound of the 95% confidence interval
#' }
#' 
#' @examples
#' \donttest{ 
#'x<-torch::torch_randn(3,2) 
#'b <- torch::torch_rand(2)
#'y <- torch::torch_matmul(x,b)
#'train_data <- torch::tensor_dataset(x,y)
#'train_loader <- torch::dataloader(train_data,batch_size = 3,shuffle=FALSE)
#'problem<-'regression'
#'sizes <- c(2,1,1) 
#'inclusion_priors <-c(0.9,0.2) 
#'inclusion_inits <- matrix(rep(c(-10,10),2),nrow = 2,ncol = 2)
#'stds <- c(1.0,1.0)
#'model <- LBBNN_Net(problem,sizes,inclusion_priors,stds,inclusion_inits,flow = FALSE,
#'input_skip = TRUE)
#'train_LBBNN(epochs = 1,LBBNN = model, lr = 0.01,train_dl = train_loader)
#'coef(model,dataset = x, num_data = 1)
#'}
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
    colnames(qs) <- c("lower","mean", "upper")
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
  colnames(qs) <- c("lower","mean", "upper")
  
  
  return(as.data.frame(qs))
  
}


#'@title Obtain predictions from the variational posterior of an \code{LBBNN model}
#'@description Draw from the (variational) posterior distribution of a trained \code{LBBNN_Net} object.
#'@param object A trained \code{LBBNN_Net} object
#'@param mpm logical, whether to use the median probability model.
#'@param newdata A \code{torch::dataloader} object containing the data with which to predict.
#'@param draws integer, the number of samples to draw from the posterior. 
#'@param device character, device for computation (default = \code{"cpu"}). 
#'@param link Optional link function to apply to the network output. Currently not implemented.
#'@param ... further arguments passed to or from other methods.
#'@return A \code{torch::torch_tensor}  of shape \code{(draws,N,C)} where \code{N} is the number of samples in \code{newdata}, and \code{C} the number of outputs.
#' @examples
#' \donttest{ 
#'x<-torch::torch_randn(3,2) 
#'b <- torch::torch_rand(2)
#'y <- torch::torch_matmul(x,b)
#'train_data <- torch::tensor_dataset(x,y)
#'train_loader <- torch::dataloader(train_data,batch_size = 3,shuffle=FALSE)
#'problem<-'regression'
#'sizes <- c(2,1,1) 
#'inclusion_priors <-c(0.9,0.2) 
#'inclusion_inits <- matrix(rep(c(-10,10),2),nrow = 2,ncol = 2)
#'stds <- c(1.0,1.0)
#'model <- LBBNN_Net(problem,sizes,inclusion_priors,stds,inclusion_inits,flow = FALSE,
#'input_skip = TRUE)
#'train_LBBNN(epochs = 1,LBBNN = model, lr = 0.01,train_dl = train_loader)
#'predict(model,mpm = FALSE,newdata = train_loader,draws = 1)
#'}
#' @export
predict.LBBNN_Net <- function(object,newdata,mpm = FALSE,draws = 10,device = 'cpu',link = NULL,...){#should newdata be a dataloader or a dataset?
  object$eval()
  object$raw_output = TRUE #skip final sigmoid/softmax
  if(! object$computed_paths){
    if(object$input_skip){object$compute_paths_input_skip()} #need this to get active paths to compute mpm
    else(object$compute_paths)
   
    
  }
  if(class(newdata)[[1]] != 'dataloader')stop('Currently only torch::dataloader objects are supported for newdata')
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



#' @title Print summary of an \code{LBBNN_Net} object
#' @description
#' Provides a summary of a trained \code{LBBNN_Net} object.
#' Includes the model type (input-skip or not), whether normalizing flows
#' are used, module and sub-module structure, number of trainable parameters, and prior
#' variance and inclusion probabilities for the weights.
#' @param x An object of class \code{LBBNN_Net}.
#' @param ... Further arguments passed to or from other methods.
#' @return Invisibly returns the input \code{x}.
#' @examples
#' \donttest{ 
#'x<-torch::torch_randn(3,2) 
#'b <- torch::torch_rand(2)
#'y <- torch::torch_matmul(x,b)
#'train_data <- torch::tensor_dataset(x,y)
#'train_loader <- torch::dataloader(train_data,batch_size = 3,shuffle=FALSE)
#'problem<-'regression'
#'sizes <- c(2,1,1) 
#'inclusion_priors <-c(0.9,0.2) 
#'inclusion_inits <- matrix(rep(c(-10,10),2),nrow = 2,ncol = 2)
#'stds <- c(1.0,1.0)
#'model <- LBBNN_Net(problem,sizes,inclusion_priors,stds,inclusion_inits,flow = FALSE,
#'input_skip = TRUE)
#'print(model)
#'}
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
  invisible(x)
}


#' @title Plot \code{LBBNN_Net} objects
#' @description
#' Given a trained \code{LBBNN_Net} model, this function produces either:
#' \itemize{
#'   \item \strong{Global plot}: a visualization of the network structure,
#'     showing only the active paths.
#'   \item \strong{Local explanation}: a plot of the local
#'     explanation for a single input sample, including error bars obtained
#'     from Monte Carlo sampling of the network weights.
#' }
#' @param x An instance of \code{LBBNN_Net}.
#' @param type Either \code{"global"} or \code{"local"}.
#' @param data If local is chosen, one sample must be provided to obtain the explanation. Must be a \code{torch::torch_tensor} of shape \code{(1,p)}.
#' @param num_samples integer, how many samples to use for model averaging over the weights in case of local explanations.
#' @param ... further arguments passed to or from other methods.
#' @return No return value. Called for its side effects of producing a plot.
#' @export
plot.LBBNN_Net <- function(x,type = c('global','local'),data = NULL,num_samples = 100, ...) {
  if(x$input_skip == FALSE)(stop('Plotting currently only implemented for input-skip'))
  if(x$computed_paths == FALSE){x$compute_paths_input_skip()}
  d <- match.arg(type)
  if(d == 'global'){
    LBBNN_plot(x,...)
  }
  else{
    if(is.null(data))stop('data must contain a sample to explain')
    plot_local_explanations_gradient(x,input_data = data,num_samples = num_samples,device = x$device,...)
    
  }
  
  
}






