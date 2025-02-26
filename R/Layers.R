library(torch)


#' Generate prior inclusion probabilities for a LBBNN layer
#' @description A function to generate prior probability values for 
#' each weight in an LBBNN layer. Currently only the same probability for each
#' weight in a layer is allowed. 
#' @param x A number between 0 and 1.
#' @param out_shape number of output neurons.
#' @param in_shape number of input neurons.
#' @return a torch tensor with shape (out_shape,in_shape) with prior probabilities.
#' @examples
#' alpha_prior(0.25, 10,5)
#' @export
alpha_prior <- function(x,out_shape,in_shape,device = 'cpu') {
  alpha_out <- torch::torch_zeros(out_shape,in_shape,device=device)
  if (!is.numeric(x) )
    stop("invalid_class:", " alpha must be numeric")
  if (any(x <=0) | any(x>=1))
    stop("invalid_value:", " alpha must be between 0 and 1")
  
  
  if (length(x) != 1)
    stop("only one value of alpha allowed per layer")
  
  return(alpha_out + x)
  
}



#' Class to generate an LBBNN feed forward layer
#' @param in_features number of input neurons.
#' @param out_features number of output neurons.
#' @param prior_inclusion Prior inclusion probability for each weight in the layer.
#' @param device The device to be used. Default is CPU.
#' @description Includes function for forward pass, where one can
#' either use the full model, or the medium probability model (MPM).
#' Also contains method to initialize parameters and compute KL-divergence.
#' @examples
#' l1 <- LBBNN_Linear(in_features = 10,out_features = 5,prior_inclusion = 0.25,device = 'cpu')
#' x <- torch::torch_rand(20,10,requires_grad = FALSE)
#' output <- l1(x,MPM = FALSE) #the forward pass, output has shape (20,5)
#' print(l1$kl_div()$item()) #compute KL-divergence after the forward pass
#' @export
LBBNN_Linear <- torch::nn_module(
  "LBBNN_Linear",
  initialize = function(in_features, out_features,prior_inclusion,device = 'cpu') {
    self$in_features  <- in_features
    self$out_features <- out_features
    self$device = device
    #weight variational parameters
    self$weight_mean <- torch::nn_parameter(torch_empty(out_features, in_features,device = device))
    self$weight_rho <- torch::nn_parameter(torch_empty(out_features, in_features,device = device))
    self$weight_sigma <- torch::torch_empty(out_features, in_features,device = device)
    
    #bias variational parameters 
    self$bias_mean <- torch::nn_parameter(torch_empty(out_features,device = device))
    self$bias_rho <- torch::nn_parameter(torch_empty(out_features,device = device))
    self$bias_sigma <- torch::torch_empty(out_features,device = device)
    
    #inclusion variational parameters
    self$lambda_l <- torch::nn_parameter(torch_empty(out_features, in_features,device = device))
    self$alpha <- torch::torch_empty(out_features, in_features,device = device)
    
    #define priors. For now, the user is only allowed to define the inclusion prior themselves
    self$alpha_prior <- alpha_prior(prior_inclusion,out_features,in_features,self$device)
    
    #standard normal prior on the weights and biases
    self$weight_mean_prior <- torch::torch_zeros(out_features, in_features, device=device)
    self$weight_sigma_prior <- torch::torch_zeros(out_features, in_features, device=device) + 1 
    self$bias_mean_prior <- torch::torch_zeros(out_features, device=device)
    self$bias_sigma_prior <- torch::torch_zeros(out_features, device=device) + 1
    
    
    
    self$reset_parameters()
  },
  reset_parameters = function() {
    torch::nn_init_normal_(self$weight_mean,mean = 0,std = 1)
    torch::nn_init_normal_(self$weight_rho,mean = -9, std = 0.1)
    torch::nn_init_uniform_(self$bias_mean,-0.2,0.2)
    torch::nn_init_normal_(self$bias_rho,mean = -9, std = 0.1)
    torch::nn_init_uniform_(self$lambda_l,-10,10)
    
    
  },
  forward = function(input,MPM=FALSE) {
    self$alpha <- 1 / (1 + torch::torch_exp(-self$lambda_l))
    self$weight_sigma <- torch::torch_log1p(torch_exp(self$weight_rho))
    self$bias_sigma <- torch::torch_log1p(torch_exp(self$bias_rho))
    
    
    if (! MPM) {#compute the mean and the variance of the activations using the LRT
      e_w <- self$weight_mean * self$alpha
      var_w <- self$alpha * (self$weight_sigma^2 + (1 - self$alpha) * self$weight_mean^2)
      
      e_b <- torch::torch_matmul(input, torch::torch_t(e_w)) + self$bias_mean
      var_b <- torch::torch_matmul(input^2, torch::torch_t(var_w)) + self$bias_sigma^2
      eps <- torch::torch_randn(size=(dim(var_b)), device=self$device)
      activations <- e_b + torch::torch_sqrt(var_b) * eps
      
    }else {#only sample from weights with inclusion prob > 0.5 aka the median probability model 
      gamma <-(torch::torch_clone(self$alpha)> 0.5) * 1.
      w <- torch::torch_normal(self$weight_mean, self$weight_sigma)
      bias <- torch::torch_normal(self$bias_mean, self$bias_sigma)
      weight <- w * gamma
      activations <- torch::torch_matmul(input, torch_t(weight)) + bias
    }
    
    
    
    return(activations)},
  kl_div = function() {
    kl_bias <- torch::torch_sum(torch::torch_log(self$bias_sigma_prior / self$bias_sigma) - 0.5 + (self$bias_sigma^2
                    + (self$bias_mean - self$bias_mean_prior)^2) / (2 * self$bias_sigma_prior^2))
    
    kl_weight <- torch::torch_sum(self$alpha * (torch::torch_log(self$weight_sigma_prior / self$weight_sigma)
                                         - 0.5 + torch::torch_log(self$alpha / self$alpha_prior)
                                         + (self$weight_sigma^2 + (self$weight_mean - self$weight_mean_prior)^2) / (
                                           2 * self$weight_sigma_prior^2))
                           + (1 - self$alpha) * torch::torch_log((1 - self$alpha) / (1 - self$alpha_prior)))
    
    
    return(kl_bias + kl_weight)}
  
  
)

#' Class to generate an LBBNN convolutional layer
#' @param in_channels number of input channels.
#' @param out_channels number of output channels.
#' @param prior_inclusion Prior inclusion probability for each weight.
#' @param kernel_size Size of the convolving kernel.
#' @param device The device to be used. Default is CPU.
#' @description Includes function for forward pass, where one can
#' either use the full model, or the medium probability model (MPM).
#' Also contains method to initialize parameters and compute KL-divergence.
#' @examples
#'layer <-  LBBNN_Conv2d(in_channels = 1,out_channels = 32,kernel_size = c(3,3),prior_inclusion = 0.2,device = 'cpu')
#'x <- torch::torch_randn(100,1,28,28)
#'out <- layer(x)
#'print(dim(out))
#' @importFrom torch torch_empty torch_long torch_zeros torch_zeros_like with_no_grad torch_rand torch_randn torch_exp
#' @export
LBBNN_Conv2d <- torch::nn_module(
  "LBBNN_Conv2d",
  initialize = function(in_channels, out_channels,kernel_size,prior_inclusion,device = 'cpu') {
    
    if(length(kernel_size) == 1){
      kernel <- c(kernel_size,kernel_size)
    }
    else if(length(kernel_size) == 2){
      kernel <- kernel_size
    }
    else(stop('kernel_size must be of either length one or two.'))
    
    
    self$in_channels  <- in_channels
    self$out_channels <- out_channels
    self$device = device
    
    #weight variational parameters
    self$weight_mean <- torch::nn_parameter(torch_empty(out_channels, in_channels,kernel[1],kernel[2],device = device))
    self$weight_rho <-  torch::nn_parameter(torch_empty(out_channels, in_channels,kernel[1],kernel[2],device = device))
    self$weight_sigma <- torch::torch_empty(out_channels, in_channels,kernel[1],kernel[2],device = device)
    
    #bias variational parameters 
    self$bias_mean <- torch::nn_parameter(torch_empty(out_channels,device = device))
    self$bias_rho <- torch::nn_parameter(torch_empty(out_channels,device = device))
    self$bias_sigma <- torch::torch_empty(out_channels,device = device)
    
    #inclusion variational parameters
    self$lambda_l <- torch::nn_parameter(torch_empty(out_channels, in_channels,kernel[1],kernel[2],device = device))
    self$alpha <- torch::torch_empty(out_channels, in_channels,kernel[1],kernel[2],device = device)
    
    #define priors. For now, the user is only allowed to define the inclusion prior themselves
    self$alpha_prior <- torch::torch_zeros_like(self$alpha,device = self$device) + prior_inclusion
    
    #standard normal prior on the weights and biases
    self$weight_mean_prior <- torch::torch_zeros_like(self$weight_mean, device=self$device)
    self$weight_sigma_prior <- torch::torch_zeros_like(self$weight_sigma,device = self$device) + 1 
    self$bias_mean_prior <- torch::torch_zeros_like(self$bias_mean,device = self$device)
    self$bias_sigma_prior <- torch::torch_zeros_like(self$bias_sigma,device = self$device) + 1
    
    
    
    self$reset_parameters()
  },
  reset_parameters = function() {
    torch::nn_init_uniform_(self$weight_mean,-0.2,0.2)
    torch::nn_init_normal_(self$weight_rho,mean = -9, std = 0.1)
    torch::nn_init_uniform_(self$bias_mean,-0.2,0.2)
    torch::nn_init_normal_(self$bias_rho,mean = -9, std = 0.1)
    torch::nn_init_uniform_(self$lambda_l,-10,10)
    
    
  },
  forward = function(input,MPM=FALSE) {
    self$alpha <- 1 / (1 + torch::torch_exp(-self$lambda_l))
    self$weight_sigma <- torch::torch_log1p(torch_exp(self$weight_rho))
    self$bias_sigma <- torch::torch_log1p(torch_exp(self$bias_rho))
    
    
    if (! MPM) {#compute the mean and the variance of the activations using the LRT
      e_w <- self$weight_mean * self$alpha
      var_w <- self$alpha * (self$weight_sigma^2 + (1 - self$alpha) * self$weight_mean^2)
      psi <- torch::nnf_conv2d(input = input,weight = e_w,bias = self$bias_mu)
      delta <- torch::nnf_conv2d(input = input^2,weight = var_w,bias = self$bias_sigma^2)
      #delta[delta<= 0] = 0 +1e-20 
      eps <- torch::torch_randn(size=(dim(delta)), device=self$device)
      activations <- psi + torch::torch_sqrt(delta) * eps
    }else {#only sample from weights with inclusion prob > 0.5 aka the mediaan probability model 
      gamma <-(torch::torch_clone(self$alpha)> 0.5) * 1.
      w <- torch::torch_normal(self$weight_mean, self$weight_sigma)
      bias <- torch::torch_normal(self$bias_mean, self$bias_sigma)
      weight <- w * gamma
      activations <- torch::nnf_conv2d(input = input,weight = weight,bias = bias)
    }
    
    
    
    return(activations)},
  kl_div = function() {
    kl_bias <- torch::torch_sum(torch::torch_log(self$bias_sigma_prior / self$bias_sigma) - 0.5 + (self$bias_sigma^2
                 + (self$bias_mean - self$bias_mean_prior)^2) / (2 * self$bias_sigma_prior^2))
    
    kl_weight <- torch::torch_sum(self$alpha * (torch::torch_log(self$weight_sigma_prior / self$weight_sigma)
                                                - 0.5 + torch::torch_log(self$alpha / self$alpha_prior)
                                                + (self$weight_sigma^2 + (self$weight_mean - self$weight_mean_prior)^2) / (
                                                  2 * self$weight_sigma_prior^2))
                                  + (1 - self$alpha) * torch::torch_log((1 - self$alpha) / (1 - self$alpha_prior)))
    


    return(kl_bias + kl_weight)}
  
  
)





