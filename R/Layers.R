library(torch)


#' @title Generate prior inclusion probabilities for the weights of a LBBNN layer (linear or convolutional).
#' @description A function to generate prior probability values for 
#' each weight in an LBBNN layer. The same probability is applied to all weights within a layer.
#' @param x A number between 0 and 1.
#' @return a numeric to be added to either (out_shape,in_shape) in case of linear layers
#' or (out_channels,in_channels,kernel0,kernel1) in case of convolutional layers.
#' @keywords internal
alpha_prior <- function(x) {
  if (!is.numeric(x) )
    stop("invalid_class:", " x must be numeric")
  if (x <=0 | x>=1)
    stop("invalid_value:", " x must be between 0 and 1")
  
  
  if (length(x) != 1)
    stop("only one value of x allowed per layer")
  
  return(x)
  
}

#' @title Generate prior standard deviation for weights and biases of either linear or convolutional layers.
#' @description A function to generate prior standard deviations for 
#' each weight and bias in an LBBNN layer.
#' @param x A number greater than 0.
#' @return a numeric to be added to either (out_shape,in_shape) in case of linear layers
#' or (out_channels,in_channels,kernel0,kernel1) in case of convolutional layers
#' @keywords internal
std_prior <- function(x) {
  if (!is.numeric(x) )
    stop("invalid_class:", "x must be numeric")
  if (x<=0)
    stop("invalid_value:", "x must be > 0")
  
  
  if (length(x) != 1)
    stop("only one value allowed per layer")
  
  return(x)
  
}

#' @title Generate initializations for the inclusion parameters. 
#' @description Controls the initial density of the network, and thus an important tuning
#' parameter. The initialization parameters are sampled from 1/(1+exp(-U)),
#' with U ~ Uniform(lower,upper). 
#' @param lower numeric scalar.
#' @param upper numeric scalar, must be greater than 
#' @return A numeric vector of length 2: \code{c(lower,upper)}
#' @keywords internal
density_initialization <- function(lower,upper) {
  if (!is.numeric(lower) )
    stop("invalid_class:", " lower must be numeric")
  if (!is.numeric(upper) )
    stop("invalid_class:", " upper must be numeric")
  
  if (length(lower) != 1)
    stop("lower bound can only have one value")
  if (length(upper) != 1)
    stop("upper bound can only have one value")
  if (lower >= upper)
    stop("the lower bound must be smaller than the upper bound")
  
  return(c(lower,upper))
  
}





#' @title Class to generate an LBBNN feed forward layer
#' @description This module implements a fully connected LBBNN layer.
#' It supports:
#' \itemize{
#'   \item Prior inclusion probabilities for weights and biases in each layer.
#'   \item Standard deviation priors for weights and biases in each layer.
#'   \item Optional normalizing flows (RNVP) for a more flexible variational posterior.
#'   \item Forward pass using either the full model or the Median Probability Model (MPM).
#'   \item Computation of the KL-divergence.
#' }
#' @param in_features integer, number of input neurons.
#' @param out_features integer, number of output neurons.
#' @param prior_inclusion numeric scalar, prior inclusion probability for each weight and bias in the layer.
#' @param standard_prior numeric scalar, prior standard deviation for weights and biases in each layer.
#' @param density_init A numeric of size 2, used to initialize the inclusion parameters, one for each layer.
#' @param flow logical, whether to use normalizing flows
#' @param num_transforms integer, number of transformations for \code{flow}. Default is 2.
#' @param hidden_dims numeric vector, dimension of the hidden layer(s) in the neural networks of the RNVP transform.
#' @param device The device to be used. Default is CPU.
#' @param bias_inclusion_prob logical, determines whether the bias should be as associated with inclusion probabilities. 
#' @param conv_net logical, whether the layer is to be used in a convolutional net. 
#' @examples
#' \donttest{
#' l1 <- lbbnn_linear(in_features = 10,out_features = 5,prior_inclusion = 0.25,
#' standard_prior = 1,density_init = c(0,1),flow = FALSE)
#' x <- torch::torch_rand(20,10,requires_grad = FALSE)
#' output <- l1(x,MPM = FALSE) #the forward pass, output has shape (20,5)
#' print(l1$kl_div()$item()) #compute KL-divergence after the forward pass
#' }
#' @return A \code{torch::nn_module} object representing a fully connected LBBNN layer. 
#' The module has the following methods:
#'   - \code{forward(input, MPM = FALSE)}: Computes activation (using the LRT at training time) of a batch of inputs. 
#'   - \code{kl_div()}: Computes the KL-divergence.
#'   - \code{sample_z()}: Samples from the flow if \code{flow = TRUE}, in addition returns the log-determinant of the Jacobian
#'  of the transformation.

#' @export
lbbnn_linear <- torch::nn_module(
  "lbbnn_linear",
  initialize = function(
    in_features, 
    out_features,
    prior_inclusion,
    standard_prior,
    density_init,
    flow = FALSE,
    num_transforms = 2,
    hidden_dims = c(200,200),
    device = 'cpu',
    bias_inclusion_prob = FALSE,
    conv_net = FALSE) {
    self$in_features  <- in_features
    self$out_features <- out_features
    self$device <- device
    self$flow <- flow #true or false
    self$bias_inclusion_prob <- bias_inclusion_prob # true or false
    self$conv_net <- conv_net
    self$num_transforms <- num_transforms
    self$hidden_dims <- hidden_dims
    self$density_init = density_initialization(density_init[1],density_init[2])
   
    #weight variational parameters
    self$weight_mean <- torch::nn_parameter(torch::torch_empty(out_features, in_features,device = device))
    self$weight_rho <- torch::nn_parameter(torch::torch_empty(out_features, in_features,device = device))
    self$weight_sigma <- torch::torch_empty(out_features, in_features,device = device)
    
    #bias variational parameters 
    self$bias_mean <- torch::nn_parameter(torch::torch_empty(out_features,device = device))
    self$bias_rho <- torch::nn_parameter(torch::torch_empty(out_features,device = device))
    self$bias_sigma <- torch::torch_empty(out_features,device = device)
    
    #inclusion variational parameters
    self$lambda_l <- torch::nn_parameter(torch::torch_empty(out_features, in_features,device = device))
    self$alpha <- torch::torch_empty(out_features, in_features,device = device)
    self$alpha_active_path <- torch::torch_empty(out_features, in_features,device = device) #used to store alpha in active paths

    #define priors. For now, the user is only allowed to define the inclusion prior themselves
    self$alpha_prior <- torch::torch_zeros(out_features, in_features, device=device) + alpha_prior(prior_inclusion)


    if(self$bias_inclusion_prob){
    #inclusion variational parameters for bias
    self$bias_lambda_l <- torch::nn_parameter(torch::torch_empty(out_features, device = device))
    self$bias_alpha <- torch::torch_empty(out_features, device = device)

    #define priors for bias. For now, the user is only allowed to define the inclusion prior themselves
    self$bias_alpha_prior <- torch::torch_zeros(out_features, device=device) + alpha_prior(prior_inclusion)
    }
    
    #standard normal prior on the weights and biases
    self$weight_mean_prior <- torch::torch_zeros(out_features, in_features, device=device)
    self$weight_sigma_prior <- torch::torch_zeros(out_features, in_features, device=device) + std_prior(standard_prior)
    self$bias_mean_prior <- torch::torch_zeros(out_features, device=device)
    self$bias_sigma_prior <- torch::torch_zeros(out_features, device=device) + std_prior(standard_prior)
    
    #flow parameters
    if(self$flow){
      self$q0_mean <- torch::nn_parameter(torch::torch_empty(in_features,device = device))
      self$q0_logvar <- torch::nn_parameter(torch::torch_empty(in_features,device = device))
      self$c1 <- torch::nn_parameter(torch::torch_empty(in_features,device = device))
      self$b1 <- torch::nn_parameter(torch::torch_empty(in_features,device = device))
      self$b2 <- torch::nn_parameter(torch::torch_empty(in_features,device = device))
      
      
      self$RNVP_flow <- normalizing_flow(c(in_features,hidden_dims),'RNVP',self$num_transforms)
      self$RNVP_flow$to(device = device)
    }
    
    
    self$reset_parameters()
  },
  reset_parameters = function() {
    torch::nn_init_uniform_(self$weight_mean,-0.01,0.01)
    torch::nn_init_normal_(self$weight_rho,mean = -9, std = 1.0)
    torch::nn_init_uniform_(self$bias_mean,-0.2,0.2)
    torch::nn_init_normal_(self$bias_rho,mean = -9, std = 1.0)
    torch::nn_init_uniform_(self$lambda_l,self$density_init[1],self$density_init[2])
    if(self$flow){
      torch::nn_init_normal_(self$q0_mean,mean = 0,std = 1)
      torch::nn_init_normal_(self$q0_logvar,mean = -9, std = 1)
      torch::nn_init_normal_(self$c1,mean = 0,std = 1)
      torch::nn_init_normal_(self$b1,mean = 0,std = 1)
      torch::nn_init_normal_(self$b2,mean = 0,std = 1)
    }
    
    
  },
  sample_z = function(){
    q0_std <- torch::torch_sqrt(torch::torch_exp(self$q0_logvar))
    epsilon_z <- torch::torch_randn_like(q0_std)
    self$z <- self$q0_mean + q0_std * epsilon_z #the initial random normal variable to be send through the flow
    out <- self$RNVP_flow(self$z) #returns z_k and the log determinant of the transformation
    return(out)
  }
  ,
  forward = function(input,MPM=FALSE) {
    self$alpha <- torch::torch_sigmoid(self$lambda_l)
    if(self$bias_inclusion_prob){
    self$bias_alpha <- torch::torch_sigmoid(self$bias_lambda_l)
    } 
    self$weight_sigma <- torch::torch_log1p(torch_exp(self$weight_rho))
    self$bias_sigma <- torch::torch_log1p(torch_exp(self$bias_rho))
    self$z_k <- torch::torch_ones_like(self$weight_mean) #if not flow, all z = 1.
    if(self$flow){
      out <- self$sample_z()
      self$z_k <- out$z
    }
    if (! MPM) {#compute the mean and the variance of the activations using the LRT
      e_w <- self$weight_mean * self$alpha * self$z_k
      var_w <- self$alpha * (self$weight_sigma^2 + (1 - self$alpha) * self$weight_mean^2*self$z_k^2)
      if(self$bias_inclusion_prob){
        e_b_alpha <- self$bias_mean * self$bias_alpha
        var_b_alpha <- self$bias_alpha * (self$bias_sigma^2 + (1-self$bias_alpha)*self$bias_mean^2)
      } else {
        e_b_alpha <- self$bias_mean
        var_b_alpha <- self$bias_sigma^2
      }
      e_b <- torch::torch_matmul(input, torch::torch_t(e_w)) + e_b_alpha
      var_b <- torch::torch_matmul(input^2, torch::torch_t(var_w)) + var_b_alpha
      eps <- torch::torch_randn(size=(dim(var_b)), device=self$device)
      activations <- e_b + torch::torch_sqrt(var_b) * eps
      
    }else {#median probability model
      w <- torch::torch_normal(self$weight_mean * self$z_k, self$weight_sigma)
      bias <- torch::torch_normal(self$bias_mean, self$bias_sigma)
      weight <- w * self$alpha_active_path
      
      #need the below if we are running a convolutional net without active paths
      if(self$conv_net){
        gamma <-(self$alpha$clone()$detach()> 0.5) * 1.
        weight <- w * gamma
      }
      
      if(self$bias_inclusion_prob){
      bias <- bias * (self$bias_alpha>0.5)
      }
      activations <- torch::torch_matmul(input, torch::torch_t(weight)) + bias
    }
    
    

    return(activations)},
  kl_div = function() {
    z2 <- torch::torch_ones_like(self$weight_mean) #if not flow
    log_q <- 0 #these are 0 if no flow is used
    log_r <- 0
    if(self$flow){
      out <- self$sample_z()
      z2 <- out$z
      log_det_q <- out$logdet
      log_q0 <- torch::torch_sum(-0.5 * torch::torch_log(pi)$to(device = self$device) - 0.5 * self$q0_logvar
                - 0.5 * ((self$z - self$q0_mean) ** 2 / torch::torch_exp(self$q0_logvar)))
      log_q <- log_q0 -log_det_q
      
      ######get log r
      W_mean <- z2 * self$weight_mean * self$alpha
      W_var <- self$alpha * (self$weight_sigma^2 + (1 - self$alpha) * self$weight_mean^2*z2^2)
      act_mean <- torch::torch_matmul(self$c1,torch::torch_t(W_mean))
      act_var <- torch::torch_matmul(self$c1^2,torch::torch_t(W_var))
      act_inner <- act_mean + torch::torch_sqrt(act_var) * torch::torch_randn_like(act_var)
      act <- torch::nnf_hardtanh(act_inner)
      mean_r <- self$b1$outer(act)$mean(-1)  # eq (9) from MNF paper
      log_var_r <- self$b2$outer(act)$mean(-1)  # eq (10) from MNF paper
      out2 <- self$sample_z()
      z_b <- out2$z
      log_det_r <- out2$logdet
      log_rb <- torch::torch_sum(-0.5 * torch::torch_log(pi)$to(device = self$device) - 0.5 * log_var_r
                                - 0.5 * ((z_b - mean_r) ** 2 / torch::torch_exp(log_var_r)))
      
      log_r <- log_det_r + log_rb

      
    }
    
    if(self$bias_inclusion_prob){
      kl_bias <- torch::torch_sum(
        self$bias_alpha * (
          torch::torch_log(self$bias_sigma_prior / self$bias_sigma) 
          - 0.5 
          + torch::torch_log(self$bias_alpha/self$bias_alpha_prior + 1e-20) 
          + (self$bias_sigma^2 + (self$bias_mean - self$bias_mean_prior)^2) / (2*self$bias_sigma_prior^2)
        ) 
        + (1-self$bias_alpha) * torch::torch_log((1-self$bias_alpha) / (1 - self$bias_alpha_prior) + 1e-20)
    )
    }else {
      kl_bias <- torch::torch_sum(
        torch::torch_log(self$bias_sigma_prior / self$bias_sigma) 
        - 0.5 
        + (self$bias_sigma^2 + (self$bias_mean - self$bias_mean_prior)^2) / (2 * self$bias_sigma_prior^2))
    }
    
    kl_weight <- torch::torch_sum(self$alpha * (torch::torch_log(self$weight_sigma_prior / self$weight_sigma)
                                         - 0.5 + torch::torch_log(self$alpha / self$alpha_prior+1e-20)
                                         + (self$weight_sigma^2 + (self$weight_mean*z2 - self$weight_mean_prior)^2) / (
                                           2 * self$weight_sigma_prior^2))
                           + (1 - self$alpha) * torch::torch_log((1 - self$alpha) / (1 - self$alpha_prior) + 1e-20))
    
    
    return(kl_bias + kl_weight + log_q - log_r)}
  
  
)

#' @title Class to generate an LBBNN convolutional layer.
#' @description
#'  It supports:
#' \itemize{
#'   \item Prior inclusion probabilities for weights and biases in each layer.
#'   \item Standard deviation priors for weights and biases in each layer.
#'   \item Optional normalizing flows (RNVP) for a more flexible variational posterior.
#'   \item Forward pass using either the full model or the Median Probability Model (MPM).
#'   \item Computation of the KL-divergence.
#' }
#' @param in_channels integer, number of input channels.
#' @param out_channels integer, number of output channels.
#' @param kernel_size size of the convolving kernel.
#' @param prior_inclusion numeric scalar, prior inclusion probability for each weight and bias in the layer.
#' @param standard_prior numeric scalar, prior standard deviation for weights and biases in each layer.
#' @param density_init A numeric of size 2, used to initialize the inclusion parameters, one for each layer.
#' @param flow logical, whether to use normalizing flows
#' @param num_transforms integer, number of transformations for \code{flow}. Default is 2.
#' @param hidden_dims numeric vector, dimension of the hidden layer(s) in the neural networks of the RNVP transform.
#' @param device The device to be used. Default is CPU.
#' @examples
#' \donttest{
#'layer <- lbbnn_conv2d(in_channels = 1,out_channels = 32,kernel_size = c(3,3),
#'prior_inclusion = 0.2,standard_prior = 1,density_init = c(-10,10),device = 'cpu')
#'x <-torch::torch_randn(100,1,28,28)
#'out <-layer(x)
#'print(dim(out))
#'}
#' @importFrom torch torch_empty torch_long torch_zeros torch_zeros_like with_no_grad torch_rand torch_randn torch_exp
#' @return A \code{torch::nn_module} object representing a convolutional LBBNN layer. 
#' The module has the following methods:
#'   - \code{forward(input, MPM = FALSE)}: Computes activation (using the LRT at training time) of a batch of inputs. 
#'   - \code{kl_div()}: Computes the KL-divergence.
#'   - \code{sample_z()}: Samples from the flow if \code{flow = TRUE}, in addition returns the log-determinant of the Jacobian
#'  of the transformation.

#' @export
lbbnn_conv2d <- torch::nn_module(
  "lbbnn_conv2d",
  initialize = function(in_channels, out_channels,kernel_size,prior_inclusion,
                        standard_prior,density_init,
                        flow = FALSE,num_transforms = 2, hidden_dims = c(200,200),device = 'cpu') {

    if(length(kernel_size) == 1){
      kernel <- c(kernel_size,kernel_size)
    }
    else if(length(kernel_size) == 2){
      kernel <- kernel_size
    }
    else(stop('kernel_size must be of either length one or two.'))
    
    
    self$in_channels  <- in_channels
    self$out_channels <- out_channels
    self$flow <- flow #true or false
    self$num_transforms <- num_transforms
    self$hidden_dims <- hidden_dims
    
    self$device = device
    self$density_init = density_initialization(density_init[1],density_init[2])
    
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
    self$alpha_prior <- torch::torch_zeros_like(self$alpha,device = self$device) + alpha_prior(prior_inclusion)
    
    #standard normal prior on the weights and biases
    self$weight_mean_prior <- torch::torch_zeros_like(self$weight_mean, device=self$device)
    self$weight_sigma_prior <- torch::torch_zeros_like(self$weight_sigma,device = self$device) + std_prior(standard_prior)
    self$bias_mean_prior <- torch::torch_zeros_like(self$bias_mean,device = self$device)
    self$bias_sigma_prior <- torch::torch_zeros_like(self$bias_sigma,device = self$device) + std_prior(standard_prior)
    
    #flow parameters
    if(self$flow){
      self$q0_mean <- torch::nn_parameter(torch::torch_empty(out_channels,device = device))
      self$q0_logvar <- torch::nn_parameter(torch::torch_empty(out_channels,device = device))
      self$c1 <- torch::nn_parameter(torch::torch_empty(out_channels,device = device))
      self$b1 <- torch::nn_parameter(torch::torch_empty(out_channels,device = device))
      self$b2 <- torch::nn_parameter(torch::torch_empty(out_channels,device = device))
      
      
      self$RNVP_flow <- normalizing_flow(c(out_channels,hidden_dims),'RNVP',self$num_transforms)
      self$RNVP_flow$to(device = device)
    }
    
    
    
    self$reset_parameters()
  },
  reset_parameters = function() {
    torch::nn_init_uniform_(self$weight_mean,-0.2,0.2)
    torch::nn_init_normal_(self$weight_rho,mean = -9, std = 0.1)
    torch::nn_init_uniform_(self$bias_mean,-0.2,0.2)
    torch::nn_init_normal_(self$bias_rho,mean = -9, std = 0.1)
    torch::nn_init_uniform_(self$lambda_l,self$density_init[1],self$density_init[2])
    if(self$flow){
      torch::nn_init_normal_(self$q0_mean,mean = 0,std = 1)
      torch::nn_init_normal_(self$q0_logvar,mean = -9, std = 1)
      torch::nn_init_normal_(self$c1,mean = 0,std = 1)
      torch::nn_init_normal_(self$b1,mean = 0,std = 1)
      torch::nn_init_normal_(self$b2,mean = 0,std = 1)
    }
    
    
  },
  forward = function(input,MPM=FALSE) {
    self$alpha <- 1 / (1 + torch::torch_exp(-self$lambda_l))
    self$weight_sigma <- torch::torch_log1p(torch_exp(self$weight_rho))
    self$bias_sigma <- torch::torch_log1p(torch_exp(self$bias_rho))
    
    z_k <- torch::torch_ones(self$out_channels,device = self$device) #if mean field
    if(self$flow){
      out <- self$sample_z()
      z_k <- out$z
    }
    
    if (! MPM) {#compute the mean and the variance of the activations using the LRT
      e_w <- self$weight_mean * self$alpha * z_k$view(c(-1,1,1,1))
      var_w <- self$alpha * (self$weight_sigma^2 + (1 - self$alpha) * self$weight_mean^2 * z_k$view(c(-1,1,1,1))^2)
      psi <- torch::nnf_conv2d(input = input,weight = e_w,bias = self$bias_mu)
      delta <- torch::nnf_conv2d(input = input^2,weight = var_w,bias = self$bias_sigma^2)
      #delta[delta<= 0] = 0 +1e-20 
      eps <- torch::torch_randn(size=(dim(delta)), device=self$device)
      activations <- psi + torch::torch_sqrt(delta) * eps
    }else {#only sample from weights with inclusion prob > 0.5 aka the median probability model 
      gamma <-(self$alpha$clone()$detach()> 0.5) * 1.
      w <- torch::torch_normal(self$weight_mean*z_k$view(c(-1,1,1,1)), self$weight_sigma)
      bias <- torch::torch_normal(self$bias_mean, self$bias_sigma)
      weight <- w * gamma
      activations <- torch::nnf_conv2d(input = input,weight = weight,bias = bias)
    }
    
    
    
    return(activations)},
  
  sample_z = function(){
    q0_std <- torch::torch_sqrt(torch::torch_exp(self$q0_logvar))
    epsilon_z <- torch::torch_randn_like(q0_std)
    self$z <- self$q0_mean + q0_std * epsilon_z #the initial random normal variable to be send through the flow
    out <- self$RNVP_flow(self$z) #returns z_k and the log determinant of the transformation
    return(out)
  },
  
  kl_div = function() {
    z2 <- torch::torch_ones(self$out_channels,device = self$device) #if not flow
    log_q <- 0 #these are 0 if no flow is used
    log_r <- 0
    if(self$flow){
      out <- self$sample_z()
      z2 <- out$z
      log_det_q <- out$logdet
      log_q0 <- torch::torch_sum(-0.5 * torch::torch_log(pi)$to(device = self$device) - 0.5 * self$q0_logvar
                                - 0.5 * ((self$z - self$q0_mean) ** 2 / torch::torch_exp(self$q0_logvar)))
      log_q <- log_q0 -log_det_q
      
      ######get log r
      W_mean <- self$weight_mean * self$alpha * z2$view(c(-1,1,1,1))
      W_var <-  self$alpha * (self$weight_sigma^2 + (1 - self$alpha) * self$weight_mean^2 * z2$view(c(-1,1,1,1))^2)
      act_mu <- torch::torch_matmul(W_mean$view(-1, length(self$c1)),self$c1) # eq. (11)
      act_var <- torch::torch_matmul(W_mean$view(-1, length(self$c1)),self$c1^2)
      
      # For convolutional layers, linear mappings empirically work better than
      # tanh. Hence no need for act = tanh(act). Christos Louizos
      # confirmed this in https://github.com/AMLab-Amsterdam/MNF_VBNN/issues/4
      # even though the paper states the use of tanh in conv layers.
      act <- act_mu + torch::torch_sqrt(act_var) * torch::torch_randn_like(act_var)
      mean_r <- self$b1$outer(act)$mean(-1)  # eq (9) from MNF paper
      log_var_r <- self$b2$outer(act)$mean(-1)  # eq (10) from MNF paper
      out2 <- self$sample_z()
      z_b <- out2$z
      log_det_r <- out2$logdet
      log_rb <- torch::torch_sum(-0.5 * torch::torch_log(pi)$to(device = self$device) - 0.5 * log_var_r
                                 - 0.5 * ((z_b - mean_r) ** 2 / torch::torch_exp(log_var_r)))
      
      log_r <- log_det_r + log_rb
      
      
    }
    kl_bias <- torch::torch_sum(torch::torch_log(self$bias_sigma_prior / self$bias_sigma) - 0.5 + (self$bias_sigma^2
                 + (self$bias_mean - self$bias_mean_prior)^2) / (2 * self$bias_sigma_prior^2))
    
    kl_weight <- torch::torch_sum(self$alpha * (torch::torch_log(self$weight_sigma_prior / self$weight_sigma)
                                                - 0.5 + torch::torch_log(self$alpha / self$alpha_prior + 1e-20)
                                                + (self$weight_sigma^2 + (self$weight_mean*z2$view(c(-1,1,1,1)) - self$weight_mean_prior)^2) / (
                                                  2 * self$weight_sigma_prior^2))
                                  + (1 - self$alpha) * torch::torch_log((1 - self$alpha) / (1 - self$alpha_prior)+1e-20))
    


    return(kl_bias + kl_weight + log_q - log_r)}
  
  
)

