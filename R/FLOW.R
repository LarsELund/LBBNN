#' Class to generate a multi-layer perceptron, used in RNVP transforms. 
#' @param hidden_sizes A vector of ints. The first is the dimensionality of the vector,
#' to be transformed by RNVP. The subsequent are hidden dimensions in the MLP.
#' @description As of now, each hidden layer (except the last) is followed
#' by a non-linear transformation. 
#' @examples
#'net <- MLP(c(50,100,200,400))
#'x <- torch_rand(50)
#'out <- net(x)
#'print(dim(out))
#' @export
MLP <- torch::nn_module(
  "MLP",
  
  initialize = function(hidden_sizes) {
    self$lay <- torch::nn_module_list() 
    
    for(i in 1:(length(hidden_sizes)-1)){
      self$lay$append(torch::nn_linear(hidden_sizes[i],hidden_sizes[i+1])) #hidden layers
      if(i < length(hidden_sizes)-1){
        self$lay$append(torch::nn_leaky_relu()) #relu after each layer except the last
      }
      
    }
  },
  forward = function(x){
    for(l in self$lay$children){
      x <- l(x)
      
    }
    return(x)  
  }
)


#' Class to generate one RNVP transform layer. 
#' @param hidden_sizes A vector of ints. The first is the dimensionality of the vector,
#' to be transformed by RNVP. The subsequent are hidden dimensions in the MLP.
#' @description Affine half flow aka Real Non-Volume Preserving (x = z * exp(s) + t),
#' where a randomly selected half z1 of the dimensions in z are transformed as an
#'affine function of the other half z2, i.e. scaled by s(z2) and shifted by t(z2).
#'From "Density estimation using Real NVP", Dinh et al. (May 2016)
#'https://arxiv.org/abs/1605.08803
#'This implementation uses the numerically stable updates introduced by IAF:
# 'https://arxiv.org/abs/1606.04934
#' @examples
#'z <- torch_rand(200)
#'layer <- RNVP_layer(c(200,50,100))
#'out <- layer(z)
#'print(dim(out))
#'print(layer$log_det())
#' @export
RNVP_layer <- torch::nn_module(
  "RNVP_layer",
  
  initialize = function(hidden_sizes) {
    self$net <- MLP(hidden_sizes)
    self$t <- torch::nn_linear(hidden_sizes[length(hidden_sizes)],hidden_sizes[1])
    self$s <- torch::nn_linear(hidden_sizes[length(hidden_sizes)],hidden_sizes[1])
  },
  forward = function(z){
    self$m <- torch::torch_bernoulli(0.5 * torch::torch_ones_like(z))
    z1 <- (1-self$m) * z
    z2 <- self$ m * z
    out <- self$net(z2)
    shift <- self$t(out)
    scale <- self$s(out)
    self$gate <- torch::torch_sigmoid(scale)
    x = z1 * (self$gate + (1 - self$gate) * shift) + z2
    return(x)
  },
  log_det = function(){
    return(torch::torch_sum((1 -self$m) * torch::torch_log(self$gate + 1e-10)))
  }
)




#' Class to generate a FLOW 
#' @param input_dim The dimensionality of each layer. First item is the input vector size.
#' @param transform_type The type of transformation. Currently only RNVP is implemented.
#' @param num_transforms How many layers of transformations to include in the flow
#' @description Contains an nn_module where the initial vector gets transformed through 
#' all the layers in the module. Also computed the log determinant for the entire 
#' transformation, which is just the sum of the independent layers.
#' @examples
#'flow <- FLOW(c(200,100,100),transform_type = 'RNVP',num_transforms = 3)
#'flow$to(device = 'cpu')
#'x <- torch_rand(200,device = 'cpu')
#'output <- flow(x)
#'z_out <- output$z
#'print(dim(z_out))
#'log_det <- output$logdet
#'print(log_det)
#' @export
FLOW <- torch::nn_module(
  "FLOW",
  
  initialize = function(input_dim,transform_type,num_transforms) {
    self$layers <- torch::nn_module_list() 
    if(transform_type == 'RNVP'){
      for(l in 1:num_transforms){
        self$layers$append(RNVP_layer(input_dim))
      }
    }
    else(stop(cat('transform type',transform_type,'not implemented, try \'RNVP\' instead')))
  },
  forward = function(z){
    logdet <- 0
    
    for(l in self$layers$children){
      z <- l(z)
      logdet <- logdet + l$log_det()
    }
    l = list('z' = z,'logdet' = logdet)
    return(l)
  }
)



