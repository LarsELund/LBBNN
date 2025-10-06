#' Class to generate a multi-layer perceptron, used in RNVP transforms. 
#' @param hidden_sizes A vector of ints. The first is the dimensionality of the vector,
#' to be transformed by RNVP. The subsequent are hidden dimensions in the MLP.
#' @param device The device to be used. Default is CPU.
#' @description As of now, each hidden layer (except the last) is followed
#' by a non-linear transformation. 
#' @examples
#'net <- MLP(c(50,100,200,400))
#'x <- torch::torch_rand(50)
#'out <- net(x)
#'print(dim(out))
#' @export
MLP <- torch::nn_module(
  "MLP",
  
  initialize = function(hidden_sizes,device = 'cpu') {
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
#' @param device The device to be used. Default is CPU.
#' @description Affine half flow aka Real Non-Volume Preserving (x = z * exp(s) + t),
#' where a randomly selected half z1 of the dimensions in z are transformed as an
#'affine function of the other half z2, i.e. scaled by s(z2) and shifted by t(z2).
#'From "Density estimation using Real NVP", Dinh et al. (May 2016)
#'https://arxiv.org/abs/1605.08803
#'This implementation uses the numerically stable updates introduced by IAF:
# 'https://arxiv.org/abs/1606.04934
#' @examples
#'z <- torch::torch_rand(200)
#'layer <- RNVP_layer(c(200,50,100))
#'out <- layer(z)
#'print(dim(out))
#'print(layer$log_det())
#' @export
RNVP_layer <- torch::nn_module(
  "RNVP_layer",
  
  initialize = function(hidden_sizes,device = 'cpu') {
    self$net <- MLP(hidden_sizes,device)
    self$t <- torch::nn_linear(hidden_sizes[length(hidden_sizes)],hidden_sizes[1])
    self$s <- torch::nn_linear(hidden_sizes[length(hidden_sizes)],hidden_sizes[1])
  },
  forward = function(z){
    self$m <- torch::torch_bernoulli(0.5 * torch::torch_ones_like(z))
    z1 <- (1-self$m) * z
    z2 <- self$m * z
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


