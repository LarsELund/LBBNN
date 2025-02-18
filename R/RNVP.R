#' Class to generate a multi-layer perceptron, used in RNVP transforms. 
#' @param hidden_sizes A vector of ints. The first is the dimensionality of the vector,
#' to be transformed by RNVP. The subsequent are hidden dimensions in the MLP.
#' @param device The device to be used. Default is CPU.
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


