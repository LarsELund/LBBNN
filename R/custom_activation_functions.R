#' @title Generate a custom activation function.
#' @description The first 3 entries are customized
#' in order to see if we can learn that structure. The rest will be relu as usual 
#' @return Returns a `\code{torch::nn_module} that can be used in an \code{LBBNN_Net}  
#' @export
Custom_activation <- nn_module(
  "Custom_activation",
  initialize = function() {
    self$act <- torch::nn_leaky_relu(0.0)
    
  },
  forward = function(x) {
    shapes <- dim(x)
    x1 <- x[,1]
    x2 <- x[,2]
    x3 <- x[,3]
    x4 <- x[,4:shapes[2]]
    x1 <- torch::torch_exp(x1)$unsqueeze(2)
    x2 <- torch::torch_log(torch::torch_abs(x2))$unsqueeze(2)
    x3 <- torch::torch_abs(x3)^(1/3)
    x4 <- self$act(x4)
    
    x <- torch::torch_cat(c(x1,x2,x3,x4),dim = 2)
    return(x)
  }
)
