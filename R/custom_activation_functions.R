#' @export
Custom_activation <- nn_module(
  "Custom_activation",
  initialize = function() {
    self$act <- torch::nn_relu()
    
  },
  forward = function(x) {
    shapes <- dim(x)
    x1 <- x[,1]
    x2 <- x[,2]
    x3 <- x[,3]
    x4 <- x[,4:shapes[2]]
    x1 <- torch::torch_exp(x1)$unsqueeze(2)
    x2 <- torch::torch_sigmoid(x2)$unsqueeze(2)
    x3 <- (x3^2)$unsqueeze(2)
    x4 <- self$act(x4)
     
    x <- torch::torch_cat(c(x1,x2,x3,x4),dim = 2)
    return(x)
  }
)

