library(torch)

#' @title Class to generate a normalizing flow
#' @param input_dim numeric vector, the dimensionality of each layer.
#' The first item is the input vector size.
#' @param transform_type Transformation type. Currently RNVP is implemented.
#' @param num_transforms integer, how many layers of transformations to include.
#' @description Used in\code{LBBNN_Net} when the argument \code{flow = TRUE}.
#' Contains a \code{torch::nn_module} where the initial vector gets transformed
#' through all the layers in the module.
#' Also computes the log-determinant of the Jacobian for the entire
#' transformation, the sum of the log-determinants of the independent layers.
#' @return
#' A \code{torch::nn_module} object representing the normalizing flow.
#' The module provides:
#' \describe{
#'   \item{\code{forward(z)}}{
#'     Applies all flow transformation layers to the input tensor \code{z}.
#'     Returns a named list containing:
#'     \describe{
#'       \item{\code{z}}{
#'         A `torch_tensor` containing the transformed version of the input,
#'         with the same shape as `z`.
#'       }
#'       \item{\code{logdet}}{
#'         A scalar `torch_tensor` equal to the sum of the log-determinants of
#'         all transformation layers.
#'       }
#'     }
#'   }
#' }
#' @examples
#' \donttest{
#'flow <- normalizing_flow(c(2,5,5), transform_type='RNVP', num_transforms = 3)
#'flow$to(device = 'cpu')
#'x <- torch::torch_rand(2, device = 'cpu')
#'output <- flow(x)
#'z_out <- output$z
#'print(dim(z_out))
#'log_det <- output$logdet
#'print(log_det)
#' }
#' @export
normalizing_flow <- torch::nn_module(
  "normalizing_flow",
  initialize = function(input_dim, transform_type, num_transforms) {
    self$layers <- torch::nn_module_list()
    if (transform_type == "RNVP") {
      for (l in 1:num_transforms) {
        self$layers$append(rnvp_layer(input_dim))
      }
    }else {
      stop(paste("transform type", transform_type,
                 "not implemented, try 'RNVP' instead"))
    }
  },
  forward = function(z) {
    logdet <- 0
    for (l in self$layers$children) {
      z <- l(z)
      logdet <- logdet + l$log_det()
    }
    l = list("z" = z, "logdet" = logdet)
    return(l)
  }
)
