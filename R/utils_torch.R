#' Check if torch/libtorch is available for examples 
#' @export
torch_available <- function() {
  requireNamespace("torch", quietly = TRUE) &&
    torch::torch_is_installed() #without this, runtime operations will fail (missing libtorch)
}