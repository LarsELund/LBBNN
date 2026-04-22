#' @keywords internal
torch_available <- function() {
  requireNamespace("torch", quietly = TRUE) &&
    isTRUE(tryCatch(
      torch::torch_is_installed(),
      error = function(e) FALSE
    ))
}
