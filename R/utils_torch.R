#' Check if torch/libtorch is available for examples
#' @export
torch_available <- function() {
  requireNamespace("torch", quietly = TRUE) &&
    torch::torch_is_installed() #without this, runtime operations will fail (missing libtorch)
}

#' Resolve a device string to a valid torch device.
#'
#' @description Maps the device aliases used throughout the package to the
#' device names understood by \code{torch}. \code{'gpu'} is treated as an alias
#' for \code{'cuda'}. If an accelerator is requested but unavailable, an error
#' is raised.
#'
#' @param device A device specification. Accepts \code{'cpu'}, \code{'gpu'},
#' \code{'cuda'} and \code{'mps'} (case-insensitive). A \code{torch_device}
#' object is returned unchanged.
#' @return A valid torch device string (\code{'cpu'}, \code{'cuda'} or
#' \code{'mps'}).
#' @export
resolve_device <- function(device = "cpu") {
  if (inherits(device, "torch_device")) {
    return(device)
  }
  if (!is.character(device) || length(device) != 1L) {
    stop("`device` must be a single string such as 'cpu', 'gpu' or 'mps'.")
  }

  dev <- tolower(device)
  if (dev %in% c("gpu", "cuda")) {
    if (!torch::cuda_is_available()) {
      stop(
        "device = '", device, "' was requested, but no CUDA device is ",
        "available (torch::cuda_is_available() is FALSE). Install a ",
        "CUDA-enabled build of torch, or use device = 'cpu'.",
        call. = FALSE
      )
    }
    return("cuda")
  }
  if (dev == "mps") {
    if (!torch::backends_mps_is_available()) {
      stop(
        "device = 'mps' was requested, but the MPS backend is not available. ",
        "Use device = 'cpu'.",
        call. = FALSE
      )
    }
    return("mps")
  }
  if (dev == "cpu") {
    return("cpu")
  }
  stop("Unrecognised device '", device,
       "'. Use one of 'cpu', 'gpu'/'cuda' or 'mps'.", call. = FALSE)
}