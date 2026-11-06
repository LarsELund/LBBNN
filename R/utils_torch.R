#' Check if torch/libtorch is available for examples
#' @export
torch_available <- function() {
  requireNamespace("torch", quietly = TRUE) &&
    torch::torch_is_installed()
}

#' @title Detect optimal compute device for Symbolic LBBNN
#' @description
#' On macOS local runs, defaults to \code{"cpu"} for small tabular symbolic
#' networks (as LibTorch MPS kernel launch overhead on tiny matrices exceeds CPU
#' speed), with an explicit toggle \code{prefer_mps = TRUE} or
#' \code{benchmark = TRUE} to benchmark \code{"mps"} vs \code{"cpu"}.
#' On Linux servers, automatically selects \code{"cuda"} when available.
#' @param prefer_mps logical, whether to use \code{"mps"} on macOS if available.
#' @param benchmark logical, whether to run a micro-benchmark comparing
#'   \code{"mps"} vs \code{"cpu"} on macOS before selecting the device.
#' @return character scalar: \code{"cpu"}, \code{"mps"}, or \code{"cuda"}.
#' @export
get_compute_device <- function(prefer_mps = FALSE, benchmark = FALSE) {
  if (!torch_available()) {
    return("cpu")
  }
  sysname <- Sys.info()[["sysname"]]
  if (identical(sysname, "Darwin")) {
    mps_ok <- tryCatch(
      torch::backends_mps_is_available(),
      error = function(e) FALSE
    )
    if (!mps_ok) {
      return("cpu")
    }
    if (isTRUE(benchmark)) {
      bench_dev <- tryCatch({
        run_one <- function(dev) {
          x <- torch::torch_randn(c(256, 10), device = dev, requires_grad = TRUE)
          w <- torch::torch_randn(c(10, 16), device = dev, requires_grad = TRUE)
          t0 <- proc.time()[[3]]
          for (i in 1:25) {
            z <- torch::torch_matmul(x, w)
            y <- torch::torch_exp(torch::torch_clamp(z, -12, 12)) +
              torch::torch_log(torch::torch_abs(z) + 1e-6)
            loss <- torch::torch_mean(y^2)
            loss$backward()
          }
          proc.time()[[3]] - t0
        }
        t_cpu <- run_one("cpu")
        t_mps <- run_one("mps")
        message(sprintf(
          "Device benchmark (25 steps): CPU = %.4fs | MPS = %.4fs",
          t_cpu, t_mps
        ))
        if (t_mps < t_cpu) "mps" else "cpu"
      }, error = function(e) "cpu")
      return(bench_dev)
    }
    if (isTRUE(prefer_mps)) {
      # Verify custom symbolic ops backward pass on MPS
      mps_verified <- tryCatch({
        z <- torch::torch_tensor(c(-10, 0, 10), device = "mps", requires_grad = TRUE)
        y <- torch::torch_log(torch::torch_abs(z) + 1e-6) +
          torch::torch_exp(torch::torch_clamp(z, -12, 12))
        y$sum()$backward()
        all(as.logical(torch::torch_isfinite(z$grad)$cpu()))
      }, error = function(e) FALSE)
      if (mps_verified) return("mps")
    }
    return("cpu")
  }

  # Linux / non-macOS server
  cuda_ok <- tryCatch(
    torch::cuda_is_available(),
    error = function(e) FALSE
  )
  if (isTRUE(cuda_ok)) {
    return("cuda")
  }
  return("cpu")
}
