#' @title Extract Record-Level Log-Likelihood Contributions
#' @description Computes an \eqn{S \times n} matrix of pointwise log-likelihood
#' evaluations \eqn{\log f(y_i \mid \theta^{(s)})} across posterior parameter draws \eqn{\theta^{(s)}}
#' for each observation \eqn{i = 1, \dots, n}.
#'
#' @param object A trained \code{lbbnn_net} object.
#' @param newdata Either a \code{torch::dataloader}, \code{torch::dataset}, \code{data.frame}, or list with \code{x} and \code{y}.
#' @param draws Integer, number of posterior parameter draws \eqn{S} (default 100).
#' @param mpm Logical, whether to evaluate under the median probability model / thresholded sparse subnetwork (default FALSE).
#' @param sigma Numeric, observation standard deviation for Gaussian regression log-likelihood (default 1.0).
#' @param target_cols Integer, number of target columns if \code{newdata} is a \code{data.frame} (default 1).
#' @param device Character, device to compute on ("cpu", "cuda", "mps").
#' @return A numeric matrix of dimension \code{draws x n} containing pointwise log-likelihoods.
#' @export
loglik_contributions <- function(object, newdata, draws = 100, mpm = FALSE,
                                 sigma = 1.0, target_cols = 1, device = "cpu") {
  device <- resolve_device(device)
  trained <- object$training
  raw_out <- object$raw_output
  on.exit({
    object$raw_output <- raw_out
    if (trained) object$train() else object$eval()
  }, add = TRUE)

  object$eval()
  object$raw_output <- FALSE

  # Normalize newdata to a dataloader with shuffle = FALSE
  if (inherits(newdata, "data.frame")) {
    p <- ncol(newdata) - target_cols
    if (p < 1) stop("ncol(newdata) must be strictly greater than target_cols")
    x_mat <- as.matrix(newdata[, 1:p, drop = FALSE])
    y_mat <- if (target_cols == 1) {
      as.numeric(newdata[, ncol(newdata)])
    } else {
      as.matrix(newdata[, (p + 1):ncol(newdata), drop = FALSE])
    }
    ds <- torch::tensor_dataset(
      torch::torch_tensor(x_mat, dtype = torch::torch_float32()),
      torch::torch_tensor(y_mat, dtype = torch::torch_float32())
    )
    dl <- torch::dataloader(ds, batch_size = nrow(newdata), shuffle = FALSE)
  } else if (inherits(newdata, "dataset") || inherits(newdata, "tensor_dataset")) {
    dl <- torch::dataloader(newdata, batch_size = length(newdata), shuffle = FALSE)
  } else if (is.list(newdata) && !inherits(newdata, "dataloader") && !is.null(newdata$x) && !is.null(newdata$y)) {
    x_mat <- as.matrix(newdata$x)
    y_mat <- as.matrix(newdata$y)
    ds <- torch::tensor_dataset(
      torch::torch_tensor(x_mat, dtype = torch::torch_float32()),
      torch::torch_tensor(y_mat, dtype = torch::torch_float32())
    )
    dl <- torch::dataloader(ds, batch_size = nrow(x_mat), shuffle = FALSE)
  } else if (inherits(newdata, "dataloader")) {
    dl <- newdata
  } else {
    stop("newdata must be a torch::dataloader, torch::dataset, data.frame, or list(x=..., y=...)")
  }

  L_list <- vector("list", draws)

  torch::with_no_grad({
    for (s in seq_len(draws)) {
      ll_draw <- numeric(0)
      coro::loop(for (b in dl) {
        data <- b[[1]]$to(device = device)
        target <- b[[2]]$to(device = device)
        output <- object(data, MPM = mpm)

        if (object$problem_type == "binary classification") {
          target_flt <- target$squeeze(-1)$to(dtype = torch::torch_float32())
          out_flt <- output$squeeze(-1)
          ll_vec <- -torch::nnf_binary_cross_entropy(out_flt, target_flt, reduction = "none")
        } else if (object$problem_type %in% c("multiclass classification", "MNIST")) {
          target_lng <- target$squeeze(-1)$to(dtype = torch::torch_long())
          ll_vec <- -torch::nnf_nll_loss(output, target_lng, reduction = "none")
        } else if (object$problem_type == "regression") {
          sigma_to_use <- if (!is.null(sigma)) sigma else if (!is.null(object$sigma)) object$sigma else 1.0
          target_flt <- target$to(dtype = torch::torch_float32())
          out_flt <- output$view_as(target_flt)
          sq_err <- (target_flt - out_flt)^2
          if (sq_err$dim() > 1 && sq_err$shape[2] > 1) {
            sq_err <- sq_err$sum(dim = 2)
          } else {
            sq_err <- sq_err$squeeze(-1)
          }
          ll_vec <- -0.5 * log(2 * pi * sigma_to_use^2) - (sq_err / (2 * sigma_to_use^2))
        } else if (object$problem_type == "custom") {
          nll_fn <- if (!is.null(object$nll)) object$nll else object$loss_fn
          nll_eval <- nll_fn(output, target, reduction = "none")
          if (nll_eval$dim() > 1 && nll_eval$shape[2] == 1) {
            nll_eval <- nll_eval$squeeze(2)
          }
          ll_vec <- -nll_eval
        } else {
          stop(paste("Unsupported problem_type:", object$problem_type))
        }

        ll_draw <- c(ll_draw, as.numeric(ll_vec$detach()$cpu()))
      })
      L_list[[s]] <- ll_draw
    }
  })

  L_mat <- do.call(rbind, L_list)
  rownames(L_mat) <- paste0("draw_", seq_len(draws))
  colnames(L_mat) <- paste0("obs_", seq_len(ncol(L_mat)))
  return(L_mat)
}


#' @title Two-Pass Differential Privacy Pseudo-Posterior Synthesizer
#' @description Implements the two-pass risk-weighted pseudo-posterior mechanism
#' from Savitsky, Williams, and Hu (2024). In Pass 1, fits an unweighted LBBNN model
#' to evaluate record-level disclosure risk / sensitivity. Constructs record-specific
#' privacy weights \eqn{\alpha_i \in [0, 1]}, downweighting sensitive records (or removing
#' them if exceeding threshold \code{M_thresh}). In Pass 2, trains a fresh model under
#' the \eqn{\alpha}-weighted pseudo-posterior objective and calculates the local Lipschitz
#' privacy guarantee \eqn{\epsilon_x}.
#'
#' @param model_spec An \code{lbbnn_net} instance, a list of parameters to pass to \code{lbbnn_net()},
#' or a zero-argument function that returns a fresh \code{lbbnn_net}.
#' @param X Matrix or data.frame of predictors.
#' @param y Vector or matrix of targets.
#' @param epochs Integer, number of training epochs per pass (default 100).
#' @param lr Numeric, learning rate for optimizer (default 0.01).
#' @param batch_size Integer, mini-batch size (default 64).
#' @param S Integer, number of posterior draws for sensitivity extraction (default 100).
#' @param c_scale Numeric, scaling factor for risk weights (default 1.0).
#' @param g_shift Numeric, shift parameter for risk weights (default 0.0).
#' @param M_thresh Optional numeric threshold \eqn{M}. If provided, any record where
#' \eqn{\alpha_i f_i > M} has its weight forced to 0.
#' @param sigma Optional numeric standard deviation for Gaussian regression log-likelihood.
#' If NULL (default), automatically estimated after Pass 1 from the posterior mean residuals.
#' @param device Character, device string ("cpu", "cuda", "mps").
#' @param verbose Logical, whether to print training progress (default FALSE).
#' @param ... Additional arguments passed to \code{train_lbbnn()}.
#' @return A list containing:
#'   \describe{
#'     \item{model}{The final private model from Pass 2 trained with weights \eqn{\alpha}.}
#'     \item{model_unweighted}{The baseline unweighted model from Pass 1.}
#'     \item{alpha}{Numeric vector of length \eqn{n} containing risk weights \eqn{\alpha_i \in [0, 1]}.}
#'     \item{f_vec}{Numeric vector of length \eqn{n} containing record-level maximum sensitivities.}
#'     \item{delta_alpha}{Local Lipschitz bound \eqn{\Delta_{\alpha, x}}.}
#'     \item{epsilon_local}{Local DP guarantee \eqn{\epsilon_x = 2\Delta_{\alpha, x}}.}
#'     \item{sigma_hat}{Calibrated observation residual standard deviation.}
#'     \item{L_pass1}{Log-likelihood contributions matrix from Pass 1.}
#'     \item{L_pass2}{Log-likelihood contributions matrix from Pass 2.}
#'   }
#' @export
fit_dp_lbbnn <- function(model_spec, X, y, epochs = 100, lr = 0.01,
                         batch_size = 64, S = 100, c_scale = 1.0,
                         g_shift = 0.0, M_thresh = NULL, sigma = NULL,
                         device = "cpu", verbose = FALSE, ...) {
  device <- resolve_device(device)
  X_mat <- as.matrix(X)
  n <- nrow(X_mat)
  y_vec <- if (is.matrix(y)) y else as.numeric(y)

  # Factory to construct fresh model instances
  create_model <- function() {
    if (is.function(model_spec)) {
      return(model_spec())
    } else if (is.list(model_spec)) {
      return(do.call(lbbnn_net, model_spec))
    } else if (inherits(model_spec, "lbbnn_net")) {
      return(lbbnn_net(
        problem_type = model_spec$problem_type,
        sizes = model_spec$sizes,
        prior = model_spec$prior_inclusion,
        std = model_spec$prior_std,
        inclusion_inits = "balanced",
        input_skip = model_spec$input_skip,
        flow = model_spec$flow,
        num_transforms = model_spec$num_transforms,
        dims = model_spec$dims,
        device = device,
        raw_output = model_spec$raw_output,
        custom_act = model_spec$act,
        link = model_spec$link,
        nll = model_spec$nll,
        bias_inclusion_prob = model_spec$bias_inclusion_prob,
        weight_init = model_spec$weight_init
      ))
    } else {
      stop("model_spec must be a function(), an argument list, or an lbbnn_net instance")
    }
  }

  # Build unweighted and evaluation dataloaders
  t_x <- torch::torch_tensor(X_mat, dtype = torch::torch_float32())
  t_y <- torch::torch_tensor(y_vec, dtype = torch::torch_float32())
  t_w_ones <- torch::torch_ones(n, dtype = torch::torch_float32())

  ds_pass1 <- torch::tensor_dataset(t_x, t_y, t_w_ones)
  dl_pass1 <- torch::dataloader(ds_pass1, batch_size = min(batch_size, n), shuffle = TRUE)

  eval_ds <- torch::tensor_dataset(t_x, t_y)
  eval_dl <- torch::dataloader(eval_ds, batch_size = min(batch_size, n), shuffle = FALSE)

  # 1. Pass 1: Fit unweighted model
  if (verbose) message("Starting Pass 1 (Unweighted LBBNN)...")
  model_pass1 <- create_model()
  train_lbbnn(epochs = epochs, LBBNN = model_pass1, lr = lr,
              train_dl = dl_pass1, device = device, verbose = verbose, ...)

  # Calibrate sigma for regression if NULL
  if (is.null(sigma)) {
    if (model_pass1$problem_type == "regression") {
      preds_p1 <- predict(model_pass1, newdata = eval_dl, draws = S, mpm = FALSE, device = device)
      y_hat_p1 <- as.numeric(preds_p1$mean(dim = 1)$cpu())
      sigma <- sd(as.numeric(y_vec) - y_hat_p1)
      if (is.na(sigma) || sigma <= 0) sigma <- 1.0
      if (verbose) message(sprintf("Calibrated residual scale sigma_hat: %.4f", sigma))
    } else {
      sigma <- 1.0
    }
  }

  # 2. Record Sensitivity Matrix
  if (verbose) message("Extracting record-level sensitivities from Pass 1...")
  L_mat <- loglik_contributions(model_pass1, eval_dl, draws = S, mpm = FALSE,
                                sigma = sigma, device = device)
  f_vec <- apply(abs(L_mat), 2, max)

  # 3. Min-Max Normalization & Risk Weights
  f_min <- min(f_vec)
  f_max <- max(f_vec)
  f_tilde <- if (f_max > f_min) (f_vec - f_min) / (f_max - f_min) else rep(0, n)
  alpha <- pmin(1, pmax(0, c_scale * (1 - f_tilde) + g_shift))

  # Monotone surgical weight construction:
  # 1. Cap alpha_i so that alpha_i * f_i never exceeds M_thresh for any active record
  # 2. Surgically truncate (alpha_i = 0) extreme tail records where f_i exceeds the tail cutoff
  if (!is.null(M_thresh) && is.finite(M_thresh)) {
    alpha <- pmin(alpha, M_thresh / pmax(f_vec, 1e-8))
    alpha[f_vec > 3.0 * M_thresh] <- 0.0
  }

  # 4. Pass 2: Fit Alpha-Weighted Pseudo-Posterior Model
  if (verbose) message("Starting Pass 2 (Alpha-Weighted LBBNN)...")
  t_w_alpha <- torch::torch_tensor(as.numeric(alpha), dtype = torch::torch_float32())
  ds_pass2 <- torch::tensor_dataset(t_x, t_y, t_w_alpha)
  dl_pass2 <- torch::dataloader(ds_pass2, batch_size = min(batch_size, n), shuffle = TRUE)

  model_spec_p2 <- model_spec
  if (identical(model_spec_p2$problem_type, "regression") && !is.null(sigma)) {
    model_spec_p2$sigma <- sigma
  }
  model_pass2 <- do.call(lbbnn_net, model_spec_p2)
  train_lbbnn(epochs = epochs, LBBNN = model_pass2, lr = lr,
              train_dl = dl_pass2, device = device, verbose = verbose, ...)

  # 5. Local Lipschitz Bound & DP Guarantee
  if (verbose) message("Evaluating local Lipschitz bound and privacy guarantee...")
  L_pass2 <- loglik_contributions(model_pass2, eval_dl, draws = S, mpm = FALSE,
                                  sigma = sigma, device = device)
  L_alpha <- sweep(abs(L_pass2), 2, alpha, "*")
  delta_alpha_x <- if (!is.null(M_thresh)) min(max(L_alpha), M_thresh) else max(L_alpha)
  epsilon_x <- 2 * delta_alpha_x

  return(list(
    model = model_pass2,
    model_unweighted = model_pass1,
    alpha = alpha,
    f_vec = f_vec,
    delta_alpha = delta_alpha_x,
    epsilon_local = epsilon_x,
    sigma_hat = sigma,
    L_pass1 = L_mat,
    L_pass2 = L_pass2
  ))
}


#' @title Generate Synthetic Microdata from DP-Trained LBBNN
#' @description Draws synthetic records from the posterior predictive distribution
#' of an LBBNN model fitted with \code{fit_dp_lbbnn()}, incorporating calibrated observation noise.
#' Returns an \code{n x draws} matrix where each column represents one complete synthetic dataset replicate.
#'
#' @param dp_fit Output list returned by \code{fit_dp_lbbnn()}.
#' @param newdata Matrix, data.frame, or tensor of predictor features.
#' @param draws Integer, number of posterior draws / synthetic replicates (default 20).
#' @param mpm Logical, whether to use Median Probability Model (default FALSE).
#' @param log_scale Logical, if TRUE exponentiates predictions \code{exp(y_synth)} (default FALSE).
#' @param device Character, device string ("cpu", "cuda", "mps").
#' @return A numeric matrix of dimensions \code{nrow(newdata) x draws} of synthetic observations.
#' @export
sample_synthetic_lbbnn <- function(dp_fit, newdata, draws = 20, mpm = FALSE,
                                   log_scale = FALSE, device = "cpu") {
  mu_draws <- predict(dp_fit$model, newdata = newdata, draws = draws, mpm = mpm, device = device)
  # mu_draws has shape (draws, n, 1) -> transpose to (n, draws)
  mu_mat <- t(as.matrix(mu_draws[, , 1]$cpu()))
  n_obs <- nrow(mu_mat)
  sigma_noise <- if (!is.null(dp_fit$sigma_hat)) dp_fit$sigma_hat else 1.0
  synth_mat <- matrix(
    rnorm(n_obs * draws, mean = as.numeric(mu_mat), sd = sigma_noise),
    nrow = n_obs, ncol = draws
  )
  if (log_scale) synth_mat <- exp(synth_mat)
  return(synth_mat)
}
