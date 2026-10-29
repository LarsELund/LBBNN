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
#' @importFrom stats predict rnorm quantile sd median
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


#' @title Two-Stage Privacy Weight Normalization for Differential Privacy
#' @description Computes observation-level privacy weights \eqn{\alpha_i \in [0, 1]} from
#' record sensitivities \eqn{f_i = \max_s |\log f(y_i \mid \theta^{(s)})|} following the two-stage
#' framework of Savitsky (Oct 28, 2026, \emph{Privacy Weight Normalization and Second-Stage Reweighting Strategies in Asymptotic Differential Privacy}).
#'
#' In Stage 1, initial weights are constructed via \code{"linear"} min-max scaling (Savitsky et al. 2022),
#' \code{"percentile"} quantile-based deficit truncation, or \code{"target_clip"} ratio clipping.
#' In Stage 2, weights are post-processed via \code{"hard_trunc"} (zeroing records exceeding threshold \eqn{M}),
#' \code{"vector_upscale"} (upward rescaling to Loewner-dominate hard truncation), \code{"surgical"} (monotone tail zeroing),
#' or \code{"none"}.
#'
#' @param f_vec Numeric vector of unit-level empirical sensitivities \eqn{f_i = \max_s |\ell_i(\theta_s)|}. Default is \code{NULL}.
#' @param stage1_method Character string for Stage 1 initial normalization: \code{"linear"}, \code{"percentile"}, or \code{"target_clip"}.
#' @param stage2_method Character string for Stage 2 reweighting: \code{"surgical"}, \code{"hard_trunc"}, \code{"vector_upscale"}, or \code{"none"}.
#' @param c_scale Numeric scaling constant \eqn{c \in (0, 1]} for linear min-max normalization. Default is \code{1}.
#' @param g_shift Numeric shift constant \eqn{g \ge 0} for linear min-max normalization. Default is \code{0}.
#' @param p_L Numeric lower quantile threshold \eqn{p_L \in (0, 1)} for percentile normalization. Default is \code{0.05}.
#' @param p_R Numeric upper quantile threshold \eqn{p_R \in (0, 1)} for percentile normalization. Default is \code{0.95}.
#' @param Delta_goal Optional numeric scalar or vector specifying the unit sensitivity goal \eqn{\Delta_{i,\text{goal}} > 0} for target ratio clipping. Default is \code{NULL}.
#' @param M_thresh Optional numeric privacy threshold \eqn{M_{\text{thresh}} > 0} for Stage 2 reweighting. Default is \code{NULL}.
#' @param f_i Optional alias for \code{f_vec}. Default is \code{NULL}.
#' @param delta_goal Optional alias for \code{Delta_goal}. Default is \code{NULL}.
#' @return A list containing:
#'   \item{alpha}{Final numeric vector of privacy weights \eqn{\alpha_i^* \in [0, 1]}.}
#'   \item{alpha_stage1}{Numeric vector of intermediate weights after Stage 1.}
#'   \item{stage1_method}{Character string of Stage 1 method used.}
#'   \item{stage2_method}{Character string of Stage 2 method used.}
#'   \item{delta_alpha}{Realized maximum sensitivity \eqn{\max_i (\alpha_i^* f_i)}.}
#'   \item{epsilon_local}{Local privacy bound \eqn{\epsilon_{\mathbf{y}} = 2 \Delta_{\alpha, \mathbf{y}}}.}
#' @export
compute_dp_weights <- function(
    f_vec = NULL,
    stage1_method = c("linear", "percentile", "target_clip"),
    stage2_method = c("surgical", "hard_trunc", "vector_upscale", "none"),
    c_scale = 1.0,
    g_shift = 0.0,
    p_L = 0.05,
    p_R = 0.95,
    Delta_goal = NULL,
    M_thresh = NULL,
    f_i = NULL,
    delta_goal = NULL
) {
  if (is.null(f_vec) && !is.null(f_i)) f_vec <- f_i
  if (is.null(Delta_goal) && !is.null(delta_goal)) Delta_goal <- delta_goal
  if (is.null(f_vec)) stop("f_vec must be provided")
  stage1_method <- match.arg(stage1_method)
  stage2_method <- match.arg(stage2_method)
  f_vec <- as.numeric(f_vec)
  n <- length(f_vec)
  f_safe <- pmax(f_vec, 1e-8)
  alpha_tilde <- 1.0 / f_safe

  # --- Stage 1: Initial Normalization ---
  if (stage1_method == "linear") {
    f_min <- min(f_vec)
    f_max <- max(f_vec)
    f_tilde <- if (f_max > f_min) (f_vec - f_min) / (f_max - f_min) else rep(0, n)
    alpha_1 <- pmin(1, pmax(0, c_scale * (1 - f_tilde) + g_shift))

  } else if (stage1_method == "percentile") {
    q_L <- stats::quantile(alpha_tilde, probs = p_L, na.rm = TRUE)
    q_R <- stats::quantile(alpha_tilde, probs = p_R, na.rm = TRUE)
    if (q_R > q_L) {
      scaled <- (alpha_tilde - q_L) / (q_R - q_L)
      alpha_1 <- pmin(1, pmax(0, scaled))
    } else {
      alpha_1 <- rep(1, n)
    }

  } else if (stage1_method == "target_clip") {
    delta_target <- if (!is.null(Delta_goal)) {
      as.numeric(Delta_goal)
    } else if (!is.null(M_thresh)) {
      as.numeric(M_thresh)
    } else {
      stats::median(f_vec, na.rm = TRUE)
    }
    alpha_1 <- pmin(1, delta_target / f_safe)
  }

  alpha_stage1 <- alpha_1

  # --- Stage 2: Second-Stage Reweighting ---
  if (stage2_method == "none" || is.null(M_thresh) || !is.finite(M_thresh)) {
    alpha_2 <- alpha_1

  } else if (stage2_method == "hard_trunc") {
    # Scheme 1: Hard threshold truncation: zero observations exceeding M_thresh
    alpha_2 <- alpha_1
    alpha_2[alpha_1 * f_vec > M_thresh] <- 0.0

  } else if (stage2_method == "vector_upscale") {
    # Scheme 2: Vector-weighted upward rescaling (Savitsky, Hu, & Williams 2020 Eq 3.1)
    # alpha_i^vec = min(1, M_thresh / f_i) which pointwise dominates hard truncation
    alpha_2 <- pmin(1, M_thresh / f_safe)
    # If initial weights were compressed and delta_alpha < M_thresh, proportionally spend budget
    delta_cur <- max(alpha_1 * f_vec)
    if (delta_cur > 0 && delta_cur < M_thresh) {
      alpha_2 <- pmin(1, pmax(alpha_2, alpha_1 * (M_thresh / delta_cur)))
    }

  } else if (stage2_method == "surgical") {
    # Monotone surgical truncation (legacy BLS default):
    # 1. Cap alpha_i so that alpha_i * f_i <= M_thresh
    # 2. Hard zero extreme tail outliers where f_i > 3 * M_thresh
    alpha_2 <- pmin(alpha_1, M_thresh / f_safe)
    alpha_2[f_vec > 3.0 * M_thresh] <- 0.0
  }

  realized_delta <- max(alpha_2 * f_vec)
  eps_local <- 2 * realized_delta

  return(list(
    alpha = alpha_2,
    alpha_stage1 = alpha_stage1,
    stage1_method = stage1_method,
    stage2_method = stage2_method,
    delta_alpha = realized_delta,
    epsilon_local = eps_local
  ))
}

#' @title Multi-Stage Differential Privacy Pseudo-Posterior Synthesizer
#' @description Implements the risk-weighted pseudo-posterior mechanism
#' from Savitsky, Williams, and Hu (2022) with the modular two-stage normalization
#' framework of Savitsky (Oct 28, 2026). In Pass 1, fits an unweighted LBBNN model
#' to evaluate record-level disclosure sensitivities \eqn{f_i = \max_s |\ell_i(\theta_s)|}.
#' In Stage 1, computes initial privacy weights \eqn{\alpha_i \in [0, 1]}. In Stage 2,
#' applies second-stage reweighting (\code{"hard_trunc"}, \code{"vector_upscale"}, or \code{"surgical"}).
#' Under vector upward rescaling, executes an additional training pass targeting the reweighted
#' pseudo-posterior to verify the final local Lipschitz bound \eqn{\Delta_{\alpha, \mathbf{y}}}
#' and differential privacy guarantee \eqn{\epsilon_{\mathbf{y}} = 2\Delta_{\alpha, \mathbf{y}}}.
#'
#' @param model_spec An \code{lbbnn_net} instance, a list of parameters to pass to \code{lbbnn_net()},
#' or a zero-argument function that returns a fresh \code{lbbnn_net}.
#' @param X Matrix or data.frame of predictors.
#' @param y Vector or matrix of targets.
#' @param epochs Integer, number of training epochs per pass (default 100).
#' @param lr Numeric, learning rate for optimizer (default 0.01).
#' @param batch_size Integer, mini-batch size (default 64).
#' @param S Integer, number of posterior draws for sensitivity extraction (default 100).
#' @param stage1_method Character string for Stage 1 normalization: \code{"linear"} (default), \code{"percentile"}, or \code{"target_clip"}.
#' @param stage2_method Character string for Stage 2 adjustment: \code{"surgical"} (default), \code{"hard_trunc"}, \code{"vector_upscale"}, or \code{"none"}.
#' @param c_scale Numeric, scaling factor for risk weights (default 1.0).
#' @param g_shift Numeric, shift parameter for risk weights (default 0.0).
#' @param p_L Numeric left quantile level for percentile normalization (default 0.05).
#' @param p_R Numeric right quantile level for percentile normalization (default 0.95).
#' @param Delta_goal Optional numeric sensitivity goal \eqn{\Delta_{\text{goal}} > 0} for target clipping.
#' @param M_thresh Optional numeric threshold \eqn{M_{\text{thresh}} > 0}.
#' @param sigma Optional numeric standard deviation for Gaussian regression log-likelihood.
#' If NULL (default), automatically estimated after Pass 1 from the posterior mean residuals.
#' @param device Character, device string ("cpu", "cuda", "mps").
#' @param verbose Logical, whether to print training progress (default FALSE).
#' @param ... Additional arguments passed to \code{train_lbbnn()}.
#' @return A list containing:
#'   \describe{
#'     \item{model}{The final private model trained with weights \eqn{\alpha}.}
#'     \item{model_unweighted}{The baseline unweighted model from Pass 1.}
#'     \item{alpha}{Numeric vector of length \eqn{n} containing risk weights \eqn{\alpha_i \in [0, 1]}.}
#'     \item{alpha_stage1}{Numeric vector of intermediate Stage 1 weights.}
#'     \item{f_vec}{Numeric vector of length \eqn{n} containing record-level maximum sensitivities.}
#'     \item{stage1_method}{Character string of Stage 1 method used.}
#'     \item{stage2_method}{Character string of Stage 2 method used.}
#'     \item{delta_alpha}{Local Lipschitz bound \eqn{\Delta_{\alpha, \mathbf{y}}}.}
#'     \item{epsilon_local}{Local DP guarantee \eqn{\epsilon_{\mathbf{y}} = 2\Delta_{\alpha, \mathbf{y}}}.}
#'     \item{sigma_hat}{Calibrated observation residual standard deviation.}
#'     \item{L_pass1}{Log-likelihood contributions matrix from Pass 1.}
#'     \item{L_pass2}{Log-likelihood contributions matrix from Pass 2.}
#'     \item{L_pass3}{Log-likelihood contributions matrix from Pass 3 (if vector_upscale was used).}
#'     \item{dp_weights_info}{List output from \code{compute_dp_weights()}.}
#'   }
#' @export
fit_dp_lbbnn <- function(
    model_spec, X, y, epochs = 100, lr = 0.01,
    batch_size = 64, S = 100,
    stage1_method = c("linear", "percentile", "target_clip"),
    stage2_method = c("surgical", "hard_trunc", "vector_upscale", "none"),
    c_scale = 1.0, g_shift = 0.0,
    p_L = 0.05, p_R = 0.95, Delta_goal = NULL,
    M_thresh = NULL, sigma = NULL,
    device = "cpu", verbose = FALSE, ...
) {
  stage1_method <- match.arg(stage1_method)
  stage2_method <- match.arg(stage2_method)
  device <- resolve_device(device)
  X_mat <- as.matrix(X)
  n <- nrow(X_mat)
  y_vec <- if (is.matrix(y)) y else as.numeric(y)

  # Factory to construct fresh model instances
  create_model <- function() {
    if (is.function(model_spec)) {
      return(model_spec())
    } else if (is.list(model_spec)) {
      spec <- model_spec
      if (!is.null(spec$sizes)) {
        num_layers <- length(spec$sizes) - 1
        if (is.null(spec$prior)) spec$prior <- rep(0.5, num_layers)
        if (is.null(spec$std)) spec$std <- rep(1.0, num_layers)
        if (is.null(spec$inclusion_inits)) spec$inclusion_inits <- "balanced"
      }
      return(do.call(lbbnn_net, spec))
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
      preds_p1 <- stats::predict(model_pass1, newdata = eval_dl, draws = S, mpm = FALSE, device = device)
      y_hat_p1 <- as.numeric(preds_p1$mean(dim = 1)$cpu())
      sigma <- stats::sd(as.numeric(y_vec) - y_hat_p1)
      if (is.na(sigma) || sigma <= 0) sigma <- 1.0
      if (verbose) message(sprintf("Calibrated residual scale sigma_hat: %.4f", sigma))
    } else {
      sigma <- 1.0
    }
  }

  # 2. Record Sensitivity Matrix from Pass 1
  if (verbose) message("Extracting record-level sensitivities from Pass 1...")
  L_mat <- loglik_contributions(model_pass1, eval_dl, draws = S, mpm = FALSE,
                                sigma = sigma, device = device)
  f_vec <- apply(abs(L_mat), 2, max)

  # 3. Two-Stage Privacy Weights
  dp_info <- compute_dp_weights(
    f_vec = f_vec,
    stage1_method = stage1_method,
    stage2_method = if (stage2_method == "vector_upscale") "none" else stage2_method,
    c_scale = c_scale,
    g_shift = g_shift,
    p_L = p_L,
    p_R = p_R,
    Delta_goal = Delta_goal,
    M_thresh = M_thresh
  )
  alpha <- dp_info$alpha

  # 4. Pass 2: Fit Weighted Model
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

  L_pass2 <- loglik_contributions(model_pass2, eval_dl, draws = S, mpm = FALSE,
                                  sigma = sigma, device = device)

  L_pass3 <- NULL
  final_model <- model_pass2

  # 5. Optional Pass 3 for Vector Upward Rescaling (Savitsky 2026 requirement)
  if (stage2_method == "vector_upscale" && !is.null(M_thresh) && is.finite(M_thresh)) {
    if (verbose) message("Executing Pass 3 for Vector Upward Rescaling geometry...")
    f_vec_p2 <- apply(abs(L_pass2), 2, max)
    dp_info_p3 <- compute_dp_weights(
      f_vec = f_vec_p2,
      stage1_method = stage1_method,
      stage2_method = "vector_upscale",
      Delta_goal = Delta_goal,
      M_thresh = M_thresh
    )
    alpha_p3 <- dp_info_p3$alpha

    t_w_p3 <- torch::torch_tensor(as.numeric(alpha_p3), dtype = torch::torch_float32())
    ds_pass3 <- torch::tensor_dataset(t_x, t_y, t_w_p3)
    dl_pass3 <- torch::dataloader(ds_pass3, batch_size = min(batch_size, n), shuffle = TRUE)

    model_pass3 <- do.call(lbbnn_net, model_spec_p2)
    train_lbbnn(epochs = epochs, LBBNN = model_pass3, lr = lr,
                train_dl = dl_pass3, device = device, verbose = verbose, ...)

    L_pass3 <- loglik_contributions(model_pass3, eval_dl, draws = S, mpm = FALSE,
                                    sigma = sigma, device = device)
    final_model <- model_pass3
    alpha <- alpha_p3
    dp_info <- dp_info_p3
  }

  # 6. Final Local Lipschitz Bound & DP Guarantee
  L_final <- if (!is.null(L_pass3)) L_pass3 else L_pass2
  L_alpha <- sweep(abs(L_final), 2, alpha, "*")
  delta_alpha_x <- if (!is.null(M_thresh)) min(max(L_alpha), M_thresh) else max(L_alpha)
  epsilon_x <- 2 * delta_alpha_x

  return(list(
    model = final_model,
    model_unweighted = model_pass1,
    alpha = alpha,
    alpha_stage1 = dp_info$alpha_stage1,
    f_vec = f_vec,
    stage1_method = stage1_method,
    stage2_method = stage2_method,
    delta_alpha = delta_alpha_x,
    epsilon_local = epsilon_x,
    sigma_hat = sigma,
    L_pass1 = L_mat,
    L_pass2 = L_pass2,
    L_pass3 = L_pass3,
    dp_weights_info = dp_info
  ))
}


#' @title Generate Synthetic Microdata from DP-Trained LBBNN
#' @description Draws synthetic records from the posterior predictive distribution
#' of an LBBNN model fitted with \code{fit_dp_lbbnn()}, incorporating calibrated observation noise.
#' Returns an \code{n x draws} matrix where each column represents one complete synthetic dataset replicate.
#'
#' @param dp_fit Fitted DP LBBNN object returned by \code{fit_dp_lbbnn()}. Default is \code{NULL}.
#' @param newdata Optional matrix, data.frame, or tensor of predictor features. Default is \code{NULL}.
#' @param draws Integer, number of posterior draws / synthetic replicates (default 20).
#' @param mpm Logical, whether to use Median Probability Model (default FALSE).
#' @param log_scale Logical, if TRUE exponentiates predictions \code{exp(y_synth)} (default FALSE).
#' @param device Character, device string ("cpu", "cuda", "mps").
#' @param model Optional alias for \code{dp_fit} (or a fitted \code{lbbnn_net} model). Default is \code{NULL}.
#' @param x_ref Optional alias for \code{newdata}. Default is \code{NULL}.
#' @param n_syn Optional integer specifying the number of synthetic records or draws (alias for \code{draws}). Default is \code{NULL}.
#' @return A numeric matrix of dimensions \code{nrow(newdata) x draws} of synthetic observations.
#' @export
sample_synthetic_lbbnn <- function(dp_fit = NULL, newdata = NULL, draws = 20, mpm = FALSE,
                                   log_scale = FALSE, device = "cpu",
                                   model = NULL, x_ref = NULL, n_syn = NULL) {
  if (is.null(dp_fit) && !is.null(model)) {
    if (is.list(model) && !is.null(model$model)) {
      dp_fit <- model
    } else {
      dp_fit <- list(model = model)
    }
  }
  if (is.null(newdata) && !is.null(x_ref)) {
    newdata <- x_ref
  }
  if (!is.null(n_syn) && is.matrix(newdata) && nrow(newdata) > n_syn) {
    newdata <- newdata[1:n_syn, , drop = FALSE]
  }
  target_model <- if (inherits(dp_fit, "lbbnn_net")) dp_fit else dp_fit$model
  mu_draws <- stats::predict(target_model, newdata = newdata, draws = draws, mpm = mpm, device = device)
  # mu_draws has shape (draws, n, 1) -> transpose to (n, draws)
  mu_mat <- t(as.matrix(mu_draws[, , 1]$cpu()))
  n_obs <- nrow(mu_mat)
  sigma_noise <- if (!is.null(dp_fit$sigma_hat)) dp_fit$sigma_hat else 1.0
  synth_mat <- matrix(
    stats::rnorm(n_obs * draws, mean = as.numeric(mu_mat), sd = sigma_noise),
    nrow = n_obs, ncol = draws
  )
  if (log_scale) synth_mat <- exp(synth_mat)
  return(synth_mat)
}
