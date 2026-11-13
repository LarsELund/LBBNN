#' Capture B Fixed Posterior Parameter Support Draws from a Trained LBBNN
#'
#' Samples and freezes B forward-pass predictive conditional parameters on covariate
#' matrix X (equivalent to fixing B parameter draws Theta = \{theta_1, ..., theta_B\}
#' from the variational posterior q(theta, gamma) when covariates X are fixed).
#'
#' @param model Trained \code{lbbnn_net} object.
#' @param X Covariate matrix or data.frame of dimension \code{[n, p]}.
#' @param B Integer number of posterior draws to save in the support (default 100).
#' @param mpm Logical; whether to draw from the Median Probability Model (default FALSE).
#' @param link Optional link function passed to \code{predict.lbbnn_net()}.
#' @return A numeric matrix of dimension \code{[B, n]} containing predicted conditional
#'   parameters \code{mu_{b, i} = mu(x_i; theta_b)} across the B saved parameter draws.
#' @export
extract_lbbnn_support <- function(model, X, B = 100, mpm = FALSE, link = NULL) {
  X_df <- as.data.frame(X)
  n <- nrow(X_df)
  preds_tensor <- stats::predict(model, newdata = X_df, draws = B, mpm = mpm, link = link)
  if (inherits(preds_tensor, "torch_tensor")) {
    if (length(dim(preds_tensor)) == 3) {
      mu_mat <- as.matrix(preds_tensor[, , 1]$cpu())
    } else {
      mu_mat <- as.matrix(preds_tensor$cpu())
    }
  } else {
    mu_mat <- as.matrix(preds_tensor)
  }
  mu_mat <- matrix(as.numeric(mu_mat), nrow = B, ncol = n)
  return(mu_mat)
}

#' Evaluate B x n Absolute Pointwise Log-Likelihood Matrix on Fixed Parameter Support
#'
#' Evaluates \eqn{|\log f(y_i \mid \mathbf{x}_i, \theta_b)|} across all B saved parameter
#' draws and n units without re-running neural network forward passes or VI training.
#'
#' @param mu_support Numeric matrix \code{[B, n]} of conditional means or class
#'   probabilities from \code{extract_lbbnn_support()}.
#' @param y Numeric response vector of length \code{n}.
#' @param problem_type Character: \code{"regression"} or \code{"binary"} (or
#'   \code{"binary classification"}).
#' @param sigma Numeric residual standard deviation for Gaussian regression.
#' @return A numeric matrix of dimension \code{[B, n]} containing \code{|log f_{theta_b}(y_i)|}.
#' @export
eval_support_abs_loglik <- function(mu_support, y, problem_type = "regression", sigma = 1.0) {
  B <- nrow(mu_support)
  n <- ncol(mu_support)
  y_mat <- matrix(as.numeric(y), nrow = B, ncol = n, byrow = TRUE)

  if (identical(problem_type, "regression")) {
    log_lik <- -0.5 * log(2 * pi * sigma^2) - ((y_mat - mu_support)^2) / (2 * sigma^2)
  } else {
    eps <- 1e-7
    p_mat <- pmin(pmax(mu_support, eps), 1.0 - eps)
    log_lik <- y_mat * log(p_mat) + (1.0 - y_mat) * log(1.0 - p_mat)
  }
  return(abs(log_lik))
}

#' Fast-Computing Finite-Sample (epsilon, delta)-DP Calibration via Dual-Support Plug-in
#'
#' Implements Algorithm 3.1 of Azizi, Savitsky, Williams & Chatterjee (2026) to
#' approximate the finite-sample replicate sensitivity distribution
#' \eqn{\{\Delta_{\alpha, x_r}\}_{r=1}^R} without additional model retraining,
#' evaluate the unadjusted privacy leakage probability \eqn{\hat{\delta}_n(0)},
#' and compute the additive calibration nugget \eqn{\eta} for any user-specified
#' target \eqn{\delta_{\text{target}} \in (0, 1)}.
#'
#' @param dp_fit Fitted object returned by \code{fit_dp_lbbnn()}.
#' @param X Covariate matrix or data.frame \code{[n, p]}.
#' @param y Observed response vector of length \code{n}.
#' @param R Integer number of posterior predictive replicate datasets (default 200).
#' @param B Integer number of saved posterior draws per support (default 100).
#' @param delta_target Numeric vector or scalar in \code{(0, 1)} specifying target
#'   finite-sample failure probability (default \code{0.05}).
#' @param stage1_method Stage 1 weight normalization scheme (defaults to \code{dp_fit$stage1_method}).
#' @param stage2_method Stage 2 reweighting scheme (defaults to \code{dp_fit$stage2_method}).
#' @param M_thresh Sensitivity threshold (defaults to \code{dp_fit$M_thresh}).
#' @param c_scale Linear scaling factor (defaults to \code{dp_fit$c_scale} or \code{1.0}).
#' @param g_shift Linear shift parameter (defaults to \code{dp_fit$g_shift} or \code{0.0}).
#' @param p_bounds Numeric length-2 quantile bounds for percentile scheme (defaults to
#'   \code{c(dp_fit$p_L, dp_fit$p_R)} or \code{c(0.05, 0.95)}).
#' @param Delta_goal Optional target clipping goal (defaults to \code{dp_fit$Delta_goal} or \code{M_thresh}).
#' @param seed Random seed for reproducibility (default 2026).
#' @return A list containing:
#'   \describe{
#'     \item{delta_alpha_obs}{Observed local sensitivity \eqn{\Delta_{\alpha, x}}.}
#'     \item{epsilon_obs}{Unadjusted local privacy parameter \eqn{2\Delta_{\alpha, x}}.}
#'     \item{delta_rep_vec}{Numeric vector of length \code{R} of replicate sensitivities \eqn{\Delta_{\alpha, x_r}}.}
#'     \item{delta_hat_0}{Empirical finite-sample exceedance rate at zero nugget.}
#'     \item{delta_target}{Primary target failure probability.}
#'     \item{eta_nugget}{Additive calibration nugget \eqn{\eta} for primary \code{delta_target}.}
#'     \item{delta_alpha_calibrated}{Calibrated sensitivity \eqn{\Delta_{\alpha, x} + \eta}.}
#'     \item{epsilon_calibrated}{Calibrated privacy guarantee \eqn{2(\Delta_{\alpha, x} + \eta)}.}
#'     \item{calibration_table}{Data frame summarizing quantiles, nuggets, calibrated epsilon,
#'       and empirical exceedance rates across target \code{delta_target} levels.}
#'   }
#' @export
calibrate_dp_finite_sample <- function(dp_fit,
                                       X,
                                       y,
                                       R = 200,
                                       B = 100,
                                       delta_target = 0.05,
                                       stage1_method = NULL,
                                       stage2_method = NULL,
                                       M_thresh = NULL,
                                       c_scale = NULL,
                                       g_shift = NULL,
                                       p_bounds = NULL,
                                       Delta_goal = NULL,
                                       seed = 2026) {
  set.seed(seed)
  torch::torch_manual_seed(seed)

  stopifnot(is.numeric(delta_target), all(delta_target > 0 & delta_target < 1))

  if (is.null(stage1_method)) {
    stage1_method <- if (!is.null(dp_fit$stage1_method)) dp_fit$stage1_method else "target_clip"
  }
  if (is.null(stage2_method)) {
    stage2_method <- if (!is.null(dp_fit$stage2_method)) dp_fit$stage2_method else "vector_upscale"
  }
  if (is.null(M_thresh)) {
    M_thresh <- if (!is.null(dp_fit$M_thresh)) dp_fit$M_thresh else 5.5
  }
  if (is.null(c_scale)) {
    c_scale <- if (!is.null(dp_fit$c_scale)) dp_fit$c_scale else 1.0
  }
  if (is.null(g_shift)) {
    g_shift <- if (!is.null(dp_fit$g_shift)) dp_fit$g_shift else 0.0
  }
  if (is.null(p_bounds)) {
    p_L_val <- if (!is.null(dp_fit$p_L)) dp_fit$p_L else 0.05
    p_R_val <- if (!is.null(dp_fit$p_R)) dp_fit$p_R else 0.95
    p_bounds <- c(p_L_val, p_R_val)
  }
  if (is.null(Delta_goal)) {
    Delta_goal <- if (!is.null(dp_fit$Delta_goal)) dp_fit$Delta_goal else M_thresh
  }

  raw_ptype <- if (!is.null(dp_fit$model$problem_type)) dp_fit$model$problem_type else "regression"
  ptype <- if (identical(raw_ptype, "regression")) "regression" else "binary"
  link_fn <- if (identical(ptype, "binary")) torch::torch_sigmoid else NULL

  n <- length(y)
  sigma_1 <- if (!is.null(dp_fit$sigma_p1)) {
    dp_fit$sigma_p1
  } else if (!is.null(dp_fit$model$sigma)) {
    dp_fit$model$sigma
  } else {
    dp_fit$sigma_hat
  }
  sigma_2 <- if (!is.null(dp_fit$model$sigma)) dp_fit$model$sigma else dp_fit$sigma_hat
  sigma_gen <- if (!is.null(dp_fit$sigma_hat)) dp_fit$sigma_hat else sigma_1

  # Step 1: Save B parameter draws from Fit 1 (unweighted) & evaluate observed risks
  Theta_1_mu <- extract_lbbnn_support(dp_fit$model_unweighted, X, B = B, mpm = FALSE, link = link_fn)
  L_obs_1    <- eval_support_abs_loglik(Theta_1_mu, y, problem_type = ptype, sigma = sigma_1)
  f_obs      <- apply(L_obs_1, 2, max)
  w_obs      <- compute_dp_weights(
    f_vec = f_obs, M_thresh = M_thresh,
    stage1_method = stage1_method, stage2_method = stage2_method,
    c_scale = c_scale, g_shift = g_shift,
    p_L = p_bounds[1], p_R = p_bounds[2], Delta_goal = Delta_goal
  )
  alpha_obs <- w_obs$alpha

  # Step 2: Save B parameter draws from Fit 2/3 (alpha-weighted) & evaluate baseline Delta_{alpha, x}
  Theta_2_mu      <- extract_lbbnn_support(dp_fit$model, X, B = B, mpm = FALSE, link = link_fn)
  L_obs_2         <- eval_support_abs_loglik(Theta_2_mu, y, problem_type = ptype, sigma = sigma_2)
  L_alpha_obs     <- sweep(L_obs_2, 2, alpha_obs, `*`)
  raw_obs_delta   <- max(L_alpha_obs)
  scale_anchor    <- if (!is.null(dp_fit$delta_alpha) && is.finite(dp_fit$delta_alpha) && raw_obs_delta > 0) {
    as.numeric(dp_fit$delta_alpha) / raw_obs_delta
  } else {
    1.0
  }
  delta_alpha_obs <- raw_obs_delta * scale_anchor

  # Step 3: Dual-Support Plug-in across R posterior predictive replicate datasets
  delta_rep_vec <- numeric(R)
  for (r in seq_len(R)) {
    # (a) Posterior predictive simulation from Fit 1 support
    b_r <- sample.int(B, size = 1)
    if (identical(ptype, "regression")) {
      y_rep <- stats::rnorm(n, mean = Theta_1_mu[b_r, ], sd = sigma_gen)
    } else {
      y_rep <- stats::rbinom(n, size = 1, prob = pmin(pmax(Theta_1_mu[b_r, ], 1e-7), 1.0 - 1e-7))
    }

    # (b) Stage 1 Plug-in: Evaluate replicate risks & weights on Fit 1 support Theta^{(1)}
    L_rep_1 <- eval_support_abs_loglik(Theta_1_mu, y_rep, problem_type = ptype, sigma = sigma_1)
    f_rep   <- apply(L_rep_1, 2, max)
    w_rep   <- compute_dp_weights(
      f_vec = f_rep, M_thresh = M_thresh,
      stage1_method = stage1_method, stage2_method = stage2_method,
      c_scale = c_scale, g_shift = g_shift,
      p_L = p_bounds[1], p_R = p_bounds[2], Delta_goal = Delta_goal
    )
    alpha_rep <- w_rep$alpha

    # (c) Stage 2 Re-plug: Evaluate replicate sensitivity on Fit 2 support Theta_alpha^{(2)}
    L_rep_2          <- eval_support_abs_loglik(Theta_2_mu, y_rep, problem_type = ptype, sigma = sigma_2)
    L_alpha_rep      <- sweep(L_rep_2, 2, alpha_rep, `*`)
    delta_rep_vec[r] <- max(L_alpha_rep) * scale_anchor
  }

  # Step 4: Evaluate unadjusted delta_hat(0) and calibrate nugget eta for delta_target
  delta_hat_0 <- mean(delta_rep_vec > delta_alpha_obs)

  targets_all <- sort(unique(c(delta_target, 0.10, 0.05, 0.01)), decreasing = TRUE)
  cal_rows <- lapply(targets_all, function(dt) {
    q_val      <- as.numeric(stats::quantile(delta_rep_vec, probs = 1.0 - dt, names = FALSE))
    eta_val    <- max(0.0, q_val - delta_alpha_obs)
    delta_cal  <- delta_alpha_obs + eta_val
    eps_cal    <- 2.0 * delta_cal
    realized_d <- mean(delta_rep_vec > delta_cal)
    data.frame(
      delta_target           = dt,
      quantile_1_minus_delta = q_val,
      eta_nugget             = eta_val,
      delta_alpha_eta        = delta_cal,
      epsilon_eta            = eps_cal,
      empirical_exceedance   = realized_d,
      stringsAsFactors       = FALSE
    )
  })
  cal_table <- do.call(rbind, cal_rows)
  rownames(cal_table) <- NULL

  q_primary   <- as.numeric(stats::quantile(delta_rep_vec, probs = 1.0 - delta_target[1], names = FALSE))
  eta_primary <- max(0.0, q_primary - delta_alpha_obs)

  list(
    delta_alpha_obs        = delta_alpha_obs,
    epsilon_obs            = 2.0 * delta_alpha_obs,
    delta_rep_vec          = delta_rep_vec,
    delta_hat_0            = delta_hat_0,
    delta_target           = delta_target[1],
    eta_nugget             = eta_primary,
    delta_alpha_calibrated = delta_alpha_obs + eta_primary,
    epsilon_calibrated     = 2.0 * (delta_alpha_obs + eta_primary),
    calibration_table      = cal_table
  )
}
