#' Safe Upper-Triangular Cholesky Decomposition with Eigenvalue Floor
#' @keywords internal
#' @noRd
.safe_chol <- function(mat, min_eig = 1e-8) {
  sym_mat <- 0.5 * (mat + t(mat))
  res <- tryCatch(base::chol(sym_mat), error = function(e) NULL)
  if (!is.null(res)) return(res)
  eig <- base::eigen(sym_mat, symmetric = TRUE)
  vals <- pmax(eig$values, min_eig)
  pd_mat <- eig$vectors %*% (vals * t(eig$vectors))
  pd_mat <- 0.5 * (pd_mat + t(pd_mat))
  base::chol(pd_mat)
}

#' Deterministic MPM Mean Forward Pass Through an LBBNN Linear Layer
#' @keywords internal
#' @noRd
.forward_layer_mpm_mean <- function(lyr, inp, active_cutoff = 0.5) {
  w_mean <- lyr$weight_mean$detach()
  b_mean <- lyr$bias_mean$detach()
  a_prob <- torch::torch_sigmoid(lyr$lambda_l$detach())
  mask   <- (a_prob > active_cutoff)$to(dtype = w_mean$dtype)
  if (!is.null(lyr$alpha_active_path)) {
    aap <- lyr$alpha_active_path$detach()
    if (all(is.finite(as.numeric(aap$cpu())))) {
      mask <- mask * (aap > 0.5)$to(dtype = w_mean$dtype)
    }
  }
  w_mpm <- w_mean * mask
  if (isTRUE(lyr$bias_inclusion_prob) && !is.null(lyr$bias_lambda_l)) {
    b_mask <- (torch::torch_sigmoid(lyr$bias_lambda_l$detach()) > active_cutoff)$to(dtype = b_mean$dtype)
    b_mpm  <- b_mean * b_mask
  } else {
    b_mpm <- b_mean
  }
  torch::torch_matmul(inp, torch::torch_t(w_mpm)) + b_mpm
}

#' Williams-Savitsky (2021) Sandwich Variance Correction for Active MPM Slab Parameters
#'
#' Applies the scale-and-shape Cholesky projection of Williams & Savitsky (2021,
#' International Statistical Review, 89(1), 72-107, doi:10.1111/insr.12376) to
#' model parameters/weights selected into the active Median Probability Model (MPM)
#' slab (\code{a > active_cutoff}) to account for the repeated-sampling distribution
#' \eqn{P_\nu} under informative and complex survey designs.
#'
#' Conditioning on the MPM active slab subspace is essential for neural networks:
#' dense BNN posteriors are multi-modal and non-identifiable, violating the
#' Bernstein-von Mises (BvM) regularity conditions, whereas the pruned MPM slab
#' isolates a low-dimensional, locally unimodal parameterization where BvM holds.
#' In hierarchical extensions where unit outcomes \eqn{y_i \sim p(y_i \mid \theta_i)}
#' are centered on latent unit truth \eqn{\theta_i}, the same Cholesky transport
#' \eqn{R_2^\top R_1^{-\top}} calibrates joint/Schur-complement credible intervals.
#'
#' @param object Trained \code{lbbnn_net} object (regression).
#' @param X Covariate matrix or data.frame \code{[n, p]}.
#' @param y Response vector of length \code{n}.
#' @param weights Survey design weights of length \code{n} (normalized internally to sum to \code{n}).
#' @param strata Optional vector of stratum indicators of length \code{n} for stratified
#'   score covariance estimation in \code{J_2}.
#' @param psu Optional vector of primary sampling unit (cluster) indicators of length \code{n}
#'   for clustered score covariance estimation in \code{J_2} (Williams & Savitsky, 2021).
#' @param draws Integer number of posterior draws to sample and transport (default 200).
#' @param active_cutoff Posterior inclusion probability cutoff defining the slab (default 0.5).
#' @param level Credible interval level (default 0.95).
#' @param sigma Optional observation residual standard deviation override; defaults to
#'   \code{object$sigma} (or \code{1.0} if unset).
#' @param seed Random seed (default 2026).
#' @return A list containing:
#'   \itemize{
#'     \item \code{summary_table}: Data frame comparing unadjusted vs. Williams-Savitsky
#'       sandwich-corrected posterior means, SDs, design effect ratios (DEFF), and credible intervals.
#'     \item \code{draws_unadjusted}: Matrix \code{[draws, d_slab]} of raw variational slab draws.
#'     \item \code{draws_adjusted}: Matrix \code{[draws, d_slab]} of transported slab draws \code{theta^*}.
#'     \item \code{Sigma_1}: Unadjusted posterior covariance matrix \code{R_1^T R_1}.
#'     \item \code{H1_bread}: Bread matrix (weighted observed Fisher information) \code{H_1}.
#'     \item \code{J2_meat}: Meat matrix (design-based weighted score covariance) \code{J_2}.
#'     \item \code{Sigma_sandwich}: Target sandwich covariance matrix \code{H_1^{-1} J_2 H_1^{-1} = R_2^T R_2}.
#'     \item \code{R1}: Cholesky factor of \code{Sigma_1}.
#'     \item \code{R2}: Cholesky factor of \code{Sigma_sandwich}.
#'   }
#' @export
adjust_slab_sandwich_lbbnn <- function(object,
                                       X,
                                       y,
                                       weights = NULL,
                                       strata = NULL,
                                       psu = NULL,
                                       draws = 200,
                                       active_cutoff = 0.5,
                                       level = 0.95,
                                       sigma = NULL,
                                       seed = 2026) {
  set.seed(seed)
  torch::torch_manual_seed(seed)

  X_mat <- as.matrix(X)
  y_vec <- as.numeric(y)
  n     <- nrow(X_mat)
  p     <- ncol(X_mat)

  if (is.null(weights)) {
    w_tilde <- rep(1.0, n)
  } else {
    w_tilde <- as.numeric(weights) * (n / sum(weights))
  }

  sigma_val <- if (!is.null(sigma) && is.finite(sigma)) {
    as.numeric(sigma)
  } else if (!is.null(object$sigma) && is.finite(object$sigma)) {
    as.numeric(object$sigma)
  } else {
    1.0
  }

  # Extract active MPM slab parameters (sigmoid(lambda_l) > active_cutoff) from output layer
  out_layer <- object$out_layer
  alpha_out <- as.numeric(torch::torch_sigmoid(out_layer$lambda_l$detach())$cpu())
  mu_w_out  <- as.numeric(out_layer$weight_mean$detach()$cpu())
  sd_w_out  <- as.numeric(torch::torch_log1p(torch::torch_exp(out_layer$weight_rho$detach()))$cpu())
  mu_b_out  <- as.numeric(out_layer$bias_mean$detach()$cpu())
  sd_b_out  <- as.numeric(torch::torch_log1p(torch::torch_exp(out_layer$bias_rho$detach()))$cpu())

  active_idx <- which(alpha_out > active_cutoff)
  if (!is.null(out_layer$alpha_active_path)) {
    aap_out <- as.numeric(out_layer$alpha_active_path$detach()$cpu())
    if (all(is.finite(aap_out))) {
      active_path_idx <- which(alpha_out > active_cutoff & aap_out > 0.5)
      if (length(active_path_idx) > 0) {
        active_idx <- active_path_idx
      }
    }
  }
  if (length(active_idx) == 0) {
    active_idx <- which.max(alpha_out)
  }

  # Evaluate deterministic MPM basis matrix H_full [n, n_hid (+ p if input_skip)]
  x_tensor <- torch::torch_tensor(X_mat, dtype = torch::torch_float32(), device = object$device)
  torch::with_no_grad({
    hid_children <- object$layers$children
    n_hid_layers <- length(hid_children)
    if (n_hid_layers == 0) {
      H_tensor <- x_tensor
    } else {
      h_curr <- object$act(.forward_layer_mpm_mean(hid_children[[1]], x_tensor, active_cutoff))
      if (n_hid_layers >= 2) {
        for (j in 2:n_hid_layers) {
          inp_j <- if (isTRUE(object$input_skip)) {
            torch::torch_cat(list(h_curr, x_tensor), dim = 2)
          } else {
            h_curr
          }
          h_curr <- object$act(.forward_layer_mpm_mean(hid_children[[j]], inp_j, active_cutoff))
        }
      }
      H_tensor <- if (isTRUE(object$input_skip)) {
        torch::torch_cat(list(h_curr, x_tensor), dim = 2)
      } else {
        h_curr
      }
    }
    H_full <- as.matrix(H_tensor$cpu())
  })

  # Drop completely degenerate (zero-variance, e.g., dead ReLU) active hidden columns if any
  col_sds <- apply(H_full[, active_idx, drop = FALSE], 2, stats::sd)
  nondeg_mask <- is.finite(col_sds) & (col_sds > 1e-8)
  if (any(nondeg_mask)) {
    active_idx <- active_idx[nondeg_mask]
  }

  n_hid <- ncol(H_full) - (if (isTRUE(object$input_skip)) p else 0L)
  x_names <- if (!is.null(colnames(X_mat))) colnames(X_mat) else paste0("x", seq_len(p))
  col_labels <- c(
    if (n_hid > 0) paste0("Hidden_h", seq_len(n_hid)) else character(0),
    if (isTRUE(object$input_skip)) paste0("Skip_", x_names) else character(0)
  )

  H_active <- cbind(Intercept = 1.0, H_full[, active_idx, drop = FALSE])
  param_names <- c("Intercept", col_labels[active_idx])
  d_slab <- ncol(H_active)

  theta_mean_vi <- c(mu_b_out[1], mu_w_out[active_idx])
  theta_sd_vi   <- c(sd_b_out[1], sd_w_out[active_idx])
  prior_sd_slab <- if (!is.null(object$prior_std) && length(object$prior_std) > 0) {
    as.numeric(object$prior_std[length(object$prior_std)])
  } else {
    1.0
  }

  # Sample S unadjusted posterior draws from active MPM slab
  draws_unadj <- matrix(0.0, nrow = draws, ncol = d_slab)
  for (j in seq_len(d_slab)) {
    draws_unadj[, j] <- stats::rnorm(draws, mean = theta_mean_vi[j], sd = theta_sd_vi[j])
  }
  colnames(draws_unadj) <- param_names
  Sigma_1 <- stats::cov(draws_unadj)
  if (d_slab == 1L) {
    Sigma_1 <- matrix(Sigma_1, nrow = 1L, ncol = 1L)
  }
  min_ev_1 <- min(base::eigen(0.5 * (Sigma_1 + t(Sigma_1)), symmetric = TRUE, only.values = TRUE)$values)
  if (!is.finite(min_ev_1) || min_ev_1 <= 1e-10) {
    Sigma_1 <- Sigma_1 + diag(1e-8, d_slab)
  }

  # 1. Bread Matrix H_1: Weighted observed Fisher information + prior precision
  W_diag   <- w_tilde / (sigma_val^2)
  H1_bread <- crossprod(H_active, H_active * W_diag) + diag(1.0 / (prior_sd_slab^2), d_slab)
  H1_bread <- 0.5 * (H1_bread + t(H1_bread))
  H1_inv   <- solve(H1_bread + diag(1e-8, d_slab))

  # 2. Meat Matrix J_2: Design-based covariance of weighted score contributions
  # Unit score: s_i = w_tilde_i * nabla_theta log f(y_i | x_i, theta_hat)
  mu_pred   <- as.numeric(H_active %*% theta_mean_vi)
  resid_i   <- y_vec - mu_pred
  score_mat <- H_active * (w_tilde * resid_i / (sigma_val^2))

  if (is.null(psu) && is.null(strata)) {
    score_centered <- scale(score_mat, center = TRUE, scale = FALSE)
    J2_meat <- (n / (n - 1)) * crossprod(score_centered) + diag(1e-10, d_slab)
  } else {
    strata_id <- if (is.null(strata)) rep(1L, n) else as.factor(strata)
    psu_id    <- if (is.null(psu)) seq_len(n) else psu
    J2_meat   <- matrix(0.0, nrow = d_slab, ncol = d_slab)
    for (st in levels(strata_id)) {
      idx_h <- which(strata_id == st)
      psu_h <- factor(psu_id[idx_h])
      k_h   <- nlevels(psu_h)
      if (k_h > 1L) {
        psu_totals   <- rowsum(score_mat[idx_h, , drop = FALSE], group = psu_h)
        psu_centered <- scale(psu_totals, center = TRUE, scale = FALSE)
        J2_meat      <- J2_meat + (k_h / (k_h - 1)) * crossprod(psu_centered)
      }
    }
    J2_meat <- J2_meat + diag(1e-10, d_slab)
  }
  J2_meat <- 0.5 * (J2_meat + t(J2_meat))

  # 3. Sandwich Covariance Target: Sigma_sand = H_1^{-1} J_2 H_1^{-1}
  Sigma_sandwich <- H1_inv %*% J2_meat %*% H1_inv
  Sigma_sandwich <- 0.5 * (Sigma_sandwich + t(Sigma_sandwich))

  # 4. Williams & Savitsky (2021, Eq. 12) Scale-and-Shape Cholesky Projection:
  # theta^{*(s)} = theta_hat + R_2^T R_1^{-T} (theta^{(s)} - theta_hat)
  R1 <- .safe_chol(Sigma_1)
  R2 <- .safe_chol(Sigma_sandwich)
  Sigma_sandwich <- crossprod(R2) # Exact R2^T R2 matching transported covariance
  transport_mat  <- solve(R1) %*% R2

  theta_hat_mat <- matrix(colMeans(draws_unadj), nrow = draws, ncol = d_slab, byrow = TRUE)
  draws_adj     <- theta_hat_mat + (draws_unadj - theta_hat_mat) %*% transport_mat
  colnames(draws_adj) <- param_names

  alpha_tail <- (1.0 - level) / 2.0
  q_unadj <- apply(draws_unadj, 2, stats::quantile, probs = c(alpha_tail, 1.0 - alpha_tail))
  q_adj   <- apply(draws_adj,   2, stats::quantile, probs = c(alpha_tail, 1.0 - alpha_tail))

  sd_unadj <- sqrt(diag(Sigma_1))
  sd_sand  <- sqrt(diag(Sigma_sandwich))

  summary_tbl <- data.frame(
    Parameter   = param_names,
    Post_Mean   = theta_mean_vi,
    Unadj_SD    = sd_unadj,
    Sandwich_SD = sd_sand,
    DEFF_Ratio  = (sd_sand / sd_unadj)^2,
    Unadj_Low   = q_unadj[1, ],
    Unadj_High  = q_unadj[2, ],
    Sand_Low    = q_adj[1, ],
    Sand_High   = q_adj[2, ],
    row.names   = NULL,
    stringsAsFactors = FALSE
  )

  list(
    summary_table    = summary_tbl,
    draws_unadjusted = draws_unadj,
    draws_adjusted   = draws_adj,
    Sigma_1          = Sigma_1,
    H1_bread         = H1_bread,
    J2_meat          = J2_meat,
    Sigma_sandwich   = Sigma_sandwich,
    R1               = R1,
    R2               = R2
  )
}
