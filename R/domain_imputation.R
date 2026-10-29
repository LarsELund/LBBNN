#' @title Domain-Level Estimation for LBBNN Models
#' @description Computes domain-level posterior means, totals, standard errors, and credible intervals
#' across sub-populations / domains defined by a domain indicator matrix or categorical grouping variable.
#' Supports both model-assisted predictions (\code{type = "latent_mean"} or \code{type = "predictive"})
#' and Dirichlet Bayesian bootstrap uncertainty quantification (\code{dirichlet_uq = TRUE}) per Savitsky & Toth (2016).
#'
#' @param object A trained \code{lbbnn_net} object.
#' @param newdata A \code{data.frame}, \code{torch::dataloader}, or list containing predictor features.
#' @param domain_matrix A binary/fractional membership matrix \eqn{\mathbf{G} \in \mathbb{R}^{n \times D}}
#' where \eqn{G_{i,d} = 1} if unit \eqn{i} belongs to domain \eqn{d}, or a factor/character/integer vector
#' of length \eqn{n} that is automatically one-hot encoded into a domain matrix.
#' @param weights Optional numeric vector of observation/survey weights \eqn{w_i \ge 0} of length \eqn{n}.
#' If NULL, uniform weights \eqn{w_i = 1} are used.
#' @param draws Integer, number of posterior parameter draws \eqn{S} (default 100).
#' @param mpm Logical, whether to evaluate predictions under the sparse Median Probability Model (default FALSE).
#' @param type Character string, either \code{"latent_mean"} (default, uses \eqn{\mu_{\theta^{(s)}}(\mathbf{x}_i)}
#' which strips out idiosyncratic observation measurement error \eqn{\varepsilon_i}) or \code{"predictive"}
#' (includes additive observation noise \eqn{\varepsilon_i^{(s)} \sim \mathcal{N}(0, \sigma^2)}).
#' @param dirichlet_uq Logical, whether to incorporate design-based finite-population uncertainty via
#' Dirichlet Bayesian bootstrap perturbations \eqn{\boldsymbol{\xi}^{(s)} \sim \text{Dirichlet}(1, \dots, 1)} (default FALSE).
#' @param sigma Optional numeric residual error standard deviation \eqn{\sigma}. If NULL, uses \code{object$sigma}
#' or defaults to 1.0.
#' @param device Character string, device for neural network computation ("cpu", "cuda", "mps").
#' @return A \code{data.frame} with one row per domain containing:
#' \describe{
#'   \item{domain}{Domain identifier / label.}
#'   \item{n_units}{Number of sample units belonging to the domain.}
#'   \item{mean}{Posterior expectation of domain mean \eqn{\hat{\bar{Y}}_d}.}
#'   \item{sd}{Posterior standard deviation of domain mean.}
#'   \item{ci_lower}{Lower 2.5% credible bound for domain mean.}
#'   \item{ci_upper}{Upper 97.5% credible bound for domain mean.}
#'   \item{total_est}{Estimated domain total \eqn{\hat{Y}_{\text{total}, d}}.}
#'   \item{total_sd}{Posterior standard deviation of domain total.}
#'   \item{total_ci_lower}{Lower 2.5% credible bound for domain total.}
#'   \item{total_ci_upper}{Upper 97.5% credible bound for domain total.}
#' }
#' @importFrom stats predict rnorm rexp quantile sd model.matrix
#' @export
predict_domains_lbbnn <- function(
    object,
    newdata,
    domain_matrix,
    weights = NULL,
    draws = 100,
    mpm = FALSE,
    type = c("latent_mean", "predictive"),
    dirichlet_uq = FALSE,
    sigma = NULL,
    device = "cpu"
) {
  type <- match.arg(type)
  device <- resolve_device(device)

  # 1. Obtain posterior draws of predictions: shape (draws, n, 1) or (draws, n)
  preds_draws <- stats::predict(object, newdata = newdata, draws = draws, mpm = mpm, device = device)
  
  if (inherits(preds_draws, "torch_tensor")) {
    preds_mat <- as.matrix(preds_draws$squeeze(-1)$cpu())
  } else {
    preds_mat <- as.matrix(preds_draws)
  }
  # Shape of preds_mat is draws x n. Transpose to n x draws
  if (nrow(preds_mat) == draws) {
    preds_mat <- t(preds_mat)
  }
  n <- nrow(preds_mat)

  # 2. Add observation noise if predictive
  if (type == "predictive") {
    sigma_val <- if (!is.null(sigma)) sigma else if (!is.null(object$sigma)) object$sigma else 1.0
    noise_mat <- matrix(stats::rnorm(n * draws, mean = 0, sd = sigma_val), nrow = n, ncol = draws)
    preds_mat <- preds_mat + noise_mat
  }

  # 3. Construct binary/fractional domain membership matrix G
  if (is.matrix(domain_matrix) || is.data.frame(domain_matrix)) {
    G <- as.matrix(domain_matrix)
    if (nrow(G) != n) {
      stop(sprintf("nrow(domain_matrix) (%d) must match number of observations (%d)", nrow(G), n))
    }
    domain_names <- colnames(G)
    if (is.null(domain_names)) domain_names <- paste0("Domain_", seq_len(ncol(G)))
  } else if (is.factor(domain_matrix) || is.character(domain_matrix) || is.numeric(domain_matrix)) {
    fac <- as.factor(domain_matrix)
    if (length(fac) != n) {
      stop(sprintf("length(domain_matrix) (%d) must match number of observations (%d)", length(fac), n))
    }
    domain_names <- levels(fac)
    G <- stats::model.matrix(~ 0 + fac)
    colnames(G) <- domain_names
  } else {
    stop("domain_matrix must be a matrix, data.frame, factor, or character vector")
  }
  n_domains <- ncol(G)

  # 4. Process base observation weights
  if (is.null(weights)) {
    base_w <- rep(1, n)
  } else {
    base_w <- as.numeric(weights)
    if (length(base_w) != n) {
      stop(sprintf("length(weights) (%d) must match number of observations (%d)", length(base_w), n))
    }
  }

  # 5. Evaluate domain estimates across draws
  domain_means <- matrix(NA_real_, nrow = n_domains, ncol = draws)
  domain_totals <- matrix(NA_real_, nrow = n_domains, ncol = draws)

  for (s in seq_len(draws)) {
    if (dirichlet_uq) {
      # Dirichlet(1, ..., 1) draw via normalized Exponential(1)
      exp_draws <- stats::rexp(n, rate = 1)
      xi_s <- exp_draws / sum(exp_draws)
      w_s <- xi_s * base_w * n  # scale so mean weight is preserved
    } else {
      w_s <- base_w
    }

    y_s <- preds_mat[, s]

    for (d in seq_len(n_domains)) {
      g_d <- G[, d]
      eff_w <- w_s * g_d
      denom <- sum(eff_w)
      if (denom > 1e-12) {
        domain_means[d, s] <- sum(eff_w * y_s) / denom
      } else {
        domain_means[d, s] <- NA_real_
      }
      domain_totals[d, s] <- sum(eff_w * y_s)
    }
  }

  # 6. Aggregate results into summary data frame
  results <- data.frame(
    domain = domain_names,
    n_units = colSums(G > 0),
    mean = apply(domain_means, 1, mean, na.rm = TRUE),
    sd = apply(domain_means, 1, stats::sd, na.rm = TRUE),
    ci_lower = apply(domain_means, 1, stats::quantile, probs = 0.025, na.rm = TRUE),
    ci_upper = apply(domain_means, 1, stats::quantile, probs = 0.975, na.rm = TRUE),
    total_est = apply(domain_totals, 1, mean, na.rm = TRUE),
    total_sd = apply(domain_totals, 1, stats::sd, na.rm = TRUE),
    total_ci_lower = apply(domain_totals, 1, stats::quantile, probs = 0.025, na.rm = TRUE),
    total_ci_upper = apply(domain_totals, 1, stats::quantile, probs = 0.975, na.rm = TRUE),
    row.names = NULL,
    stringsAsFactors = FALSE
  )

  return(results)
}


#' @title Bayesian Response Imputation for Missing Outcomes
#' @description Performs Bayesian multiple imputation for missing survey response values (\code{y_i = NA})
#' using posterior predictive draws from a trained LBBNN model. Missing units are identified,
#' and predictive draws are generated conditionally on observed covariates \eqn{\mathbf{x}_i}.
#'
#' @param object A trained \code{lbbnn_net} model.
#' @param newdata A \code{data.frame} containing covariates and target column \code{y} (which may contain \code{NA}s).
#' @param target_col Name or index of the response column in \code{newdata}. Default is the last column.
#' @param draws Integer, number of multiple imputation posterior draws (default 100).
#' @param mpm Logical, whether to use the Median Probability Model for imputation (default FALSE).
#' @param type Character string, either \code{"latent_mean"} (imputes conditional expectation \eqn{\mu_{\theta^{(s)}}(\mathbf{x}_i)},
#' removing measurement error \eqn{\varepsilon_i}) or \code{"predictive"} (imputes \eqn{\mu_{\theta^{(s)}}(\mathbf{x}_i) + \varepsilon_i^{(s)}}
#' with observation noise \eqn{\varepsilon_i \sim \mathcal{N}(0, \sigma^2)}). Default is \code{"predictive"}.
#' @param sigma Optional numeric residual error scale \eqn{\sigma}. If NULL, uses \code{object$sigma} or defaults to 1.0.
#' @param device Character string, compute device ("cpu", "cuda", "mps").
#' @return A list containing:
#' \describe{
#'   \item{imputed_draws}{A matrix of dimension \eqn{n_{\text{mis}} \times \text{draws}} containing the imputed values.}
#'   \item{completed_data}{A copy of \code{newdata} with missing \code{y} values replaced by the posterior mean of the imputations.}
#'   \item{imputed_summary}{A \code{data.frame} summarizing each missing unit: row index, mean, sd, and 95% credible interval.}
#'   \item{missing_indices}{Integer vector of row indices in \code{newdata} that were missing.}
#' }
#' @export
impute_response_lbbnn <- function(
    object,
    newdata,
    target_col = ncol(newdata),
    draws = 100,
    mpm = FALSE,
    type = c("predictive", "latent_mean"),
    sigma = NULL,
    device = "cpu"
) {
  type <- match.arg(type)
  device <- resolve_device(device)

  if (!inherits(newdata, "data.frame")) {
    stop("newdata must be a data.frame")
  }

  y_all <- newdata[[target_col]]
  mis_idx <- which(is.na(y_all))

  if (length(mis_idx) == 0) {
    message("No missing values detected in target column. Returning original data.")
    return(list(
      imputed_draws = matrix(numeric(0), nrow = 0, ncol = draws),
      completed_data = newdata,
      imputed_summary = data.frame(),
      missing_indices = integer(0)
    ))
  }

  # Subset covariates for missing units
  if (is.character(target_col)) {
    covar_cols <- setdiff(names(newdata), target_col)
  } else {
    covar_cols <- setdiff(seq_len(ncol(newdata)), target_col)
  }
  mis_covars <- newdata[mis_idx, covar_cols, drop = FALSE]

  # Generate predictive draws for missing units
  preds <- stats::predict(object, newdata = mis_covars, draws = draws, mpm = mpm, device = device)
  
  if (inherits(preds, "torch_tensor")) {
    preds_mat <- as.matrix(preds$squeeze(-1)$cpu())
  } else {
    preds_mat <- as.matrix(preds)
  }
  if (nrow(preds_mat) == draws) {
    preds_mat <- t(preds_mat)
  }
  # Shape is n_mis x draws

  if (type == "predictive") {
    sigma_val <- if (!is.null(sigma)) sigma else if (!is.null(object$sigma)) object$sigma else 1.0
    noise_mat <- matrix(stats::rnorm(length(mis_idx) * draws, mean = 0, sd = sigma_val),
                        nrow = length(mis_idx), ncol = draws)
    imputed_draws <- preds_mat + noise_mat
  } else {
    imputed_draws <- preds_mat
  }

  imp_mean <- apply(imputed_draws, 1, mean)
  imp_sd   <- apply(imputed_draws, 1, stats::sd)
  imp_q025 <- apply(imputed_draws, 1, stats::quantile, probs = 0.025)
  imp_q975 <- apply(imputed_draws, 1, stats::quantile, probs = 0.975)

  imputed_summary <- data.frame(
    row_id = mis_idx,
    mean = imp_mean,
    sd = imp_sd,
    ci_lower = imp_q025,
    ci_upper = imp_q975,
    stringsAsFactors = FALSE
  )

  completed_data <- newdata
  completed_data[[target_col]][mis_idx] <- imp_mean

  return(list(
    imputed_draws = imputed_draws,
    completed_data = completed_data,
    imputed_summary = imputed_summary,
    missing_indices = mis_idx
  ))
}
