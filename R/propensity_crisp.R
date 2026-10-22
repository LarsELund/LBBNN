#' @title Composite Bernoulli Negative Log-Likelihood for Unlinked Stacked Sample (STACK)
#' @description Evaluates the composite Bernoulli negative log-likelihood on the unlinked stacked sample
#' \eqn{S = S_c \oplus S_r} of size \eqn{n_c + n_r} without linkage information.
#' The composite probability is \eqn{\pi_{zi} = \pi_{ci} / (\pi_{ci} + \pi_{ri})}.
#' @param input Torch tensor of shape \code{(batch, 1)} containing unconstrained model logits \eqn{\eta_i}.
#' @param target Torch tensor of shape \code{(batch, 2)} where column 1 is \eqn{I_{zi}} (1 for \eqn{S_c}, 0 for \eqn{S_r})
#' and column 2 is \eqn{\pi_{ri}}.
#' @param reduction Character, either \code{"sum"} or \code{"none"}. Default is \code{"sum"}.
#' @return A scalar torch tensor (if reduction is "sum") or 1D tensor of loss contributions.
#' @export
nll_stack <- function(input, target, reduction = "sum") {
  eta <- input$squeeze(-1)
  pi_c <- torch::torch_sigmoid(eta)
  I_z <- target[, 1]
  pi_r <- target[, 2]
  pi_z <- pi_c / (pi_c + pi_r + 1e-12)
  pi_z <- torch::torch_clamp(pi_z, 1e-7, 1 - 1e-7)
  nll_i <- -(I_z * torch::torch_log(pi_z) + (1 - I_z) * torch::torch_log(1 - pi_z))
  if (reduction == "none") nll_i else torch::torch_sum(nll_i)
}

#' @title Composite Bernoulli Negative Log-Likelihood for Deterministically Linked Union (LINK1)
#' @description Evaluates the composite Bernoulli negative log-likelihood on the sample union
#' \eqn{S = S_c \cup (S_r \setminus S_c)} where overlapping units in \eqn{S_r} are removed.
#' The composite probability is \eqn{\pi_{zi} = \pi_{ci} / (\pi_{ci} + \pi_{ri} - \pi_{ci}\pi_{ri})}.
#' @param input Torch tensor of shape \code{(batch, 1)} containing unconstrained model logits \eqn{\eta_i}.
#' @param target Torch tensor of shape \code{(batch, 2)} where column 1 is \eqn{I_{zi}} (1 for \eqn{S_c}, 0 for \eqn{S_r \setminus S_c})
#' and column 2 is \eqn{\pi_{ri}}.
#' @param reduction Character, either \code{"sum"} or \code{"none"}. Default is \code{"sum"}.
#' @return A scalar torch tensor (if reduction is "sum") or 1D tensor of loss contributions.
#' @export
nll_link1 <- function(input, target, reduction = "sum") {
  eta <- input$squeeze(-1)
  pi_c <- torch::torch_sigmoid(eta)
  I_z <- target[, 1]
  pi_r <- target[, 2]
  denom <- pi_c + pi_r - (pi_c * pi_r) + 1e-12
  pi_z <- pi_c / denom
  pi_z <- torch::torch_clamp(pi_z, 1e-7, 1 - 1e-7)
  nll_i <- -(I_z * torch::torch_log(pi_z) + (1 - I_z) * torch::torch_log(1 - pi_z))
  if (reduction == "none") nll_i else torch::torch_sum(nll_i)
}

#' @title Composite Bernoulli Negative Log-Likelihood for Deterministically Linked Stacked Sample (LINK2)
#' @description Evaluates the composite Bernoulli negative log-likelihood on the stacked sample
#' \eqn{S = S_c \oplus S_r} where overlapping units in \eqn{S_r} are assigned \eqn{I_{zi} = 1}.
#' The composite probability is \eqn{\pi_{zi} = \pi_{ci}(1 + \pi_{ri}) / (\pi_{ci} + \pi_{ri})}.
#' @param input Torch tensor of shape \code{(batch, 1)} containing unconstrained model logits \eqn{\eta_i}.
#' @param target Torch tensor of shape \code{(batch, 2)} where column 1 is grouping indicator \eqn{I_{zi}}
#' (1 for \eqn{S_c} and \eqn{S_c \cap S_r}, 0 for \eqn{S_r \setminus S_c}) and column 2 is \eqn{\pi_{ri}}.
#' @param reduction Character, either \code{"sum"} or \code{"none"}. Default is \code{"sum"}.
#' @return A scalar torch tensor (if reduction is "sum") or 1D tensor of loss contributions.
#' @export
nll_link2 <- function(input, target, reduction = "sum") {
  eta <- input$squeeze(-1)
  pi_c <- torch::torch_sigmoid(eta)
  I_z <- target[, 1]
  pi_r <- target[, 2]
  pi_z <- (pi_c * (1 + pi_r)) / (pi_c + pi_r + 1e-12)
  pi_z <- torch::torch_clamp(pi_z, 1e-7, 1 - 1e-7)
  nll_i <- -(I_z * torch::torch_log(pi_z) + (1 - I_z) * torch::torch_log(1 - pi_z))
  if (reduction == "none") nll_i else torch::torch_sum(nll_i)
}

#' @title Factory for Missing Information Principle (MIP) Composite Negative Log-Likelihood
#' @description Constructs a composite loss function for probabilistically linked stacked samples
#' \eqn{S = S_c \oplus S_r} with linkage informativeness \eqn{p_{\text{link}} \in [0, 1]}.
#' Under the Missing Information Principle, the soft target indicator for \eqn{S_r} units is
#' \eqn{p_i = p_{\text{link}} I_{ci} + (1 - p_{\text{link}}) \hat{\pi}_{ci}}.
#' @param p_link Numeric between 0 and 1 specifying linkage informativeness.
#' Default is 0.5 (intermediate probabilistic linkage). When 1, coincides with exact linkage (LINK2);
#' when 0, represents unlinked data.
#' @return A closure function accepting \code{(input, target, reduction = "sum")}.
#' @export
make_nll_mip <- function(p_link = 0.5) {
  function(input, target, reduction = "sum") {
    eta <- input$squeeze(-1)
    pi_c <- torch::torch_sigmoid(eta)
    is_Sc <- target[, 1]
    I_c_true <- target[, 2]
    pi_r <- target[, 3]

    # Expected overlap indicator under Missing Information Principle
    p_i <- (p_link * I_c_true) + ((1 - p_link) * pi_c$detach())
    I_z_mip <- is_Sc + (1 - is_Sc) * p_i

    pi_z <- (pi_c * (1 + pi_r)) / (pi_c + pi_r + 1e-12)
    pi_z <- torch::torch_clamp(pi_z, 1e-7, 1 - 1e-7)
    nll_i <- -(I_z_mip * torch::torch_log(pi_z) + (1 - I_z_mip) * torch::torch_log(1 - pi_z))
    if (reduction == "none") nll_i else torch::torch_sum(nll_i)
  }
}

#' @title Pseudo-Log-Likelihood Negative Loss (Chen, Li & Wu 2020)
#' @description Evaluates the negative pseudo-log-likelihood loss using design survey weights \eqn{w_{ri} = 1/\pi_{ri}}
#' on the reference sample \eqn{S_r}.
#' \eqn{\ell_i = \text{is\_Sc}_i \cdot \eta_i - (1 - \text{is\_Sc}_i) \cdot (1 / \pi_{ri}) \cdot \log(1 + e^{\eta_i})}.
#' @param input Torch tensor of shape \code{(batch, 1)} containing unconstrained model logits \eqn{\eta_i}.
#' @param target Torch tensor of shape \code{(batch, 2)} where column 1 is indicator \code{is_Sc} (1 for \eqn{S_c}, 0 for \eqn{S_r})
#' and column 2 is \eqn{\pi_{ri}}.
#' @param reduction Character, either \code{"sum"} or \code{"none"}. Default is \code{"sum"}.
#' @return A scalar torch tensor (if reduction is "sum") or 1D tensor of loss contributions.
#' @export
nll_pseudo <- function(input, target, reduction = "sum") {
  eta <- input$squeeze(-1)
  is_Sc <- target[, 1]
  pi_r <- target[, 2]
  # Numerically stable softplus formulation
  nll_i <- -is_Sc * eta + (1 - is_Sc) * (1 / (pi_r + 1e-12)) * torch::nnf_softplus(eta)
  if (reduction == "none") nll_i else torch::torch_sum(nll_i)
}

#' @title Helper to solve for intercept guaranteeing target sample size
#' @description Solves \eqn{\sum_{i=1}^N \text{logit}^{-1}(\beta_0 + \eta_i) = n} using root finding.
#' @param eta Numeric vector of linear or non-linear predictor values across the population.
#' @param n Target sample size.
#' @param interval Numeric vector of length 2 giving the search bracket. Default \code{c(-15, 15)}.
#' @return Numeric root \eqn{\beta_0}.
#' @keywords internal
approx_int <- function(eta, n, interval = c(-15, 15)) {
  f <- function(b0) {
    sum(stats::plogis(b0 + eta)) - n
  }
  res <- stats::uniroot(f, interval = interval)
  res$root
}

#' @title Generate Finite Population with Linear or Non-Linear Covariates and Outcomes
#' @description Generates a synthetic finite population of size \eqn{N} with \eqn{p} multivariate normal covariates,
#' outcome variable \eqn{Y}, reference sample inclusion probabilities \eqn{\pi_{ri}}, and convenience sample
#' response propensities \eqn{\pi_{ci}}. Supports both baseline linear specifications (Beresovsky et al.)
#' and non-linear specifications with uninformative noise features.
#' @param N Integer, population size. Default is 10,000.
#' @param p Integer, total number of covariates. Default is 10 (with \eqn{X_4, \dots, X_{10}} being pure noise).
#' @param nonlinear_y Logical, whether the outcome \eqn{Y} includes non-linear polynomial and trigonometric terms. Default is FALSE.
#' @param nonlinear_propensity Logical, whether the convenience propensity \eqn{\pi_{ci}} includes non-linear interactions and curves. Default is FALSE.
#' @param common_support Character, either \code{"high"} (\eqn{\beta_1 = 1.0, \beta_2 = -0.3}) or \code{"low"} (\eqn{\beta_1 = -1.0, \beta_2 = -0.3}). Default is \code{"high"}.
#' @param b2 Optional numeric coefficient for X2 in convenience propensity. Default NULL (uses -0.5 for high support, -0.3 for low support).
#' @param n_r Target reference sample size \eqn{n_r}. Default is 1,000.
#' @param n_c Target convenience sample size \eqn{n_c}. Default is 1,000.
#' @param seed Integer seed for reproducibility. Default is 100.
#' @return A \code{data.frame} containing the generated finite population with columns \code{X1, ..., Xp, y, pi.r, pi.c}.
#' @export
gen_pop_nonlinear <- function(
    N = 10000,
    p = 10,
    nonlinear_y = FALSE,
    nonlinear_propensity = FALSE,
    common_support = c("high", "low"),
    b2 = NULL,
    n_r = 1000,
    n_c = 1000,
    seed = 100
) {
  common_support <- match.arg(common_support)
  if (!is.null(seed)) set.seed(seed)

  # 1. Generate Covariates X ~ MVN(0, Sigma)
  # Vladislav 3x3 covariance block for X1, X2, X3; independent standard normals for X4..Xp
  SigmaX <- diag(1, p)
  SigmaX[1, 2] <- SigmaX[2, 1] <- 0.5
  SigmaX[1, 3] <- SigmaX[3, 1] <- 0.3
  SigmaX[2, 3] <- SigmaX[3, 2] <- 0.3

  X <- MASS::mvrnorm(n = N, mu = rep(0, p), Sigma = SigmaX)
  colnames(X) <- paste0("X", 1:p)

  # 2. Outcome Y
  if (!nonlinear_y) {
    # Baseline linear mode: Y = 1 + X1 + X2 + X3 + eps
    mu_y <- 1.0 + X[, "X1"] + X[, "X2"] + X[, "X3"]
  } else {
    # Non-linear mode: polynomial + sine interaction
    mu_y <- 1.0 + X[, "X1"] + X[, "X2"] + X[, "X3"] + 0.5 * (X[, "X1"]^2) - 0.5 * sin(pi * X[, "X2"])
  }
  y <- stats::rnorm(N, mean = mu_y, sd = 1.0)

  # 3. Probability Reference Sample Inclusion Probabilities pi.r
  # Measure of size based on X1, X2
  sz_pr <- stats::plogis(X[, "X1"] + X[, "X2"])
  pi_r <- sampling::inclusionprobabilities(sz_pr, n_r)

  # 4. Convenience Sample Response Propensity pi.c
  b1 <- if (common_support == "high") 1.0 else -1.0
  if (is.null(b2)) {
    b2 <- if (common_support == "high") -0.5 else -0.3
  }

  if (!nonlinear_propensity) {
    # Linear propensity: eta_c = b1*X1 + b2*X2
    eta_c_raw <- b1 * X[, "X1"] + b2 * X[, "X2"]
  } else {
    # Non-linear propensity: interaction + cosine + quadratic
    eta_c_raw <- b1 * X[, "X1"] + b2 * X[, "X2"] +
      0.6 * X[, "X1"] * X[, "X2"] +
      0.5 * cos(pi * X[, "X1"]) -
      0.4 * (X[, "X2"]^2)
  }

  b0 <- approx_int(eta_c_raw, n = n_c)
  pi_c <- stats::plogis(b0 + eta_c_raw)

  pop_df <- as.data.frame(X)
  pop_df$y <- y
  pop_df$pi.r <- pi_r
  pop_df$pi.c <- pi_c
  pop_df$id <- 1:N

  return(pop_df)
}

#' @title Draw Combined Probability and Non-Probability Samples from Population
#' @description Draws selection indicator \code{I.r} using systematic PPS sampling
#' and \code{I.c} using Bernoulli sampling with parameter \code{pi.c}.
#' @param pop Data frame returned by \code{\link{gen_pop_nonlinear}}.
#' @param seed Optional integer seed.
#' @return The input data frame with added columns \code{I.r} and \code{I.c}.
#' @export
draw_combined_samples <- function(pop, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  pop$I.r <- 0
  sel_r <- which(sampling::UPrandomsystematic(pop$pi.r) == 1)
  pop$I.r[sel_r] <- 1
  pop$I.c <- stats::rbinom(nrow(pop), 1, pop$pi.c)
  return(pop)
}

#' @title Construct Training Dataset for Composite Propensity Estimators
#' @description Formats covariates and target matrices for the 5 composite likelihood methods
#' (\code{"STACK"}, \code{"LINK1"}, \code{"LINK2"}, \code{"MIP"}, \code{"PSEUDO"}).
#' @param pop Data frame with drawn samples (\code{I.c} and \code{I.r}).
#' @param method Character string: \code{"STACK"}, \code{"LINK1"}, \code{"LINK2"}, \code{"MIP"}, or \code{"PSEUDO"}.
#' @param covar_cols Character vector of covariate column names. Default is \code{paste0("X", 1:10)}.
#' @return A list containing:
#'   \item{data}{A \code{data.frame} with covariates and the target columns packed at the end.}
#'   \item{target_cols}{Integer number of target columns (2 for STACK/LINK1/LINK2/PSEUDO, 3 for MIP).}
#'   \item{Sc_indices}{Row indices in \code{data} corresponding to the convenience sample \eqn{S_c}.}
#' @export
build_propensity_dataset <- function(
    pop,
    method = c("STACK", "LINK1", "LINK2", "MIP", "PSEUDO"),
    covar_cols = paste0("X", 1:10)
) {
  method <- match.arg(method)
  Sc <- pop[pop$I.c == 1, ]
  Sr <- pop[pop$I.r == 1, ]

  p <- length(covar_cols)

  if (method == "STACK") {
    # Sc: I_z = 1, Sr: I_z = 0
    df_c <- data.frame(Sc[, covar_cols, drop = FALSE], I_z = 1, pi_r = Sc$pi.r)
    df_r <- data.frame(Sr[, covar_cols, drop = FALSE], I_z = 0, pi_r = Sr$pi.r)
    combined <- rbind(df_c, df_r)
    return(list(data = combined, target_cols = 2, Sc_indices = 1:nrow(df_c)))

  } else if (method == "LINK1") {
    # Union Sc U (Sr \ Sc). Sr units that also belong to Sc are omitted from Sr.
    Sr_not_c <- Sr[Sr$I.c == 0, ]
    df_c <- data.frame(Sc[, covar_cols, drop = FALSE], I_z = 1, pi_r = Sc$pi.r)
    df_r <- data.frame(Sr_not_c[, covar_cols, drop = FALSE], I_z = 0, pi_r = Sr_not_c$pi.r)
    combined <- rbind(df_c, df_r)
    return(list(data = combined, target_cols = 2, Sc_indices = 1:nrow(df_c)))

  } else if (method == "LINK2") {
    # Stacked Sc (+) Sr. I_z = 1 for all Sc, AND I_z = 1 for Sr units where I.c == 1.
    df_c <- data.frame(Sc[, covar_cols, drop = FALSE], I_z = 1, pi_r = Sc$pi.r)
    df_r <- data.frame(Sr[, covar_cols, drop = FALSE], I_z = as.numeric(Sr$I.c == 1), pi_r = Sr$pi.r)
    combined <- rbind(df_c, df_r)
    return(list(data = combined, target_cols = 2, Sc_indices = 1:nrow(df_c)))

  } else if (method == "MIP") {
    # Stacked Sc (+) Sr with 3-column target: is_Sc, I_c_true, pi_r
    df_c <- data.frame(Sc[, covar_cols, drop = FALSE], is_Sc = 1, I_c_true = 1, pi_r = Sc$pi.r)
    df_r <- data.frame(Sr[, covar_cols, drop = FALSE], is_Sc = 0, I_c_true = as.numeric(Sr$I.c == 1), pi_r = Sr$pi.r)
    combined <- rbind(df_c, df_r)
    return(list(data = combined, target_cols = 3, Sc_indices = 1:nrow(df_c)))

  } else if (method == "PSEUDO") {
    # Stacked Sc (+) Sr: is_Sc = 1 for Sc, 0 for Sr; pi_r
    df_c <- data.frame(Sc[, covar_cols, drop = FALSE], is_Sc = 1, pi_r = Sc$pi.r)
    df_r <- data.frame(Sr[, covar_cols, drop = FALSE], is_Sc = 0, pi_r = Sr$pi.r)
    combined <- rbind(df_c, df_r)
    return(list(data = combined, target_cols = 2, Sc_indices = 1:nrow(df_c)))
  }
}

#' @title Compute Hajek Estimate of Population Mean
#' @description Computes the self-normalizing Hajek estimator \eqn{\hat{\bar{Y}}_{\text{Hajek}} = \frac{\sum_{i \in S_c} w_i y_i}{\sum_{i \in S_c} w_i}}
#' using estimated propensity weights \eqn{w_i = 1 / \hat{\pi}_{ci}}.
#' @param y Numeric vector of observed responses \eqn{y_i} in the convenience sample \eqn{S_c}.
#' @param pi_hat Numeric vector of estimated participation propensities \eqn{\hat{\pi}_{ci}} in \eqn{S_c}.
#' @param clip_min Numeric minimum bound for \eqn{\hat{\pi}_{ci}} to prevent division by zero or extreme weight inflation. Default 1e-4.
#' @return Numeric scalar Hajek estimate of the population mean.
#' @export
hajek_estimate <- function(y, pi_hat, clip_min = 1e-4) {
  pi_clipped <- pmax(as.numeric(pi_hat), clip_min)
  weights <- 1.0 / pi_clipped
  sum(weights * y) / sum(weights)
}
