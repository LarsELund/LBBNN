#' @title Compute BGNLM-Inspired Symbolic Complexity Priors
#' @description
#' Computes prior inclusion probabilities \eqn{\pi_{i,j}^{(l)}} per neuron
#' \eqn{i}, incoming connection \eqn{j}, and layer \eqn{l}:
#' \deqn{\pi_{i,j}^{(l)} = \frac{1}{1 + \exp\left(\text{logit}(1 - \pi_0) +
#'   \beta_{\text{depth}} \cdot (l - 1) + \beta_{\text{comp}} \cdot c(g_i^{(l)})\right)}}
#' where \eqn{c(g_i^{(l)})} is the operator complexity cost from
#' \code{\link{get_symbolic_op}}. For input-skip connections directly entering
#' deeper layers or the output layer, \eqn{l_{\text{eff}} = 1} can be used on
#' the skip columns so shallow linear skip paths face lower prior skepticism
#' than deep compositions.
#' @param sizes Integer vector of layer sizes \code{c(p, d_1, ..., d_L, out)}.
#' @param layer_activations_list List of length \code{length(sizes) - 2}
#'   giving the activation specification for each hidden layer.
#' @param base_prior Numeric scalar \eqn{\pi_0 \in (0, 1)}, default \code{0.2}.
#' @param depth_decay Numeric scalar \eqn{\beta_{\text{depth}} \ge 0},
#'   default \code{0.5}.
#' @param complexity_weight Numeric scalar \eqn{\beta_{\text{comp}} \ge 0},
#'   default \code{0.4}.
#' @param input_skip Logical, whether the network uses input-skip connections,
#'   default \code{TRUE}.
#' @return A list of length \code{length(sizes) - 1}, where each element is a
#'   numeric matrix of shape \code{(out_features, in_features)} containing
#'   \eqn{\pi_{i,j}^{(l)} \in (0, 1)}.
#' @export
compute_symbolic_priors <- function(sizes,
                                    layer_activations_list,
                                    base_prior = 0.2,
                                    depth_decay = 0.5,
                                    complexity_weight = 0.4,
                                    input_skip = TRUE) {
  if (!is.numeric(base_prior) || length(base_prior) != 1L ||
      base_prior <= 0 || base_prior >= 1) {
    stop("base_prior must be a scalar strictly between 0 and 1.")
  }
  n_weight_mats <- length(sizes) - 1L
  n_hidden <- length(sizes) - 2L
  if (length(layer_activations_list) != n_hidden) {
    stop(sprintf(
      "layer_activations_list must have length %d (number of hidden layers).",
      n_hidden
    ))
  }

  logit_1_minus_pi0 <- log((1.0 - base_prior) / base_prior)
  p <- as.integer(sizes[1])
  priors_list <- vector("list", n_weight_mats)

  # Hidden layers l = 1, ..., n_hidden
  for (l in seq_len(n_hidden)) {
    in_dim <- if (l == 1L || !isTRUE(input_skip)) {
      as.integer(sizes[l])
    } else {
      as.integer(sizes[l] + p)
    }
    out_dim <- as.integer(sizes[l + 1L])
    parsed <- parse_layer_activations(layer_activations_list[[l]], out_dim)
    costs <- vapply(parsed$act_vec, function(op) get_symbolic_op(op)$cost, numeric(1))

    # Row-wise prior based on layer depth l and operator cost c(g_i^(l))
    eta_row <- logit_1_minus_pi0 + depth_decay * (l - 1.0) + complexity_weight * costs
    pi_row <- 1.0 / (1.0 + exp(eta_row))
    mat <- matrix(rep(pi_row, times = in_dim), nrow = out_dim, ncol = in_dim)

    # If input_skip is enabled in deeper hidden layers (l > 1), direct skip columns
    # have depth 1 rather than depth l
    if (l > 1L && isTRUE(input_skip)) {
      eta_skip <- logit_1_minus_pi0 + complexity_weight * costs
      pi_skip <- 1.0 / (1.0 + exp(eta_skip))
      skip_cols <- seq.int(in_dim - p + 1L, in_dim)
      mat[, skip_cols] <- rep(pi_skip, times = length(skip_cols))
    }
    priors_list[[l]] <- mat
  }

  # Output layer (l = n_weight_mats, linear/id activation cost = 0.5)
  l_out <- n_weight_mats
  in_dim_out <- if (isTRUE(input_skip) && n_hidden >= 1L) {
    as.integer(sizes[n_weight_mats] + p)
  } else {
    as.integer(sizes[n_weight_mats])
  }
  out_dim_out <- as.integer(sizes[n_weight_mats + 1L])
  c_linear <- get_symbolic_op("linear")$cost

  eta_deep <- logit_1_minus_pi0 + depth_decay * (l_out - 1.0) + complexity_weight * c_linear
  pi_deep <- 1.0 / (1.0 + exp(eta_deep))
  mat_out <- matrix(pi_deep, nrow = out_dim_out, ncol = in_dim_out)

  if (isTRUE(input_skip) && n_hidden >= 1L) {
    eta_skip_out <- logit_1_minus_pi0 + complexity_weight * c_linear
    pi_skip_out <- 1.0 / (1.0 + exp(eta_skip_out))
    skip_cols <- seq.int(in_dim_out - p + 1L, in_dim_out)
    mat_out[, skip_cols] <- pi_skip_out
  }
  priors_list[[l_out]] <- mat_out

  priors_list
}

