#' @title Heterogeneous Neuro-Symbolic LBBNN Linear Layer
#' @description
#' Implements a feed-forward LBBNN layer with per-neuron symbolic activations
#' (including canonical positive-input log \code{"clog"}, protected algebraic
#' operators \code{"exp"}, \code{"log"}, \code{"sqrt"}, \code{"inv"},
#' \code{"square"}, \code{"sin"}, \code{"cos"}, \code{"id"}, and differentiable
#' Boolean gates \code{"logic_and"}, \code{"logic_or"} with bimodal literal
#' priors).
#' @param in_features Integer, number of input features to this layer.
#' @param out_features Integer, number of output neurons in this layer.
#' @param layer_activations Character vector of length \code{out_features} or
#'   named list of operator counts (e.g., \code{list(clog = 4, exp = 4, id = 2)}).
#' @param prior_inclusion Numeric scalar, vector of length \code{out_features},
#'   or matrix of shape \code{(out_features, in_features)} giving prior
#'   inclusion probabilities \eqn{\pi_{i,j}}.
#' @param standard_prior Numeric scalar, prior standard deviation for standard
#'   weights and biases.
#' @param density_init Numeric vector of length 2 for initializing
#'   \code{lambda_l}.
#' @param input_skip_dim Optional integer giving the number of raw input
#'   variables \eqn{p} appended at the end of the layer input when
#'   \code{input_skip = TRUE} in deeper layers.
#' @param is_first_layer Logical, whether this is the first hidden layer.
#' @param sigma_logic Numeric scalar \eqn{\sigma_L}, standard deviation for the
#'   bimodal literal mixture prior \eqn{\frac{1}{2}\mathcal{N}(+1, \sigma_L^2) +
#'   \frac{1}{2}\mathcal{N}(-1, \sigma_L^2)} on logic neurons. Default \code{0.25}.
#' @param tau_logic Numeric scalar \eqn{\tau_L}, temperature for smooth logic
#'   literal sigmoids. Default \code{10.0}.
#' @param logic_mode Character, \code{"product_tnorm"} (default) or
#'   \code{"lukasiewicz"}.
#' @param use_concrete Logical, whether to sample Concrete / Gumbel-Softmax
#'   relaxed gates during training. Default \code{TRUE}.
#' @param flow Logical, whether to use normalizing flows (default \code{FALSE}).
#' @param num_transforms Integer, number of flow transforms.
#' @param hidden_dims Integer vector of flow hidden dimensions.
#' @param device Character, compute device (\code{"cpu"}, \code{"cuda"},
#'   \code{"mps"}).
#' @param bias_inclusion_prob Logical, whether bias has inclusion probabilities.
#' @param eps Numeric stability constant, default \code{1e-6}.
#' @return A \code{torch::nn_module} object of class \code{lbbnn_symbolic_linear}.
#' @export
lbbnn_symbolic_linear <- torch::nn_module(
  "lbbnn_symbolic_linear",
  initialize = function(in_features,
                        out_features,
                        layer_activations,
                        prior_inclusion = 0.25,
                        standard_prior = 1.0,
                        density_init = c(-2, 2),
                        input_skip_dim = NULL,
                        is_first_layer = TRUE,
                        sigma_logic = 0.25,
                        tau_logic = 10.0,
                        logic_mode = "product_tnorm",
                        use_concrete = TRUE,
                        flow = FALSE,
                        num_transforms = 2,
                        hidden_dims = c(200, 200),
                        device = "cpu",
                        bias_inclusion_prob = FALSE,
                        eps = 1e-6) {
    self$in_features <- as.integer(in_features)
    self$out_features <- as.integer(out_features)
    self$device <- device
    self$flow <- flow
    self$bias_inclusion_prob <- bias_inclusion_prob
    self$conv_net <- FALSE
    self$num_transforms <- num_transforms
    self$hidden_dims <- hidden_dims
    self$density_init <- density_initialization(density_init[1], density_init[2])
    self$is_first_layer <- isTRUE(is_first_layer)
    self$input_skip_dim <- if (is.null(input_skip_dim)) self$in_features else as.integer(input_skip_dim)
    self$sigma_logic <- sigma_logic
    self$tau_logic <- tau_logic
    self$logic_mode <- logic_mode
    self$use_concrete <- isTRUE(use_concrete)
    self$temperature <- 0.5
    self$debias_mode <- FALSE
    self$deterministic_mpm <- FALSE
    self$eps <- eps
    self$snapped_rationals <- list()

    # Parse heterogeneous activation specification
    parsed <- parse_layer_activations(layer_activations, self$out_features)
    self$act_vec <- parsed$act_vec
    self$op_groups <- parsed$groups

    # Identify row types
    self$clog_rows <- which(self$act_vec == "clog")
    self$logic_rows <- which(self$act_vec %in% c("logic_and", "logic_or"))
    self$standard_rows <- setdiff(seq_len(self$out_features), c(self$clog_rows, self$logic_rows))

    # Map each clog neuron to its dedicated positive input column
    p_raw <- self$input_skip_dim
    raw_col_offset <- if (self$is_first_layer) 0L else (self$in_features - p_raw)
    self$clog_input_cols <- integer(length(self$clog_rows))
    if (length(self$clog_rows) > 0L) {
      for (m in seq_along(self$clog_rows)) {
        col_idx <- raw_col_offset + ((m - 1L) %% p_raw) + 1L
        self$clog_input_cols[m] <- as.integer(col_idx)
      }
    }

    # Variational parameters
    self$weight_mean <- torch::nn_parameter(
      torch::torch_empty(self$out_features, self$in_features, device = device)
    )
    self$weight_rho <- torch::nn_parameter(
      torch::torch_empty(self$out_features, self$in_features, device = device)
    )
    self$weight_sigma <- torch::torch_empty(self$out_features, self$in_features, device = device)

    self$bias_mean <- torch::nn_parameter(
      torch::torch_empty(self$out_features, device = device)
    )
    self$bias_rho <- torch::nn_parameter(
      torch::torch_empty(self$out_features, device = device)
    )
    self$bias_sigma <- torch::torch_empty(self$out_features, device = device)

    self$lambda_l <- torch::nn_parameter(
      torch::torch_empty(self$out_features, self$in_features, device = device)
    )
    self$alpha <- torch::torch_empty(self$out_features, self$in_features, device = device)
    self$alpha_active_path <- torch::torch_zeros(self$out_features, self$in_features, device = device)

    # Prior inclusion probabilities (scalar, vector, or matrix)
    if (inherits(prior_inclusion, "torch_tensor")) {
      self$alpha_prior <- prior_inclusion$to( dtype = torch::torch_float32(), device = device)
    } else if (is.matrix(prior_inclusion)) {
      self$alpha_prior <- torch::torch_tensor(
        prior_inclusion, dtype = torch::torch_float32(), device = device
      )
    } else if (is.numeric(prior_inclusion) && length(prior_inclusion) == self$out_features) {
      mat_p <- matrix(rep(prior_inclusion, times = self$in_features),
                      nrow = self$out_features, ncol = self$in_features)
      self$alpha_prior <- torch::torch_tensor(mat_p, dtype = torch::torch_float32(), device = device)
    } else {
      self$alpha_prior <- torch::torch_zeros(self$out_features, self$in_features, device = device) +
        alpha_prior(prior_inclusion)
    }

    if (self$bias_inclusion_prob) {
      self$bias_lambda_l <- torch::nn_parameter(torch::torch_empty(self$out_features, device = device))
      self$bias_alpha <- torch::torch_empty(self$out_features, device = device)
      self$bias_alpha_prior <- torch::torch_mean(self$alpha_prior, dim = 2)
    }

    # Weight and bias priors
    self$weight_mean_prior <- torch::torch_zeros(self$out_features, self$in_features, device = device)
    self$weight_sigma_prior <- torch::torch_zeros(self$out_features, self$in_features, device = device) +
      std_prior(standard_prior)
    self$bias_mean_prior <- torch::torch_zeros(self$out_features, device = device)
    self$bias_sigma_prior <- torch::torch_zeros(self$out_features, device = device) +
      std_prior(standard_prior)

    # Gradient / parameter freeze masks for clog and logic neurons
    w_mask <- matrix(1.0, nrow = self$out_features, ncol = self$in_features)
    b_mask <- rep(1.0, self$out_features)
    lam_mask <- matrix(1.0, nrow = self$out_features, ncol = self$in_features)

    if (length(self$clog_rows) > 0L) {
      w_mask[self$clog_rows, ] <- 0.0
      b_mask[self$clog_rows] <- 0.0
      lam_mask[self$clog_rows, ] <- 0.0
      for (m in seq_along(self$clog_rows)) {
        lam_mask[self$clog_rows[m], self$clog_input_cols[m]] <- 1.0
      }
    }
    if (length(self$logic_rows) > 0L) {
      # Logic gates do not use an additive bias
      b_mask[self$logic_rows] <- 0.0
    }

    self$weight_grad_mask <- torch::torch_tensor(w_mask, dtype = torch::torch_float32(), device = device)
    self$bias_grad_mask <- torch::torch_tensor(b_mask, dtype = torch::torch_float32(), device = device)
    self$lambda_grad_mask <- torch::torch_tensor(lam_mask, dtype = torch::torch_float32(), device = device)

    self$last_domain_loss <- torch::torch_tensor(0.0, device = device)
    self$reset_parameters()
  },

  reset_parameters = function() {
    torch::with_no_grad({
      torch::nn_init_uniform_(self$weight_mean, -0.1, 0.1)
      torch::nn_init_normal_(self$weight_rho, mean = -6.0, std = 0.5)
      torch::nn_init_uniform_(self$bias_mean, -0.05, 0.05)
      torch::nn_init_normal_(self$bias_rho, mean = -6.0, std = 0.5)
      torch::nn_init_uniform_(self$lambda_l, self$density_init[1], self$density_init[2])

      # Initialize logic neurons in the high-gradient region so literal signs can freely polarize to +/- 1
      if (length(self$logic_rows) > 0L) {
        n_log <- length(self$logic_rows)
        w_log <- matrix(
          runif(n_log * self$in_features, min = -0.5, max = 0.5),
          nrow = n_log, ncol = self$in_features
        )
        self$weight_mean[self$logic_rows, ]$copy_(
          torch::torch_tensor(w_log, dtype = torch::torch_float32(), device = self$device)
        )
        self$bias_mean[self$logic_rows]$fill_(0.0)
        self$bias_rho[self$logic_rows]$fill_(-10.0)
      }

      # Enforce frozen canonical log (clog) parameters: w = 1 on dedicated input, b = 0
      if (length(self$clog_rows) > 0L) {
        self$weight_mean[self$clog_rows, ]$fill_(0.0)
        self$weight_rho[self$clog_rows, ]$fill_(-10.0)
        self$bias_mean[self$clog_rows]$fill_(0.0)
        self$bias_rho[self$clog_rows]$fill_(-10.0)
        for (m in seq_along(self$clog_rows)) {
          r_idx <- self$clog_rows[m]
          c_idx <- self$clog_input_cols[m]
          self$weight_mean[r_idx, c_idx]$fill_(1.0)
          # Non-dedicated incoming edges for clog are permanently off
          for (j in seq_len(self$in_features)) {
            if (j != c_idx) {
              self$lambda_l[r_idx, j]$fill_(-20.0)
            }
          }
        }
      }
    })
  },

  enforce_constraints = function() {
    torch::with_no_grad({
      if (length(self$clog_rows) > 0L) {
        self$weight_mean[self$clog_rows, ]$fill_(0.0)
        self$weight_rho[self$clog_rows, ]$fill_(-10.0)
        self$bias_mean[self$clog_rows]$fill_(0.0)
        self$bias_rho[self$clog_rows]$fill_(-10.0)
        for (m in seq_along(self$clog_rows)) {
          r_idx <- self$clog_rows[m]
          c_idx <- self$clog_input_cols[m]
          self$weight_mean[r_idx, c_idx]$fill_(1.0)
          for (j in seq_len(self$in_features)) {
            if (j != c_idx) {
              self$lambda_l[r_idx, j]$fill_(-20.0)
            }
          }
        }
      }
      if (length(self$logic_rows) > 0L) {
        self$bias_mean[self$logic_rows]$fill_(0.0)
        self$bias_rho[self$logic_rows]$fill_(-10.0)
      }
    })
  },

  mask_frozen_grads = function() {
    torch::with_no_grad({
      if (!is.null(self$weight_mean$grad) &&
          length(dim(self$weight_mean$grad)) > 0L) {
        self$weight_mean$grad$mul_(self$weight_grad_mask)
      }
      if (!is.null(self$weight_rho$grad) &&
          length(dim(self$weight_rho$grad)) > 0L) {
        self$weight_rho$grad$mul_(self$weight_grad_mask)
      }
      if (!is.null(self$bias_mean$grad) &&
          length(dim(self$bias_mean$grad)) > 0L) {
        self$bias_mean$grad$mul_(self$bias_grad_mask)
      }
      if (!is.null(self$bias_rho$grad) &&
          length(dim(self$bias_rho$grad)) > 0L) {
        self$bias_rho$grad$mul_(self$bias_grad_mask)
      }
      if (!is.null(self$lambda_l$grad) &&
          length(dim(self$lambda_l$grad)) > 0L) {
        self$lambda_l$grad$mul_(self$lambda_grad_mask)
      }
      if (isTRUE(self$debias_mode)) {
        if (!is.null(self$weight_mean$grad) &&
            length(dim(self$weight_mean$grad)) > 0L) {
          self$weight_mean$grad$mul_(self$alpha_active_path * self$weight_grad_mask)
        }
        if (!is.null(self$weight_rho$grad) &&
            length(dim(self$weight_rho$grad)) > 0L) {
          self$weight_rho$grad$zero_()
        }
        if (!is.null(self$bias_rho$grad) &&
            length(dim(self$bias_rho$grad)) > 0L) {
          self$bias_rho$grad$zero_()
        }
        if (!is.null(self$lambda_l$grad) &&
            length(dim(self$lambda_l$grad)) > 0L) {
          self$lambda_l$grad$zero_()
        }
        if (!is.null(self$bias_mean$grad) &&
            length(dim(self$bias_mean$grad)) > 0L) {
          active_row_mask <- (torch::torch_sum(self$alpha_active_path, dim = 2) > 0)$to(
            dtype = torch::torch_float32()
          )
          self$bias_mean$grad$mul_(active_row_mask * self$bias_grad_mask)
        }
      }
    })
  },

  sample_gate = function(MPM = FALSE) {
    self$alpha <- torch::torch_sigmoid(self$lambda_l)
    if (isTRUE(self$debias_mode) || isTRUE(MPM)) {
      return(self$alpha_active_path)
    }
    if (isTRUE(self$use_concrete) && isTRUE(self$training)) {
      u <- torch::torch_rand_like(self$lambda_l)$clamp(min = 1e-6, max = 1.0 - 1e-6)
      logistic_noise <- torch::torch_log(u) - torch::torch_log(1.0 - u)
      tau <- max(self$temperature, 0.01)
      gate <- torch::torch_sigmoid((self$lambda_l + logistic_noise) / tau)
      # Ensure non-dedicated clog gates stay 0
      gate <- gate * self$lambda_grad_mask
      return(gate)
    }
    self$alpha * self$lambda_grad_mask
  },

  forward = function(input, MPM = FALSE) {
    if (length(dim(input)) == 1L) {
      input <- input$unsqueeze(1)
    }
    self$alpha <- torch::torch_sigmoid(self$lambda_l) * self$lambda_grad_mask
    if (self$bias_inclusion_prob) {
      self$bias_alpha <- torch::torch_sigmoid(self$bias_lambda_l)
    }
    self$weight_sigma <- torch::torch_log1p(torch::torch_exp(self$weight_rho))
    self$bias_sigma <- torch::torch_log1p(torch::torch_exp(self$bias_rho))
    self$last_domain_loss <- torch::torch_tensor(0.0, device = self$device)

    gate_mat <- self$sample_gate(MPM = MPM)
    out_chunks <- vector("list", length(self$op_groups))

    for (g_idx in seq_along(self$op_groups)) {
      grp <- self$op_groups[[g_idx]]
      idx <- grp$indices
      op_name <- grp$op
      op_spec <- get_symbolic_op(op_name)

      if (op_spec$type == "clog") {
        # Canonical positive input log: w = 1, b = 0, gated by learnable gamma_{i, c_i}
        clog_cols <- self$clog_input_cols[match(idx, self$clog_rows)]
        x_pos <- input[, clog_cols, drop = FALSE]
        log_x <- clog(x_pos, eps = self$eps)
        # Extract dedicated gate for each clog neuron in this group
        g_vec <- vector("list", length(idx))
        for (ii in seq_along(idx)) {
          g_vec[[ii]] <- gate_mat[idx[ii], clog_cols[ii]]$unsqueeze(1)
        }
        g_diag <- torch::torch_cat(g_vec, dim = 1)$unsqueeze(1) # (1, n_clog)
        out_chunks[[g_idx]] <- log_x * g_diag

      } else if (op_spec$type == "logic") {
        w_mean_sub <- self$weight_mean[idx, , drop = FALSE]
        w_sig_sub <- self$weight_sigma[idx, , drop = FALSE]
        g_sub <- gate_mat[idx, , drop = FALSE]

        if (isTRUE(self$debias_mode) || isTRUE(MPM)) {
          out_chunks[[g_idx]] <- op_spec$forward(
            input, weight = w_mean_sub, gate = g_sub, MPM = TRUE,
            tau_L = self$tau_logic, mode = self$logic_mode, eps = self$eps
          )
        } else {
          w_sub <- if (isTRUE(self$training)) {
            w_mean_sub + w_sig_sub * torch::torch_randn_like(w_sig_sub)
          } else {
            w_mean_sub
          }
          out_chunks[[g_idx]] <- op_spec$forward(
            input, weight = w_sub, gate = g_sub, MPM = FALSE,
            tau_L = self$tau_logic, mode = self$logic_mode, eps = self$eps
          )
        }

      } else {
        # Standard unary / algebraic symbolic operators
        w_mean_sub <- self$weight_mean[idx, , drop = FALSE]
        w_sig_sub <- self$weight_sigma[idx, , drop = FALSE]
        b_mean_sub <- self$bias_mean[idx]
        b_sig_sub <- self$bias_sigma[idx]
        g_sub <- gate_mat[idx, , drop = FALSE]
        alpha_sub <- self$alpha[idx, , drop = FALSE]

        if (isTRUE(self$debias_mode)) {
          w_eff <- w_mean_sub * self$alpha_active_path[idx, , drop = FALSE]
          b_eff <- b_mean_sub
          z_sub <- torch::torch_matmul(input, torch::torch_t(w_eff)) + b_eff
        } else if (isTRUE(MPM)) {
          if (isTRUE(self$deterministic_mpm)) {
            w_samp <- w_mean_sub
            b_samp <- b_mean_sub
          } else {
            w_samp <- torch::torch_normal(w_mean_sub, w_sig_sub)
            b_samp <- torch::torch_normal(b_mean_sub, b_sig_sub)
          }
          w_eff <- w_samp * self$alpha_active_path[idx, , drop = FALSE]
          if (self$bias_inclusion_prob) {
            b_samp <- b_samp * (self$bias_alpha[idx] > 0.5)
          }
          z_sub <- torch::torch_matmul(input, torch::torch_t(w_eff)) + b_samp
        } else {
          # Smooth training: for nonlinear exponential/symbolic operators, avoid
          # double-counting discrete Bernoulli variance (1 - alpha)*w^2 inside exp()
          e_w <- w_mean_sub * g_sub
          if (op_name == "exp") {
            var_w <- alpha_sub * (w_sig_sub^2)
          } else {
            var_w <- alpha_sub * (w_sig_sub^2 + (1.0 - alpha_sub) * w_mean_sub^2)
          }
          if (self$bias_inclusion_prob) {
            b_alp <- self$bias_alpha[idx]
            e_b_alpha <- b_mean_sub * b_alp
            var_b_alpha <- b_alp * (b_sig_sub^2 + (1.0 - b_alp) * b_mean_sub^2)
          } else {
            e_b_alpha <- b_mean_sub
            var_b_alpha <- b_sig_sub^2
          }
          e_b <- torch::torch_matmul(input, torch::torch_t(e_w)) + e_b_alpha
          var_b <- torch::torch_matmul(input^2, torch::torch_t(var_w)) + var_b_alpha
          eps_noise <- if (isTRUE(self$training)) torch::torch_randn_like(var_b) else torch::torch_zeros_like(var_b)
          z_sub <- e_b + torch::torch_sqrt(var_b + 1e-12) * eps_noise
        }

        if (op_name == "exp") {
          self$last_domain_loss <- self$last_domain_loss + compute_exp_domain_penalty(z_sub, limit = 12.0)
        }
        out_chunks[[g_idx]] <- op_spec$forward(z_sub, eps = self$eps)
      }
    }

    if (length(out_chunks) == 1L) {
      return(out_chunks[[1]])
    }
    torch::torch_cat(out_chunks, dim = 2)
  },

  kl_div = function() {
    self$alpha <- torch::torch_sigmoid(self$lambda_l) * self$lambda_grad_mask
    self$weight_sigma <- torch::torch_log1p(torch::torch_exp(self$weight_rho))
    self$bias_sigma <- torch::torch_log1p(torch::torch_exp(self$bias_rho))

    total_kl <- torch::torch_tensor(0.0, device = self$device)

    # 1. Bernoulli gate KL across all active-eligible entries (lambda_grad_mask == 1)
    alp <- torch::torch_clamp(self$alpha, min = 1e-7, max = 1.0 - 1e-7)
    alp_p <- torch::torch_clamp(self$alpha_prior, min = 1e-7, max = 1.0 - 1e-7)
    kl_bern_mat <- alp * torch::torch_log(alp / alp_p) +
      (1.0 - alp) * torch::torch_log((1.0 - alp) / (1.0 - alp_p))
    total_kl <- total_kl + torch::torch_sum(kl_bern_mat * self$lambda_grad_mask)

    # 2. Standard Gaussian slab KL for standard_rows
    if (length(self$standard_rows) > 0L) {
      idx_s <- self$standard_rows
      a_s <- self$alpha[idx_s, , drop = FALSE]
      w_m <- self$weight_mean[idx_s, , drop = FALSE]
      w_s <- self$weight_sigma[idx_s, , drop = FALSE]
      w_mp <- self$weight_mean_prior[idx_s, , drop = FALSE]
      w_sp <- self$weight_sigma_prior[idx_s, , drop = FALSE]

      kl_gauss_s <- a_s * (
        torch::torch_log(w_sp / w_s) - 0.5 +
          (w_s^2 + (w_m - w_mp)^2) / (2.0 * w_sp^2)
      )
      total_kl <- total_kl + torch::torch_sum(kl_gauss_s)

      b_m <- self$bias_mean[idx_s]
      b_s <- self$bias_sigma[idx_s]
      b_mp <- self$bias_mean_prior[idx_s]
      b_sp <- self$bias_sigma_prior[idx_s]
      kl_bias_s <- torch::torch_sum(
        torch::torch_log(b_sp / b_s) - 0.5 + (b_s^2 + (b_m - b_mp)^2) / (2.0 * b_sp^2)
      )
      total_kl <- total_kl + kl_bias_s
    }

    # 3. Bimodal mixture slab KL for logic_rows: 0.5 N(+1, sigma_L^2) + 0.5 N(-1, sigma_L^2)
    if (length(self$logic_rows) > 0L) {
      idx_l <- self$logic_rows
      a_l <- self$alpha[idx_l, , drop = FALSE]
      w_m <- self$weight_mean[idx_l, , drop = FALSE]
      w_s <- self$weight_sigma[idx_l, , drop = FALSE]
      sig_L <- self$sigma_logic

      # Entropy of q(w) = N(w_m, w_s^2): E_q[log q(w)] = -0.5 log(2*pi*e*w_s^2)
      # Cross-entropy against 0.5 N(+1, sig_L^2) + 0.5 N(-1, sig_L^2)
      d_pos <- -0.5 * ((w_m - 1.0)^2 + w_s^2) / (sig_L^2)
      d_neg <- -0.5 * ((w_m + 1.0)^2 + w_s^2) / (sig_L^2)
      m_max <- torch::torch_maximum(d_pos, d_neg)
      log_mix <- m_max + torch::torch_log(
        0.5 * torch::torch_exp(d_pos - m_max) + 0.5 * torch::torch_exp(d_neg - m_max)
      )
      kl_bimodal <- a_l * (torch::torch_log(sig_L / w_s) - 0.5 - log_mix)
      total_kl <- total_kl + torch::torch_sum(kl_bimodal)
    }

    total_kl
  }
)
