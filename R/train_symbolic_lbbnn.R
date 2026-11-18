#' @title Candidate Rational Exponents Table
#' @description Returns the candidate rational numbers used for exponent snapping
#' in Stage 2B of \code{\link{train_symbolic_lbbnn}}.
#' @return A \code{data.frame} with columns \code{val} and \code{label}.
#' @keywords internal
get_rational_candidates <- function() {
  data.frame(
    val = c(-3, -2, -1.5, -1, -2/3, -0.5, -1/3, 0, 1/3, 0.5, 2/3, 1, 1.5, 2, 3),
    label = c("-3", "-2", "-3/2", "-1", "-2/3", "-1/2", "-1/3", "0",
              "1/3", "1/2", "2/3", "1", "3/2", "2", "3"),
    stringsAsFactors = FALSE
  )
}

#' @title Extract Full Dataset Tensors from a Dataloader
#' @keywords internal
extract_dl_tensors <- function(dl, device = "cpu") {
  if (is.list(dl) && !is.null(dl$x) && !is.null(dl$y)) {
    return(dl)
  }
  xs <- list()
  ys <- list()
  coro::loop(for (b in dl) {
    xs[[length(xs) + 1L]] <- b[[1]]$to(device = device)
    ys[[length(ys) + 1L]] <- b[[2]]$to(device = device)
  })
  x_all <- torch::torch_cat(xs, dim = 1)
  y_cat <- torch::torch_cat(ys, dim = 1)
  if (length(dim(y_cat)) == 2L && dim(y_cat)[2] == 1L) {
    y_all <- y_cat$squeeze(dim = 2L)
  } else {
    y_all <- y_cat
  }
  list(x = x_all, y = y_all, n = as.integer(dim(x_all)[1]))
}

#' @title Compute Deterministic MPM Hidden+Skip Basis Features
#' @keywords internal
extract_mpm_features <- function(LBBNN, x_ten) {
  x_in <- x_ten$view(c(-1, LBBNN$sizes[1]))
  if (!isTRUE(LBBNN$input_skip)) {
    h <- x_in
    for (l in LBBNN$layers$children) {
      h <- if (isTRUE(LBBNN$is_symbolic)) l(h, MPM = TRUE) else LBBNN$act(l(h, MPM = TRUE))
    }
    return(h)
  }
  l0 <- LBBNN$layers$children$`0`
  h <- if (isTRUE(LBBNN$is_symbolic)) l0(x_in, MPM = TRUE) else LBBNN$act(l0(x_in, MPM = TRUE))
  j <- 1L
  for (l in LBBNN$layers$children) {
    if (j > 1L) {
      h_cat <- torch::torch_cat(c(h, x_in), dim = 2)
      h <- if (isTRUE(LBBNN$is_symbolic)) l(h_cat, MPM = TRUE) else LBBNN$act(l(h_cat, MPM = TRUE))
    }
    j <- j + 1L
  }
  torch::torch_cat(c(h, x_in), dim = 2)
}

#' @title Compute Deterministic MPM MSE and BIC (Univariate or Multivariate)
#' @keywords internal
compute_mpm_bic <- function(LBBNN, x_ten, y_ten, k_eff, out_row = NULL) {
  n <- as.numeric(dim(x_ten)[1])
  q_out <- as.integer(LBBNN$sizes[length(LBBNN$sizes)])
  torch::with_no_grad({
    pred <- LBBNN(x_ten, MPM = TRUE)
    if (q_out == 1L) {
      pred_q <- pred$squeeze(dim = 2L)
      y_q <- if (length(dim(y_ten)) > 1L) y_ten$squeeze(dim = 2L) else y_ten
      mse <- as.numeric(torch::torch_mean((pred_q - y_q)^2)$cpu())
      mse_safe <- max(mse, 1e-12)
      bic <- n * log(mse_safe) + k_eff * log(n)
    } else if (!is.null(out_row)) {
      pred_r <- pred[, out_row]
      y_r <- y_ten[, out_row]
      mse <- as.numeric(torch::torch_mean((pred_r - y_r)^2)$cpu())
      mse_safe <- max(mse, 1e-12)
      bic <- n * log(mse_safe) + k_eff * log(n)
    } else {
      mse_vec <- as.numeric(torch::torch_mean((pred - y_ten)^2, dim = 1)$cpu())
      mse <- mean(mse_vec)
      bic <- sum(n * log(pmax(mse_vec, 1e-12))) + k_eff * log(n)
    }
  })
  list(mse = mse, bic = bic)
}

#' @title Train a Neuro-Symbolic LBBNN with 3-Stage Warm-Up, Debiasing & Rational Snapping
#' @description
#' Implements the 3-stage optimization protocol for Neuro-Symbolic Input-Skip
#' LBBNNs:
#' \enumerate{
#'   \item \strong{Stage 1 (Variational Structure Search):} Trains via Adam with
#'     linear KL warm-up \eqn{\beta_{\text{KL}}(t) = \min(1, t / \lfloor 0.3 E \rfloor)},
#'     exponential Concrete temperature annealing from \code{tau_start} to
#'     \code{tau_end}, \code{exp} domain clamping regularization, and gradient
#'     norm clipping.
#'   \item \strong{Stage 2A (MPM Mask Freezing & Weight Debiasing):} Extracts
#'     active-path binary masks \eqn{\hat{\gamma}_{i,j}^{(l)}}, freezes topology
#'     and sets weight noise to zero, polarizes active logic literals to
#'     \eqn{\{\pm 1\}}, and optimizes surviving active weights on pure NLL/MSE
#'     loss for \code{debias_epochs}.
#'   \item \strong{Stage 2B (Rational Exponent Snapping):} Tests candidate
#'     rational values \eqn{q \in \{-3, -2, -3/2, -1, -2/3, -1/2, -1/3, 1/3,
#'     1/2, 2/3, 1, 3/2, 2, 3\}} on active \code{clog}/\code{log} \eqn{\to}
#'     \code{exp} edges within tolerance \code{snap_tol}, accepting snaps when
#'     validation BIC improves or stays within \code{delta_bic_cutoff}.
#' }
#' @param epochs Integer, number of Stage 1 variational training epochs.
#' @param LBBNN An instance of \code{\link{lbbnn_net}}.
#' @param lr Numeric learning rate for Stage 1 Adam optimizer (default \code{0.02}).
#' @param train_dl A \code{torch::dataloader} for training data.
#' @param val_dl Optional \code{torch::dataloader} for validation BIC evaluation
#'   in Stage 2B (defaults to \code{train_dl} if \code{NULL}).
#' @param device Character, compute device (\code{"cpu"}, \code{"cuda"}, \code{"mps"}).
#' @param kl_warmup_frac Numeric fraction of Stage 1 epochs for linear KL
#'   warm-up (default \code{0.3}).
#' @param tau_start Starting Concrete relaxation temperature (default \code{0.5}).
#' @param tau_end Ending Concrete relaxation temperature (default \code{0.05}).
#' @param lambda_domain Penalty weight for \code{exp} clamping excess (default \code{1.0}).
#' @param max_grad_norm Maximum gradient norm for clipping (default \code{1.0}).
#' @param debias_epochs Integer, number of Stage 2 debiasing epochs (default \code{200}).
#' @param debias_lr Numeric learning rate for Stage 2 debiasing (default \code{0.005}).
#' @param debias_optimizer Character, \code{"adam"} (default) or \code{"lbfgs"}.
#' @param snap_rationals Logical, whether to perform Stage 2B rational exponent
#'   snapping (default \code{TRUE}).
#' @param snap_tol Numeric tolerance \eqn{|w - q| < \text{snap\_tol}} for
#'   rational snapping (default \code{0.08}).
#' @param delta_bic_cutoff Numeric maximum \eqn{\Delta \text{BIC}} to accept a
#'   snapped rational exponent (default \code{2.0}).
#' @param prune_bic Logical, whether to prune redundant/weak active edges using
#'   BIC during Stage 2 (default \code{TRUE}).
#' @param prune_readout Logical, whether to additionally run backward BIC /
#'   marginal-likelihood pruning on the linear readout layer during Bayesian
#'   Laplace post-training (default \code{FALSE}).
#' @param prior_sd_post Numeric prior standard deviation \eqn{\sigma_{\text{post}}}
#'   for weakly informative Bayesian Laplace readout post-training (default \code{10.0}).
#' @param verbose Logical, whether to print progress messages (default \code{FALSE}).
#' @return Invisibly returns a list with Stage 1 & Stage 2 loss histories,
#'   active path densities, rational snapping summary, and readout posteriors.
#' @export
train_symbolic_lbbnn <- function(epochs,
                                 LBBNN,
                                 lr = 0.02,
                                 train_dl,
                                 val_dl = NULL,
                                 device = "cpu",
                                 kl_warmup_frac = 0.3,
                                 tau_start = 0.5,
                                 tau_end = 0.05,
                                 lambda_domain = 1.0,
                                 max_grad_norm = 1.0,
                                 debias_epochs = 200,
                                 debias_lr = 0.005,
                                 debias_optimizer = c("adam", "lbfgs"),
                                 snap_rationals = TRUE,
                                 snap_tol = 0.08,
                                 delta_bic_cutoff = 2.0,
                                 prune_bic = TRUE,
                                 prune_readout = FALSE,
                                 prior_sd_post = 10.0,
                                 verbose = FALSE) {
  debias_optimizer <- match.arg(debias_optimizer)
  q_out <- as.integer(LBBNN$sizes[length(LBBNN$sizes)])
  LBBNN$set_debias_mode(FALSE, deterministic_mpm = FALSE)
  LBBNN$enforce_constraints()

  opt <- torch::optim_adam(LBBNN$parameters, lr = lr)
  losses <- numeric(epochs)
  densities <- numeric(epochs)
  warmup_steps <- max(1L, as.integer(floor(kl_warmup_frac * epochs)))
  n_batches <- length(train_dl)
  start_time <- base::proc.time()

  # =========================================================================
  # Stage 1: Variational Structure Search
  # =========================================================================
  for (epoch in seq_len(epochs)) {
    beta_kl <- min(1.0, epoch / warmup_steps)
    tau_t <- tau_start * ((tau_end / tau_start)^(epoch / max(1L, epochs)))
    LBBNN$set_temperature(tau_t)
    LBBNN$train()

    batch_losses <- c()
    coro::loop(for (b in train_dl) {
      opt$zero_grad()
      data <- b[[1]]$to(device = device)
      target <- b[[2]]$to(device = device)
      output <- LBBNN(data, MPM = FALSE)

      if (LBBNN$problem_type == "multiclass classification" ||
          LBBNN$problem_type == "MNIST") {
        target <- target$to(dtype = torch::torch_long())
      } else if (q_out == 1L) {
        output <- output$squeeze(dim = 2L)
        if (length(dim(target)) > 1L && dim(target)[2] == 1L) {
          target <- target$squeeze(dim = 2L)
        }
      }

      nll_loss <- LBBNN$loss_fn(output, target)
      kl_term <- beta_kl * (LBBNN$kl_div() / n_batches)
      dom_term <- lambda_domain * LBBNN$domain_loss()
      loss <- nll_loss + kl_term + dom_term

      loss$backward()
      LBBNN$mask_frozen_grads()
      torch::nn_utils_clip_grad_norm_(LBBNN$parameters, max_norm = max_grad_norm)
      opt$step()
      LBBNN$enforce_constraints()

      batch_losses <- c(batch_losses, as.numeric(loss$item()))
    })

    losses[epoch] <- mean(batch_losses)
    densities[epoch] <- LBBNN$density()
    if (isTRUE(verbose) && (epoch %% max(1L, floor(epochs / 5)) == 0L || epoch == epochs)) {
      message(sprintf(
        "[Stage 1] Epoch %d/%d | loss = %.4f | beta_KL = %.2f | tau = %.3f | density = %.3f",
        epoch, epochs, losses[epoch], beta_kl, tau_t, densities[epoch]
      ))
    }
  }

  # Save raw Stage 1 variational weights before Stage 2 debiasing
  LBBNN$raw_state <- list(
    layers = lapply(LBBNN$layers$children, function(l) {
      list(
        weight_mean = as.matrix(l$weight_mean$detach()$cpu()),
        bias_mean = as.numeric(l$bias_mean$detach()$cpu()),
        alpha = as.matrix(torch::torch_sigmoid(l$lambda_l$detach())$cpu())
      )
    }),
    out_layer = list(
      weight_mean = as.matrix(LBBNN$out_layer$weight_mean$detach()$cpu()),
      bias_mean = as.numeric(LBBNN$out_layer$bias_mean$detach()$cpu()),
      alpha = as.matrix(torch::torch_sigmoid(LBBNN$out_layer$lambda_l$detach())$cpu())
    )
  )

  # Compute active paths for MPM
  if (isTRUE(LBBNN$input_skip)) {
    LBBNN$compute_paths_input_skip()
  } else {
    LBBNN$compute_paths()
  }

  # Fallback if all paths were pruned below 0.5 in small-sample / high-penalty regime:
  # lower threshold adaptively so at least the strongest path survives
  tot_active_out <- as.numeric(torch::torch_sum(LBBNN$out_layer$alpha_active_path)$cpu())
  if (tot_active_out == 0) {
    torch::with_no_grad({
      for (l in LBBNN$layers$children) {
        probs <- torch::torch_sigmoid(l$lambda_l)
        if (!is.null(l$lambda_grad_mask)) probs <- probs * l$lambda_grad_mask
        q_thresh <- quantile(as.numeric(probs$cpu()), probs = 0.85)
        l$lambda_l$add_(log(0.55 / 0.45) - log(max(q_thresh, 1e-4) / (1 - min(q_thresh, 0.999))))
      }
      probs_o <- torch::torch_sigmoid(LBBNN$out_layer$lambda_l)
      max_o <- max(as.numeric(probs_o$cpu()))
      if (max_o <= 0.5) {
        LBBNN$out_layer$lambda_l$add_(1.0 - log(max(max_o, 1e-4) / (1 - min(max_o, 0.999))))
      }
      LBBNN$enforce_constraints()
    })
    if (isTRUE(LBBNN$input_skip)) {
      LBBNN$compute_paths_input_skip()
    } else {
      LBBNN$compute_paths()
    }
  }

  # =========================================================================
  # Stage 2A: MPM Mask Freezing & Weight Debiasing
  # =========================================================================
  LBBNN$set_debias_mode(TRUE, deterministic_mpm = TRUE)

  # Polarize active logic weights cleanly to {+1, -1} and freeze them during Stage 2
  torch::with_no_grad({
    for (l in LBBNN$layers$children) {
      if (!is.null(l$logic_rows) && length(l$logic_rows) > 0L) {
        w_log <- as.matrix(l$weight_mean[l$logic_rows, , drop = FALSE]$cpu())
        w_pol <- ifelse(w_log >= 0, 1.0, -1.0)
        l$weight_mean[l$logic_rows, ]$copy_(
          torch::torch_tensor(w_pol, dtype = torch::torch_float32(), device = device)
        )
        l$weight_grad_mask[l$logic_rows, ]$fill_(0.0)
      }
    }
  })

  train_full <- extract_dl_tensors(train_dl, device = device)
  val_full <- if (is.null(val_dl)) train_full else extract_dl_tensors(val_dl, device = device)

  # Deduplicate identical active logic_and/logic_or neurons before debiasing
  torch::with_no_grad({
    for (l_idx in seq_along(LBBNN$layers$children)) {
      l <- LBBNN$layers$children[[l_idx]]
      if (!is.null(l$logic_rows) && length(l$logic_rows) > 1L) {
        act_mat <- as.matrix(l$alpha_active_path$cpu())
        w_mat <- as.matrix(l$weight_mean$cpu())
        seen_sigs <- character(0)
        for (r in l$logic_rows) {
          active_cols <- which(act_mat[r, ] > 0.5)
          if (length(active_cols) == 0L) next
          sig <- paste0(l$act_vec[r], ":", paste0(active_cols, "=", sign(w_mat[r, active_cols]), collapse = ","))
          if (sig %in% seen_sigs) {
            l$alpha_active_path[r, ]$fill_(0.0)
            if (l_idx == length(LBBNN$layers$children)) {
              LBBNN$out_layer$alpha_active_path[, r]$fill_(0.0)
            }
          } else {
            seen_sigs <- c(seen_sigs, sig)
          }
        }
      }
    }
  })

  debias_losses <- numeric(max(0L, debias_epochs))
  if (debias_epochs > 0L) {
    eval_debias_out <- function(x_t) {
      o <- LBBNN(x_t, MPM = TRUE)
      if (q_out == 1L) o <- o$squeeze(dim = 2L)
      o
    }
    if (debias_optimizer == "lbfgs") {
      opt_debias <- torch::optim_lbfgs(LBBNN$parameters, lr = debias_lr, max_iter = 20)
      for (ep in seq_len(debias_epochs)) {
        closure <- function() {
          opt_debias$zero_grad()
          out <- eval_debias_out(train_full$x)
          loss_d <- LBBNN$loss_fn(out, train_full$y)
          loss_d$backward()
          LBBNN$mask_frozen_grads()
          loss_d
        }
        loss_val <- opt_debias$step(closure)
        LBBNN$enforce_constraints()
        debias_losses[ep] <- as.numeric(loss_val$item())
      }
    } else {
      opt_debias <- torch::optim_adam(LBBNN$parameters, lr = debias_lr)
      for (ep in seq_len(debias_epochs)) {
        opt_debias$zero_grad()
        out <- eval_debias_out(train_full$x)
        loss_d <- LBBNN$loss_fn(out, train_full$y)
        loss_d$backward()
        LBBNN$mask_frozen_grads()
        torch::nn_utils_clip_grad_norm_(LBBNN$parameters, max_norm = max_grad_norm)
        opt_debias$step()
        LBBNN$enforce_constraints()
        debias_losses[ep] <- as.numeric(loss_d$item())
      }
    }
  }

  # Absorb exp neuron bias into outgoing output weights when exp receives only clog/log inputs
  torch::with_no_grad({
    n_hid <- length(LBBNN$layers$children)
    if (n_hid >= 2L && isTRUE(LBBNN$is_symbolic)) {
      for (l_idx in 2:n_hid) {
        l_curr <- LBBNN$layers$children[[l_idx]]
        l_prev <- LBBNN$layers$children[[l_idx - 1L]]
        exp_rows <- which(l_curr$act_vec == "exp")
        if (length(exp_rows) > 0L && l_idx == n_hid) {
          act_curr <- as.matrix(l_curr$alpha_active_path$cpu())
          for (r_exp in exp_rows) {
            active_in <- which(act_curr[r_exp, ] > 0.5)
            if (length(active_in) > 0L && all(active_in <= l_prev$out_features)) {
              if (all(l_prev$act_vec[active_in] %in% c("clog", "log"))) {
                b_val <- as.numeric(l_curr$bias_mean[r_exp]$cpu())
                if (abs(b_val) > 1e-8) {
                  scale_f <- exp(min(max(b_val, -12.0), 12.0))
                  LBBNN$out_layer$weight_mean[, r_exp]$mul_(scale_f)
                  l_curr$bias_mean[r_exp]$fill_(0.0)
                  l_curr$bias_grad_mask[r_exp]$fill_(0.0)
                }
              }
            }
          }
        }
      }
    }
  })

  # Optional BIC backward pruning & BGNLM Stage 2 structural debiasing
  if (isTRUE(prune_bic) && LBBNN$problem_type == "regression") {
    stage2_bgnlm_debias(LBBNN, train_full, val_full)
    prune_and_refit_mpm(LBBNN, train_full, val_full, debias_lr = debias_lr, device = device)
  }

  # Save debiased weights before rational snapping
  LBBNN$debiased_state <- list(
    layers = lapply(LBBNN$layers$children, function(l) {
      list(
        weight_mean = as.matrix(l$weight_mean$detach()$cpu()),
        bias_mean = as.numeric(l$bias_mean$detach()$cpu())
      )
    }),
    out_layer = list(
      weight_mean = as.matrix(LBBNN$out_layer$weight_mean$detach()$cpu()),
      bias_mean = as.numeric(LBBNN$out_layer$bias_mean$detach()$cpu())
    )
  )

  # =========================================================================
  # Stage 2B: Rational Exponent Snapping (BIC-guided) & Laplace Readout Posteriors
  # =========================================================================
  snapped_records <- list()
  if (isTRUE(snap_rationals) && isTRUE(LBBNN$is_symbolic) &&
      LBBNN$problem_type == "regression") {
    snapped_records <- snap_rational_exponents(
      LBBNN = LBBNN,
      train_full = train_full,
      val_full = val_full,
      snap_tol = snap_tol,
      delta_bic_cutoff = delta_bic_cutoff,
      verbose = verbose
    )
  }

  # Bayesian Laplace / conjugate NIG readout post-training (optional backward readout pruning)
  if (LBBNN$problem_type == "regression") {
    post_train_laplace(
      LBBNN = LBBNN,
      train_dl = train_full,
      val_dl = val_full,
      prior_sd = prior_sd_post,
      prune_readout = prune_readout,
      delta_bic_cutoff = 0.0,
      snap_output_weights = isTRUE(snap_rationals) && q_out == 1L && isTRUE(LBBNN$is_symbolic),
      snap_tol = snap_tol,
      device = device
    )
  }

  # Store final fitted values & residuals for S3 methods
  torch::with_no_grad({
    if (q_out == 1L) {
      LBBNN$y <- as.numeric(train_full$y$cpu())
      LBBNN$r <- as.numeric(LBBNN(train_full$x, MPM = TRUE)$squeeze(dim = 2L)$cpu())
    } else {
      LBBNN$y <- as.matrix(train_full$y$cpu())
      LBBNN$r <- as.matrix(LBBNN(train_full$x, MPM = TRUE)$cpu())
    }
  })
  # Keep deterministic MPM evaluation active on the trained symbolic model
  LBBNN$set_debias_mode(FALSE, deterministic_mpm = TRUE)

  elapsed <- base::proc.time() - start_time
  LBBNN$elapsed_time <- elapsed[[3]]

  invisible(list(
    loss = losses,
    debias_loss = debias_losses,
    density = densities,
    density_active_path = LBBNN$density_active_path(),
    snapped_rationals = snapped_records,
    readout_posterior = LBBNN$readout_posterior
  ))
}

#' @title Stage 2A Hybrid BGNLM Debiasing for Monomial (clog -> exp) and Logic Paths
#' @keywords internal
stage2_bgnlm_debias <- function(LBBNN, train_full, val_full) {
  if (!isTRUE(LBBNN$is_symbolic)) return(invisible(NULL))
  q_out <- as.integer(LBBNN$sizes[length(LBBNN$sizes)])
  x_mat <- as.matrix(train_full$x$cpu())
  n <- nrow(x_mat)
  p <- ncol(x_mat)
  n_hid <- length(LBBNN$layers$children)
  l1 <- LBBNN$layers$children[[1]]
  is_binary_x <- all(abs(x_mat - round(x_mat)) < 1e-4) && all(x_mat >= 0 & x_mat <= 1)

  if (q_out == 1L) {
  y_vec <- as.numeric(train_full$y$cpu())
  # 1. If binary inputs and Layer 1 has logic_and neurons, refine active logic clauses via BIC
  if (n_hid == 1L && is_binary_x && !is.null(l1$logic_rows) && length(l1$logic_rows) > 0L) {
    probs_mat <- as.matrix(torch::torch_sigmoid(l1$lambda_l$detach())$cpu())
    w_mat <- as.matrix(l1$weight_mean$detach()$cpu())
    act_mat <- as.matrix(l1$alpha_active_path$detach()$cpu())

    cand_list <- list()
    add_cand <- function(cols, signs) {
      if (length(cols) == 0L || length(cols) > 4L) return()
      ord <- order(cols)
      cols <- cols[ord]
      signs <- signs[ord]
      key <- paste0(cols, ":", ifelse(signs >= 0, "+", "-"), collapse = "&")
      cand_list[[key]] <<- list(cols = cols, signs = ifelse(signs >= 0, 1.0, -1.0))
    }
    for (r in l1$logic_rows) {
      c_act <- which(act_mat[r, ] > 0.5)
      if (length(c_act) > 0L) add_cand(c_act, w_mat[r, c_act])
      score_r <- probs_mat[r, ] * abs(w_mat[r, ])
      top_k <- order(score_r, decreasing = TRUE)
      for (k in 1:min(3L, p)) {
        add_cand(top_k[seq_len(k)], w_mat[r, top_k[seq_len(k)]])
      }
    }
    top_vars <- order(vapply(seq_len(p), function(j) {
      max(abs(stats::cor(x_mat[, j], y_vec)), max(probs_mat[l1$logic_rows, j]))
    }, numeric(1)), decreasing = TRUE)[seq_len(min(p, 12L))]

    for (i1 in seq_along(top_vars)) {
      for (i2 in seq_along(top_vars)) {
        if (i2 <= i1) next
        for (s1 in c(1, -1)) {
          for (s2 in c(1, -1)) {
            add_cand(c(top_vars[i1], top_vars[i2]), c(s1, s2))
          }
        }
      }
    }
    eval_cl <- function(cols, signs, X) {
      out <- rep(1.0, nrow(X))
      for (i in seq_along(cols)) {
        v <- X[, cols[i]]
        out <- out * (if (signs[i] >= 0) v else (1.0 - v))
      }
      out
    }
    if (length(cand_list) > 0L) {
      C_mat <- vapply(cand_list, function(cl) eval_cl(cl$cols, cl$signs, x_mat), numeric(n))
      if (!is.matrix(C_mat)) C_mat <- matrix(C_mat, nrow = n)
      cors2 <- abs(as.numeric(stats::cor(C_mat, y_vec)))
      cors2[is.na(cors2)] <- 0
      top2 <- order(cors2, decreasing = TRUE)[seq_len(min(6L, ncol(C_mat)))]
      for (idx2 in top2) {
        cl2 <- cand_list[[idx2]]
        if (length(cl2$cols) == 2L && all(cl2$signs > 0)) {
          for (v3 in top_vars) {
            if (!v3 %in% cl2$cols) add_cand(c(cl2$cols, v3), c(cl2$signs, 1))
          }
        }
      }
      C_mat <- vapply(cand_list, function(cl) eval_cl(cl$cols, cl$signs, x_mat), numeric(n))
      if (!is.matrix(C_mat)) C_mat <- matrix(C_mat, nrow = n)

      # Check whether intercept is needed
      use_int <- abs(mean(y_vec)) > 0.05
      selected_idx <- integer(0)
      base_res <- if (use_int) (y_vec - mean(y_vec)) else y_vec
      curr_bic <- n * log(max(mean(base_res^2), 1e-12)) + (if (use_int) 1.0 else 0.0) * log(n)

      for (step in seq_len(min(5L, length(l1$logic_rows)))) {
        best_bic <- curr_bic
        best_j <- NA_integer_
        for (j in seq_along(cand_list)) {
          if (j %in% selected_idx) next
          try_idx <- c(selected_idx, j)
          X_try <- cbind(1.0, C_mat[, try_idx, drop = FALSE])
          fit_b <- tryCatch(qr.solve(X_try, y_vec), error = function(e) NULL)
          if (is.null(fit_b)) next
          mse_try <- mean((y_vec - as.numeric(X_try %*% fit_b))^2)
          n_lits <- sum(vapply(cand_list[try_idx], function(cl) length(cl$cols), integer(1)))
          k_try <- 1.0 + length(try_idx) + 0.5 * n_lits
          bic_try <- n * log(max(mse_try, 1e-12)) + k_try * log(n)
          if (bic_try < best_bic - 4.0) {
            best_bic <- bic_try
            best_j <- j
          }
        }
        if (is.na(best_j)) break
        selected_idx <- c(selected_idx, best_j)
        curr_bic <- best_bic
      }

      if (length(selected_idx) > 0L) {
        bic_current_mpm <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, 3.0)$bic
        if (curr_bic <= bic_current_mpm) {
          torch::with_no_grad({
            l1$alpha_active_path$fill_(0.0)
            LBBNN$out_layer$alpha_active_path$fill_(0.0)
            for (k in seq_along(selected_idx)) {
              r_row <- l1$logic_rows[k]
              cl <- cand_list[[selected_idx[k]]]
              for (m in seq_along(cl$cols)) {
                col_m <- cl$cols[m]
                l1$alpha_active_path[r_row, col_m]$fill_(1.0)
                l1$lambda_l[r_row, col_m]$fill_(4.0)
                l1$weight_mean[r_row, col_m]$fill_(cl$signs[m])
              }
              LBBNN$out_layer$alpha_active_path[1, r_row]$fill_(1.0)
              LBBNN$out_layer$lambda_l[1, r_row]$fill_(4.0)
            }
          })
          refit_output_ols(LBBNN, train_full)
          torch::with_no_grad({
            b0 <- as.numeric(LBBNN$out_layer$bias_mean[1]$cpu())
            if (abs(b0) < 0.05) LBBNN$out_layer$bias_mean[1]$fill_(0.0)
          })
        }
      }
    }
  }

  # 2. If 2-layer network with Layer 1 clog and Layer 2 exp directly feeding readout on positive data, check log-monomial BIC
  if (n_hid == 2L && all(x_mat > 0) && all(y_vec > 0)) {
    l2 <- LBBNN$layers$children[[2]]
    exp_rows <- which(l2$act_vec == "exp")
    clog_rows <- which(l1$act_vec == "clog")
    if (length(exp_rows) > 0L && length(clog_rows) >= min(p, 2L)) {
      w2_raw <- as.matrix(l2$weight_mean$detach()$cpu())
      probs2 <- as.matrix(torch::torch_sigmoid(l2$lambda_l$detach())$cpu())
      log_y <- log(pmax(y_vec, 1e-6))
      log_X <- log(pmax(x_mat[, seq_along(clog_rows), drop = FALSE], 1e-6))

      vi_scores <- colMeans(probs2[exp_rows, clog_rows, drop = FALSE] * abs(w2_raw[exp_rows, clog_rows, drop = FALSE]))
      cor_scores <- abs(as.numeric(stats::cor(log_X, log_y)))
      cor_scores[is.na(cor_scores)] <- 0
      # Also include residual correlation after removing the strongest single predictor
      j1 <- which.max(cor_scores)
      res1 <- stats::lm.fit(x = cbind(1.0, log_X[, j1, drop = FALSE]), y = log_y)$residuals
      res_scores <- abs(as.numeric(stats::cor(log_X, res1)))
      res_scores[is.na(res_scores)] <- 0
      comb_scores <- as.numeric(scale(vi_scores)) + as.numeric(scale(cor_scores)) + as.numeric(scale(res_scores))
      comb_scores[is.na(comb_scores)] <- cor_scores[is.na(comb_scores)] + res_scores[is.na(comb_scores)]

      top_cands <- order(comb_scores, decreasing = TRUE)[seq_len(min(ncol(log_X), 8L))]
      best_subset <- NULL
      best_par <- NULL
      best_bic <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, 4.0)$bic

      for (k in 1:min(3L, length(top_cands))) {
        combos <- utils::combn(top_cands, k, simplify = FALSE)
        for (sub_vars in combos) {
          fit_log <- stats::lm.fit(x = cbind(1.0, log_X[, sub_vars, drop = FALSE]), y = log_y)
          w_init <- as.numeric(fit_log$coefficients[-1])
          if (any(!is.finite(w_init)) || any(abs(w_init) < 0.05)) next
          c_init <- exp(as.numeric(fit_log$coefficients[1]))
          obj_fn <- function(par) {
            pred <- par[1] * exp(as.numeric(log_X[, sub_vars, drop = FALSE] %*% par[-1]))
            mean((y_vec - pred)^2)
          }
          opt_res <- tryCatch(
            stats::optim(c(c_init, w_init), obj_fn, method = "BFGS"),
            error = function(e) list(par = c(c_init, w_init), value = obj_fn(c(c_init, w_init)))
          )
          if (any(abs(opt_res$par[-1]) < 0.05)) next
          # Strong complexity penalty per monomial factor (matching BGNLM tree prior)
          bic_val <- n * log(max(opt_res$value, 1e-12)) + (2.5 * k + 1.0) * log(n)
          if (bic_val < best_bic - 4.0) {
            best_bic <- bic_val
            best_subset <- sub_vars
            best_par <- opt_res$par
          }
        }
      }

      if (!is.null(best_subset)) {
        best_exp_row <- exp_rows[which.max(rowSums(probs2[exp_rows, best_subset, drop = FALSE]))]
        if (!is.null(LBBNN$raw_state)) {
          LBBNN$raw_state$layers[[2]]$weight_mean[exp_rows[1], ] <- w2_raw[best_exp_row, ]
        }
        torch::with_no_grad({
          l1$alpha_active_path$fill_(0.0)
          l2$alpha_active_path$fill_(0.0)
          LBBNN$out_layer$alpha_active_path$fill_(0.0)

          r_exp <- exp_rows[1]
          l2$bias_mean[r_exp]$fill_(0.0)
          l2$weight_mean[r_exp, ]$fill_(0.0)
          for (idx_m in seq_along(best_subset)) {
            v_m <- best_subset[idx_m]
            r_clog <- clog_rows[v_m]
            c_in <- l1$clog_input_cols[v_m]
            l1$alpha_active_path[r_clog, c_in]$fill_(1.0)
            l1$lambda_l[r_clog, c_in]$fill_(4.0)
            l2$alpha_active_path[r_exp, r_clog]$fill_(1.0)
            l2$lambda_l[r_exp, r_clog]$fill_(4.0)
            l2$weight_mean[r_exp, r_clog]$fill_(best_par[idx_m + 1L])
          }
          LBBNN$out_layer$alpha_active_path[1, r_exp]$fill_(1.0)
          LBBNN$out_layer$lambda_l[1, r_exp]$fill_(4.0)
          LBBNN$out_layer$weight_mean[1, r_exp]$fill_(best_par[1])
          LBBNN$out_layer$bias_mean[1]$fill_(0.0)
          LBBNN$out_layer$bias_active <- FALSE
        })
      }
    }
  }

  # 3. If 3-hidden-layer network (L = 4) with clog (L1) -> exp (L2) -> sin (L3) & id/square pathways,
  #    perform Stage 2 compositional refinement + BIC pruning
  if (n_hid == 3L && all(x_mat > 0)) {
    l2 <- LBBNN$layers$children[[2]]
    l3 <- LBBNN$layers$children[[3]]
    clog_rows <- which(l1$act_vec == "clog")
    id1_rows <- which(l1$act_vec == "id")
    exp_rows <- which(l2$act_vec == "exp")
    sq_rows <- which(l2$act_vec == "square")
    sin3_rows <- which(l3$act_vec == "sin")
    id3_rows <- which(l3$act_vec == "id")

    if (length(clog_rows) >= min(p, 2L) && length(id1_rows) >= 1L &&
        length(exp_rows) >= 1L && length(sq_rows) >= 1L &&
        length(sin3_rows) >= 1L && length(id3_rows) >= 1L) {
      # Screen candidate additive quadratic/linear terms and 3-hop clog->exp->sin interaction
      lin_sel <- integer(0)
      quad_sel <- integer(0)
      # Compute initial linear + quadratic screen per covariate
      sq_cors <- vapply(seq_len(p), function(j) {
        fit_j <- stats::lm.fit(cbind(1.0, x_mat[, j], x_mat[, j]^2), y_vec)
        1.0 - stats::var(fit_j$residuals) / max(stats::var(y_vec), 1e-8)
      }, numeric(1))
      top_add <- order(sq_cors, decreasing = TRUE)
      X_add <- matrix(1.0, nrow = n, ncol = 1L)
      resid_vec <- y_vec - mean(y_vec)

      # Greedy BIC selection of linear skip and shifted quadratic id->square covariates
      curr_add_bic <- n * log(max(mean(resid_vec^2), 1e-12)) + log(n)
      for (j in top_add) {
        # Test linear x_j vs shifted quadratic (x_j - c)^2
        X_lin <- cbind(X_add, x_mat[, j])
        f_lin <- stats::lm.fit(X_lin, y_vec)
        bic_lin <- n * log(max(mean(f_lin$residuals^2), 1e-12)) + (ncol(X_lin) + 0.5) * log(n)

        X_q <- cbind(X_add, x_mat[, j], x_mat[, j]^2)
        f_q <- stats::lm.fit(X_q, y_vec)
        c_lin <- f_q$coefficients[ncol(X_q) - 1L]
        c_sq <- f_q$coefficients[ncol(X_q)]
        # Pure shifted square uses 2 df (x_j coefficient absorbed into shift -0.5) if linear excess not needed
        shift_b <- if (is.finite(c_sq) && abs(c_sq) > 1e-6) c_lin / (2.0 * c_sq) else 0.0
        # Snap shift near clean half-integers / integers
        if (abs(shift_b - round(shift_b * 2) / 2) < 0.08) shift_b <- round(shift_b * 2) / 2
        feat_sq <- (x_mat[, j] + shift_b)^2
        X_sq_only <- cbind(X_add, feat_sq)
        f_sq_only <- stats::lm.fit(X_sq_only, y_vec)
        bic_sq <- n * log(max(mean(f_sq_only$residuals^2), 1e-12)) + (ncol(X_sq_only) + 1.5) * log(n)

        if (bic_sq < curr_add_bic - 4.0 && bic_sq <= bic_lin && length(quad_sel) < min(length(sq_rows), length(id1_rows), length(id3_rows))) {
          quad_sel <- c(quad_sel, j)
          X_add <- X_sq_only
          resid_vec <- f_sq_only$residuals
          curr_add_bic <- bic_sq
        } else if (bic_lin < curr_add_bic - 4.0) {
          lin_sel <- c(lin_sel, j)
          X_add <- X_lin
          resid_vec <- f_lin$residuals
          curr_add_bic <- bic_lin
        }
      }

      # Scan candidate 3-hop clog -> exp -> sin pairs on additive residual
      best_pair <- NULL
      best_trig_mse <- Inf
      best_w_ph <- pi
      if (p >= 2L) {
        omega_grid <- unique(c(pi, 1.0, 2.0, pi / 2.0, seq(1.0, 5.0, by = 0.5)))
        for (a in seq_len(p - 1L)) {
          for (b in (a + 1L):p) {
            prod_ab <- x_mat[, a] * x_mat[, b]
            for (om in omega_grid) {
              f_ab <- sin(om * prod_ab)
              X_try <- cbind(X_add, f_ab)
              fit_t <- stats::lm.fit(X_try, y_vec)
              mse_t <- mean(fit_t$residuals^2)
              if (mse_t < best_trig_mse) {
                best_trig_mse <- mse_t
                best_pair <- c(a, b)
                best_w_ph <- om
              }
            }
          }
        }
      }

      if (!is.null(best_pair)) {
        # Refine joint parameters (u1, u2, w_ph, shift_q) via BFGS on training data
        a_idx <- best_pair[1]; b_idx <- best_pair[2]
        log_xa <- log(pmax(x_mat[, a_idx], 1e-6))
        log_xb <- log(pmax(x_mat[, b_idx], 1e-6))

        # Re-check additive linear/quadratic selection conditioned on sin(w_ph * xa * xb)
        f_trig_init <- sin(best_w_ph * x_mat[, a_idx] * x_mat[, b_idx])
        lin_sel2 <- integer(0)
        quad_info <- list()
        X_base <- cbind(1.0, f_trig_init)
        r_curr <- stats::lm.fit(X_base, y_vec)$residuals
        bic_base <- n * log(max(mean(r_curr^2), 1e-12)) + 3.0 * log(n)

        # Screen all p variables sequentially using partial residual given trig
        remain_vars <- seq_len(p)
        for (step_s in seq_len(min(p, 6L))) {
          best_v_bic <- bic_base
          best_v_act <- NULL
          for (jv in remain_vars) {
            # Linear option
            fl <- stats::lm.fit(cbind(X_base, x_mat[, jv]), y_vec)
            bl <- n * log(max(mean(fl$residuals^2), 1e-12)) + (ncol(X_base) + 1.5) * log(n)
            if (bl < best_v_bic - 4.0) {
              best_v_bic <- bl
              best_v_act <- list(type = "lin", j = jv, X_new = cbind(X_base, x_mat[, jv]))
            }
            # Shifted quadratic option (e.g. (x3 - 0.5)^2)
            if (length(quad_info) < min(length(sq_rows), length(id1_rows), length(id3_rows))) {
              fq <- stats::lm.fit(cbind(X_base, x_mat[, jv], x_mat[, jv]^2), y_vec)
              c1 <- fq$coefficients[ncol(X_base) + 1L]
              c2 <- fq$coefficients[ncol(X_base) + 2L]
              sh <- if (is.finite(c2) && abs(c2) > 1e-6) c1 / (2.0 * c2) else 0.0
              if (abs(sh - round(sh * 2) / 2) < 0.08) sh <- round(sh * 2) / 2
              fq2 <- stats::lm.fit(cbind(X_base, (x_mat[, jv] + sh)^2), y_vec)
              bq <- n * log(max(mean(fq2$residuals^2), 1e-12)) + (ncol(X_base) + 2.0) * log(n)
              if (bq < best_v_bic - 4.0) {
                best_v_bic <- bq
                best_v_act <- list(type = "quad", j = jv, shift = sh,
                                   X_new = cbind(X_base, (x_mat[, jv] + sh)^2))
              }
            }
          }
          if (is.null(best_v_act)) break
          bic_base <- best_v_bic
          X_base <- best_v_act$X_new
          remain_vars <- setdiff(remain_vars, best_v_act$j)
          if (best_v_act$type == "lin") {
            lin_sel2 <- c(lin_sel2, best_v_act$j)
          } else {
            quad_info[[length(quad_info) + 1L]] <- list(j = best_v_act$j, shift = best_v_act$shift)
          }
        }

        # BFGS refinement on internal parameters (u1, u2, log_w_ph, quad_shifts)
        n_q <- length(quad_info)
        init_internal <- c(1.0, 1.0, best_w_ph, vapply(quad_info, function(qi) qi$shift, numeric(1)))
        eval_internal_rss <- function(par_int) {
          u1 <- par_int[1]; u2 <- par_int[2]; w_p <- par_int[3]
          t_feat <- sin(w_p * exp(u1 * log_xa + u2 * log_xb))
          X_f <- cbind(1.0, t_feat)
          if (n_q > 0L) {
            for (qi_idx in seq_len(n_q)) {
              sh_i <- par_int[3L + qi_idx]
              X_f <- cbind(X_f, (x_mat[, quad_info[[qi_idx]]$j] + sh_i)^2)
            }
          }
          if (length(lin_sel2) > 0L) X_f <- cbind(X_f, x_mat[, lin_sel2, drop = FALSE])
          fit_i <- stats::lm.fit(X_f, y_vec)
          mean(fit_i$residuals^2)
        }
        opt_int <- tryCatch(
          stats::optim(init_internal, eval_internal_rss, method = "BFGS"),
          error = function(e) list(par = init_internal, value = eval_internal_rss(init_internal))
        )
        # Compute candidate validation BIC against current Stage 1 MPM model
        cand_bic <- n * log(max(opt_int$value, 1e-12)) + (length(opt_int$par) + length(lin_sel2) + 3.0) * log(n)
        k_mpm_curr <- 1.0 + sum(as.matrix(LBBNN$out_layer$alpha_active_path$cpu()) > 0.5)
        bic_mpm_curr <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_mpm_curr)$bic

        if (cand_bic < bic_mpm_curr) {
          torch::with_no_grad({
            l1$alpha_active_path$fill_(0.0)
            l2$alpha_active_path$fill_(0.0)
            l3$alpha_active_path$fill_(0.0)
            LBBNN$out_layer$alpha_active_path$fill_(0.0)

            # 1) Wire 3-hop clog(a_idx, b_idx) -> exp -> sin
            r_c1 <- clog_rows[a_idx]; c_c1 <- l1$clog_input_cols[a_idx]
            r_c2 <- clog_rows[b_idx]; c_c2 <- l1$clog_input_cols[b_idx]
            r_exp <- exp_rows[1]
            r_sin <- sin3_rows[1]

            l1$alpha_active_path[r_c1, c_c1]$fill_(1.0); l1$lambda_l[r_c1, c_c1]$fill_(4.0)
            l1$alpha_active_path[r_c2, c_c2]$fill_(1.0); l1$lambda_l[r_c2, c_c2]$fill_(4.0)

            l2$alpha_active_path[r_exp, r_c1]$fill_(1.0); l2$lambda_l[r_exp, r_c1]$fill_(4.0)
            l2$alpha_active_path[r_exp, r_c2]$fill_(1.0); l2$lambda_l[r_exp, r_c2]$fill_(4.0)
            l2$weight_mean[r_exp, r_c1]$fill_(opt_int$par[1])
            l2$weight_mean[r_exp, r_c2]$fill_(opt_int$par[2])
            l2$bias_mean[r_exp]$fill_(0.0)

            l3$alpha_active_path[r_sin, r_exp]$fill_(1.0); l3$lambda_l[r_sin, r_exp]$fill_(4.0)
            l3$weight_mean[r_sin, r_exp]$fill_(opt_int$par[3])
            l3$bias_mean[r_sin]$fill_(0.0)

            LBBNN$out_layer$alpha_active_path[1, r_sin]$fill_(1.0)
            LBBNN$out_layer$lambda_l[1, r_sin]$fill_(4.0)

            # 2) Wire shifted quadratic paths: L1_id -> L2_square -> L3_id -> out_layer
            if (n_q > 0L) {
              for (qi_idx in seq_len(n_q)) {
                jq <- quad_info[[qi_idx]]$j
                sh_q <- opt_int$par[3L + qi_idx]
                if (abs(sh_q - round(sh_q * 2) / 2) < 0.08) sh_q <- round(sh_q * 2) / 2
                r_id1 <- id1_rows[qi_idx]
                r_sq <- sq_rows[qi_idx]
                r_id3 <- id3_rows[qi_idx]

                l1$alpha_active_path[r_id1, jq]$fill_(1.0); l1$lambda_l[r_id1, jq]$fill_(4.0)
                l1$weight_mean[r_id1, jq]$fill_(1.0)
                l1$bias_mean[r_id1]$fill_(sh_q)

                l2$alpha_active_path[r_sq, r_id1]$fill_(1.0); l2$lambda_l[r_sq, r_id1]$fill_(4.0)
                l2$weight_mean[r_sq, r_id1]$fill_(1.0)
                l2$bias_mean[r_sq]$fill_(0.0)

                l3$alpha_active_path[r_id3, r_sq]$fill_(1.0); l3$lambda_l[r_id3, r_sq]$fill_(4.0)
                l3$weight_mean[r_id3, r_sq]$fill_(1.0)
                l3$bias_mean[r_id3]$fill_(0.0)

                LBBNN$out_layer$alpha_active_path[1, r_id3]$fill_(1.0)
                LBBNN$out_layer$lambda_l[1, r_id3]$fill_(4.0)
              }
            }

            # 3) Wire linear Input-Skip paths directly to out_layer
            for (jl in lin_sel2) {
              col_skip <- l3$out_features + jl
              LBBNN$out_layer$alpha_active_path[1, col_skip]$fill_(1.0)
              LBBNN$out_layer$lambda_l[1, col_skip]$fill_(4.0)
            }
          })
          refit_output_ols(LBBNN, train_full)
        }
      }
    }
  }
  } # end if (q_out == 1L)

  # 4. Multi-output (q_out > 1L) Stage 2 compositional refinement over shared sin/square/skip features
  if (q_out > 1L && n_hid == 2L && isTRUE(LBBNN$input_skip)) {
    l2 <- LBBNN$layers$children[[2]]
    sin1_rows <- which(l1$act_vec == "sin")
    id2_rows <- which(l2$act_vec == "id")
    sin2_rows <- which(l2$act_vec == "sin")
    sq2_rows <- which(l2$act_vec == "square")
    if (length(sin2_rows) + min(length(sin1_rows), length(id2_rows)) >= 1L &&
        length(sq2_rows) >= 2L) {
      y_mat <- as.matrix(train_full$y$cpu())
      # Candidate dictionary: linear skip (x_j), unary sin(x_j), unary x_j^2, bilinear pairs (x_a*x_b)
      cands <- list()
      cand_mats <- list()
      for (j in seq_len(p)) {
        cands[[length(cands) + 1L]] <- list(type = "lin", vars = j)
        cand_mats[[length(cand_mats) + 1L]] <- x_mat[, j]
        cands[[length(cands) + 1L]] <- list(type = "sin", vars = j)
        cand_mats[[length(cand_mats) + 1L]] <- sin(x_mat[, j])
        cands[[length(cands) + 1L]] <- list(type = "sq", vars = j)
        cand_mats[[length(cand_mats) + 1L]] <- x_mat[, j]^2
      }
      if (p >= 2L) {
        for (a in seq_len(p - 1L)) {
          for (b in (a + 1L):p) {
            cands[[length(cands) + 1L]] <- list(type = "pair", vars = c(a, b))
            cand_mats[[length(cand_mats) + 1L]] <- x_mat[, a] * x_mat[, b]
          }
        }
      }
      C_all <- do.call(cbind, cand_mats)

      # Stepwise BIC selection per equation r = 1, ..., q_out
      eq_sel <- vector("list", q_out)
      eq_int <- logical(q_out)
      tot_cand_bic <- 0.0
      max_sin_slots <- length(sin2_rows) + min(length(sin1_rows), length(id2_rows))

      for (r in seq_len(q_out)) {
        yr <- y_mat[, r]
        sel_idx <- integer(0)
        # Decide intercept vs zero-intercept by BIC as variables enter
        curr_bic <- Inf
        best_r <- list(sel = integer(0), use_int = TRUE)

        for (use_int_init in c(FALSE, TRUE)) {
          s_idx <- integer(0)
          X0 <- if (use_int_init) matrix(1.0, n, 1L) else matrix(nrow = n, ncol = 0L)
          rss0 <- if (use_int_init) mean((yr - mean(yr))^2) else mean(yr^2)
          b_curr <- n * log(max(rss0, 1e-12)) + (if (use_int_init) 1.0 else 0.0) * log(n)

          for (step_k in seq_len(min(p, 5L))) {
            b_best <- b_curr
            j_best <- NA_integer_
            for (jc in seq_len(ncol(C_all))) {
              if (jc %in% s_idx) next
              try_s <- c(s_idx, jc)
              X_try <- cbind(X0, C_all[, try_s, drop = FALSE])
              ft <- stats::lm.fit(X_try, yr)
              rss_t <- mean(ft$residuals^2)
              df_pen <- if (cands[[jc]]$type == "pair") 2.0 else 1.0
              # Count existing polarization pairs
              k_tot <- (if (use_int_init) 1.0 else 0.0) + sum(vapply(try_s, function(u) {
                if (cands[[u]]$type == "pair") 2.0 else if (cands[[u]]$type == "sin") 1.5 else 1.0
              }, numeric(1)))
              b_try <- n * log(max(rss_t, 1e-12)) + k_tot * log(n)
              if (b_try < b_best - 4.0) {
                b_best <- b_try
                j_best <- jc
              }
            }
            if (is.na(j_best)) break
            s_idx <- c(s_idx, j_best)
            b_curr <- b_best
          }
          if (b_curr < curr_bic) {
            curr_bic <- b_curr
            best_r <- list(sel = s_idx, use_int = use_int_init)
          }
        }
        eq_sel[[r]] <- best_r$sel
        eq_int[r] <- best_r$use_int
        tot_cand_bic <- tot_cand_bic + curr_bic
      }

      # Allocate required shared hidden neurons across all q_out equations
      uniq_sel <- unique(unlist(eq_sel))
      need_sin <- uniq_sel[vapply(uniq_sel, function(u) cands[[u]]$type == "sin", logical(1))]
      need_sq  <- uniq_sel[vapply(uniq_sel, function(u) cands[[u]]$type == "sq", logical(1))]
      need_pair <- uniq_sel[vapply(uniq_sel, function(u) cands[[u]]$type == "pair", logical(1))]

      # Check slot capacity in L1/L2 and validation BIC improvement
      has_cap <- (length(need_sin) <= max_sin_slots) &&
                 ((length(need_sq) + 2L * length(need_pair)) <= length(sq2_rows))

      if (has_cap) {
        k_curr_m <- as.numeric(q_out) + sum(as.matrix(LBBNN$out_layer$alpha_active_path$cpu()) > 0.5)
        bic_curr_m <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_curr_m)$bic

        if (tot_cand_bic < bic_curr_m) {
          torch::with_no_grad({
            l1$alpha_active_path$fill_(0.0)
            l2$alpha_active_path$fill_(0.0)
            LBBNN$out_layer$alpha_active_path$fill_(0.0)
            LBBNN$out_layer$bias_active <- eq_int

            feat_to_out_cols <- list()
            # 1) Direct skip linear features
            for (u in uniq_sel) {
              if (cands[[u]]$type == "lin") {
                feat_to_out_cols[[as.character(u)]] <- l2$out_features + cands[[u]]$vars
              }
            }
            # 2) Allocate sin(x_j) features onto L2 sin2_rows first (via Input-Skip col l1$out_features + j), then L1 sin->L2 id
            s_ptr <- 1L
            for (u in need_sin) {
              vj <- cands[[u]]$vars
              if (s_ptr <= length(sin2_rows)) {
                r2 <- sin2_rows[s_ptr]
                c_skip2 <- l1$out_features + vj
                l2$alpha_active_path[r2, c_skip2]$fill_(1.0); l2$lambda_l[r2, c_skip2]$fill_(4.0)
                l2$weight_mean[r2, c_skip2]$fill_(1.0); l2$bias_mean[r2]$fill_(0.0)
                feat_to_out_cols[[as.character(u)]] <- r2
              } else {
                off_s <- s_ptr - length(sin2_rows)
                r1 <- sin1_rows[off_s]; r2 <- id2_rows[off_s]
                l1$alpha_active_path[r1, vj]$fill_(1.0); l1$lambda_l[r1, vj]$fill_(4.0)
                l1$weight_mean[r1, vj]$fill_(1.0); l1$bias_mean[r1]$fill_(0.0)
                l2$alpha_active_path[r2, r1]$fill_(1.0); l2$lambda_l[r2, r1]$fill_(4.0)
                l2$weight_mean[r2, r1]$fill_(1.0); l2$bias_mean[r2]$fill_(0.0)
                feat_to_out_cols[[as.character(u)]] <- r2
              }
              s_ptr <- s_ptr + 1L
            }
            # 3) Allocate x_j^2 and polarization pairs (x_a+x_b)^2 - (x_a-x_b)^2 onto L2 sq2_rows via Input-Skip
            sq_ptr <- 1L
            for (u in need_sq) {
              vj <- cands[[u]]$vars
              r2 <- sq2_rows[sq_ptr]; sq_ptr <- sq_ptr + 1L
              c_skip2 <- l1$out_features + vj
              l2$alpha_active_path[r2, c_skip2]$fill_(1.0); l2$lambda_l[r2, c_skip2]$fill_(4.0)
              l2$weight_mean[r2, c_skip2]$fill_(1.0); l2$bias_mean[r2]$fill_(0.0)
              feat_to_out_cols[[as.character(u)]] <- r2
            }
            for (u in need_pair) {
              va <- cands[[u]]$vars[1]; vb <- cands[[u]]$vars[2]
              r2p <- sq2_rows[sq_ptr]; sq_ptr <- sq_ptr + 1L
              r2m <- sq2_rows[sq_ptr]; sq_ptr <- sq_ptr + 1L
              ca <- l1$out_features + va; cb <- l1$out_features + vb
              # (x_a + x_b)^2
              l2$alpha_active_path[r2p, ca]$fill_(1.0); l2$lambda_l[r2p, ca]$fill_(4.0)
              l2$alpha_active_path[r2p, cb]$fill_(1.0); l2$lambda_l[r2p, cb]$fill_(4.0)
              l2$weight_mean[r2p, ca]$fill_(1.0); l2$weight_mean[r2p, cb]$fill_(1.0)
              l2$bias_mean[r2p]$fill_(0.0)
              # (x_a - x_b)^2
              l2$alpha_active_path[r2m, ca]$fill_(1.0); l2$lambda_l[r2m, ca]$fill_(4.0)
              l2$alpha_active_path[r2m, cb]$fill_(1.0); l2$lambda_l[r2m, cb]$fill_(4.0)
              l2$weight_mean[r2m, ca]$fill_(1.0); l2$weight_mean[r2m, cb]$fill_(-1.0)
              l2$bias_mean[r2m]$fill_(0.0)
              feat_to_out_cols[[as.character(u)]] <- c(r2p, r2m)
            }

            # Wire out_layer per output equation r
            for (r in seq_len(q_out)) {
              for (u in eq_sel[[r]]) {
                cols_r <- feat_to_out_cols[[as.character(u)]]
                for (co in cols_r) {
                  LBBNN$out_layer$alpha_active_path[r, co]$fill_(1.0)
                  LBBNN$out_layer$lambda_l[r, co]$fill_(4.0)
                }
              }
            }
          })
          refit_output_ols(LBBNN, train_full)
        }
      }
    }
  }
}

#' @title Refit Output Linear Layer via Closed-Form Conjugate OLS on Active Basis Features
#' @keywords internal
refit_output_ols <- function(LBBNN, train_full, prior_prec = 1e-7) {
  torch::with_no_grad({
    feat_ten <- extract_mpm_features(LBBNN, train_full$x)
    F_all <- as.matrix(feat_ten$cpu())
    q_out <- as.integer(LBBNN$sizes[length(LBBNN$sizes)])
    y_raw <- if (q_out == 1L) {
      matrix(as.numeric(train_full$y$cpu()), ncol = 1L)
    } else {
      as.matrix(train_full$y$cpu())
    }

    for (r_out in seq_len(q_out)) {
      y_r <- y_raw[, r_out]
      act_out <- as.numeric(LBBNN$out_layer$alpha_active_path[r_out, ]$cpu())
      active_cols <- which(act_out > 0.5)
      use_int <- if (!is.null(LBBNN$out_layer$bias_active) && length(LBBNN$out_layer$bias_active) >= r_out) {
        isTRUE(LBBNN$out_layer$bias_active[r_out])
      } else {
        TRUE
      }

      if (length(active_cols) == 0L) {
        LBBNN$out_layer$bias_mean[r_out]$fill_(if (use_int) mean(y_r) else 0.0)
        next
      }

      F_mat <- F_all[, active_cols, drop = FALSE]
      X_des <- if (use_int) cbind(1.0, F_mat) else F_mat
      XtX <- crossprod(X_des) + diag(prior_prec, ncol(X_des))
      Xty <- crossprod(X_des, y_r)
      beta <- tryCatch(solve(XtX, Xty), error = function(e) qr.solve(X_des, y_r))

      offset_idx <- 0L
      if (use_int) {
        LBBNN$out_layer$bias_mean[r_out]$fill_(as.numeric(beta[1]))
        offset_idx <- 1L
      } else {
        LBBNN$out_layer$bias_mean[r_out]$fill_(0.0)
      }
      for (idx_k in seq_along(active_cols)) {
        col_j <- active_cols[idx_k]
        LBBNN$out_layer$weight_mean[r_out, col_j]$fill_(as.numeric(beta[idx_k + offset_idx]))
      }
    }
  })
}

#' @title Bayesian Laplace & Normal-Inverse-Gamma Readout Post-Training
#' @description
#' Computes exact conjugate normal-inverse-gamma (or weakly informative Laplace
#' prior) readout posteriors conditional on the active MPM symbolic features for
#' univariate or multivariate outputs \eqn{\bm{y}_i \in \mathbb{R}^q}. Optionally
#' runs backward marginal-likelihood / BIC pruning on each output row when
#' \code{prune_readout = TRUE}, propagates surviving active paths backwards
#' through hidden layers, and stores posterior means, standard errors,
#' \code{cred_level} Student-\eqn{t} credible intervals, and noise variance
#' estimates \eqn{\hat{\sigma}_{\epsilon,r}^2} on \code{LBBNN$readout_posterior}.
#' @param LBBNN A trained \code{\link{lbbnn_net}} object.
#' @param train_dl Training \code{torch::dataloader} or extracted tensor list.
#' @param val_dl Optional validation dataloader or extracted tensor list for
#'   readout pruning BIC comparisons (defaults to \code{train_dl}).
#' @param prior_sd Numeric weakly informative post-training slab SD
#'   \eqn{\sigma_{\text{post}}} (default \code{10.0}).
#' @param cred_level Numeric credible level (default \code{0.95}).
#' @param prune_readout Logical, whether to backward-prune low-amplitude readout
#'   paths via validation BIC (default \code{FALSE}).
#' @param delta_bic_cutoff Numeric BIC acceptance threshold for pruning
#'   (default \code{0.0}).
#' @param snap_output_weights Logical, whether to test rational/integer snapping
#'   on readout coefficients after storing continuous posteriors (default \code{FALSE}).
#' @param snap_tol Numeric tolerance for optional readout snapping (default \code{0.08}).
#' @param var_names Optional covariate names for labeling skip features.
#' @param device Compute device.
#' @return Invisibly returns the \code{readout_posterior} list.
#' @export
post_train_laplace <- function(LBBNN,
                               train_dl,
                               val_dl = NULL,
                               prior_sd = 10.0,
                               cred_level = 0.95,
                               prune_readout = FALSE,
                               delta_bic_cutoff = 0.0,
                               snap_output_weights = FALSE,
                               snap_tol = 0.08,
                               var_names = NULL,
                               device = "cpu") {
  train_full <- extract_dl_tensors(train_dl, device = device)
  val_full <- if (is.null(val_dl)) train_full else extract_dl_tensors(val_dl, device = device)
  q_out <- as.integer(LBBNN$sizes[length(LBBNN$sizes)])
  prior_prec <- 1.0 / (prior_sd^2)

  torch::with_no_grad({
    F_tr_all <- as.matrix(extract_mpm_features(LBBNN, train_full$x)$cpu())
    F_va_all <- as.matrix(extract_mpm_features(LBBNN, val_full$x)$cpu())
    y_tr_all <- if (q_out == 1L) {
      matrix(as.numeric(train_full$y$cpu()), ncol = 1L)
    } else {
      as.matrix(train_full$y$cpu())
    }
    y_va_all <- if (q_out == 1L) {
      matrix(as.numeric(val_full$y$cpu()), ncol = 1L)
    } else {
      as.matrix(val_full$y$cpu())
    }
    n_tr <- nrow(F_tr_all)
    n_va <- nrow(F_va_all)

    bias_active <- logical(q_out)
    for (r_out in seq_len(q_out)) {
      b_val <- as.numeric(LBBNN$out_layer$bias_mean[r_out]$cpu())
      if (!is.null(LBBNN$out_layer$bias_active) && length(LBBNN$out_layer$bias_active) >= r_out) {
        bias_active[r_out] <- isTRUE(LBBNN$out_layer$bias_active[r_out])
      } else {
        bias_active[r_out] <- abs(b_val) > 1e-9
      }
    }
    LBBNN$out_layer$bias_active <- bias_active

    # Helper to compute conjugate Laplace/NIG fit for row r_out
    fit_nig_row <- function(F_tr, y_tr, F_va, y_va, cols, use_int) {
      if (length(cols) == 0L && !use_int) return(NULL)
      X_tr <- if (length(cols) > 0L) F_tr[, cols, drop = FALSE] else matrix(nrow = n_tr, ncol = 0L)
      X_va <- if (length(cols) > 0L) F_va[, cols, drop = FALSE] else matrix(nrow = n_va, ncol = 0L)
      if (use_int) {
        X_tr <- cbind(1.0, X_tr)
        X_va <- cbind(1.0, X_va)
      }
      XtX <- crossprod(X_tr) + diag(prior_prec, ncol(X_tr))
      V_mat <- tryCatch(solve(XtX), error = function(e) MASS::ginv(XtX))
      beta_hat <- as.numeric(V_mat %*% crossprod(X_tr, y_tr))
      pred_va <- as.numeric(X_va %*% beta_hat)
      mse_va <- max(mean((y_va - pred_va)^2), 1e-12)
      k_eff <- ncol(X_tr)
      bic_va <- n_va * log(mse_va) + k_eff * log(n_va)

      res_tr <- y_tr - as.numeric(X_tr %*% beta_hat)
      df_res <- max(1L, n_tr - ncol(X_tr))
      sigma2_hat <- sum(res_tr^2) / df_res
      se_hat <- sqrt(pmax(sigma2_hat * diag(V_mat), 1e-16))
      t_crit <- stats::qt(1.0 - (1.0 - cred_level) / 2.0, df = df_res)
      list(
        beta = beta_hat,
        se = se_hat,
        ci_low = beta_hat - t_crit * se_hat,
        ci_high = beta_hat + t_crit * se_hat,
        sigma_eps = sqrt(sigma2_hat),
        df = df_res,
        bic = bic_va,
        mse_va = mse_va
      )
    }

    # Optional backward BIC readout pruning per output equation
    any_pruned <- FALSE
    if (isTRUE(prune_readout)) {
      for (r_out in seq_len(q_out)) {
        act_cols <- which(as.numeric(LBBNN$out_layer$alpha_active_path[r_out, ]$cpu()) > 0.5)
        use_int <- bias_active[r_out]
        y_tr <- y_tr_all[, r_out]
        y_va <- y_va_all[, r_out]
        if (length(act_cols) <= 1L && !use_int) next

        repeat {
          curr_fit <- fit_nig_row(F_tr_all, y_tr, F_va_all, y_va, act_cols, use_int)
          if (is.null(curr_fit)) break
          best_bic <- curr_fit$bic
          best_action <- NULL

          if (length(act_cols) > 1L) {
            for (c_try in act_cols) {
              cand_cols <- setdiff(act_cols, c_try)
              cand_fit <- fit_nig_row(F_tr_all, y_tr, F_va_all, y_va, cand_cols, use_int)
              if (!is.null(cand_fit) && (cand_fit$bic - curr_fit$bic) < delta_bic_cutoff) {
                if (cand_fit$bic < best_bic) {
                  best_bic <- cand_fit$bic
                  best_action <- list(type = "col", val = c_try)
                }
              }
            }
          }
          if (isTRUE(use_int) && length(act_cols) >= 1L) {
            cand_fit <- fit_nig_row(F_tr_all, y_tr, F_va_all, y_va, act_cols, FALSE)
            if (!is.null(cand_fit) && (cand_fit$bic - curr_fit$bic) < delta_bic_cutoff) {
              if (cand_fit$bic < best_bic) {
                best_bic <- cand_fit$bic
                best_action <- list(type = "int")
              }
            }
          }
          if (is.null(best_action)) break
          any_pruned <- TRUE
          if (best_action$type == "col") {
            act_cols <- setdiff(act_cols, best_action$val)
            LBBNN$out_layer$alpha_active_path[r_out, best_action$val]$fill_(0.0)
            LBBNN$out_layer$weight_mean[r_out, best_action$val]$fill_(0.0)
          } else {
            use_int <- FALSE
            bias_active[r_out] <- FALSE
            LBBNN$out_layer$bias_mean[r_out]$fill_(0.0)
          }
        }
      }
      LBBNN$out_layer$bias_active <- bias_active
      if (any_pruned && !is.null(LBBNN$propagate_active_paths)) {
        LBBNN$propagate_active_paths()
      }
    }

    # Compute & store exact conjugate posteriors across all outputs r = 1, ..., q_out
    eq_posts <- vector("list", q_out)
    sum_rows <- list()
    n_hid <- length(LBBNN$layers$children)
    n_prev <- if (n_hid >= 1L) LBBNN$layers$children[[n_hid]]$out_features else LBBNN$sizes[1]

    for (r_out in seq_len(q_out)) {
      act_cols <- which(as.numeric(LBBNN$out_layer$alpha_active_path[r_out, ]$cpu()) > 0.5)
      use_int <- bias_active[r_out]
      y_tr <- y_tr_all[, r_out]
      y_va <- y_va_all[, r_out]

      if (length(act_cols) == 0L && !use_int) {
        use_int <- TRUE
        bias_active[r_out] <- TRUE
      }
      fit_r <- fit_nig_row(F_tr_all, y_tr, F_va_all, y_va, act_cols, use_int)

      # Write continuous posterior mean onto out_layer
      if (use_int) {
        LBBNN$out_layer$bias_mean[r_out]$fill_(fit_r$beta[1])
        off <- 1L
      } else {
        LBBNN$out_layer$bias_mean[r_out]$fill_(0.0)
        off <- 0L
      }
      LBBNN$out_layer$weight_mean[r_out, ]$fill_(0.0)
      for (idx_k in seq_along(act_cols)) {
        LBBNN$out_layer$weight_mean[r_out, act_cols[idx_k]]$fill_(fit_r$beta[idx_k + off])
      }

      term_labels <- character(0)
      col_indices <- integer(0)
      if (use_int) {
        term_labels <- c(term_labels, "(Intercept)")
        col_indices <- c(col_indices, 0L)
      }
      for (c_j in act_cols) {
        lbl <- if (isTRUE(LBBNN$input_skip) && c_j > n_prev) {
          v_idx <- c_j - n_prev
          if (!is.null(var_names) && v_idx <= length(var_names)) paste0("skip:", var_names[v_idx]) else paste0("skip:x", v_idx)
        } else {
          paste0("L", n_hid, "_N", c_j)
        }
        term_labels <- c(term_labels, lbl)
        col_indices <- c(col_indices, c_j)
      }

      df_r <- data.frame(
        output_idx = r_out,
        term = term_labels,
        col_idx = col_indices,
        estimate = fit_r$beta,
        se = fit_r$se,
        ci_low = fit_r$ci_low,
        ci_high = fit_r$ci_high,
        sigma_eps = fit_r$sigma_eps,
        df = fit_r$df,
        stringsAsFactors = FALSE
      )
      eq_posts[[r_out]] <- df_r
      sum_rows[[length(sum_rows) + 1L]] <- df_r

      # Optional integer/rational snapping on readout multiplier after recording continuous posterior
      if (isTRUE(snap_output_weights) && !isTRUE(prune_readout)) {
        cands <- get_rational_candidates()
        k_curr_r <- as.numeric(length(act_cols) + if (use_int) 1L else 0L)
        for (idx_k in seq_along(act_cols)) {
          c_out <- act_cols[idx_k]
          w_o <- fit_r$beta[idx_k + off]
          diffs_o <- abs(cands$val - w_o)
          bi <- which.min(diffs_o)
          if (diffs_o[bi] < (0.5 * snap_tol) && cands$val[bi] != 0) {
            bic_b <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_curr_r, out_row = r_out)$bic
            saved_wo <- LBBNN$out_layer$weight_mean[r_out, c_out]$clone()
            LBBNN$out_layer$weight_mean[r_out, c_out]$fill_(cands$val[bi])
            bic_a <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_curr_r - 0.5, out_row = r_out)$bic
            if ((bic_a - bic_b) < 2.0) {
              k_curr_r <- k_curr_r - 0.5
              LBBNN$out_layer$snapped_rationals[[paste0(r_out, "_", c_out)]] <- list(
                row = r_out, col = c_out, w_debiased = w_o,
                q_val = cands$val[bi], q_label = cands$label[bi]
              )
            } else {
              LBBNN$out_layer$weight_mean[r_out, c_out]$copy_(saved_wo)
            }
          }
        }
      }
    }

    LBBNN$readout_posterior <- list(
      equations = eq_posts,
      summary_df = do.call(rbind, sum_rows),
      cred_level = cred_level,
      prior_sd = prior_sd
    )
  })

  invisible(LBBNN$readout_posterior)
}

#' @title BIC-Guided Pruning of Weak Active Paths and Output OLS Refit
#' @keywords internal
prune_and_refit_mpm <- function(LBBNN, train_full, val_full, debias_lr = 0.005, device = "cpu") {
  refit_output_ols(LBBNN, train_full)
  q_out <- as.integer(LBBNN$sizes[length(LBBNN$sizes)])

  # Count active parameters
  count_active <- function() {
    k <- as.numeric(q_out)
    for (l in LBBNN$layers$children) {
      act_m <- as.matrix(l$alpha_active_path$cpu())
      w_m <- if (!is.null(l$weight_grad_mask)) as.matrix(l$weight_grad_mask$cpu()) else matrix(1, nrow(act_m), ncol(act_m))
      k <- k + sum(act_m * w_m > 0.5)
    }
    k <- k + sum(as.matrix(LBBNN$out_layer$alpha_active_path$cpu()) > 0.5)
    k
  }

  # 1. Prune weak incoming edges into exp or logic neurons if BIC improves
  for (l_idx in seq_along(LBBNN$layers$children)) {
    l <- LBBNN$layers$children[[l_idx]]
    if (is.null(l$act_vec)) next
    target_rows <- which(l$act_vec %in% c("exp", "logic_and", "logic_or"))
    for (r in target_rows) {
      act_row <- as.numeric(l$alpha_active_path[r, ]$cpu())
      active_cols <- which(act_row > 0.5)
      if (length(active_cols) <= 1L) next
      w_row <- abs(as.numeric(l$weight_mean[r, active_cols]$cpu()))
      ord <- active_cols[order(w_row)]
      for (c_try in ord) {
        if (sum(as.numeric(l$alpha_active_path[r, ]$cpu()) > 0.5) <= 1L) break
        k_curr <- count_active()
        bic_curr <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_curr)$bic
        saved_out_w <- LBBNN$out_layer$weight_mean$clone()
        saved_out_b <- LBBNN$out_layer$bias_mean$clone()

        torch::with_no_grad({
          l$alpha_active_path[r, c_try]$fill_(0.0)
        })
        refit_output_ols(LBBNN, train_full)
        bic_new <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_curr - 1.0)$bic

        if (bic_new > bic_curr) {
          torch::with_no_grad({
            l$alpha_active_path[r, c_try]$fill_(1.0)
            LBBNN$out_layer$weight_mean$copy_(saved_out_w)
            LBBNN$out_layer$bias_mean$copy_(saved_out_b)
          })
        }
      }
    }
  }

  # 2. Prune weak output-layer edges per output dimension r = 1, ..., q_out if BIC improves
  for (r_out in seq_len(q_out)) {
    act_out <- as.numeric(LBBNN$out_layer$alpha_active_path[r_out, ]$cpu())
    active_cols <- which(act_out > 0.5)
    if (length(active_cols) > 1L) {
      w_out <- abs(as.numeric(LBBNN$out_layer$weight_mean[r_out, active_cols]$cpu()))
      ord_out <- active_cols[order(w_out)]
      for (c_try in ord_out) {
        if (sum(as.numeric(LBBNN$out_layer$alpha_active_path[r_out, ]$cpu()) > 0.5) <= 1L) break
        k_curr <- 1.0 + sum(as.numeric(LBBNN$out_layer$alpha_active_path[r_out, ]$cpu()) > 0.5)
        bic_curr <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_curr, out_row = r_out)$bic
        saved_out_w <- LBBNN$out_layer$weight_mean$clone()
        saved_out_b <- LBBNN$out_layer$bias_mean$clone()

        torch::with_no_grad({
          LBBNN$out_layer$alpha_active_path[r_out, c_try]$fill_(0.0)
        })
        refit_output_ols(LBBNN, train_full)
        bic_new <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_curr - 1.0, out_row = r_out)$bic

        if (bic_new > bic_curr) {
          torch::with_no_grad({
            LBBNN$out_layer$alpha_active_path[r_out, c_try]$fill_(1.0)
            LBBNN$out_layer$weight_mean$copy_(saved_out_w)
            LBBNN$out_layer$bias_mean$copy_(saved_out_b)
          })
        }
      }
    }
  }

  # Propagate output pruning backwards across arbitrary hidden depth L >= 2
  LBBNN$propagate_active_paths()
}

#' @title Stage 2B Rational Exponent & Phase Constant Snapping
#' @keywords internal
snap_rational_exponents <- function(LBBNN,
                                    train_full,
                                    val_full,
                                    snap_tol = 0.08,
                                    delta_bic_cutoff = 2.0,
                                    verbose = FALSE) {
  cands <- get_rational_candidates()
  phase_cands <- data.frame(
    val = c(-2 * pi, -pi, -pi / 2, -2, -1, 0, 1, 2, pi / 2, pi, 2 * pi),
    label = c("-2*pi", "-pi", "-pi/2", "-2", "-1", "0", "1", "2", "pi/2", "pi", "2*pi"),
    stringsAsFactors = FALSE
  )
  records <- list()
  n_hid <- length(LBBNN$layers$children)
  q_out <- as.integer(LBBNN$sizes[length(LBBNN$sizes)])
  if (n_hid < 2L) return(records)

  # Base active free parameter count
  k_free <- as.numeric(q_out) + sum(as.matrix(LBBNN$out_layer$alpha_active_path$cpu()) > 0.5)
  for (l in LBBNN$layers$children) {
    k_free <- k_free + sum(as.matrix((l$alpha_active_path * l$weight_grad_mask)$cpu()) > 0.5)
  }

  for (l_idx in 2:n_hid) {
    l_curr <- LBBNN$layers$children[[l_idx]]
    l_prev <- LBBNN$layers$children[[l_idx - 1L]]
    act_curr <- as.matrix(l_curr$alpha_active_path$cpu())
    w_curr <- as.matrix(l_curr$weight_mean$cpu())

    # 1. Snap clog/log -> exp exponents
    exp_rows <- which(l_curr$act_vec == "exp")
    for (r_exp in exp_rows) {
      active_cols <- which(act_curr[r_exp, ] > 0.5)
      for (c_in in active_cols) {
        if (c_in > l_prev$out_features) next
        src_op <- l_prev$act_vec[c_in]
        if (!src_op %in% c("clog", "log")) next

        w_val <- w_curr[r_exp, c_in]
        diffs <- abs(cands$val - w_val)
        best_idx <- which.min(diffs)
        q_val <- cands$val[best_idx]
        q_label <- cands$label[best_idx]
        dist_q <- diffs[best_idx]

        if (dist_q < snap_tol) {
          bic_before <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_free)$bic
          saved_w <- l_curr$weight_mean[r_exp, c_in]$clone()
          saved_act <- l_curr$alpha_active_path[r_exp, c_in]$clone()
          saved_l1_act <- l_prev$alpha_active_path[c_in, ]$clone()
          saved_out_w <- LBBNN$out_layer$weight_mean$clone()
          saved_out_b <- LBBNN$out_layer$bias_mean$clone()

          torch::with_no_grad({
            l_curr$weight_mean[r_exp, c_in]$fill_(q_val)
            if (q_val == 0) {
              l_curr$alpha_active_path[r_exp, c_in]$fill_(0.0)
              if (sum(as.numeric(l_curr$alpha_active_path[, c_in]$cpu()) > 0.5) == 0L) {
                l_prev$alpha_active_path[c_in, ]$fill_(0.0)
              }
            }
          })
          refit_output_ols(LBBNN, train_full)
          k_new <- if (q_val == 0) (k_free - 1.0) else (k_free - 0.5)
          bic_after <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_new)$bic
          delta_bic <- bic_after - bic_before

          if (delta_bic < delta_bic_cutoff) {
            k_free <- k_new
            if (q_val != 0) {
              key <- paste0(r_exp, "_", c_in)
              l_curr$snapped_rationals[[key]] <- list(
                row = r_exp, col = c_in,
                w_debiased = w_val, q_val = q_val, q_label = q_label,
                delta_bic = delta_bic
              )
              records[[length(records) + 1L]] <- l_curr$snapped_rationals[[key]]
            }
            if (isTRUE(verbose)) {
              message(sprintf(
                "[Stage 2B] Snapped Layer %d exp[%d, %d] weight %.4f -> %s (dBIC = %.2f)",
                l_idx, r_exp, c_in, w_val, q_label, delta_bic
              ))
            }
          } else {
            torch::with_no_grad({
              l_curr$weight_mean[r_exp, c_in]$copy_(saved_w)
              l_curr$alpha_active_path[r_exp, c_in]$copy_(saved_act)
              l_prev$alpha_active_path[c_in, ]$copy_(saved_l1_act)
              LBBNN$out_layer$weight_mean$copy_(saved_out_w)
              LBBNN$out_layer$bias_mean$copy_(saved_out_b)
            })
          }
        }
      }
    }

    # 2. Snap phase constants (e.g. pi) on incoming edges to sin/cos neurons in 3+ layer compositions
    trig_rows <- which(l_curr$act_vec %in% c("sin", "cos"))
    for (r_tr in trig_rows) {
      active_cols <- which(act_curr[r_tr, ] > 0.5)
      for (c_in in active_cols) {
        w_val <- as.numeric(l_curr$weight_mean[r_tr, c_in]$cpu())
        diffs_p <- abs(phase_cands$val - w_val)
        bp <- which.min(diffs_p)
        if (diffs_p[bp] < snap_tol && phase_cands$val[bp] != 0) {
          bic_before <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_free)$bic
          saved_w <- l_curr$weight_mean[r_tr, c_in]$clone()
          saved_out_w <- LBBNN$out_layer$weight_mean$clone()
          saved_out_b <- LBBNN$out_layer$bias_mean$clone()

          torch::with_no_grad({
            l_curr$weight_mean[r_tr, c_in]$fill_(phase_cands$val[bp])
          })
          refit_output_ols(LBBNN, train_full)
          bic_after <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_free - 0.5)$bic
          if ((bic_after - bic_before) < delta_bic_cutoff) {
            k_free <- k_free - 0.5
            key <- paste0(r_tr, "_", c_in)
            l_curr$snapped_rationals[[key]] <- list(
              row = r_tr, col = c_in, w_debiased = w_val,
              q_val = phase_cands$val[bp], q_label = phase_cands$label[bp],
              delta_bic = bic_after - bic_before
            )
            records[[length(records) + 1L]] <- l_curr$snapped_rationals[[key]]
          } else {
            torch::with_no_grad({
              l_curr$weight_mean[r_tr, c_in]$copy_(saved_w)
              LBBNN$out_layer$weight_mean$copy_(saved_out_w)
              LBBNN$out_layer$bias_mean$copy_(saved_out_b)
            })
          }
        }
      }
    }
  }

  # 3. Check if output-layer intercepts are within snap_tol of zero across all q_out outputs
  torch::with_no_grad({
    if (is.null(LBBNN$out_layer$bias_active)) {
      LBBNN$out_layer$bias_active <- rep(TRUE, q_out)
    }
    for (r_out in seq_len(q_out)) {
      b_out <- as.numeric(LBBNN$out_layer$bias_mean[r_out]$cpu())
      if (abs(b_out) < snap_tol) {
        bic_before <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_free, out_row = r_out)$bic
        saved_b <- LBBNN$out_layer$bias_mean[r_out]$clone()
        LBBNN$out_layer$bias_mean[r_out]$fill_(0.0)
        bic_after <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_free - 0.5, out_row = r_out)$bic
        if ((bic_after - bic_before) < delta_bic_cutoff) {
          k_free <- k_free - 0.5
          LBBNN$out_layer$bias_active[r_out] <- FALSE
          LBBNN$out_layer$snapped_rationals[[paste0("bias_", r_out)]] <- list(
            row = r_out, col = 0L, w_debiased = b_out, q_val = 0, q_label = "0"
          )
        } else {
          LBBNN$out_layer$bias_mean[r_out]$copy_(saved_b)
        }
      }
    }
  })

  records
}
