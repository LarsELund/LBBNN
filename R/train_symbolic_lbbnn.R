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
  xs <- list()
  ys <- list()
  coro::loop(for (b in dl) {
    xs[[length(xs) + 1L]] <- b[[1]]$to(device = device)
    ys[[length(ys) + 1L]] <- b[[2]]$to(device = device)
  })
  x_all <- torch::torch_cat(xs, dim = 1)
  y_all <- torch::torch_cat(ys, dim = 1)$squeeze()
  list(x = x_all, y = y_all, n = as.integer(dim(x_all)[1]))
}

#' @title Compute Deterministic MPM MSE and BIC
#' @keywords internal
compute_mpm_bic <- function(LBBNN, x_ten, y_ten, k_eff) {
  n <- as.numeric(dim(x_ten)[1])
  torch::with_no_grad({
    pred <- LBBNN(x_ten, MPM = TRUE)$squeeze()
    mse <- as.numeric(torch::torch_mean((pred - y_ten)^2)$cpu())
  })
  mse_safe <- max(mse, 1e-12)
  bic <- n * log(mse_safe) + k_eff * log(n)
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
#' @param verbose Logical, whether to print progress messages (default \code{FALSE}).
#' @return Invisibly returns a list with Stage 1 & Stage 2 loss histories,
#'   active path densities, and rational snapping summary.
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
                                 verbose = FALSE) {
  debias_optimizer <- match.arg(debias_optimizer)
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
      } else {
        output <- output$squeeze()
        target <- target$squeeze()
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
    if (debias_optimizer == "lbfgs") {
      opt_debias <- torch::optim_lbfgs(LBBNN$parameters, lr = debias_lr, max_iter = 20)
      for (ep in seq_len(debias_epochs)) {
        closure <- function() {
          opt_debias$zero_grad()
          out <- LBBNN(train_full$x, MPM = TRUE)$squeeze()
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
        out <- LBBNN(train_full$x, MPM = TRUE)$squeeze()
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
  # Stage 2B: Rational Exponent Snapping (BIC-guided)
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

  # Store final fitted values & residuals for S3 methods
  torch::with_no_grad({
    LBBNN$y <- as.numeric(train_full$y$cpu())
    LBBNN$r <- as.numeric(LBBNN(train_full$x, MPM = TRUE)$squeeze()$cpu())
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
    snapped_rationals = snapped_records
  ))
}

#' @title Stage 2A Hybrid BGNLM Debiasing for Monomial (clog -> exp) and Logic Paths
#' @keywords internal
stage2_bgnlm_debias <- function(LBBNN, train_full, val_full) {
  if (!isTRUE(LBBNN$is_symbolic)) return(invisible(NULL))
  x_mat <- as.matrix(train_full$x$cpu())
  y_vec <- as.numeric(train_full$y$cpu())
  n <- nrow(x_mat)
  p <- ncol(x_mat)
  n_hid <- length(LBBNN$layers$children)

  # 1. If binary inputs and Layer 1 has logic_and neurons, refine active logic clauses via BIC
  l1 <- LBBNN$layers$children[[1]]
  is_binary_x <- all(abs(x_mat - round(x_mat)) < 1e-4) && all(x_mat >= 0 & x_mat <= 1)
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

  # 2. If 2-layer network with Layer 1 clog and Layer 2 exp on positive data, check log-monomial BIC
  if (n_hid >= 2L && all(x_mat > 0) && all(y_vec > 0)) {
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
        })
      }
    }
  }
}

#' @title Refit Output Linear Layer via Closed-Form OLS on Active Basis Features
#' @keywords internal
refit_output_ols <- function(LBBNN, train_full) {
  torch::with_no_grad({
    x <- train_full$x
    y <- as.numeric(train_full$y$cpu())
    n <- length(y)
    if (!isTRUE(LBBNN$input_skip)) {
      h <- x$view(c(-1, LBBNN$sizes[1]))
      for (l in LBBNN$layers$children) {
        h <- if (isTRUE(LBBNN$is_symbolic)) l(h, MPM = TRUE) else LBBNN$act(l(h, MPM = TRUE))
      }
      feat_ten <- h
    } else {
      x_in <- x$view(c(-1, LBBNN$sizes[1]))
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
      feat_ten <- torch::torch_cat(c(h, x_in), dim = 2)
    }

    act_out <- as.numeric(LBBNN$out_layer$alpha_active_path[1, ]$cpu())
    active_cols <- which(act_out > 0.5)
    if (length(active_cols) == 0L) {
      LBBNN$out_layer$bias_mean[1]$fill_(mean(y))
      return(invisible(NULL))
    }

    F_mat <- as.matrix(feat_ten[, active_cols, drop = FALSE]$cpu())
    X_des <- cbind(1.0, F_mat)
    # Ridge-stabilized OLS solve
    XtX <- crossprod(X_des) + diag(1e-7, ncol(X_des))
    Xty <- crossprod(X_des, y)
    beta <- tryCatch(solve(XtX, Xty), error = function(e) qr.solve(X_des, y))

    LBBNN$out_layer$bias_mean[1]$fill_(as.numeric(beta[1]))
    for (idx_k in seq_along(active_cols)) {
      col_j <- active_cols[idx_k]
      LBBNN$out_layer$weight_mean[1, col_j]$fill_(as.numeric(beta[idx_k + 1L]))
    }
  })
}

#' @title BIC-Guided Pruning of Weak Active Paths and Output OLS Refit
#' @keywords internal
prune_and_refit_mpm <- function(LBBNN, train_full, val_full, debias_lr = 0.005, device = "cpu") {
  refit_output_ols(LBBNN, train_full)

  # Count active parameters
  count_active <- function() {
    k <- 1.0 # output intercept
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
      # Order candidate columns by smallest |weight|
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

        if (bic_new <= bic_curr) {
          # Accept pruning of edge (r, c_try)
        } else {
          torch::with_no_grad({
            l$alpha_active_path[r, c_try]$fill_(1.0)
            LBBNN$out_layer$weight_mean$copy_(saved_out_w)
            LBBNN$out_layer$bias_mean$copy_(saved_out_b)
          })
        }
      }
    }
  }

  # 2. Prune weak output-layer edges if BIC improves
  act_out <- as.numeric(LBBNN$out_layer$alpha_active_path[1, ]$cpu())
  active_cols <- which(act_out > 0.5)
  if (length(active_cols) > 1L) {
    w_out <- abs(as.numeric(LBBNN$out_layer$weight_mean[1, active_cols]$cpu()))
    ord_out <- active_cols[order(w_out)]
    for (c_try in ord_out) {
      if (sum(as.numeric(LBBNN$out_layer$alpha_active_path[1, ]$cpu()) > 0.5) <= 1L) break
      k_curr <- count_active()
      bic_curr <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_curr)$bic
      saved_out_w <- LBBNN$out_layer$weight_mean$clone()
      saved_out_b <- LBBNN$out_layer$bias_mean$clone()

      torch::with_no_grad({
        LBBNN$out_layer$alpha_active_path[1, c_try]$fill_(0.0)
      })
      refit_output_ols(LBBNN, train_full)
      bic_new <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_curr - 1.0)$bic

      if (bic_new > bic_curr) {
        torch::with_no_grad({
          LBBNN$out_layer$alpha_active_path[1, c_try]$fill_(1.0)
          LBBNN$out_layer$weight_mean$copy_(saved_out_w)
          LBBNN$out_layer$bias_mean$copy_(saved_out_b)
        })
      }
    }
  }

  # Propagate output pruning backwards to update hidden layer alpha_active_path
  torch::with_no_grad({
    n_hid <- length(LBBNN$layers$children)
    if (n_hid == 1L) {
      l1 <- LBBNN$layers$children[[1]]
      out_act <- as.numeric(LBBNN$out_layer$alpha_active_path[1, 1:l1$out_features]$cpu())
      for (r in seq_len(l1$out_features)) {
        if (out_act[r] < 0.5) {
          l1$alpha_active_path[r, ]$fill_(0.0)
        }
      }
    } else if (n_hid >= 2L) {
      l2 <- LBBNN$layers$children[[2]]
      l1 <- LBBNN$layers$children[[1]]
      out_act <- as.numeric(LBBNN$out_layer$alpha_active_path[1, 1:l2$out_features]$cpu())
      for (r in seq_len(l2$out_features)) {
        if (out_act[r] < 0.5) {
          l2$alpha_active_path[r, ]$fill_(0.0)
        }
      }
      l2_act <- as.matrix(l2$alpha_active_path[, 1:l1$out_features, drop = FALSE]$cpu())
      for (r1 in seq_len(l1$out_features)) {
        if (sum(l2_act[, r1] > 0.5) == 0L) {
          l1$alpha_active_path[r1, ]$fill_(0.0)
        }
      }
    }
  })
}

#' @title Stage 2B Rational Exponent Snapping on Active clog/log -> exp Edges
#' @keywords internal
snap_rational_exponents <- function(LBBNN,
                                    train_full,
                                    val_full,
                                    snap_tol = 0.08,
                                    delta_bic_cutoff = 2.0,
                                    verbose = FALSE) {
  cands <- get_rational_candidates()
  records <- list()
  n_hid <- length(LBBNN$layers$children)
  if (n_hid < 2L) return(records)

  # Base active free parameter count
  k_free <- 1.0 + sum(as.matrix(LBBNN$out_layer$alpha_active_path$cpu()) > 0.5)
  for (l in LBBNN$layers$children) {
    k_free <- k_free + sum(as.matrix((l$alpha_active_path * l$weight_grad_mask)$cpu()) > 0.5)
  }

  for (l_idx in 2:n_hid) {
    l_curr <- LBBNN$layers$children[[l_idx]]
    l_prev <- LBBNN$layers$children[[l_idx - 1L]]
    exp_rows <- which(l_curr$act_vec == "exp")
    if (length(exp_rows) == 0L) next

    act_curr <- as.matrix(l_curr$alpha_active_path$cpu())
    w_curr <- as.matrix(l_curr$weight_mean$cpu())

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
          # Snapped rational counts as 0.5 effective parameter (or 0 if pruned to 0)
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
  }

  # Also check if output-layer weights/intercept on pure monomial paths are within snap_tol of integer/rational (e.g. 1.0, 0.0)
  torch::with_no_grad({
    b_out <- as.numeric(LBBNN$out_layer$bias_mean[1]$cpu())
    if (abs(b_out) < snap_tol) {
      bic_before <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_free)$bic
      saved_b <- LBBNN$out_layer$bias_mean[1]$clone()
      LBBNN$out_layer$bias_mean[1]$fill_(0.0)
      bic_after <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_free - 0.5)$bic
      if ((bic_after - bic_before) < delta_bic_cutoff) {
        k_free <- k_free - 0.5
      } else {
        LBBNN$out_layer$bias_mean[1]$copy_(saved_b)
      }
    }

    act_out <- as.numeric(LBBNN$out_layer$alpha_active_path[1, ]$cpu())
    for (c_out in which(act_out > 0.5)) {
      w_o <- as.numeric(LBBNN$out_layer$weight_mean[1, c_out]$cpu())
      diffs_o <- abs(cands$val - w_o)
      bi <- which.min(diffs_o)
      if (diffs_o[bi] < (0.5 * snap_tol) && cands$val[bi] != 0) {
        bic_before <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_free)$bic
        saved_wo <- LBBNN$out_layer$weight_mean[1, c_out]$clone()
        LBBNN$out_layer$weight_mean[1, c_out]$fill_(cands$val[bi])
        bic_after <- compute_mpm_bic(LBBNN, val_full$x, val_full$y, k_free - 0.5)$bic
        if ((bic_after - bic_before) < delta_bic_cutoff) {
          k_free <- k_free - 0.5
          LBBNN$out_layer$snapped_rationals[[paste0("1_", c_out)]] <- list(
            row = 1L, col = c_out, w_debiased = w_o,
            q_val = cands$val[bi], q_label = cands$label[bi]
          )
        } else {
          LBBNN$out_layer$weight_mean[1, c_out]$copy_(saved_wo)
        }
      }
    }
  })

  records
}
