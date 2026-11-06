#' @title Format Numeric Value or Rational Label
#' @keywords internal
fmt_num <- function(x, digits = 3, rational_label = NULL) {
  if (!is.null(rational_label) && nzchar(rational_label)) {
    if (grepl("/", rational_label, fixed = TRUE)) {
      return(paste0("(", rational_label, ")"))
    }
    return(rational_label)
  }
  if (abs(x - round(x)) < 1e-6) {
    return(as.character(as.integer(round(x))))
  }
  as.character(round(x, digits))
}

#' @title Format LaTeX Exponent or Coefficient
#' @keywords internal
fmt_latex_exp <- function(x, digits = 3, rational_label = NULL) {
  if (!is.null(rational_label) && nzchar(rational_label)) {
    return(rational_label)
  }
  if (abs(x - round(x)) < 1e-6) {
    return(as.character(as.integer(round(x))))
  }
  as.character(round(x, digits))
}

#' @title Format Variable Name for LaTeX
#' @keywords internal
var_to_latex <- function(v) {
  if (grepl("^([a-zA-Z]+)_?([0-9]+)$", v)) {
    sub("^([a-zA-Z]+)_?([0-9]+)$", "\\1_{\\2}", v)
  } else {
    gsub("_", "\\_", v, fixed = TRUE)
  }
}

#' @title Extract Closed-Form Symbolic Expression from Active MPM Paths
#' @description
#' Performs a bottom-up symbolic traversal across all active paths of a trained
#' Neuro-Symbolic \code{\link{lbbnn_net}} model:
#' \enumerate{
#'   \item Collapses \eqn{\exp(b + \sum_j w_j \log(x_j))} directly into canonical
#'     monomial form \code{C * x1^(w1) * x2^(w2)}.
#'   \item Collapses \code{"logic_and"} and \code{"logic_or"} neurons into
#'     canonical Boolean clauses \code{(x1 & !x2 & x4)}.
#'   \item Collapses linear combinations of identical symbolic basis terms.
#' }
#' @param model A trained \code{\link{lbbnn_net}} object.
#' @param var_names Optional character vector of length \eqn{p} giving input
#'   variable names. Defaults to \code{c("x1", "x2", ..., "xp")}.
#' @param digits Integer number of decimal places for floating-point constants
#'   (default \code{3}).
#' @param simplify Logical, whether to combine like symbolic basis terms and
#'   simplify unit coefficients / zero intercepts (default \code{TRUE}).
#' @param zero_tol Numeric threshold below which linear coefficients are dropped
#'   (default \code{1e-4}).
#' @return An S3 object of class \code{"lbbnn_symbolic"} containing:
#'   \describe{
#'     \item{\code{formula_str}}{Human-readable symbolic equation string.}
#'     \item{\code{latex}}{Formatted LaTeX equation string.}
#'     \item{\code{r_expr}}{Evaluable R \code{expression} object.}
#'     \item{\code{active_paths_df}}{Data frame of all surviving active edges,
#'       operator types, inclusion probabilities \eqn{\alpha_{i,j}}, raw
#'       variational weights, and debiased/snapped weights.}
#'     \item{\code{basis_terms}}{Named numeric vector of simplified basis terms
#'       and their coefficients.}
#'   }
#' @export
extract_symbolic_mpm <- function(model,
                                 var_names = NULL,
                                 digits = 3,
                                 simplify = TRUE,
                                 zero_tol = 1e-4) {
  p <- as.integer(model$sizes[1])
  if (is.null(var_names)) {
    var_names <- paste0("x", seq_len(p))
  } else if (length(var_names) != p) {
    stop(sprintf("var_names must have length %d (matching model$sizes[1]).", p))
  }
  latex_vars <- vapply(var_names, var_to_latex, character(1))

  if (!isTRUE(model$computed_paths)) {
    if (isTRUE(model$input_skip)) {
      model$compute_paths_input_skip()
    } else {
      model$compute_paths()
    }
  }

  n_hid <- length(model$layers$children)
  edge_rows <- list()

  # Helper to get raw & debiased weights if stored by train_symbolic_lbbnn
  get_w_raw <- function(l_idx, r, c, is_out = FALSE) {
    if (!is.null(model$raw_state)) {
      if (is_out) return(model$raw_state$out_layer$weight_mean[r, c])
      return(model$raw_state$layers[[l_idx]]$weight_mean[r, c])
    }
    if (is_out) return(as.numeric(model$out_layer$weight_mean[r, c]$cpu()))
    as.numeric(model$layers$children[[l_idx]]$weight_mean[r, c]$cpu())
  }
  get_w_deb <- function(l_idx, r, c, is_out = FALSE) {
    if (!is.null(model$debiased_state)) {
      if (is_out) return(model$debiased_state$out_layer$weight_mean[r, c])
      return(model$debiased_state$layers[[l_idx]]$weight_mean[r, c])
    }
    if (is_out) return(as.numeric(model$out_layer$weight_mean[r, c]$cpu()))
    as.numeric(model$layers$children[[l_idx]]$weight_mean[r, c]$cpu())
  }

  # Represent each input node
  input_nodes <- vector("list", p)
  for (j in seq_len(p)) {
    input_nodes[[j]] <- list(
      type = "input",
      var_idx = j,
      r_str = var_names[j],
      latex_str = latex_vars[j],
      scale = 1.0
    )
  }

  prev_nodes <- input_nodes

  # Traverse hidden layers l = 1, ..., n_hid
  for (l_idx in seq_len(n_hid)) {
    l <- model$layers$children[[l_idx]]
    d_out <- as.integer(l$out_features)
    d_in <- as.integer(l$in_features)
    act_mat <- as.matrix(l$alpha_active_path$detach()$cpu())
    probs_mat <- as.matrix(torch::torch_sigmoid(l$lambda_l$detach())$cpu())
    if (!is.null(l$lambda_grad_mask)) {
      probs_mat <- probs_mat * as.matrix(l$lambda_grad_mask$detach()$cpu())
    }
    w_mat <- as.matrix(l$weight_mean$detach()$cpu())
    b_vec <- as.numeric(l$bias_mean$detach()$cpu())
    act_vec <- if (!is.null(l$act_vec)) l$act_vec else rep("relu", d_out)

    # Combined incoming node list (previous layer nodes + optional input-skip nodes)
    in_nodes <- if (l_idx == 1L || !isTRUE(model$input_skip)) {
      prev_nodes
    } else {
      c(prev_nodes, input_nodes)
    }

    curr_nodes <- vector("list", d_out)
    for (r in seq_len(d_out)) {
      op_name <- act_vec[r]
      active_cols <- which(act_mat[r, ] > 0.5)
      if (length(active_cols) == 0L) {
        curr_nodes[[r]] <- list(type = "zero", r_str = "0", latex_str = "0", scale = 0.0)
        next
      }

      # Record active edges in active_paths_df
      for (c_in in active_cols) {
        from_lbl <- if (l_idx == 1L) {
          var_names[c_in]
        } else if (isTRUE(model$input_skip) && c_in > length(prev_nodes)) {
          paste0("skip:", var_names[c_in - length(prev_nodes)])
        } else {
          paste0("L", l_idx - 1L, "_N", c_in)
        }
        snap_key <- paste0(r, "_", c_in)
        rat_lbl <- if (!is.null(l$snapped_rationals) && !is.null(l$snapped_rationals[[snap_key]])) {
          l$snapped_rationals[[snap_key]]$q_label
        } else {
          ""
        }
        edge_rows[[length(edge_rows) + 1L]] <- data.frame(
          layer = paste0("L", l_idx),
          from_node = from_lbl,
          to_node = paste0("L", l_idx, "_N", r),
          operator = op_name,
          alpha = probs_mat[r, c_in],
          weight_raw = get_w_raw(l_idx, r, c_in, FALSE),
          weight_debiased = get_w_deb(l_idx, r, c_in, FALSE),
          weight_snapped = w_mat[r, c_in],
          rational_str = rat_lbl,
          stringsAsFactors = FALSE
        )
      }

      if (op_name == "clog") {
        c_in <- active_cols[1]
        src <- in_nodes[[c_in]]
        curr_nodes[[r]] <- list(
          type = "clog",
          var_idx = src$var_idx,
          var_r = src$r_str,
          var_latex = src$latex_str,
          r_str = paste0("log(", src$r_str, ")"),
          latex_str = paste0("\\log(", src$latex_str, ")"),
          scale = 1.0
        )

      } else if (op_name %in% c("logic_and", "logic_or")) {
        # Sort active columns by underlying variable index for canonical ordering
        var_indices <- vapply(active_cols, function(ci) {
          nd <- in_nodes[[ci]]
          if (!is.null(nd$var_idx)) nd$var_idx else ci
        }, integer(1))
        ord <- order(var_indices)
        active_cols_sorted <- active_cols[ord]

        lits_r <- character(length(active_cols_sorted))
        lits_tex <- character(length(active_cols_sorted))
        for (k in seq_along(active_cols_sorted)) {
          ci <- active_cols_sorted[k]
          src <- in_nodes[[ci]]
          w_val <- w_mat[r, ci]
          if (w_val >= 0) {
            lits_r[k] <- src$r_str
            lits_tex[k] <- src$latex_str
          } else {
            lits_r[k] <- paste0("!", src$r_str)
            lits_tex[k] <- paste0("\\neg ", src$latex_str)
          }
        }
        join_r <- if (op_name == "logic_and") " & " else " | "
        join_tex <- if (op_name == "logic_and") " \\land " else " \\lor "
        curr_nodes[[r]] <- list(
          type = op_name,
          r_str = paste0("(", paste(lits_r, collapse = join_r), ")"),
          latex_str = paste0("\\left(", paste(lits_tex, collapse = join_tex), "\\right)"),
          scale = 1.0
        )

      } else if (op_name == "exp") {
        # Separate incoming clog/log terms from general terms
        log_terms <- list()
        other_r <- character(0)
        other_tex <- character(0)

        for (ci in active_cols) {
          src <- in_nodes[[ci]]
          if (identical(src$type, "zero")) next
          w_val <- w_mat[r, ci]
          if (abs(w_val) < zero_tol) next
          snap_key <- paste0(r, "_", ci)
          rat_lbl <- if (!is.null(l$snapped_rationals) && !is.null(l$snapped_rationals[[snap_key]])) {
            l$snapped_rationals[[snap_key]]$q_label
          } else {
            NULL
          }

          if (src$type %in% c("clog", "log_var")) {
            v_r <- src$var_r
            if (is.null(log_terms[[v_r]])) {
              log_terms[[v_r]] <- list(
                var_r = v_r,
                var_latex = src$var_latex,
                var_idx = src$var_idx,
                w = w_val,
                rat_lbl = rat_lbl
              )
            } else {
              log_terms[[v_r]]$w <- log_terms[[v_r]]$w + w_val
              log_terms[[v_r]]$rat_lbl <- NULL
            }
          } else {
            w_s <- fmt_num(w_val, digits = digits, rational_label = rat_lbl)
            other_r <- c(other_r, paste0(w_s, " * ", src$r_str))
            other_tex <- c(other_tex, paste0(fmt_latex_exp(w_val, digits, rat_lbl), " ", src$latex_str))
          }
        }

        c_mult <- exp(b_vec[r])
        monomial_r <- character(0)
        monomial_tex <- character(0)
        if (length(log_terms) > 0L) {
          # Order by variable index
          v_idxs <- vapply(log_terms, function(lt) lt$var_idx, integer(1))
          log_terms <- log_terms[order(v_idxs)]
          for (lt in log_terms) {
            if (abs(lt$w) < zero_tol) next
            if (abs(lt$w - 1.0) < 1e-6) {
              monomial_r <- c(monomial_r, lt$var_r)
              monomial_tex <- c(monomial_tex, lt$var_latex)
            } else {
              exp_r <- fmt_num(lt$w, digits = digits, rational_label = lt$rat_lbl)
              exp_tex <- fmt_latex_exp(lt$w, digits = digits, rational_label = lt$rat_lbl)
              monomial_r <- c(monomial_r, paste0(lt$var_r, "^", exp_r))
              monomial_tex <- c(monomial_tex, paste0(lt$var_latex, "^{", exp_tex, "}"))
            }
          }
        }

        if (length(other_r) > 0L) {
          monomial_r <- c(monomial_r, paste0("exp(", paste(other_r, collapse = " + "), ")"))
          monomial_tex <- c(monomial_tex, paste0("\\exp\\left(", paste(other_tex, collapse = " + "), "\\right)"))
        }

        if (length(monomial_r) == 0L) {
          curr_nodes[[r]] <- list(type = "const", r_str = "1", latex_str = "1", scale = c_mult)
        } else {
          curr_nodes[[r]] <- list(
            type = "monomial",
            r_str = paste(monomial_r, collapse = " * "),
            latex_str = paste(monomial_tex, collapse = " "),
            scale = c_mult
          )
        }

      } else {
        # General unary operator (id, linear, log, sqrt, inv, square, sin, cos)
        lin_r <- character(0)
        lin_tex <- character(0)
        for (ci in active_cols) {
          src <- in_nodes[[ci]]
          if (identical(src$type, "zero")) next
          w_eff <- w_mat[r, ci] * (if (!is.null(src$scale)) src$scale else 1.0)
          if (abs(w_eff) < zero_tol) next
          if (abs(w_eff - 1.0) < 1e-5) {
            lin_r <- c(lin_r, src$r_str)
            lin_tex <- c(lin_tex, src$latex_str)
          } else if (abs(w_eff + 1.0) < 1e-5) {
            lin_r <- c(lin_r, paste0("-", src$r_str))
            lin_tex <- c(lin_tex, paste0("-", src$latex_str))
          } else {
            lin_r <- c(lin_r, paste0(round(w_eff, digits), " * ", src$r_str))
            lin_tex <- c(lin_tex, paste0(round(w_eff, digits), " ", src$latex_str))
          }
        }
        if (abs(b_vec[r]) >= zero_tol) {
          lin_r <- c(lin_r, as.character(round(b_vec[r], digits)))
          lin_tex <- c(lin_tex, as.character(round(b_vec[r], digits)))
        }
        if (length(lin_r) == 0L) {
          curr_nodes[[r]] <- list(type = "zero", r_str = "0", latex_str = "0", scale = 0.0)
        } else {
          inner_r <- paste(lin_r, collapse = " + ")
          inner_tex <- paste(lin_tex, collapse = " + ")
          if (op_name %in% c("id", "linear")) {
            if (length(lin_r) > 1L) {
              inner_r <- paste0("(", inner_r, ")")
              inner_tex <- paste0("\\left(", inner_tex, "\\right)")
            }
            curr_nodes[[r]] <- list(type = "linear", r_str = inner_r, latex_str = inner_tex, scale = 1.0)
          } else {
            op_spec <- get_symbolic_op(op_name)
            curr_nodes[[r]] <- list(
              type = op_name,
              r_str = op_spec$sym_str(inner_r),
              latex_str = op_spec$latex_str(inner_tex),
              scale = 1.0
            )
          }
        }
      }
    }
    prev_nodes <- curr_nodes
  }

  # Output Layer Aggregation & Simplification
  out_l <- model$out_layer
  act_out <- as.matrix(out_l$alpha_active_path$detach()$cpu())
  probs_out <- as.matrix(torch::torch_sigmoid(out_l$lambda_l$detach())$cpu())
  w_out <- as.matrix(out_l$weight_mean$detach()$cpu())
  b_out <- as.numeric(out_l$bias_mean$detach()$cpu())[1]

  out_in_nodes <- if (isTRUE(model$input_skip) && n_hid >= 1L) {
    c(prev_nodes, input_nodes)
  } else {
    prev_nodes
  }

  active_out_cols <- which(act_out[1, ] > 0.5)
  basis_env <- list() # key = r_str -> list(coef, r_str, latex_str)
  intercept <- b_out

  for (co in active_out_cols) {
    from_lbl <- if (isTRUE(model$input_skip) && co > length(prev_nodes)) {
      paste0("skip:", var_names[co - length(prev_nodes)])
    } else {
      paste0("L", n_hid, "_N", co)
    }
    snap_key <- paste0("1_", co)
    rat_lbl <- if (!is.null(out_l$snapped_rationals) && !is.null(out_l$snapped_rationals[[snap_key]])) {
      out_l$snapped_rationals[[snap_key]]$q_label
    } else {
      ""
    }
    edge_rows[[length(edge_rows) + 1L]] <- data.frame(
      layer = "Output",
      from_node = from_lbl,
      to_node = "y",
      operator = "linear",
      alpha = probs_out[1, co],
      weight_raw = get_w_raw(n_hid + 1L, 1L, co, TRUE),
      weight_debiased = get_w_deb(n_hid + 1L, 1L, co, TRUE),
      weight_snapped = w_out[1, co],
      rational_str = rat_lbl,
      stringsAsFactors = FALSE
    )

    src <- out_in_nodes[[co]]
    if (is.null(src) || identical(src$type, "zero")) next
    sc <- if (!is.null(src$scale)) src$scale else 1.0
    coef_val <- w_out[1, co] * sc
    if (abs(coef_val) < zero_tol) next

    if (identical(src$type, "const")) {
      intercept <- intercept + coef_val
    } else {
      b_key <- src$r_str
      if (is.null(basis_env[[b_key]])) {
        basis_env[[b_key]] <- list(
          coef = coef_val,
          r_str = src$r_str,
          latex_str = src$latex_str
        )
      } else {
        basis_env[[b_key]]$coef <- basis_env[[b_key]]$coef + coef_val
      }
    }
  }

  # Assemble human-readable formula_str and LaTeX
  terms_r <- character(0)
  terms_tex <- character(0)
  basis_vec <- numeric(0)

  if (abs(intercept) >= zero_tol) {
    c0_str <- fmt_num(intercept, digits = digits)
    terms_r <- c(terms_r, c0_str)
    terms_tex <- c(terms_tex, c0_str)
    basis_vec["(Intercept)"] <- intercept
  }

  for (b_key in names(basis_env)) {
    item <- basis_env[[b_key]]
    cf <- item$coef
    if (abs(cf) < zero_tol) next
    basis_vec[b_key] <- cf

    if (isTRUE(simplify) && abs(cf - 1.0) < 1e-4) {
      terms_r <- c(terms_r, item$r_str)
      terms_tex <- c(terms_tex, item$latex_str)
    } else if (isTRUE(simplify) && abs(cf + 1.0) < 1e-4) {
      terms_r <- c(terms_r, paste0("-1 * ", item$r_str))
      terms_tex <- c(terms_tex, paste0("-", item$latex_str))
    } else {
      cf_str <- fmt_num(cf, digits = digits)
      terms_r <- c(terms_r, paste0(cf_str, " * ", item$r_str))
      terms_tex <- c(terms_tex, paste0(cf_str, " ", item$latex_str))
    }
  }

  if (length(terms_r) == 0L) {
    formula_str <- "0"
    latex_str <- "0"
  } else {
    formula_str <- paste(terms_r, collapse = " + ")
    formula_str <- gsub("\\+ -", "- ", formula_str)
    latex_str <- paste(terms_tex, collapse = " + ")
    latex_str <- gsub("\\+ -", "- ", latex_str)
  }

  r_expr <- parse(text = formula_str)
  active_paths_df <- if (length(edge_rows) > 0L) {
    do.call(rbind, edge_rows)
  } else {
    data.frame(
      layer = character(0), from_node = character(0), to_node = character(0),
      operator = character(0), alpha = numeric(0), weight_raw = numeric(0),
      weight_debiased = numeric(0), weight_snapped = numeric(0),
      rational_str = character(0), stringsAsFactors = FALSE
    )
  }

  res <- list(
    formula_str = formula_str,
    latex = latex_str,
    r_expr = r_expr,
    active_paths_df = active_paths_df,
    basis_terms = basis_vec,
    var_names = var_names
  )
  class(res) <- "lbbnn_symbolic"
  res
}

#' @title Print Method for Extracted Symbolic LBBNN Objects
#' @param x An object of class \code{"lbbnn_symbolic"}.
#' @param ... Further arguments passed to or from other methods.
#' @export
print.lbbnn_symbolic <- function(x, ...) {
  cat("Neuro-Symbolic LBBNN Median Probability Model (MPM)\n")
  cat("===================================================\n")
  cat("Formula: ", x$formula_str, "\n")
  cat("LaTeX:   $", x$latex, "$\n", sep = "")
  cat("Active Edges:", nrow(x$active_paths_df), "\n")
  invisible(x)
}

