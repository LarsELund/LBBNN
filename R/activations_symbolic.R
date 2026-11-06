#' @title Protected Logarithmic Activation
#' @description Computes \eqn{\log(|z| + \epsilon)} for numerical stability.
#' @param z A \code{torch_tensor}.
#' @param eps Positive numeric constant, default \code{1e-6}.
#' @return A \code{torch_tensor}.
#' @export
plog <- function(z, eps = 1e-6) {
  torch::torch_log(torch::torch_abs(z) + eps)
}

#' @title Canonical Positive Input Logarithmic Activation
#' @description Computes \eqn{\log(\max(x, \epsilon))} for positive-domain inputs.
#' @param x A \code{torch_tensor}.
#' @param eps Positive numeric constant, default \code{1e-6}.
#' @return A \code{torch_tensor}.
#' @export
clog <- function(x, eps = 1e-6) {
  torch::torch_log(torch::torch_clamp(x, min = eps))
}

#' @title Clamped Exponential Activation
#' @description Computes \eqn{\exp(\text{clamp}(z, -12, 12))} to prevent overflow.
#' @param z A \code{torch_tensor}.
#' @param limit Numeric clamping bound, default \code{12.0}.
#' @return A \code{torch_tensor}.
#' @export
pexp <- function(z, limit = 12.0) {
  torch::torch_exp(torch::torch_clamp(z, min = -limit, max = limit))
}

#' @title Compute Domain Clamping Excess Penalty for Exponential Units
#' @description Computes \eqn{\text{mean}(\text{ReLU}(|z| - \text{limit})^2)}.
#' @param z A \code{torch_tensor}.
#' @param limit Numeric clamping bound, default \code{12.0}.
#' @return A scalar \code{torch_tensor}.
#' @export
compute_exp_domain_penalty <- function(z, limit = 12.0) {
  excess <- torch::nnf_relu(torch::torch_abs(z) - limit)
  torch::torch_mean(excess^2)
}

#' @title Protected Square Root Activation
#' @description Computes \eqn{\sqrt{|z| + \epsilon}}.
#' @param z A \code{torch_tensor}.
#' @param eps Positive numeric constant, default \code{1e-6}.
#' @return A \code{torch_tensor}.
#' @export
psqrt <- function(z, eps = 1e-6) {
  torch::torch_sqrt(torch::torch_abs(z) + eps)
}

#' @title Protected Multiplicative Inverse Activation
#' @description Computes \eqn{z / (z^2 + \epsilon)} with bounded gradient near 0.
#' @param z A \code{torch_tensor}.
#' @param eps Positive numeric constant, default \code{1e-6}.
#' @return A \code{torch_tensor}.
#' @export
pinv <- function(z, eps = 1e-6) {
  z / (z^2 + eps)
}

#' @title Square Activation
#' @description Computes \eqn{z^2}, clamped to avoid float32 overflow on extreme inputs.
#' @param z A \code{torch_tensor}.
#' @param limit Numeric clamping bound, default \code{1e6}.
#' @return A \code{torch_tensor}.
#' @export
psquare <- function(z, limit = 1e6) {
  z_c <- torch::torch_clamp(z, min = -limit, max = limit)
  z_c^2
}

#' @title Differentiable Logical AND Gate
#' @description
#' Computes a differentiable conjunction over binary/continuous features
#' \code{x} with literal polarity weights \code{weight} and inclusion gates
#' \code{gate}. Supports both temperature-sharpened Product T-norm (via
#' stabilized log-exp sum of literals) and Lukasiewicz T-norm during smooth
#' training, and exact hard Boolean conjunction during MPM evaluation.
#' @param x Input \code{torch_tensor} of shape \code{(batch, in_features)} or
#'   1D/2D tensor when called standalone.
#' @param weight Weight \code{torch_tensor} of shape \code{(out_logic, in_features)}.
#' @param gate Gate \code{torch_tensor} of shape \code{(out_logic, in_features)}.
#' @param MPM logical, whether to evaluate hard Boolean conjunction.
#' @param tau_L numeric, temperature sharpening for literal sigmoid.
#' @param mode character, either \code{"product_tnorm"} or \code{"lukasiewicz"}.
#' @param eps numeric stability constant, default \code{1e-6}.
#' @return A \code{torch_tensor} of shape \code{(batch, out_logic)}.
#' @export
logic_and <- function(x, weight = NULL, gate = NULL, MPM = FALSE,
                      tau_L = 10.0, mode = c("product_tnorm", "lukasiewicz"),
                      eps = 1e-6) {
  mode <- match.arg(mode)
  if (length(dim(x)) == 1L) {
    x <- x$unsqueeze(1)
  }
  # Standalone unary fallback if weight/gate are omitted (e.g. direct tensor test)
  if (is.null(weight)) {
    weight <- torch::torch_ones(c(1L, dim(x)[2]), dtype = x$dtype, device = x$device)
  }
  if (is.null(gate)) {
    gate <- torch::torch_ones_like(weight)
  }
  if (MPM) {
    x_bin <- torch::torch_clamp(x, 0.0, 1.0)$unsqueeze(2) # (B, 1, D)
    w_pos <- (weight >= 0)$to(dtype = x$dtype)$unsqueeze(1) # (1, O, D)
    g_mask <- (gate > 0.5)$to(dtype = x$dtype)$unsqueeze(1) # (1, O, D)
    lit <- w_pos * x_bin + (1.0 - w_pos) * (1.0 - x_bin)
    factors <- 1.0 - g_mask * (1.0 - lit)
    has_active <- (torch::torch_sum(g_mask, dim = 3) > 0)$to(dtype = x$dtype)
    return(torch::torch_prod(factors, dim = 3) * has_active)
  }

  x_c <- torch::torch_clamp(x, min = -10.0, max = 10.0)$unsqueeze(2) # (B, 1, D)
  w_e <- torch::torch_clamp(weight, min = -10.0, max = 10.0)$unsqueeze(1) # (1, O, D)
  g_e <- torch::torch_clamp(gate, min = 0.0, max = 1.0)$unsqueeze(1) # (1, O, D)

  if (mode == "product_tnorm") {
    lit_prob <- torch::torch_sigmoid(tau_L * w_e * (x_c - 0.5))
    factor <- torch::torch_clamp(1.0 - g_e * (1.0 - lit_prob), min = eps, max = 1.0)
    log_sum <- torch::torch_sum(torch::torch_log(factor), dim = 3)
    return(torch::torch_exp(log_sum))
  } else {
    s_w <- torch::torch_tanh(10.0 * w_e)
    violation <- torch::nnf_relu(s_w - w_e * x_c)
    return(torch::nnf_relu(1.0 - torch::torch_sum(g_e * violation, dim = 3)))
  }
}

#' @title Differentiable Logical OR Gate (De Morgan Dual of logic_and)
#' @description
#' Computes the De Morgan dual disjunction over features \code{x} with literal
#' weights \code{weight} and inclusion gates \code{gate}.
#' @inheritParams logic_and
#' @return A \code{torch_tensor} of shape \code{(batch, out_logic)}.
#' @export
logic_or <- function(x, weight = NULL, gate = NULL, MPM = FALSE,
                     tau_L = 10.0, mode = c("product_tnorm", "lukasiewicz"),
                     eps = 1e-6) {
  mode <- match.arg(mode)
  if (length(dim(x)) == 1L) {
    x <- x$unsqueeze(1)
  }
  if (is.null(weight)) {
    weight <- torch::torch_ones(c(1L, dim(x)[2]), dtype = x$dtype, device = x$device)
  }
  if (is.null(gate)) {
    gate <- torch::torch_ones_like(weight)
  }
  if (MPM) {
    x_bin <- torch::torch_clamp(x, 0.0, 1.0)$unsqueeze(2)
    w_pos <- (weight >= 0)$to(dtype = x$dtype)$unsqueeze(1)
    g_mask <- (gate > 0.5)$to(dtype = x$dtype)$unsqueeze(1)
    lit <- w_pos * x_bin + (1.0 - w_pos) * (1.0 - x_bin)
    factors <- 1.0 - g_mask * lit
    return(1.0 - torch::torch_prod(factors, dim = 3))
  }

  x_c <- torch::torch_clamp(x, min = -10.0, max = 10.0)$unsqueeze(2)
  w_e <- torch::torch_clamp(weight, min = -10.0, max = 10.0)$unsqueeze(1)
  g_e <- torch::torch_clamp(gate, min = 0.0, max = 1.0)$unsqueeze(1)

  if (mode == "product_tnorm") {
    lit_prob <- torch::torch_sigmoid(tau_L * w_e * (x_c - 0.5))
    factor <- torch::torch_clamp(1.0 - g_e * lit_prob, min = eps, max = 1.0)
    log_sum <- torch::torch_sum(torch::torch_log(factor), dim = 3)
    return(1.0 - torch::torch_exp(log_sum))
  } else {
    s_w <- torch::torch_tanh(10.0 * w_e)
    sat <- 1.0 - torch::nnf_relu(s_w - w_e * x_c)
    return(torch::torch_clamp(torch::torch_sum(g_e * sat, dim = 3), min = 0.0, max = 1.0))
  }
}

#' @title Dictionary of Protected Symbolic Operators
#' @description
#' Returns metadata, complexity cost, forward function, and symbolic/LaTeX
#' constructors for all supported neuro-symbolic operators.
#' @param op_name Optional character string of operator name. If \code{NULL},
#'   returns the full dictionary list.
#' @return A list of operator specifications or a single operator entry.
#' @export
get_symbolic_op <- function(op_name = NULL) {
  ops <- list(
    id = list(
      name = "id",
      cost = 0.5,
      type = "unary",
      forward = function(z, eps = 1e-6) z,
      sym_str = function(s) s,
      latex_str = function(s) s
    ),
    linear = list(
      name = "linear",
      cost = 0.5,
      type = "unary",
      forward = function(z, eps = 1e-6) z,
      sym_str = function(s) s,
      latex_str = function(s) s
    ),
    log = list(
      name = "log",
      cost = 1.5,
      type = "unary",
      forward = function(z, eps = 1e-6) plog(z, eps = eps),
      sym_str = function(s) paste0("log(abs(", s, "))"),
      latex_str = function(s) paste0("\\log(|", s, "|)")
    ),
    clog = list(
      name = "clog",
      cost = 1.0,
      type = "clog",
      forward = function(z, eps = 1e-6) clog(z, eps = eps),
      sym_str = function(s) paste0("log(", s, ")"),
      latex_str = function(s) paste0("\\log(", s, ")")
    ),
    exp = list(
      name = "exp",
      cost = 1.5,
      type = "unary",
      forward = function(z, eps = 1e-6) pexp(z, limit = 12.0),
      sym_str = function(s) paste0("exp(", s, ")"),
      latex_str = function(s) paste0("\\exp\\left(", s, "\\right)")
    ),
    sqrt = list(
      name = "sqrt",
      cost = 1.5,
      type = "unary",
      forward = function(z, eps = 1e-6) psqrt(z, eps = eps),
      sym_str = function(s) paste0("sqrt(abs(", s, "))"),
      latex_str = function(s) paste0("\\sqrt{|", s, "|}")
    ),
    inv = list(
      name = "inv",
      cost = 2.0,
      type = "unary",
      forward = function(z, eps = 1e-6) pinv(z, eps = eps),
      sym_str = function(s) paste0("(1 / (", s, "))"),
      latex_str = function(s) paste0("\\frac{1}{", s, "}")
    ),
    square = list(
      name = "square",
      cost = 1.0,
      type = "unary",
      forward = function(z, eps = 1e-6) psquare(z),
      sym_str = function(s) paste0("(", s, ")^2"),
      latex_str = function(s) paste0("\\left(", s, "\\right)^2")
    ),
    sin = list(
      name = "sin",
      cost = 1.5,
      type = "unary",
      forward = function(z, eps = 1e-6) torch::torch_sin(z),
      sym_str = function(s) paste0("sin(", s, ")"),
      latex_str = function(s) paste0("\\sin\\left(", s, "\\right)")
    ),
    cos = list(
      name = "cos",
      cost = 1.5,
      type = "unary",
      forward = function(z, eps = 1e-6) torch::torch_cos(z),
      sym_str = function(s) paste0("cos(", s, ")"),
      latex_str = function(s) paste0("\\cos\\left(", s, "\\right)")
    ),
    logic_and = list(
      name = "logic_and",
      cost = 1.0,
      type = "logic",
      forward = logic_and,
      sym_str = function(lits) paste0("(", paste(lits, collapse = " & "), ")"),
      latex_str = function(lits) paste0("\\left(", paste(lits, collapse = " \\land "), "\\right)")
    ),
    logic_or = list(
      name = "logic_or",
      cost = 1.0,
      type = "logic",
      forward = logic_or,
      sym_str = function(lits) paste0("(", paste(lits, collapse = " | "), ")"),
      latex_str = function(lits) paste0("\\left(", paste(lits, collapse = " \\lor "), "\\right)")
    )
  )
  if (is.null(op_name)) {
    return(ops)
  }
  if (!op_name %in% names(ops)) {
    stop(sprintf(
      "Unknown symbolic activation '%s'. Supported: %s",
      op_name, paste(names(ops), collapse = ", ")
    ))
  }
  ops[[op_name]]
}

#' @title Parse Layer Activation Specification into Contiguous Groups
#' @description
#' Converts a character vector of length \code{out_features} or a named list of
#' counts (e.g., \code{list(clog = 4, exp = 4, id = 2)}) into a canonical
#' character vector and contiguous index slices for vectorized layer execution.
#' @param layer_activations Character vector of length \code{out_features} or
#'   named list/vector of integer counts.
#' @param out_features Integer number of output neurons in the layer.
#' @return A list with \code{act_vec} (character vector of length
#'   \code{out_features}) and \code{groups} (list of contiguous slices).
#' @export
parse_layer_activations <- function(layer_activations, out_features) {
  if (is.list(layer_activations) ||
      (is.numeric(layer_activations) && !is.null(names(layer_activations)))) {
    counts <- unlist(layer_activations)
    if (sum(counts) != out_features) {
      stop(sprintf(
        "Sum of layer_activations counts (%d) must equal out_features (%d).",
        as.integer(sum(counts)), as.integer(out_features)
      ))
    }
    act_vec <- rep(names(counts), times = as.integer(counts))
  } else if (is.character(layer_activations)) {
    if (length(layer_activations) == 1L && out_features > 1L) {
      act_vec <- rep(layer_activations, out_features)
    } else if (length(layer_activations) == out_features) {
      act_vec <- layer_activations
    } else {
      stop(sprintf(
        "Length of layer_activations (%d) must equal out_features (%d).",
        length(layer_activations), as.integer(out_features)
      ))
    }
  } else {
    stop("layer_activations must be a character vector or a named list of counts.")
  }

  # Validate all operator names
  for (op in unique(act_vec)) {
    get_symbolic_op(op)
  }

  # Group contiguous runs via rle
  runs <- rle(act_vec)
  ends <- cumsum(runs$lengths)
  starts <- ends - runs$lengths + 1L
  groups <- vector("list", length(runs$values))
  for (k in seq_along(runs$values)) {
    op_name <- runs$values[k]
    op_spec <- get_symbolic_op(op_name)
    groups[[k]] <- list(
      op = op_name,
      type = op_spec$type,
      cost = op_spec$cost,
      start = as.integer(starts[k]),
      end = as.integer(ends[k]),
      indices = seq.int(starts[k], ends[k])
    )
  }
  list(act_vec = act_vec, groups = groups)
}

