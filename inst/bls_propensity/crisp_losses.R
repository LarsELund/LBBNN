# ==============================================================================
# CRISP Loss Functions for Combining Probability and Non-Probability Samples
# Based on Beresovsky, Gershunskaya & Savitsky (2026) and Chen, Li & Wu (2020)
# ==============================================================================
library(torch)

#' 1. Composite Negative Log-Likelihood: Unlinked Stacked Sample (STACK)
#' S = S_c (+) S_r, size n_c + n_r. No linkage information required.
#' Composite probability: pi_zi = pi_ci / (pi_ci + pi_ri)
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

#' 2. Composite Negative Log-Likelihood: Deterministically Linked Union (LINK1)
#' S = S_c U (S_r \ S_c). Overlapping units in S_r are omitted.
#' Composite probability: pi_zi = pi_ci / (pi_ci + pi_ri - pi_ci * pi_ri)
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

#' 3. Composite Negative Log-Likelihood: Deterministically Linked Stacked Sample (LINK2)
#' S = S_c (+) S_r. Overlapping units in S_r are marked as I_zi = 1.
#' Composite probability: pi_zi = pi_ci * (1 + pi_ri) / (pi_ci + pi_ri)
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

#' 4. Missing Information Principle Factory (MIP)
#' S = S_c (+) S_r with probabilistic linkage informativeness p_link in [0, 1].
#' For S_r units, soft indicator is p_i = p_link * I_c_true + (1 - p_link) * pi_c
make_nll_mip <- function(p_link = 0.5) {
  function(input, target, reduction = "sum") {
    eta <- input$squeeze(-1)
    pi_c <- torch::torch_sigmoid(eta)
    is_Sc <- target[, 1]
    I_c_true <- target[, 2]
    pi_r <- target[, 3]

    p_i <- (p_link * I_c_true) + ((1 - p_link) * pi_c$detach())
    I_z_mip <- is_Sc + (1 - is_Sc) * p_i

    pi_z <- (pi_c * (1 + pi_r)) / (pi_c + pi_r + 1e-12)
    pi_z <- torch::torch_clamp(pi_z, 1e-7, 1 - 1e-7)
    nll_i <- -(I_z_mip * torch::torch_log(pi_z) + (1 - I_z_mip) * torch::torch_log(1 - pi_z))
    if (reduction == "none") nll_i else torch::torch_sum(nll_i)
  }
}

#' 5. Pseudo-Log-Likelihood Negative Loss (Chen, Li & Wu 2020)
#' S = S_c (+) S_r with design survey weights w_ri = 1 / pi_ri on S_r.
#' Numerically stable softplus formulation.
nll_pseudo <- function(input, target, reduction = "sum") {
  eta <- input$squeeze(-1)
  is_Sc <- target[, 1]
  pi_r <- target[, 2]
  nll_i <- -is_Sc * eta + (1 - is_Sc) * (1 / (pi_r + 1e-12)) * torch::nnf_softplus(eta)
  if (reduction == "none") nll_i else torch::torch_sum(nll_i)
}
