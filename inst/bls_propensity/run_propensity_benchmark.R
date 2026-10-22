# ==============================================================================
# End-to-End Simulation Benchmark: Propensity Estimation with LBBNN vs GLM
# Comparing STACK, LINK1, LINK2, MIP, PSEUDO under High & Low Common Support
# ==============================================================================

suppressPackageStartupMessages({
  library(LBBNN)
  library(torch)
  library(sampling)
  library(MASS)
  library(ggplot2)
})

# Source local files if running standalone
if (file.exists("inst/bls_propensity/crisp_losses.R")) {
  source("inst/bls_propensity/crisp_losses.R")
  source("inst/bls_propensity/gen_pop_nonlinear.R")
} else if (file.exists("crisp_losses.R")) {
  source("crisp_losses.R")
  source("gen_pop_nonlinear.R")
}

# Ensure results directory exists
dir.create("inst/bls_propensity/results", showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------------------------
# Helper: Fit Single LBBNN Propensity Model
# ------------------------------------------------------------------------------
fit_lbbnn_propensity <- function(pop, method = "STACK", covar_cols = paste0("X", 1:6),
                                 epochs = 40, lr = 0.01, p_link = 0.5, seed = 42) {
  set.seed(seed)
  torch::torch_manual_seed(seed)

  ds <- build_propensity_dataset(pop, method = method, covar_cols = covar_cols)
  dl <- get_dataloaders(ds$data, train_proportion = 1.0, train_batch_size = 64,
                        test_batch_size = 64, standardize = FALSE,
                        shuffle_train = TRUE, target_cols = ds$target_cols, seed = seed)

  nll_func <- switch(
    method,
    "STACK"  = nll_stack,
    "LINK1"  = nll_link1,
    "LINK2"  = nll_link2,
    "MIP"    = make_nll_mip(p_link = p_link),
    "PSEUDO" = nll_pseudo
  )

  p <- length(covar_cols)
  model <- lbbnn_net(
    problem_type = "custom",
    sizes = c(p, 16, 1),
    prior = c(0.2, 0.2),
    std = c(1.0, 1.0),
    inclusion_inits = "balanced",
    input_skip = TRUE,
    flow = FALSE,
    raw_output = TRUE,
    nll = nll_func,
    device = "cpu"
  )

  train_lbbnn(epochs = epochs, LBBNN = model, lr = lr, train_dl = dl$train_loader, verbose = FALSE)

  # Predict pi_c on Sc
  Sc <- pop[pop$I.c == 1, ]
  preds_bma <- predict(model, newdata = as.matrix(Sc[, covar_cols]), draws = 50, link = torch::torch_sigmoid)
  preds_mpm <- predict(model, newdata = as.matrix(Sc[, covar_cols]), draws = 50, mpm = TRUE, link = torch::torch_sigmoid)

  pi_hat_bma <- apply(as.matrix(preds_bma[, , 1]$cpu()), 2, mean)
  pi_hat_mpm <- apply(as.matrix(preds_mpm[, , 1]$cpu()), 2, mean)

  sm <- summary(model)

  return(list(
    model = model,
    pi_hat_bma = pi_hat_bma,
    pi_hat_mpm = pi_hat_mpm,
    summary = sm
  ))
}

# ------------------------------------------------------------------------------
# Helper: Fit GLM Baselines
# ------------------------------------------------------------------------------
fit_glm_baselines <- function(pop, covar_cols = paste0("X", 1:6)) {
  Sc <- pop[pop$I.c == 1, ]
  Sr <- pop[pop$I.r == 1, ]

  # 1. GLM-STACK
  df_c <- data.frame(Sc[, covar_cols], I_z = 1)
  df_r <- data.frame(Sr[, covar_cols], I_z = 0)
  df_stack <- rbind(df_c, df_r)

  fit_stack <- stats::glm(I_z ~ ., data = df_stack, family = stats::binomial())
  eta_stack <- stats::predict(fit_stack, newdata = Sc[, covar_cols])
  pi_hat_glm_stack <- Sc$pi.r * exp(eta_stack)

  # 2. GLM-PSEUDO
  X_c <- as.matrix(cbind(1, Sc[, covar_cols]))
  X_r <- as.matrix(cbind(1, Sr[, covar_cols]))
  pi_r <- Sr$pi.r

  loss_pseudo <- function(beta) {
    eta_c <- X_c %*% beta
    eta_r <- X_r %*% beta
    -sum(eta_c) + sum((1 / pi_r) * log1p(exp(eta_r)))
  }
  fit_pseudo <- stats::optim(rep(0, ncol(X_c)), loss_pseudo, method = "BFGS")
  eta_pseudo <- X_c %*% fit_pseudo$par
  pi_hat_glm_pseudo <- stats::plogis(eta_pseudo)

  return(list(
    pi_hat_glm_stack = as.numeric(pi_hat_glm_stack),
    pi_hat_glm_pseudo = as.numeric(pi_hat_glm_pseudo)
  ))
}

# ------------------------------------------------------------------------------
# Run Comparative Benchmark Simulation
# ------------------------------------------------------------------------------
run_single_benchmark <- function(N = 5000, p = 6, nonlinear_y = TRUE,
                                 nonlinear_propensity = TRUE,
                                 common_support = "high", n_r = 500, n_c = 500,
                                 epochs = 40, lr = 0.01, seed = 2026) {

  cat(sprintf("\n=== Running Benchmark: Common Support = %s, NonLinear = %s ===\n",
              common_support, nonlinear_propensity))

  pop <- gen_pop_nonlinear(
    N = N, p = p, nonlinear_y = nonlinear_y,
    nonlinear_propensity = nonlinear_propensity,
    common_support = common_support, n_r = n_r, n_c = n_c, seed = seed
  )
  pop <- draw_combined_samples(pop, seed = seed)

  Sc <- pop[pop$I.c == 1, ]
  Sr <- pop[pop$I.r == 1, ]
  covar_cols <- paste0("X", 1:p)

  true_pop_mean <- mean(pop$y)
  naive_mean <- mean(Sc$y)
  oracle_hajek <- hajek_estimate(Sc$y, Sc$pi.c)

  cat(sprintf("Population Size N = %d, Sc = %d, Sr = %d\n", N, nrow(Sc), nrow(Sr)))
  cat(sprintf("True Population Mean Y:  %.4f\n", true_pop_mean))
  cat(sprintf("Naive Sample Mean:       %.4f (Bias: %+.4f)\n", naive_mean, naive_mean - true_pop_mean))
  cat(sprintf("Oracle Hajek Mean:       %.4f (Bias: %+.4f)\n", oracle_hajek, oracle_hajek - true_pop_mean))

  # GLM Baselines
  glm_res <- fit_glm_baselines(pop, covar_cols = covar_cols)
  hajek_glm_stack <- hajek_estimate(Sc$y, glm_res$pi_hat_glm_stack)
  hajek_glm_pseudo <- hajek_estimate(Sc$y, glm_res$pi_hat_glm_pseudo)

  # LBBNN Estimators
  methods <- c("STACK", "LINK1", "LINK2", "MIP", "PSEUDO")
  lbbnn_results <- list()

  for (m in methods) {
    cat(sprintf("Fitting LBBNN-%s...\n", m))
    fit_m <- fit_lbbnn_propensity(pop, method = m, covar_cols = covar_cols,
                                  epochs = epochs, lr = lr, seed = seed)
    h_bma <- hajek_estimate(Sc$y, fit_m$pi_hat_bma)
    h_mpm <- hajek_estimate(Sc$y, fit_m$pi_hat_mpm)
    lbbnn_results[[m]] <- list(
      fit = fit_m,
      hajek_bma = h_bma,
      hajek_mpm = h_mpm
    )
  }

  # Build Comparison Table
  est_names <- c(
    "Naive Convenience Mean",
    "Oracle Hajek (True Propensity)",
    "GLM-STACK",
    "GLM-PSEUDO",
    "LBBNN-STACK (BMA)",
    "LBBNN-STACK (MPM)",
    "LBBNN-LINK1 (BMA)",
    "LBBNN-LINK1 (MPM)",
    "LBBNN-LINK2 (BMA)",
    "LBBNN-LINK2 (MPM)",
    "LBBNN-MIP50 (BMA)",
    "LBBNN-MIP50 (MPM)",
    "LBBNN-PSEUDO (BMA)",
    "LBBNN-PSEUDO (MPM)"
  )

  estimates <- c(
    naive_mean,
    oracle_hajek,
    hajek_glm_stack,
    hajek_glm_pseudo,
    lbbnn_results[["STACK"]]$hajek_bma,
    lbbnn_results[["STACK"]]$hajek_mpm,
    lbbnn_results[["LINK1"]]$hajek_bma,
    lbbnn_results[["LINK1"]]$hajek_mpm,
    lbbnn_results[["LINK2"]]$hajek_bma,
    lbbnn_results[["LINK2"]]$hajek_mpm,
    lbbnn_results[["MIP"]]$hajek_bma,
    lbbnn_results[["MIP"]]$hajek_mpm,
    lbbnn_results[["PSEUDO"]]$hajek_bma,
    lbbnn_results[["PSEUDO"]]$hajek_mpm
  )

  res_table <- data.frame(
    Estimator = est_names,
    Estimate = round(estimates, 4),
    Bias = round(estimates - true_pop_mean, 4),
    Abs_Error = round(abs(estimates - true_pop_mean), 4)
  )

  print(res_table)
  return(list(pop = pop, table = res_table, lbbnn_models = lbbnn_results))
}

if (sys.nframe() == 0) {
  # Run both scenarios if called directly
  cat("Running Scenario 1: High Common Support...\n")
  res_scen1 <- run_single_benchmark(common_support = "high", nonlinear_propensity = TRUE, epochs = 35)
  write.csv(res_scen1$table, "inst/bls_propensity/results/scenario1_high_support.csv", row.names = FALSE)

  cat("\nRunning Scenario 2: Low Common Support...\n")
  res_scen2 <- run_single_benchmark(common_support = "low", nonlinear_propensity = TRUE, epochs = 35)
  write.csv(res_scen2$table, "inst/bls_propensity/results/scenario2_low_support.csv", row.names = FALSE)

  cat("\nBenchmark runs finished successfully! Results saved to inst/bls_propensity/results/\n")
}
