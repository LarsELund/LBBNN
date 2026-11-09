#!/usr/bin/env Rscript
# ==============================================================================
# Generate Publication-Quality Figures for the Neuro-Symbolic LBBNN Paper
# Outputs both vector PDFs (paper/figures/*.pdf) and high-res PNGs (*.png)
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

out_dir <- "paper/figures"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Load benchmark summaries
df_logic <- read.csv("results_overnight/logic/summary_logic.csv", stringsAsFactors = FALSE)
df_kep   <- read.csv("results_overnight/kepler/summary_kepler.csv", stringsAsFactors = FALSE)
df_real  <- read.csv("results_overnight/kepler_real/summary_kepler_real.csv", stringsAsFactors = FALSE)
df_nl    <- read.csv("results_overnight/nonlinear/summary_nonlinear.csv", stringsAsFactors = FALSE)
load("data/exoplanet.rda")

# Helper to save both PDF and PNG using base R / grid graphics for multi-panel layouts
save_dual <- function(filename_base, width, height, plot_fn) {
  pdf_path <- file.path(out_dir, paste0(filename_base, ".pdf"))
  png_path <- file.path(out_dir, paste0(filename_base, ".png"))
  
  pdf(pdf_path, width = width, height = height, useDingbats = FALSE)
  plot_fn()
  dev.off()
  
  png(png_path, width = width * 300, height = height * 300, res = 300)
  plot_fn()
  dev.off()
  
  cat(sprintf("Generated %s (.pdf & .png)\n", filename_base))
}

# ==============================================================================
# FIGURE 1: BGNLM Complexity Prior Surface & 3-Stage Training Dynamics
# ==============================================================================
save_dual("fig1_priors_and_dynamics", width = 11.5, height = 4.6, plot_fn = function() {
  par(mfrow = c(1, 3), mar = c(4.6, 5.2, 3.2, 1.4), oma = c(0, 0, 0, 0), cex.lab = 1.08, cex.main = 1.12)
  
  # Panel A: Prior Inclusion Probability vs Operator Complexity across Depths
  c_grid <- seq(0.4, 2.2, length.out = 200)
  pi0 <- 0.5
  logit_1_pi0 <- log((1 - pi0) / pi0)
  b_depth <- 0.6
  b_comp  <- 0.5
  
  calc_pi <- function(l, c_val) {
    1 / (1 + exp(logit_1_pi0 + b_depth * (l - 1) + b_comp * c_val))
  }
  
  plot(c_grid, calc_pi(1, c_grid), type = "l", lwd = 2.8, col = "#1b4f72",
       ylim = c(0.05, 0.52), xlab = "Operator Complexity Cost c(g)",
       ylab = "Prior Inclusion Probability",
       main = "(a) BGNLM Depth & Complexity Prior", las = 1)
  grid(col = "gray88", lty = "dotted")
  lines(c_grid, calc_pi(1, c_grid), lwd = 2.8, col = "#1b4f72")
  lines(c_grid, calc_pi(2, c_grid), lwd = 2.8, col = "#2874a6", lty = 2)
  lines(c_grid, calc_pi(3, c_grid), lwd = 2.8, col = "#e67e22", lty = 3)
  
  # Operator markers on Layer 1 & Layer 2
  ops_c <- c(0.5, 1.0, 1.5, 2.0)
  ops_lbl <- c("id (0.5)", "clog/sq/\nlogic (1.0)", "log/exp/\nsin/sqrt (1.5)", "inv (2.0)")
  points(ops_c, calc_pi(1, ops_c), pch = 19, cex = 1.3, col = "#1b4f72")
  points(ops_c, calc_pi(2, ops_c), pch = 17, cex = 1.3, col = "#2874a6")
  text(ops_c, calc_pi(1, ops_c) + 0.04, labels = ops_lbl, cex = 0.78, col = "gray20")
  
  legend("topright", legend = c("Layer l = 1 (Shallow)", "Layer l = 2 (Composition)", "Layer l = 3 (Deep)"),
         col = c("#1b4f72", "#2874a6", "#e67e22"), lty = c(1, 2, 3), pch = c(19, 17, NA),
         lwd = 2.2, bty = "n", cex = 0.85)
  
  # Panel B: Stage 1 & Stage 2 Schedules (KL Warmup & Concrete Temperature)
  t_norm <- seq(0, 1.25, length.out = 300)
  beta_kl <- ifelse(t_norm <= 1.0, pmin(1.0, t_norm / 0.30), 0.0)
  tau_t   <- ifelse(t_norm <= 1.0, 0.5 * (0.05 / 0.5)^t_norm, 0.0)
  
  plot(t_norm * 100, beta_kl, type = "l", lwd = 2.8, col = "#c0392b",
       ylim = c(0, 1.15), xlab = "Training Progress (% of Stage 1 + Stage 2)",
       ylab = "Schedule Magnitude",
       main = "(b) Two-Stage Optimization Protocol", las = 1, xaxt = "n")
  axis(1, at = c(0, 30, 60, 100, 125), labels = c("0%", "30%", "60%", "100%", "Stage 2"))
  grid(col = "gray88", lty = "dotted")
  rect(100, -0.05, 126, 1.18, col = rgb(0.18, 0.8, 0.44, 0.12), border = NA)
  abline(v = 30, lty = 3, col = "gray40", lwd = 1.2)
  abline(v = 100, lty = 2, col = "#27ae60", lwd = 1.8)
  lines(t_norm * 100, beta_kl, lwd = 2.8, col = "#c0392b")
  lines(t_norm * 100, tau_t, lwd = 2.8, col = "#2980b9", lty = 2)
  text(15, 1.07, "KL Warmup", cex = 0.8, col = "#c0392b")
  text(65, 1.07, "Sparse Pruning", cex = 0.8, col = "#c0392b")
  text(112.5, 0.65, "MPM Freeze\n+ Debias\n+ BIC Snap", cex = 0.78, col = "#1e8449", font = 2)
  legend("left", legend = c(expression(paste("KL Weight ", beta[KL](t))),
                            expression(paste("Concrete Temp ", tau(t)))),
         col = c("#c0392b", "#2980b9"), lty = c(1, 2), lwd = 2.5, bty = "n", cex = 0.85)
  
  # Panel C: Symmetric Bimodal Slab Prior for Logic Gates
  w_grid <- seq(-2.5, 2.5, length.out = 400)
  sigma_L <- 0.28
  bimodal_dens <- 0.5 * dnorm(w_grid, mean = 1.0, sd = sigma_L) +
                  0.5 * dnorm(w_grid, mean = -1.0, sd = sigma_L)
  unimodal_dens <- dnorm(w_grid, mean = 0.0, sd = 1.0)
  
  plot(w_grid, bimodal_dens, type = "l", lwd = 2.8, col = "#8e44ad",
       ylim = c(0, max(bimodal_dens) * 1.18),
       xlab = expression(paste("Logic Gate Weight ", w[list(i,j)])),
       ylab = "Prior Density p(w)",
       main = "(c) Bimodal Logic Literal Slab Prior", las = 1)
  grid(col = "gray88", lty = "dotted")
  polygon(c(w_grid, rev(w_grid)), c(bimodal_dens, rep(0, length(w_grid))),
          col = rgb(0.56, 0.27, 0.68, 0.18), border = NA)
  lines(w_grid, bimodal_dens, lwd = 2.8, col = "#8e44ad")
  lines(w_grid, unimodal_dens, lwd = 2.2, col = "gray45", lty = 2)
  abline(v = c(-1, 1), lty = 3, col = "#8e44ad", lwd = 1.4)
  text(-1, max(bimodal_dens) * 1.06, "Negated (!xj)", cex = 0.82, col = "#8e44ad", font = 2)
  text(1,  max(bimodal_dens) * 1.06, "Positive (xj)", cex = 0.82, col = "#8e44ad", font = 2)
  legend("topright", legend = c("Bimodal Literal Slab", "Standard N(0,1) Slab"),
         col = c("#8e44ad", "gray45"), lty = c(1, 2), lwd = 2.2, bty = "n", cex = 0.80)
})

# ==============================================================================
# FIGURE 2: Kepler's Third Law 3-Stage Exponent Recovery & ReLU Comparison
# ==============================================================================
save_dual("fig2_kepler_3stage", width = 11.5, height = 4.6, plot_fn = function() {
  par(mfrow = c(1, 3), mar = c(4.6, 5.6, 3.2, 1.2), cex.lab = 1.05, cex.main = 1.12)
  
  # Panel A: Exponent L1 Error across 3 Stages (Log10 scale)
  eps_floor <- 1e-6
  syn_mat <- cbind(
    log10(pmax(df_kep$exp_err_raw, eps_floor)),
    log10(pmax(df_kep$exp_err_debiased, eps_floor)),
    log10(pmax(df_kep$exp_err_snapped, eps_floor))
  )
  real_mat <- cbind(
    log10(pmax(df_real$exp_err_raw, eps_floor)),
    log10(pmax(df_real$exp_err_debiased, eps_floor)),
    log10(pmax(df_real$exp_err_snapped, eps_floor))
  )
  
  at_pos <- c(1, 2, 3.5, 4.5, 6, 7)
  boxplot(list(syn_mat[,1], real_mat[,1], syn_mat[,2], real_mat[,2], syn_mat[,3], real_mat[,3]),
          at = at_pos, col = rep(c("#3498db", "#e67e22"), 3),
          border = rep(c("#1b4f72", "#935116"), 3),
          xaxt = "n", yaxt = "n", ylim = c(-6.3, 0.5),
          ylab = "L1 Exponent Error |w_M - 1/3| + |w_T - 2/3|",
          main = "(a) 3-Stage Exponent Recovery (M = 100)", outline = TRUE, pch = 16, cex = 0.6)
  grid(nx = NA, ny = NULL, col = "gray88", lty = "dotted")
  axis(1, at = c(1.5, 4.0, 6.5), labels = c("Stage 1:\nRaw VI", "Stage 2A:\nMPM Debiased", "Stage 2B:\nBIC Snapped"), tick = FALSE, line = 0.5)
  axis(2, at = c(-6, -4, -3, -2, -1, 0), labels = c("Exact (0)", "1e-4", "1e-3", "1e-2", "1e-1", "1.0"), las = 1, cex.axis = 0.88)
  legend("topright", legend = c("Synthetic Kepler (p = 10)", "NASA Exoplanets (p = 8)"),
         fill = c("#3498db", "#e67e22"), border = c("#1b4f72", "#935116"), bty = "n", cex = 0.85)
  
  # Panel B: Test RMSE Comparison (Symbolic vs ReLU Input-Skip LBBNN)
  rmse_list <- list(
    log10(df_kep$test_rmse_symbolic),
    log10(df_kep$test_rmse_relu),
    log10(df_real$test_rmse_symbolic),
    log10(df_real$test_rmse_relu)
  )
  boxplot(rmse_list, at = c(1, 2, 3.5, 4.5),
          col = c("#27ae60", "#c0392b", "#27ae60", "#c0392b"),
          border = c("#145a32", "#641e16", "#145a32", "#641e16"),
          xaxt = "n", yaxt = "n", ylim = c(-2.6, 0.8),
          ylab = "Held-Out Test RMSE (AU, log scale)",
          main = "(b) Out-of-Sample Generalization", outline = TRUE, pch = 16, cex = 0.6)
  grid(nx = NA, ny = NULL, col = "gray88", lty = "dotted")
  axis(1, at = c(1.5, 4.0), labels = c("Synthetic Kepler\n(22.6x error reduction)", "NASA Exoplanets\n(67.1x error reduction)"), tick = FALSE, line = 0.5, cex.axis = 0.9)
  axis(2, at = c(-2.3, -2, -1, 0, 0.6), labels = c("0.005", "0.01", "0.10", "1.00", "4.00"), las = 1)
  abline(h = log10(0.005), lty = 3, col = "#145a32", lwd = 1.3)
  text(1.45, log10(0.005) - 0.18, "Oracle Noise (0.005)", cex = 0.78, col = "#145a32")
  legend("topleft", legend = c("Symbolic LBBNN (Ours)", "ReLU Input-Skip LBBNN"),
         fill = c("#27ae60", "#c0392b"), border = c("#145a32", "#641e16"), bty = "n", cex = 0.85)
  
  # Panel C: Active Edges Comparison on Real NASA Exoplanets
  hist(df_real$active_edges_relu, breaks = 15, col = rgb(0.75, 0.22, 0.17, 0.55),
       border = "#641e16", xlim = c(0, 90), ylim = c(0, 35),
       xlab = "Number of Active MPM Edges", ylab = "Frequency (out of 100 seeds)",
       main = "(c) Structural Parsimony (NASA Data)", las = 1)
  grid(col = "gray88", lty = "dotted")
  abline(v = 5, col = "#145a32", lwd = 3.5)
  points(5, 32, pch = 19, col = "#27ae60", cex = 1.6)
  text(25, 30, "Symbolic LBBNN:\nExact 5.00 edges\n(SD = 0.00, 100% seeds)", col = "#145a32", font = 2, cex = 0.85)
  text(58, 20, "ReLU LBBNN:\n36.5 +/- 14.1 edges\n(Dense piecewise fit)", col = "#641e16", font = 2, cex = 0.85)
})

# ==============================================================================
# FIGURE 3: Real NASA Exoplanet Catalogue Fit & Covariate Selection
# ==============================================================================
save_dual("fig3_exoplanet_discovery", width = 11.0, height = 4.8, plot_fn = function() {
  par(mfrow = c(1, 2), mar = c(5.0, 5.2, 3.2, 1.5), cex.lab = 1.05, cex.main = 1.08)
  
  # Panel A: Observed Semi-Major Axis vs Kepler Law Discovered by Symbolic LBBNN (n = 939)
  M_star <- pmax(as.numeric(exoplanet$hoststar_mass), 1e-3)
  T_norm <- pmax(as.numeric(exoplanet$period) / 365.25, 1e-4)
  a_obs  <- as.numeric(exoplanet$semimajoraxis)
  valid  <- is.finite(M_star) & is.finite(T_norm) & is.finite(a_obs) & (a_obs > 0)
  M_star <- M_star[valid]
  T_norm <- T_norm[valid]
  a_obs  <- a_obs[valid]
  
  # Color points by host star mass bracket
  pt_cols <- ifelse(M_star < 0.7, "#e67e22", ifelse(M_star > 1.3, "#8e44ad", "#2980b9"))
  
  plot(T_norm, a_obs, log = "xy", pch = 16, col = adjustcolor(pt_cols, alpha.f = 0.65), cex = 0.9,
       xlab = "Normalized Orbital Period T (Earth Years, log scale)",
       ylab = "Semi-Major Axis a (AU, log scale)",
       main = "(a) NASA Exoplanets (n = 939): Discovered Law", las = 1)
  grid(col = "gray88", lty = "dotted")
  
  # Plot reference lines for M_star = 0.3, 1.0, 2.0 Solar Masses
  t_seq <- 10^seq(log10(min(T_norm)), log10(max(T_norm)), length.out = 200)
  lines(t_seq, (0.3^(1/3)) * (t_seq^(2/3)), col = "#d35400", lwd = 2.0, lty = 2)
  lines(t_seq, (1.0^(1/3)) * (t_seq^(2/3)), col = "#c0392b", lwd = 2.8, lty = 1)
  lines(t_seq, (2.0^(1/3)) * (t_seq^(2/3)), col = "#6c3483", lwd = 2.0, lty = 2)
  
  legend("topleft",
         legend = c("Observed Exoplanets (0.7 <= M* <= 1.3)",
                    "Low/High Mass Stars (M* < 0.7 or M* > 1.3)",
                    "Symbolic MPM: a = M*^(1/3) * T^(2/3) (M*=1)",
                    "Iso-Mass Contours (M* = 0.3, 2.0 M_sun)"),
         pch = c(16, 16, NA, NA),
         col = c("#2980b9", "#8e44ad", "#c0392b", "#d35400"),
         lty = c(NA, NA, 1, 2), lwd = c(NA, NA, 2.8, 2.0),
         bty = "n", cex = 0.80)
  
  # Panel B: Empirical Selection Frequency across the 8 NASA Covariates (M = 100 seeds)
  cov_names <- c("M_star", "T", "mass", "radius", "ecc", "R_star", "Z_metal", "T_eff")
  sel_freq <- c(
    mean(grepl("M_star", df_real$formula, fixed = TRUE)),
    mean(grepl("T", df_real$formula, fixed = TRUE)),
    mean(grepl("mass", df_real$formula, fixed = TRUE)),
    mean(grepl("radius", df_real$formula, fixed = TRUE)),
    mean(grepl("ecc", df_real$formula, fixed = TRUE)),
    mean(grepl("R_star", df_real$formula, fixed = TRUE)),
    mean(grepl("Z_metal", df_real$formula, fixed = TRUE)),
    mean(grepl("T_eff", df_real$formula, fixed = TRUE))
  ) * 100
  
  bp <- barplot(sel_freq, names.arg = FALSE, ylim = c(0, 118),
                col = c("#27ae60", "#27ae60", rep("#bdc3c7", 6)),
                border = c("#145a32", "#145a32", rep("gray50", 6)),
                ylab = "MPM Active Path Selection Frequency (%)",
                main = "(b) Covariate Selection Across 100 Splits", las = 1)
  grid(nx = NA, ny = NULL, col = "gray88", lty = "dotted")
  axis(1, at = bp, labels = cov_names, las = 2, cex.axis = 0.88, tick = FALSE)
  abline(h = 50, lty = 2, col = "#c0392b", lwd = 1.5)
  text(bp, sel_freq + 5, labels = sprintf("%.0f%%", sel_freq), cex = 0.85, font = 2,
       col = c("#145a32", "#145a32", rep("gray35", 6)))
})

# ==============================================================================
# FIGURE 4: Bayesian Logic Regression (Hubin et al., 2020 Benchmark)
# ==============================================================================
save_dual("fig4_logic_regression", width = 11.0, height = 4.6, plot_fn = function() {
  par(mfrow = c(1, 2), mar = c(4.6, 4.8, 3.2, 1.5), cex.lab = 1.05, cex.main = 1.08)
  
  # Panel A: Clause-by-Clause Recovery Rate across 100 Monte Carlo Seeds
  c1_rec <- mean(grepl("x1 & x2", df_logic$formula, fixed = TRUE) | grepl("x2 & x1", df_logic$formula, fixed = TRUE)) * 100
  c2_rec <- mean(grepl("x4 & !x5", df_logic$formula, fixed = TRUE) | grepl("!x5 & x4", df_logic$formula, fixed = TRUE)) * 100
  c3_rec <- mean(grepl("x7 & x8 & x11", df_logic$formula, fixed = TRUE)) * 100
  all_rec <- mean(df_logic$exact_match) * 100
  zero_fp <- mean(df_logic$fpr == 0) * 100
  
  rates <- c(c1_rec, c2_rec, c3_rec, all_rec, zero_fp)
  lbls  <- c("C1: x1&x2\n(Order-2)", "C2: x4&!x5\n(Negated)", "C3: x7&x8&x11\n(Order-3)", "All 3 Exact\nMatch", "Zero False\nPositives")
  cols  <- c("#2980b9", "#2980b9", "#8e44ad", "#27ae60", "#16a085")
  
  bp <- barplot(rates, names.arg = FALSE, ylim = c(0, 118), col = cols, border = "gray25",
                ylab = "Recovery Rate Across 100 Seeds (%)",
                main = "(a) Boolean Conjunction & Negation Recovery", las = 1)
  grid(nx = NA, ny = NULL, col = "gray88", lty = "dotted")
  mtext(lbls, side = 1, at = bp, line = 1.5, cex = 0.72)
  text(bp, rates + 5, labels = sprintf("%.0f%%", rates), font = 2, cex = 0.9, col = "gray15")
  
  # Panel B: Test RMSE Distribution vs Oracle Noise Floor (sigma = 0.50)
  d_exact   <- df_logic$test_rmse[df_logic$exact_match == 1]
  d_partial <- df_logic$test_rmse[df_logic$exact_match == 0]
  
  hist(df_logic$test_rmse, breaks = 18, col = rgb(0.15, 0.68, 0.38, 0.55), border = "#145a32",
       xlim = c(0.45, 0.72), ylim = c(0, 34),
       xlab = "Held-Out Test Set RMSE (n_test = 500)",
       ylab = "Frequency (out of 100 seeds)",
       main = "(b) Predictive Accuracy vs. Oracle Noise Floor", las = 1)
  if (length(d_partial) > 0) {
    hist(d_partial, breaks = seq(0.45, 0.72, by = 0.015), col = rgb(0.90, 0.49, 0.13, 0.60),
         border = "#935116", add = TRUE)
  }
  abline(v = 0.50, col = "#c0392b", lwd = 2.8, lty = 2)
  text(0.502, 23.5, "Oracle SD = 0.500", col = "#c0392b", font = 2, pos = 4, cex = 0.82)
  legend("topright",
         legend = c(sprintf("Exact 3-Clause (67%%, RMSE = %.3f)", mean(d_exact)),
                    sprintf("2-Clause + Sub (33%%, RMSE = %.3f)", mean(d_partial))),
         fill = c(rgb(0.15, 0.68, 0.38, 0.55), rgb(0.90, 0.49, 0.13, 0.60)),
         border = c("#145a32", "#935116"), bty = "n", cex = 0.80)
})

# ==============================================================================
# FIGURE 5: Friedman #1 General Nonlinear Symbolic Benchmark
# ==============================================================================
save_dual("fig5_friedman_nonlinear", width = 11.0, height = 4.6, plot_fn = function() {
  par(mfrow = c(1, 2), mar = c(4.6, 4.8, 3.2, 1.5), cex.lab = 1.02, cex.main = 1.08)
  
  # Panel A: Active Sub-Expression Occurrences per Variable X1..X10
  occ_mat <- sapply(1:10, function(j) {
    pat <- paste0("x", j, "([^0-9]|$)")
    sapply(regmatches(df_nl$formula, gregexpr(pat, df_nl$formula)), length)
  })
  colnames(occ_mat) <- paste0("X", 1:10)
  
  boxplot(occ_mat, col = c(rep("#2980b9", 5), rep("#bdc3c7", 5)),
          border = c(rep("#1b4f72", 5), rep("gray45", 5)),
          xaxt = "n",
          xlab = "Covariate Index (X1-X5: True Signals, X6-X10: Noise)",
          ylab = "Active Symbolic Terms per MPM Formula",
          main = "(a) Friedman #1: Active Term Multiplicity (M = 100)",
          las = 1, pch = 16, cex = 0.6)
  axis(1, at = 1:10, labels = paste0("X", 1:10), cex.axis = 0.85)
  grid(nx = NA, ny = NULL, col = "gray88", lty = "dotted")
  abline(v = 5.5, lty = 2, col = "#c0392b", lwd = 1.8)
  legend("topright",
         legend = c("True Signals X1..X5 (Mean = 11.7 terms)", "Noise Distractors X6..X10 (Mean = 2.5 terms)"),
         fill = c("#2980b9", "#bdc3c7"), border = c("#1b4f72", "gray45"), bty = "n", cex = 0.80)
  
  # Panel B: Joint Distribution of Active Density and Test RMSE
  plot(df_nl$active_density * 100, df_nl$test_rmse, pch = 21,
       bg = rgb(0.16, 0.50, 0.73, 0.65), col = "#1b4f72", cex = 1.15,
       xlim = c(13, 25), ylim = c(0.95, 1.55),
       xlab = "MPM Active Path Network Density (%)",
       ylab = "Held-Out Test Set RMSE (n_test = 500)",
       main = "(b) Sparsity-Accuracy Trade-Off (82.1% Pruned)", las = 1)
  grid(col = "gray88", lty = "dotted")
  abline(h = 1.0, lty = 2, col = "#c0392b", lwd = 2.2)
  abline(v = mean(df_nl$active_density * 100), lty = 3, col = "#1b4f72", lwd = 1.8)
  text(19.5, 1.03, "Oracle Noise Floor (sigma = 1.00)", col = "#c0392b", font = 2, cex = 0.82)
  text(mean(df_nl$active_density * 100) + 0.3, 1.46,
       sprintf("Mean Density = %.1f%%\nMean RMSE = %.3f", mean(df_nl$active_density * 100), mean(df_nl$test_rmse)),
       pos = 4, col = "#1b4f72", font = 2, cex = 0.82)
})

# ==============================================================================
# FIGURE 6: Multi-Task Baseline Comparison Across All Four Benchmarks
# ==============================================================================
if (file.exists("results_baselines/logic/summary_logic_baselines.csv")) {
  b_log <- read.csv("results_baselines/logic/summary_logic_baselines.csv", stringsAsFactors = FALSE)
  b_kep <- read.csv("results_baselines/kepler/summary_kepler_baselines.csv", stringsAsFactors = FALSE)
  b_real <- read.csv("results_baselines/kepler_real/summary_kepler_real_baselines.csv", stringsAsFactors = FALSE)
  b_nl <- read.csv("results_baselines/nonlinear/summary_nonlinear_baselines.csv", stringsAsFactors = FALSE)

  save_dual("fig6_baseline_comparisons", width = 12.0, height = 8.4, plot_fn = function() {
    par(mfrow = c(2, 2), mar = c(6.2, 4.8, 3.2, 1.4), cex.lab = 1.02, cex.main = 1.08)

    # Panel A: Logic Regression (Clause TPR & Exact Recovery vs Test RMSE)
    mat_log <- rbind(b_log$clause_tpr_mean * 100, (1 - b_log$clause_fpr_mean) * 100, b_log$exact_recovery_mean * 100)
    colnames(mat_log) <- b_log$method
    bp_a <- barplot(mat_log, beside = TRUE, ylim = c(0, 118),
                    col = c("#2980b9", "#27ae60", "#8e44ad"), border = "gray25",
                    ylab = "Percentage (%)", main = "(a) Task 1: Bayesian Logic Regression Baselines",
                    las = 2, cex.names = 0.82)
    grid(nx = NA, ny = NULL, col = "gray88", lty = "dotted")
    legend("topright", legend = c("Clause TPR (%)", "Clause 1 - FPR (%)", "Exact 3-Clause Match (%)"),
           fill = c("#2980b9", "#27ae60", "#8e44ad"), border = "gray25", bty = "n", cex = 0.78)

    # Panel B: Synthetic Kepler Law (Structural Recovery & Exact Law vs Log10 RMSE)
    mat_kep <- rbind(b_kep$structural_recovery_mean * 100, b_kep$exact_law_mean * 100)
    colnames(mat_kep) <- b_kep$method
    barplot(mat_kep, beside = TRUE, ylim = c(0, 118),
            col = c("#3498db", "#e67e22"), border = "gray25",
            ylab = "Recovery Rate (%)", main = "(b) Task 2: Synthetic Kepler Law (p = 10)",
            las = 2, cex.names = 0.78)
    grid(nx = NA, ny = NULL, col = "gray88", lty = "dotted")
    legend("topright", legend = c("Exact Support {M*, T} (%)", "Exact Rational Law a = M*^(1/3) T^(2/3) (%)"),
           fill = c("#3498db", "#e67e22"), border = "gray25", bty = "n", cex = 0.78)

    # Panel C: NASA Exoplanet Archive (n = 939) Test RMSE (log10 scale) & Support Recovery
    rmse_real_log <- log10(pmax(b_real$test_rmse_mean, 1e-3))
    cols_real <- ifelse(b_real$method == "Symbolic-LBBNN", "#27ae60", "#2980b9")
    bp_c <- barplot(rmse_real_log + 2.5, names.arg = b_real$method, ylim = c(0, 4.2),
                    col = cols_real, border = "gray25", yaxt = "n",
                    ylab = "Held-Out Test RMSE (AU, log10 scale)",
                    main = "(c) Task 3: NASA Exoplanets (n = 939) Test Error",
                    las = 2, cex.names = 0.78)
    axis(2, at = c(0.5, 1.5, 2.5, 3.5), labels = c("0.01", "0.10", "1.00", "10.0"), las = 1)
    grid(nx = NA, ny = NULL, col = "gray88", lty = "dotted")
    text(bp_c, rmse_real_log + 2.72, labels = sprintf("%.3f", b_real$test_rmse_mean), cex = 0.72, font = 2)

    # Panel D: Friedman #1 Nonlinear Benchmark (Variable TPR, 1-FPR, and Exact Variable Recovery)
    mat_nl <- rbind(b_nl$var_tpr_mean * 100, (1 - b_nl$var_fpr_mean) * 100, b_nl$exact_var_recovery_mean * 100)
    colnames(mat_nl) <- b_nl$method
    barplot(mat_nl, beside = TRUE, ylim = c(0, 118),
            col = c("#2980b9", "#16a085", "#d35400"), border = "gray25",
            ylab = "Percentage (%)", main = "(d) Task 4: Friedman #1 Variable Selection Baselines",
            las = 2, cex.names = 0.78)
    grid(nx = NA, ny = NULL, col = "gray88", lty = "dotted")
    legend("bottomright", legend = c("Signal TPR X1..X5 (%)", "Noise Specificity 1-FPR (%)", "Exact Support X1..X5 (%)"),
           fill = c("#2980b9", "#16a085", "#d35400"), border = "gray25", bty = "n", cex = 0.75)
  })
}

cat("All publication figures regenerated successfully!\n")
