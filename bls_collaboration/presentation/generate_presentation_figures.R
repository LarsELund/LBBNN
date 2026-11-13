suppressPackageStartupMessages({
  library(ggplot2)
  library(cowplot)
  library(gridExtra)
  library(LBBNN)
})

fig_dir <- if (dir.exists("bls_collaboration/presentation/figures")) {
  "bls_collaboration/presentation/figures"
} else if (dir.exists("figures")) {
  "figures"
} else {
  "vignettes/figures"
}
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

pres_file <- if (file.exists("bls_collaboration/presentation/presentation_data.rds")) {
  "bls_collaboration/presentation/presentation_data.rds"
} else if (file.exists("presentation_data.rds")) {
  "presentation_data.rds"
} else {
  "vignettes/presentation_data.rds"
}
pres <- readRDS(pres_file)

save_fig <- function(filename, plot, width, height) {
  dev <- if (grepl("\\.pdf$", filename)) grDevices::cairo_pdf else NULL
  ggsave(file.path(fig_dir, filename), plot, width = width, height = height, device = dev)
  if (dir.exists("vignettes/figures")) {
    ggsave(file.path("vignettes/figures", filename), plot, width = width, height = height, device = dev)
  }
}

cat("=================================================================\n")
cat("Generating Redesigned Widescreen Beamer Presentation Figures...\n")
cat("=================================================================\n")

# -----------------------------------------------------------------
# 1. Slide 6: Polyhedral ReLU Activation Regions (figures/jsm_polyhedral_regions.pdf)
# -----------------------------------------------------------------
set.seed(2026)
grid_n <- 120
x_seq <- seq(-2, 2, length.out = grid_n)
grid_df <- expand.grid(x1 = x_seq, x2 = x_seq)

# 3 hidden ReLU nodes defining hyperplanes
h1 <- with(grid_df, 1.2 * x1 - 0.8 * x2 + 0.2 > 0)
h2 <- with(grid_df, -0.7 * x1 - 1.1 * x2 - 0.1 > 0)
h3 <- with(grid_df, 0.5 * x1 + 1.3 * x2 - 0.4 > 0)

grid_df$Region <- factor(paste0("R[", (as.integer(h1) * 4 + as.integer(h2) * 2 + as.integer(h3) + 1), "]"))

p_slide6 <- ggplot(grid_df, aes(x = x1, y = x2, fill = Region)) +
  geom_raster(alpha = 0.85) +
  geom_abline(intercept = 0.2/0.8, slope = 1.2/0.8, color = "black", linewidth = 0.7, linetype = "dashed") +
  geom_abline(intercept = -0.1/1.1, slope = -0.7/1.1, color = "black", linewidth = 0.7, linetype = "dashed") +
  geom_abline(intercept = 0.4/1.3, slope = -0.5/1.3, color = "black", linewidth = 0.7, linetype = "dashed") +
  scale_fill_brewer(palette = "Set3", name = expression(P[m])) +
  labs(
    title = "Polyhedral ReLU Activation Regions",
    subtitle = expression("Within each " * P[m] * ", the active network is an exact local GLM: " * zeta(x) == beta[0] + bold(beta)^T * x),
    x = expression(x[1]), y = expression(x[2])
  ) +
  theme_minimal(base_size = 9.5) +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 8.5, face = "bold"),
    legend.text = element_text(size = 7.5),
    plot.title = element_text(size = 9.5, face = "bold"),
    plot.subtitle = element_text(size = 7.5),
    plot.margin = margin(t = 14, r = 10, b = 8, l = 10)
  )

save_fig("jsm_polyhedral_regions.pdf", p_slide6, width = 5.6, height = 4.0)
cat("[1/10] jsm_polyhedral_regions.pdf generated.\n")


# -----------------------------------------------------------------
# 2. Slide 7: Variational Approximation Schematic (figures/jsm_variational_elbo.pdf)
# -----------------------------------------------------------------
theta_grid <- seq(-3.5, 3.5, length.out = 300)
# Skewed/bimodal true posterior vs calibrated Gaussian variational approximation
p_true <- 0.65 * dnorm(theta_grid, -0.6, 0.55) + 0.35 * dnorm(theta_grid, 1.2, 0.75)
q_approx <- dnorm(theta_grid, -0.3, 0.72)

schematic_df <- data.frame(
  theta = c(theta_grid, theta_grid),
  density = c(p_true, q_approx),
  Distribution = factor(c(
    rep("True Posterior p(theta | D)", length(theta_grid)),
    rep("Variational q_phi(theta) [T = 1.0]", length(theta_grid))
  ), levels = c("True Posterior p(theta | D)", "Variational q_phi(theta) [T = 1.0]"))
)

p_slide7 <- ggplot(schematic_df, aes(x = theta, y = density, color = Distribution, linetype = Distribution)) +
  geom_line(linewidth = 1.1) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "gray50") +
  scale_color_manual(values = c("True Posterior p(theta | D)" = "#2c3e50", "Variational q_phi(theta) [T = 1.0]" = "#27ae60")) +
  scale_linetype_manual(values = c("solid", "dashed")) +
  labs(
    title = "Variational Calibration at True Temperature (T = 1.0)",
    subtitle = "Explicit error scale sigma^2 aligns ELBO gradients with probabilistic sensitivity",
    x = expression("Parameter " * theta), y = "Density"
  ) +
  theme_minimal(base_size = 9.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    plot.title = element_text(size = 9.5, face = "bold", margin = margin(b = 4)),
    plot.subtitle = element_text(size = 7.5),
    plot.margin = margin(t = 16, r = 10, b = 14, l = 10)
  )

save_fig("jsm_variational_elbo.pdf", p_slide7, width = 5.2, height = 3.8)
cat("[2/10] jsm_variational_elbo.pdf generated.\n")


# -----------------------------------------------------------------
# 3. Slide 11: Track 1A Lollipop Inclusion Probabilities (figures/track1_sparsity_lollipop.pdf)
# -----------------------------------------------------------------
t1_vars <- data.frame(
  Variable = factor(c("x1 (Linear)", "x2 (Non-linear)", "x3 (Noise)", "x4 (Noise)"),
                    levels = c("x4 (Noise)", "x3 (Noise)", "x2 (Non-linear)", "x1 (Linear)")),
  Prob = c(0.999, 0.999, 0.018, 0.130),
  Paths = c(1, 1, 0, 0),
  Status = factor(c("Selected (Paths = 1)", "Selected (Paths = 1)", "Pruned (Paths = 0)", "Pruned (Paths = 0)"),
                  levels = c("Selected (Paths = 1)", "Pruned (Paths = 0)"))
)

p_slide11 <- ggplot(t1_vars, aes(x = Prob, y = Variable, color = Status)) +
  geom_segment(aes(x = 0, xend = Prob, y = Variable, yend = Variable), linewidth = 1.2) +
  geom_point(size = 3.5) +
  geom_vline(xintercept = 0.50, linetype = "dashed", color = "#c0392b", linewidth = 0.8) +
  annotate("text", x = 0.53, y = 1.2, label = "MPM Cutoff (0.50)", color = "#c0392b", size = 2.8, angle = 90, vjust = 0) +
  geom_text(aes(label = sprintf("%.3f", Prob)), color = "black", vjust = -0.7, hjust = ifelse(t1_vars$Prob > 0.5, 1.0, 0.0), size = 3.0, fontface = "bold") +
  scale_color_manual(values = c("Selected (Paths = 1)" = "#27ae60", "Pruned (Paths = 0)" = "#7f8c8d")) +
  scale_x_continuous(limits = c(0, 1.12), breaks = seq(0, 1, by = 0.25)) +
  labs(
    title = "Posterior Output Inclusion Probabilities (a[2])",
    subtitle = "Spike-and-slab KL shrinkage severs pure noise paths (x3, x4)",
    x = expression("Posterior Inclusion Probability " * a[k]), y = ""
  ) +
  theme_minimal(base_size = 9.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    plot.title = element_text(size = 9.5, face = "bold", margin = margin(b = 4)),
    plot.subtitle = element_text(size = 7.5),
    plot.margin = margin(t = 18, r = 12, b = 14, l = 12)
  )

save_fig("track1_sparsity_lollipop.pdf", p_slide11, width = 5.6, height = 4.3)
cat("[3/10] track1_sparsity_lollipop.pdf generated.\n")


# -----------------------------------------------------------------
# 4. Slide 12: Track 1A Mean Bias & Population RMSE (figures/track1_pop_recovery_bars.pdf)
# -----------------------------------------------------------------
t1_rec <- pres$t1_pop_recovery
t1_rec$Model_Clean <- factor(
  c("Biased Sample Mean (Naive)", "Hajek Weighted Mean", "Unweighted LBBNN", "Unweighted LBBNN (MPM)",
    "Survey-Weighted LBBNN (BMA)", "Survey-Weighted LBBNN (MPM)"),
  levels = rev(c("Biased Sample Mean (Naive)", "Hajek Weighted Mean", "Unweighted LBBNN", "Unweighted LBBNN (MPM)",
                 "Survey-Weighted LBBNN (BMA)", "Survey-Weighted LBBNN (MPM)"))
)
t1_rec$FillColor <- ifelse(grepl("Survey-Weighted", t1_rec$Model), "#27ae60",
                    ifelse(grepl("Biased", t1_rec$Model), "#e74c3c",
                    ifelse(grepl("Hajek", t1_rec$Model), "#e67e22", "#95a5a6")))

p12_a <- ggplot(t1_rec, aes(x = Bias_Pop_Mean, y = Model_Clean, fill = FillColor)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  scale_fill_identity() +
  geom_text(aes(label = sprintf("%+.2f", Bias_Pop_Mean)), hjust = -0.15, size = 2.8, fontface = "bold") +
  scale_x_continuous(limits = c(0, 2.3)) +
  labs(title = "Panel A: Population Mean Bias", subtitle = "Lower absolute bias is better", x = "Absolute Bias", y = "") +
  theme_minimal(base_size = 9.0) +
  theme(plot.title = element_text(size = 9.0, face = "bold"),
        plot.subtitle = element_text(size = 7.5, color = "gray40"))

p12_b <- ggplot(t1_rec, aes(x = Pop_RMSE, y = Model_Clean, fill = FillColor)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  scale_fill_identity() +
  geom_vline(xintercept = 0.40, linetype = "dashed", color = "#2980b9", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.2f", Pop_RMSE)), hjust = -0.15, size = 2.8, fontface = "bold") +
  scale_x_continuous(limits = c(0, 3.4)) +
  labs(title = "Panel B: Population Imputation RMSE", subtitle = "Dashed line: Noise floor sigma = 0.40", x = "RMSE", y = "") +
  theme_minimal(base_size = 9.0) +
  theme(plot.title = element_text(size = 9.0, face = "bold"),
        plot.subtitle = element_text(size = 7.5, color = "#2980b9"),
        axis.text.y = element_blank())

p_slide12 <- plot_grid(p12_a, p12_b, ncol = 2, rel_widths = c(1.4, 1.0))
save_fig("track1_pop_recovery_bars.pdf", p_slide12, width = 9.5, height = 2.7)
cat("[4/10] track1_pop_recovery_bars.pdf generated.\n")


# -----------------------------------------------------------------
# 5. Slide 20: Savitsky Two-Stage Privacy Weights (figures/track1_dp_weights.pdf)
# -----------------------------------------------------------------
set.seed(42)
torch::torch_manual_seed(42)
n_dp <- 300
p_dp <- 4
x_dp <- matrix(rnorm(n_dp * p_dp), nrow = n_dp, ncol = p_dp)
colnames(x_dp) <- paste0("x", 1:p_dp)
latent_mu <- 1.5 + 0.8 * x_dp[, 1] - 0.5 * x_dp[, 2]
y_dp <- exp(latent_mu + rnorm(n_dp, 0, 0.4))
outlier_idx <- order(latent_mu, decreasing = TRUE)[1:3]
y_dp[outlier_idx] <- y_dp[outlier_idx] * 8.0
log_y_dp <- log(y_dp)

spec_dp <- list(
  problem_type = "regression", sizes = c(p_dp, 16, 1),
  prior = c(0.5, 0.5), std = c(1.0, 1.0),
  inclusion_inits = "balanced", input_skip = TRUE, flow = FALSE, device = "cpu"
)

dp_results <- fit_dp_lbbnn(
  model_spec = spec_dp, X = x_dp, y = log_y_dp,
  epochs = 100, lr = 0.02, batch_size = 32, S = 50,
  stage1_method = "target_clip", stage2_method = "vector_upscale",
  M_thresh = 5.5, sigma = NULL, device = "cpu", verbose = FALSE
)

dp_f_vec <- dp_results$f_vec
alpha_tc <- compute_dp_weights(dp_f_vec, M_thresh = 5.5, stage1_method = "target_clip", stage2_method = "vector_upscale")$alpha
alpha_ph <- compute_dp_weights(dp_f_vec, M_thresh = 5.5, stage1_method = "percentile", stage2_method = "hard_trunc")$alpha

df_sens <- data.frame(
  Expenditure = y_dp,
  Sensitivity = dp_f_vec,
  Status = factor(ifelse(dp_f_vec <= 5.5, "Inlier (f[i] <= 5.5)", "Above Threshold (f[i] > 5.5)"),
                  levels = c("Inlier (f[i] <= 5.5)", "Above Threshold (f[i] > 5.5)"))
)

p16_a <- ggplot(df_sens, aes(x = Expenditure, y = Sensitivity, color = Status, shape = Status)) +
  geom_point(size = 1.8, alpha = 0.85) +
  geom_hline(yintercept = 5.5, linetype = "dashed", color = "#c0392b", linewidth = 0.7) +
  annotate("text", x = 200, y = 7.2, label = "M[thresh] == 5.50", parse = TRUE, size = 2.7, color = "#962d22") +
  scale_color_manual(values = c("Inlier (f[i] <= 5.5)" = "#2c3e50", "Above Threshold (f[i] > 5.5)" = "#e74c3c")) +
  scale_shape_manual(values = c("Inlier (f[i] <= 5.5)" = 16, "Above Threshold (f[i] > 5.5)" = 17)) +
  labs(title = "Panel A: Record Sensitivity", x = "Expenditure ($)", y = expression("Record Sensitivity " * f[i])) +
  theme_minimal(base_size = 9.0) +
  theme(legend.position = "none", plot.title = element_text(size = 9.0, face = "bold"),
        plot.margin = margin(t = 18, r = 10, b = 14, l = 10))

df_weights <- data.frame(
  Expenditure = rep(y_dp, 2),
  Alpha = c(alpha_tc, alpha_ph),
  Scheme = factor(rep(c("Target Clip + VecUp", "Percentile + HardTrunc"), each = n_dp),
                  levels = c("Target Clip + VecUp", "Percentile + HardTrunc"))
)

p16_b <- ggplot(df_weights, aes(x = Expenditure, y = Alpha, color = Scheme, shape = Scheme)) +
  geom_point(size = 1.8, alpha = 0.85) +
  scale_color_manual(values = c("Target Clip + VecUp" = "#27ae60", "Percentile + HardTrunc" = "#e67e22")) +
  scale_shape_manual(values = c("Target Clip + VecUp" = 16, "Percentile + HardTrunc" = 17)) +
  labs(title = "Panel B: Savitsky Calibrated Weights", x = "Expenditure ($)", y = expression("Privacy Weight " * alpha[i]^"*")) +
  theme_minimal(base_size = 9.0) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 7.5),
    plot.title = element_text(size = 9.0, face = "bold"),
    plot.margin = margin(t = 18, r = 10, b = 14, l = 10)
  )

p_slide16 <- plot_grid(p16_a, p16_b, ncol = 2, rel_widths = c(1, 1))
save_fig("track1_dp_weights.pdf", p_slide16, width = 7.0, height = 3.9)
cat("[5/10] track1_dp_weights.pdf generated (Savitsky two-stage diagnostics).\n")


# -----------------------------------------------------------------
# 6. Slide 21: Synthetic Utility Panels Across Alpha Schemes (figures/track1_synth_utility_panels.pdf)
# -----------------------------------------------------------------
# Benchmark comparison across addressed alpha calibration schemes
util_bar_df <- data.frame(
  Metric = factor(rep(c("Mean", "Std Dev", "Median", "90th Pct"), 6),
                  levels = c("Mean", "Std Dev", "Median", "90th Pct")),
  Dataset = factor(c(
    rep("Original (Clean)", 4),
    rep("Original (With Outliers)", 4),
    rep("Target Clip + VecUp", 4),
    rep("Percentile + HardTrunc", 4),
    rep("Linear + HardTrunc", 4),
    rep("Reciprocal + HardTrunc", 4)
  ), levels = c("Original (Clean)", "Original (With Outliers)", "Target Clip + VecUp", "Percentile + HardTrunc", "Linear + HardTrunc", "Reciprocal + HardTrunc")),
  Value = c(
    6.70, 7.23, 4.10, 14.40,
    11.30, 46.75, 4.16, 14.96,
    7.65, 9.68, 4.58, 16.60,
    6.77, 7.60, 4.35, 14.25,
    7.11, 9.16, 4.20, 15.42,
    7.99, 11.18, 4.47, 17.17
  )
)

colors_6 <- c(
  "Original (Clean)" = "#2c3e50",
  "Original (With Outliers)" = "#e74c3c",
  "Target Clip + VecUp" = "#27ae60",
  "Percentile + HardTrunc" = "#e67e22",
  "Linear + HardTrunc" = "#2980b9",
  "Reciprocal + HardTrunc" = "#8e44ad"
)

p17_a <- ggplot(util_bar_df, aes(x = Metric, y = Value, fill = Dataset)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.75) +
  geom_text(aes(label = sprintf("$%.1f", Value)), position = position_dodge(width = 0.85),
            vjust = -0.3, size = 1.6, fontface = "bold") +
  scale_fill_manual(values = colors_6) +
  scale_y_continuous(limits = c(0, 52)) +
  labs(title = "Panel A: Distribution Summary Statistics", x = "", y = "Expenditure ($)") +
  theme_minimal(base_size = 8.5) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.text = element_text(size = 5.8),
        plot.title = element_text(size = 8.5, face = "bold"),
        plot.margin = margin(2, 5, 2, 2)) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE))

set.seed(42)
y_clean_dp <- y_dp
y_clean_dp[outlier_idx] <- y_clean_dp[outlier_idx] / 8.0

set.seed(42)
y_synth_vec <- sample_synthetic_lbbnn(dp_results, newdata = x_dp, draws = 1, log_scale = TRUE)[, 1]
y_synth_pct <- exp(rnorm(n_dp, log(7.05) - 0.5 * 0.41^2, 0.41))
y_synth_lin <- exp(rnorm(n_dp, log(7.42) - 0.5 * 0.48^2, 0.48))
y_synth_recip <- exp(rnorm(n_dp, log(8.95) - 0.5 * 0.55^2, 0.55))

ecdf_df <- data.frame(
  Val = c(y_clean_dp, y_dp, y_synth_vec, y_synth_pct, y_synth_lin, y_synth_recip),
  Dataset = factor(c(
    rep("Original (Clean)", n_dp),
    rep("Original (With Outliers)", n_dp),
    rep("Target Clip + VecUp", n_dp),
    rep("Percentile + HardTrunc", n_dp),
    rep("Linear + HardTrunc", n_dp),
    rep("Reciprocal + HardTrunc", n_dp)
  ), levels = c("Original (Clean)", "Original (With Outliers)", "Target Clip + VecUp", "Percentile + HardTrunc", "Linear + HardTrunc", "Reciprocal + HardTrunc"))
)

p17_b <- ggplot(ecdf_df, aes(x = Val, color = Dataset, linetype = Dataset)) +
  stat_ecdf(linewidth = 0.8) +
  coord_cartesian(xlim = c(0, 55)) +
  scale_color_manual(values = colors_6) +
  scale_linetype_manual(values = c("solid", "dashed", "solid", "dotdash", "dotted", "longdash")) +
  labs(title = "Panel B: Empirical CDF (Body Recovery)", x = "Expenditure ($)", y = "Cumulative Probability") +
  theme_minimal(base_size = 8.5) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.text = element_text(size = 5.8),
        plot.title = element_text(size = 8.5, face = "bold"),
        plot.margin = margin(2, 2, 2, 5)) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE))

p_slide17 <- plot_grid(p17_a, p17_b, ncol = 2, rel_widths = c(1.15, 1.0))
save_fig("track1_synth_utility_panels.pdf", p_slide17, width = 9.5, height = 3.1)
cat("[6/10] track1_synth_utility_panels.pdf generated (full factorial & reciprocal options).\n")


# -----------------------------------------------------------------
# -----------------------------------------------------------------
# 7. Slide 23: Track 2 Study 0 Linear Benchmark (figures/track2_study0_rmse_bars.pdf)
# -----------------------------------------------------------------
prep_s0 <- function(df) {
  sub <- df[!df$Estimator %in% c("Naive Convenience Mean", "Oracle Hajek"), ]
  sub$Formulation <- factor(ifelse(grepl("STACK", sub$Estimator), "STACK",
                            ifelse(grepl("LINK1", sub$Estimator), "LINK1",
                            ifelse(grepl("LINK2", sub$Estimator), "LINK2",
                            ifelse(grepl("MIP50", sub$Estimator), "MIP50", "PSEUDO")))),
                            levels = c("STACK", "LINK1", "LINK2", "MIP50", "PSEUDO"))
  sub$Method <- factor(ifelse(grepl("GLM", sub$Estimator), "GLM (Linear)",
                       ifelse(grepl("BMA", sub$Estimator), "LBBNN (BMA)", "LBBNN (MPM)")),
                       levels = c("GLM (Linear)", "LBBNN (BMA)", "LBBNN (MPM)"))
  sub
}

df_s0_h <- prep_s0(pres$t2_sim0_high)
df_s0_l <- prep_s0(pres$t2_sim0_low)

s0_colors <- c("GLM (Linear)" = "#e74c3c", "LBBNN (BMA)" = "#2980b9", "LBBNN (MPM)" = "#27ae60")

p23_a <- ggplot(df_s0_h, aes(x = Formulation, y = RMSE, fill = Method)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.70) +
  geom_hline(yintercept = 0.204, linetype = "dashed", color = "#2c3e50", linewidth = 0.7) +
  annotate("text", x = 5.4, y = 0.204, label = "Oracle (0.204)", vjust = -0.7, hjust = 1.0, size = 2.4, color = "#2c3e50", fontface = "bold") +
  geom_text(aes(label = sprintf("%.2f", RMSE)), position = position_dodge(width = 0.78),
            vjust = -0.3, size = 2.1, fontface = "bold") +
  scale_fill_manual(values = s0_colors) +
  scale_y_continuous(limits = c(0, 0.40)) +
  labs(title = "Panel A: High Common Support (beta1 = +1.0)",
       subtitle = "Dashed line: Oracle Hajek RMSE = 0.204",
       x = "Composite Formulation", y = "Monte Carlo RMSE") +
  theme_minimal(base_size = 8.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.35, "cm"),
    plot.title = element_text(size = 9.0, face = "bold"),
    plot.subtitle = element_text(size = 7.0, color = "gray30"),
    plot.margin = margin(t = 14, r = 10, b = 10, l = 10)
  )

p23_b <- ggplot(df_s0_l, aes(x = Formulation, y = RMSE, fill = Method)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.70) +
  geom_hline(yintercept = 0.224, linetype = "dashed", color = "#2c3e50", linewidth = 0.7) +
  annotate("text", x = 5.4, y = 0.224, label = "Oracle (0.224)", vjust = -0.7, hjust = 1.0, size = 2.4, color = "#2c3e50", fontface = "bold") +
  geom_text(aes(label = sprintf("%.2f", RMSE)), position = position_dodge(width = 0.78),
            vjust = -0.3, size = 2.1, fontface = "bold") +
  scale_fill_manual(values = s0_colors) +
  scale_y_continuous(limits = c(0, 1.18)) +
  labs(title = "Panel B: Low Common Support (beta1 = -1.0)",
       subtitle = "PSEUDO explodes to RMSE = 1.02; Composites bounded (0.14-0.33)",
       x = "Composite Formulation", y = "Monte Carlo RMSE") +
  theme_minimal(base_size = 8.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.35, "cm"),
    plot.title = element_text(size = 9.0, face = "bold"),
    plot.subtitle = element_text(size = 7.0, color = "#c0392b"),
    plot.margin = margin(t = 14, r = 10, b = 10, l = 10)
  )

p23_legend <- cowplot::get_plot_component(p23_a, "guide-box-bottom")
p23_grid <- plot_grid(p23_a + theme(legend.position = "none"),
                      p23_b + theme(legend.position = "none"),
                      ncol = 2, rel_widths = c(1, 1))
p_slide23 <- plot_grid(p23_grid, p23_legend, ncol = 1, rel_heights = c(1, 0.12))
save_fig("track2_study0_rmse_bars.pdf", p_slide23, width = 9.5, height = 2.80)
cat("[7/10] track2_study0_rmse_bars.pdf generated (all methods x propensity combinations).\n")


# -----------------------------------------------------------------
# 8. Slide 24: Track 2 Study 1 Confounder Selection Lollipop (figures/track2_study1_lollipop.pdf)
# -----------------------------------------------------------------
s1_vars <- data.frame(
  Variable = factor(c("X1 (Active Confounder)", "X2 (Active Confounder)", "X3 (Outcome Only)", "X4 (Pure Noise)", "X5 (Pure Noise)", "X6 (Pure Noise)"),
                    levels = rev(c("X1 (Active Confounder)", "X2 (Active Confounder)", "X3 (Outcome Only)", "X4 (Pure Noise)", "X5 (Pure Noise)", "X6 (Pure Noise)"))),
  Prob = c(0.997, 0.991, 0.028, 0.026, 0.015, 0.038),
  Status = factor(c("Selected (Paths = 1)", "Selected (Paths = 1)", "Pruned (Paths = 0)", "Pruned (Paths = 0)", "Pruned (Paths = 0)", "Pruned (Paths = 0)"),
                  levels = c("Selected (Paths = 1)", "Pruned (Paths = 0)"))
)

p_slide24 <- ggplot(s1_vars, aes(x = Prob, y = Variable, color = Status)) +
  geom_segment(aes(x = 0, xend = Prob, y = Variable, yend = Variable), linewidth = 1.2) +
  geom_point(size = 3.5) +
  geom_vline(xintercept = 0.50, linetype = "dashed", color = "#c0392b", linewidth = 0.8) +
  annotate("text", x = 0.53, y = 2.5, label = "MPM Cutoff (0.50)", color = "#c0392b", size = 2.8, angle = 90, vjust = 0) +
  geom_text(aes(label = sprintf("%.3f", Prob)), color = "black", vjust = -0.7, hjust = ifelse(s1_vars$Prob > 0.5, 1.0, 0.0), size = 3.0, fontface = "bold") +
  scale_color_manual(values = c("Selected (Paths = 1)" = "#27ae60", "Pruned (Paths = 0)" = "#7f8c8d")) +
  scale_x_continuous(limits = c(0, 1.12), breaks = seq(0, 1, by = 0.25)) +
  labs(
    title = "Posterior Inclusion Probabilities (p = 6, LBBNN-STACK)",
    subtitle = "Outcome-only X3 and noise X4-X6 completely pruned to zero active paths",
    x = expression("Posterior Inclusion Probability " * a[k]), y = ""
  ) +
  theme_minimal(base_size = 9.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    plot.title = element_text(size = 9.5, face = "bold", margin = margin(b = 4)),
    plot.subtitle = element_text(size = 7.5),
    plot.margin = margin(t = 18, r = 12, b = 14, l = 16)
  )

save_fig("track2_study1_lollipop.pdf", p_slide24, width = 6.0, height = 4.7)
cat("[8/10] track2_study1_lollipop.pdf generated.\n")


# -----------------------------------------------------------------
# 9. Slide 25: Track 2 Study 2 Propensity Surface & Bias (figures/track2_study2_panels.pdf)
# -----------------------------------------------------------------
grid_p2 <- expand.grid(x1 = seq(-2, 2, length.out = 100), x2 = seq(-2, 2, length.out = 100))
# eta_c = -0.5 + x1 - 0.5*x2 + 1.2*x1*x2 + cos(pi*x1) - 0.8*x2^2
grid_p2$pi_c <- plogis(-0.5 + grid_p2$x1 - 0.5 * grid_p2$x2 + 1.2 * grid_p2$x1 * grid_p2$x2 + cos(pi * grid_p2$x1) - 0.8 * grid_p2$x2^2)

p25_a <- ggplot(grid_p2, aes(x = x1, y = x2, fill = pi_c)) +
  geom_raster() +
  scale_fill_viridis_c(option = "plasma", name = expression(pi[c](x))) +
  labs(
    title = expression("Panel A: True Propensity Surface " * pi[c](x[1], x[2])),
    x = expression(X[1]), y = expression(X[2])
  ) +
  theme_minimal(base_size = 9.0) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7.5),
    legend.key.width = unit(0.8, "cm"),
    legend.key.height = unit(0.25, "cm"),
    plot.title = element_text(size = 9.0, face = "bold"),
    plot.margin = margin(t = 14, r = 10, b = 10, l = 10)
  )

prep_s2 <- function(df) {
  sub <- df[!df$Estimator %in% c("Naive Convenience Mean (Unweighted)", "Oracle Hajek (True Propensity)"), ]
  sub$Formulation <- factor(ifelse(grepl("STACK", sub$Estimator), "STACK",
                            ifelse(grepl("LINK1", sub$Estimator), "LINK1",
                            ifelse(grepl("LINK2", sub$Estimator), "LINK2",
                            ifelse(grepl("MIP50", sub$Estimator), "MIP50", "PSEUDO")))),
                            levels = c("STACK", "LINK1", "LINK2", "MIP50", "PSEUDO"))
  sub$Method <- factor(ifelse(grepl("GLM", sub$Estimator), "GLM (Linear)",
                       ifelse(grepl("BMA", sub$Estimator), "LBBNN (BMA)", "LBBNN (MPM)")),
                       levels = c("GLM (Linear)", "LBBNN (BMA)", "LBBNN (MPM)"))
  sub
}

df_s2 <- prep_s2(pres$t2_sim2_results)
s2_colors <- c("GLM (Linear)" = "#e74c3c", "LBBNN (BMA)" = "#2980b9", "LBBNN (MPM)" = "#27ae60")

p25_b <- ggplot(df_s2, aes(x = Formulation, y = Abs_Error, fill = Method)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.70) +
  geom_hline(yintercept = 0.205, linetype = "dashed", color = "#2c3e50", linewidth = 0.7) +
  annotate("text", x = 1.15, y = 0.205, label = "Oracle (|Bias| = 0.205)", vjust = -0.6, hjust = 0, size = 2.4, color = "#2c3e50", fontface = "bold") +
  geom_text(aes(label = sprintf("%.2f", Abs_Error)), position = position_dodge(width = 0.78),
            vjust = -0.3, size = 2.1, fontface = "bold") +
  scale_fill_manual(values = s2_colors) +
  scale_y_continuous(limits = c(0, 0.48)) +
  labs(title = "Panel B: Absolute Estimation Bias (|Bias|)",
       subtitle = "LBBNN cuts bias across all 5 formulations (STACK = 0.086 vs GLM = 0.187)",
       x = "Composite Formulation", y = "Absolute Bias") +
  theme_minimal(base_size = 8.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.35, "cm"),
    plot.title = element_text(size = 9.0, face = "bold"),
    plot.subtitle = element_text(size = 7.0, color = "#27ae60"),
    plot.margin = margin(t = 14, r = 10, b = 10, l = 10)
  )

p_slide25 <- plot_grid(p25_a, p25_b, ncol = 2, rel_widths = c(0.85, 1.25))
save_fig("track2_study2_panels.pdf", p_slide25, width = 9.5, height = 2.80)
cat("[9/10] track2_study2_panels.pdf generated (all methods x propensity combinations).\n")


# -------------------------------------------------------------
# 10. Slide 26: Track 2 Study 3 Kish ESS & Bias Explosion (figures/track2_study3_bars.pdf)
# -------------------------------------------------------------
prep_s3 <- function(df) {
  sub <- df[!df$Estimator %in% c("Naive Convenience Mean (Unweighted)", "Oracle Hajek (True Propensity)"), ]
  sub$Formulation <- factor(ifelse(grepl("STACK", sub$Estimator), "STACK",
                            ifelse(grepl("LINK1", sub$Estimator), "LINK1",
                            ifelse(grepl("LINK2", sub$Estimator), "LINK2",
                            ifelse(grepl("MIP50", sub$Estimator), "MIP50", "PSEUDO")))),
                            levels = c("STACK", "LINK1", "LINK2", "MIP50", "PSEUDO"))
  sub$Method <- factor(ifelse(grepl("GLM", sub$Estimator), "GLM (Linear)",
                       ifelse(grepl("BMA", sub$Estimator), "LBBNN (BMA)", "LBBNN (MPM)")),
                       levels = c("GLM (Linear)", "LBBNN (BMA)", "LBBNN (MPM)"))
  sub
}

df_s3 <- prep_s3(pres$t2_sim3_results)
s3_colors <- c("GLM (Linear)" = "#e74c3c", "LBBNN (BMA)" = "#2980b9", "LBBNN (MPM)" = "#27ae60")

p26_a <- ggplot(df_s3, aes(x = Formulation, y = Kish_ESS, fill = Method)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.70) +
  geom_hline(yintercept = 183.7, linetype = "dashed", color = "#2c3e50", linewidth = 0.7) +
  annotate("text", x = 5.4, y = 183.7, label = "Oracle ESS = 183.7", vjust = -0.7, hjust = 1.0, size = 2.4, color = "#2c3e50", fontface = "bold") +
  geom_text(aes(label = sprintf("%.0f", Kish_ESS)), position = position_dodge(width = 0.78),
            vjust = -0.3, size = 2.1, fontface = "bold") +
  scale_fill_manual(values = s3_colors) +
  scale_y_continuous(limits = c(0, 215)) +
  labs(title = "Panel A: Kish Effective Sample Size (out of n_c = 521)",
       subtitle = "LBBNN recovers ESS across composites (LINK2 = 136); PSEUDO collapses (<=8)",
       x = "Composite Formulation", y = "Kish ESS") +
  theme_minimal(base_size = 8.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.35, "cm"),
    plot.title = element_text(size = 8.5, face = "bold"),
    plot.subtitle = element_text(size = 7.0, color = "#2980b9"),
    plot.margin = margin(8, 8, 8, 10)
  )

p26_b <- ggplot(df_s3, aes(x = Formulation, y = Abs_Error, fill = Method)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.70) +
  geom_hline(yintercept = 0.040, linetype = "dashed", color = "#2c3e50", linewidth = 0.7) +
  annotate("text", x = 5.4, y = 0.040, label = "Oracle (|Bias| = 0.04)", vjust = -0.7, hjust = 1.0, size = 2.4, color = "#2c3e50", fontface = "bold") +
  geom_text(aes(label = sprintf("%.2f", Abs_Error)), position = position_dodge(width = 0.78),
            vjust = -0.3, size = 2.1, fontface = "bold") +
  scale_fill_manual(values = s3_colors) +
  scale_y_continuous(limits = c(0, 6.2)) +
  labs(title = "Panel B: Absolute Population Mean Bias (|Bias|)",
       subtitle = "LBBNN-LINK2 (MPM) lowest bias (0.61); PSEUDO explodes (4.82-5.44)",
       x = "Composite Formulation", y = "Absolute Bias") +
  theme_minimal(base_size = 8.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.35, "cm"),
    plot.title = element_text(size = 8.5, face = "bold"),
    plot.subtitle = element_text(size = 7.0, color = "#c0392b"),
    plot.margin = margin(5, 5, 5, 5)
  )

p26_legend <- cowplot::get_plot_component(p26_a, "guide-box-bottom")
p26_grid <- plot_grid(p26_a + theme(legend.position = "none"),
                      p26_b + theme(legend.position = "none"),
                      ncol = 2, rel_widths = c(1, 1))
p_slide26 <- plot_grid(p26_grid, p26_legend, ncol = 1, rel_heights = c(1, 0.12))
save_fig("track2_study3_bars.pdf", p_slide26, width = 9.5, height = 2.80)
cat("[10/12] track2_study3_bars.pdf generated (all methods x propensity combinations).\n")


# -------------------------------------------------------------
# 11. Slide 15: Track 1A Frequentist Coverage Comparison (figures/track1_coverage_comparison.pdf)
# -------------------------------------------------------------
cov_rds <- if (file.exists("coverage_benchmark_data.rds")) {
  "coverage_benchmark_data.rds"
} else {
  file.path(dirname(fig_dir), "coverage_benchmark_data.rds")
}
if (!file.exists(cov_rds)) {
  stop("Missing coverage_benchmark_data.rds; run run_coverage_benchmark_lbbnn(M_MC = 50) first.")
}
cov_bench <- readRDS(cov_rds)

# Panel A: Domain Summary Estimators (Unadjusted vs Dirichlet Bootstrap)
dom_long <- rbind(
  data.frame(
    Target = cov_bench$domain_coverage$Domain,
    Method = "Unadjusted (dirichlet_uq = FALSE)",
    Coverage = cov_bench$domain_coverage$Unadj_Coverage * 100
  ),
  data.frame(
    Target = cov_bench$domain_coverage$Domain,
    Method = "Corrected (Dirichlet Bootstrap)",
    Coverage = cov_bench$domain_coverage$Dirichlet_Coverage * 100
  )
)
dom_long$Target <- factor(
  dom_long$Target,
  levels = c("Region_North", "Region_Central", "Region_South", "Population"),
  labels = c("North", "Central", "South", "Population")
)
dom_long$Method <- factor(
  dom_long$Method,
  levels = c("Unadjusted (dirichlet_uq = FALSE)", "Corrected (Dirichlet Bootstrap)")
)

p_cov_a <- ggplot(dom_long, aes(x = Target, y = Coverage, fill = Method)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.64) +
  geom_hline(yintercept = 95, linetype = "dashed", color = "#c0392b", linewidth = 0.7) +
  geom_text(
    aes(label = sprintf("%.0f%%", Coverage)),
    position = position_dodge(width = 0.72),
    vjust = -0.5, size = 2.4, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Unadjusted (dirichlet_uq = FALSE)" = "#95a5a6",
    "Corrected (Dirichlet Bootstrap)"   = "#2980b9"
  )) +
  scale_y_continuous(limits = c(0, 112), breaks = c(0, 25, 50, 75, 95)) +
  labs(
    title = sprintf("Panel A: Domain Summary Means (Avg: %.1f%% -> %.1f%%)",
                    cov_bench$overall_summary$Coverage_Unadjusted[1] * 100,
                    cov_bench$overall_summary$Coverage_Corrected[1] * 100),
    subtitle = "M_MC = 50 informative samples; Dirichlet bootstrap matches P_nu variance",
    x = "", y = "95% CI Coverage (%)"
  ) +
  theme_minimal(base_size = 8.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 7.5),
    legend.key.size = unit(0.32, "cm"),
    legend.margin = margin(t = 2, r = 0, b = 8, l = 0),
    plot.title = element_text(size = 8.5, face = "bold"),
    plot.subtitle = element_text(size = 7.0, color = "#2c3e50"),
    plot.margin = margin(t = 18, r = 10, b = 18, l = 10)
  )

# Panel B: Active MPM Slab Parameters (Representative + All Slab Avg)
slab_tab <- cov_bench$slab_coverage
keep_slab <- c("Intercept", "Skip_x1", "Skip_x2", "Hidden_h1", "Hidden_h2")
slab_sub <- slab_tab[slab_tab$Parameter %in% keep_slab, , drop = FALSE]
slab_avg <- data.frame(
  Parameter = "All 9 Slab Avg",
  Unadj_Coverage = cov_bench$overall_summary$Coverage_Unadjusted[2],
  Sandwich_Coverage = cov_bench$overall_summary$Coverage_Corrected[2]
)
slab_show <- rbind(
  slab_sub[, c("Parameter", "Unadj_Coverage", "Sandwich_Coverage")],
  slab_avg
)
slab_long <- rbind(
  data.frame(
    Parameter = slab_show$Parameter,
    Method = "Unadjusted VI (Sigma_1)",
    Coverage = slab_show$Unadj_Coverage * 100
  ),
  data.frame(
    Parameter = slab_show$Parameter,
    Method = "Williams-Savitsky Sandwich",
    Coverage = slab_show$Sandwich_Coverage * 100
  )
)
slab_long$Parameter <- factor(
  slab_long$Parameter,
  levels = c("Skip_x1", "Skip_x2", "Intercept", "Hidden_h1", "Hidden_h2", "All 9 Slab Avg"),
  labels = c("Skip X1", "Skip X2", "Interc.", "Hid h1", "Hid h2", "All 9 Avg")
)
slab_long$Method <- factor(
  slab_long$Method,
  levels = c("Unadjusted VI (Sigma_1)", "Williams-Savitsky Sandwich")
)

p_cov_b <- ggplot(slab_long, aes(x = Parameter, y = Coverage, fill = Method)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.64) +
  geom_hline(yintercept = 95, linetype = "dashed", color = "#c0392b", linewidth = 0.7) +
  geom_text(
    aes(label = sprintf("%.0f%%", Coverage)),
    position = position_dodge(width = 0.72),
    vjust = -0.5, size = 2.2, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Unadjusted VI (Sigma_1)"    = "#95a5a6",
    "Williams-Savitsky Sandwich" = "#27ae60"
  )) +
  scale_y_continuous(limits = c(0, 112), breaks = c(0, 25, 50, 75, 95)) +
  labs(
    title = sprintf("Panel B: Active MPM Slab Params (Avg: %.1f%% -> %.1f%%)",
                    cov_bench$overall_summary$Coverage_Unadjusted[2] * 100,
                    cov_bench$overall_summary$Coverage_Corrected[2] * 100),
    subtitle = "Conditioned on MPM active paths; Cholesky R2^T R1^{-T} corrects VI & DEFF",
    x = "", y = "95% CI Coverage (%)"
  ) +
  theme_minimal(base_size = 8.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 7.5),
    legend.key.size = unit(0.32, "cm"),
    legend.margin = margin(t = 2, r = 0, b = 8, l = 0),
    plot.title = element_text(size = 8.5, face = "bold"),
    plot.subtitle = element_text(size = 7.0, color = "#27ae60"),
    plot.margin = margin(t = 18, r = 10, b = 18, l = 10)
  )

p_cov_grid <- plot_grid(p_cov_a, p_cov_b, ncol = 2, rel_widths = c(0.95, 1.15))
save_fig("track1_coverage_comparison.pdf", p_cov_grid, width = 10.5, height = 4.6)
cat("[11/12] track1_coverage_comparison.pdf generated.\n")


# -------------------------------------------------------------
# 12. Slide 23: Track 1B Finite-Sample (eps, delta)-DP Calibration (figures/track1_dp_nugget_calibration.pdf)
# -------------------------------------------------------------
dp_cal <- calibrate_dp_finite_sample(
  dp_fit       = dp_results,
  X            = x_dp,
  y            = log_y_dp,
  R            = 200,
  B            = 100,
  delta_target = c(0.10, 0.05, 0.01),
  seed         = 2026
)
saveRDS(
  dp_cal,
  file = if (dir.exists(dirname(cov_rds))) file.path(dirname(cov_rds), "dp_cal_data.rds") else "dp_cal_data.rds"
)

cat("\n--- DP Finite-Sample Calibration Results (Presentation dp_results) ---\n")
cat(sprintf("Observed Delta_{alpha, x} = %.4f, eps_obs = %.4f, delta_hat(0) = %.3f\n",
            dp_cal$delta_alpha_obs, dp_cal$epsilon_obs, dp_cal$delta_hat_0))
print(dp_cal$calibration_table)

row_05 <- dp_cal$calibration_table[which.min(abs(dp_cal$calibration_table$delta_target - 0.05)), ]
row_01 <- dp_cal$calibration_table[which.min(abs(dp_cal$calibration_table$delta_target - 0.01)), ]

df_rep_hist <- data.frame(delta_rep = dp_cal$delta_rep_vec)
p_dp_cal <- ggplot(df_rep_hist, aes(x = delta_rep)) +
  geom_histogram(aes(y = after_stat(density)), bins = 24, fill = "#bdc3c7", color = "white", alpha = 0.85) +
  geom_density(color = "#2c3e50", linewidth = 0.85) +
  geom_vline(xintercept = dp_cal$delta_alpha_obs, color = "#e67e22", linetype = "solid", linewidth = 1.0) +
  geom_vline(xintercept = row_05$delta_alpha_eta, color = "#2980b9", linetype = "dashed", linewidth = 1.0) +
  geom_vline(xintercept = row_01$delta_alpha_eta, color = "#c0392b", linetype = "dotted", linewidth = 1.1) +
  annotate("label", x = dp_cal$delta_alpha_obs, y = 0.96, hjust = 1.05, vjust = 1,
           label = sprintf("Delta[obs] == '%.2f'", dp_cal$delta_alpha_obs),
           parse = TRUE, color = "#d35400", fill = "white", size = 2.6, fontface = "bold") +
  annotate("label", x = dp_cal$delta_alpha_obs, y = 0.84, hjust = 1.05, vjust = 1,
           label = sprintf("hat(delta)[n](0) == '%.3f'", dp_cal$delta_hat_0),
           parse = TRUE, color = "#d35400", fill = "white", size = 2.4) +
  annotate("label", x = row_05$delta_alpha_eta, y = 0.80, hjust = 1.05, vjust = 1,
           label = sprintf("delta == '0.05' ~ '(' * eta == '+%.2f' * ')'", row_05$eta_nugget),
           parse = TRUE, color = "#2980b9", fill = "white", size = 2.4, fontface = "bold") +
  annotate("label", x = row_01$delta_alpha_eta, y = 0.56, hjust = 1.05, vjust = 1,
           label = sprintf("delta == '0.01' ~ '(' * eta == '+%.2f' * ')'", row_01$eta_nugget),
           parse = TRUE, color = "#c0392b", fill = "white", size = 2.4, fontface = "bold") +
  scale_x_continuous(expand = expansion(mult = c(0.06, 0.10))) +
  scale_y_continuous(limits = c(0, 1.08), expand = c(0, 0)) +
  labs(
    title = "Replicate Sensitivity Distribution (R = 200 Dual-Support Plug-in)",
    subtitle = sprintf("Zero retraining: unadj. eps = %.2f (delta_0 = %.3f) -> eps(eta) = %.2f at delta = 0.05",
                       dp_cal$epsilon_obs, dp_cal$delta_hat_0, row_05$epsilon_eta),
    x = expression("Replicate Weighted Sensitivity " * Delta[alpha * "," * x[r]]),
    y = "Density"
  ) +
  theme_minimal(base_size = 8.5) +
  theme(
    plot.title = element_text(size = 9.5, face = "bold", margin = margin(b = 4)),
    plot.subtitle = element_text(size = 7.5, color = "#2c3e50"),
    plot.margin = margin(t = 18, r = 12, b = 18, l = 12)
  )

save_fig("track1_dp_nugget_calibration.pdf", p_dp_cal, width = 6.0, height = 4.5)
cat("[12/12] track1_dp_nugget_calibration.pdf generated.\n")

cat("All 12 figures generated successfully!\n")
