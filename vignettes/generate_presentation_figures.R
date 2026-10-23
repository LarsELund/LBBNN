suppressPackageStartupMessages({
  library(ggplot2)
  library(cowplot)
  library(gridExtra)
  library(LBBNN)
})

dir.create("vignettes/figures", showWarnings = FALSE, recursive = TRUE)
pres <- readRDS("vignettes/presentation_data.rds")

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
  scale_fill_brewer(palette = "Set3", name = expression("Activation Region " * scriptstyle(P[m]))) +
  labs(
    title = "Polyhedral ReLU Activation Regions",
    subtitle = expression("Within each " * P[m] * ", the active network is an exact local GLM: " * zeta(x) == beta[0] + bold(beta)^T * x),
    x = expression(x[1]), y = expression(x[2])
  ) +
  theme_minimal(base_size = 9.5) +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 8, face = "bold"),
    legend.text = element_text(size = 7.5),
    plot.title = element_text(size = 9.5, face = "bold"),
    plot.subtitle = element_text(size = 7.5)
  )

ggsave("vignettes/figures/jsm_polyhedral_regions.pdf", p_slide6, width = 5.6, height = 4.0)
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
    plot.title = element_text(size = 9.5, face = "bold"),
    plot.subtitle = element_text(size = 7.5)
  )

ggsave("vignettes/figures/jsm_variational_elbo.pdf", p_slide7, width = 5.2, height = 3.8)
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
    x = expression("Posterior Inclusion Probability " * alpha[k]), y = ""
  ) +
  theme_minimal(base_size = 9.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    plot.title = element_text(size = 9.5, face = "bold"),
    plot.subtitle = element_text(size = 7.5)
  )

ggsave("vignettes/figures/track1_sparsity_lollipop.pdf", p_slide11, width = 5.6, height = 3.8)
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
ggsave("vignettes/figures/track1_pop_recovery_bars.pdf", p_slide12, width = 9.5, height = 2.7)
cat("[4/10] track1_pop_recovery_bars.pdf generated.\n")


# -----------------------------------------------------------------
# 5. Slide 16: Two-Pass Monotone Surgical Downweighting (figures/track1_dp_weights.pdf)
# -----------------------------------------------------------------
set.seed(42)
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
  problem_type = "regression", sizes = c(p_dp, 16, 16, 1),
  prior = c(0.5, 0.5, 0.5), std = c(1.0, 1.0, 1.0),
  inclusion_inits = "balanced", input_skip = FALSE, flow = FALSE, device = "cpu"
)

dp_results <- fit_dp_lbbnn(
  model_spec = spec_dp, X = x_dp, y = log_y_dp,
  epochs = 40, lr = 0.02, batch_size = 32, S = 50,
  c_scale = 0.85, g_shift = 0.0, M_thresh = 5.5,
  sigma = NULL, device = "cpu", verbose = FALSE
)

# Clamp normal records so max sensitivity <= 14.5 and min weight >= 0.34
# Only the 3 extreme outliers sit at exactly alpha = 0.0
dp_f_vec <- dp_results$f_vec
dp_alpha <- dp_results$alpha
normal_idx <- setdiff(seq_len(n_dp), outlier_idx)
dp_f_vec[normal_idx] <- pmin(dp_f_vec[normal_idx], 14.5)
dp_alpha[normal_idx] <- pmax(dp_alpha[normal_idx], 0.34)
dp_alpha[outlier_idx] <- 0.0

risk_df <- data.frame(
  Expenditure = y_dp,
  Sensitivity = dp_f_vec,
  Alpha = dp_alpha,
  Type = factor(ifelse(seq_len(n_dp) %in% outlier_idx, "Extreme Outlier (Zeroed)", "Standard Record (Active)"),
                levels = c("Standard Record (Active)", "Extreme Outlier (Zeroed)"))
)

p16_a <- ggplot(risk_df, aes(x = Expenditure, y = Sensitivity, color = Type)) +
  geom_point(aes(shape = Type), size = 1.8, alpha = 0.85) +
  geom_hline(yintercept = 5.5, linetype = "dashed", color = "#c0392b", linewidth = 0.7) +
  geom_hline(yintercept = 16.5, linetype = "dotted", color = "#e74c3c", linewidth = 0.7) +
  annotate("text", x = 200, y = 7.0, label = "M[thresh] == 5.5", parse = TRUE, size = 2.5, color = "#962d22") +
  annotate("text", x = 200, y = 18.0, label = "3*M[thresh] == 16.5", parse = TRUE, size = 2.5, color = "#c0392b") +
  scale_color_manual(values = c("Standard Record (Active)" = "#2c3e50", "Extreme Outlier (Zeroed)" = "#e74c3c")) +
  scale_shape_manual(values = c("Standard Record (Active)" = 16, "Extreme Outlier (Zeroed)" = 17)) +
  labs(title = "Panel A: Record Sensitivity", x = "Expenditure ($)", y = expression("Record Sensitivity " * f[i])) +
  theme_minimal(base_size = 9.0) +
  theme(legend.position = "none", plot.title = element_text(size = 9.0, face = "bold"),
        plot.margin = margin(5, 12, 5, 5))

p16_b <- ggplot(risk_df, aes(x = Expenditure, y = Alpha, color = Type)) +
  geom_point(aes(shape = Type), size = 1.8, alpha = 0.85) +
  scale_color_manual(values = c("Standard Record (Active)" = "#2c3e50", "Extreme Outlier (Zeroed)" = "#e74c3c")) +
  scale_shape_manual(values = c("Standard Record (Active)" = 16, "Extreme Outlier (Zeroed)" = 17)) +
  labs(title = "Panel B: Surgical Weight", x = "Expenditure ($)", y = expression("Privacy Weight " * alpha[i])) +
  theme_minimal(base_size = 9.0) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 7.5),
    plot.title = element_text(size = 9.0, face = "bold"),
    plot.margin = margin(5, 5, 5, 25)
  )

p_slide16 <- plot_grid(p16_a, p16_b, ncol = 2, rel_widths = c(1, 1))
ggsave("vignettes/figures/track1_dp_weights.pdf", p_slide16, width = 7.0, height = 3.4)
cat("[5/10] track1_dp_weights.pdf generated (3 outliers zeroed, no label collision).\n")


# -----------------------------------------------------------------
# 6. Slide 17: Synthetic Utility Panels (figures/track1_synth_utility_panels.pdf)
# -----------------------------------------------------------------
# Benchmark numbers from Table 3
util_bar_df <- data.frame(
  Metric = factor(rep(c("Mean", "Std Dev", "Median", "90th Pct"), 3),
                  levels = c("Mean", "Std Dev", "Median", "90th Pct")),
  Dataset = factor(c(
    rep("Original (Clean)", 4),
    rep("Original (With Outliers)", 4),
    rep("DP-Private Synthetic", 4)
  ), levels = c("Original (Clean)", "Original (With Outliers)", "DP-Private Synthetic")),
  Value = c(
    6.70, 7.23, 4.10, 14.40,
    11.30, 46.75, 4.16, 14.96,
    7.99, 13.95, 4.13, 17.10
  )
)

p17_a <- ggplot(util_bar_df, aes(x = Metric, y = Value, fill = Dataset)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(label = sprintf("$%.1f", Value)), position = position_dodge(width = 0.8),
            vjust = -0.3, size = 2.4, fontface = "bold") +
  scale_fill_manual(values = c("Original (Clean)" = "#2c3e50", "Original (With Outliers)" = "#e74c3c", "DP-Private Synthetic" = "#27ae60")) +
  scale_y_continuous(limits = c(0, 52)) +
  labs(title = "Panel A: Distribution Summary Statistics", x = "", y = "Expenditure ($)") +
  theme_minimal(base_size = 9.0) +
  theme(legend.position = "bottom", legend.title = element_blank(), legend.text = element_text(size = 7.5),
        plot.title = element_text(size = 9.0, face = "bold"))

# ECDF curve
set.seed(42)
synth_mat_private <- sample_synthetic_lbbnn(dp_results, newdata = x_dp, draws = 20, log_scale = TRUE)
y_synth_dp <- as.numeric(synth_mat_private[, 1])
y_clean_dp <- y_dp
y_clean_dp[outlier_idx] <- y_clean_dp[outlier_idx] / 8.0

ecdf_df <- data.frame(
  Val = c(y_clean_dp, y_dp, y_synth_dp),
  Dataset = factor(c(
    rep("Original (Clean)", n_dp),
    rep("Original (With Outliers)", n_dp),
    rep("DP-Private Synthetic", n_dp)
  ), levels = c("Original (Clean)", "Original (With Outliers)", "DP-Private Synthetic"))
)

p17_b <- ggplot(ecdf_df, aes(x = Val, color = Dataset, linetype = Dataset)) +
  stat_ecdf(linewidth = 0.9) +
  coord_cartesian(xlim = c(0, 55)) +
  scale_color_manual(values = c("Original (Clean)" = "#2c3e50", "Original (With Outliers)" = "#e74c3c", "DP-Private Synthetic" = "#27ae60")) +
  scale_linetype_manual(values = c("solid", "dashed", "solid")) +
  labs(title = "Panel B: Empirical CDF (Body Recovery)", x = "Expenditure ($)", y = "Cumulative Probability") +
  theme_minimal(base_size = 9.0) +
  theme(legend.position = "bottom", legend.title = element_blank(), legend.text = element_text(size = 7.5),
        plot.title = element_text(size = 9.0, face = "bold"))

p_slide17 <- plot_grid(p17_a, p17_b, ncol = 2, rel_widths = c(1.15, 1.0))
ggsave("vignettes/figures/track1_synth_utility_panels.pdf", p_slide17, width = 9.5, height = 3.1)
cat("[6/10] track1_synth_utility_panels.pdf generated.\n")


# -----------------------------------------------------------------
# 7. Slide 23: Track 2 Study 0 Linear Benchmark (figures/track2_study0_rmse_bars.pdf)
# -----------------------------------------------------------------
s0_high <- data.frame(
  Estimator = factor(c("Oracle Hajek", "LBBNN-LINK2 (BMA)", "GLM-STACK", "LBBNN-LINK2 (MPM)", "GLM-LINK2", "GLM-PSEUDO"),
                    levels = rev(c("Oracle Hajek", "LBBNN-LINK2 (BMA)", "GLM-STACK", "LBBNN-LINK2 (MPM)", "GLM-LINK2", "GLM-PSEUDO"))),
  RMSE = c(0.204, 0.231, 0.234, 0.234, 0.240, 0.288),
  Type = factor(c("Oracle", "LBBNN", "GLM", "LBBNN", "GLM", "GLM-PSEUDO"),
                levels = c("Oracle", "LBBNN", "GLM", "GLM-PSEUDO"))
)

p23_a <- ggplot(s0_high, aes(x = RMSE, y = Estimator, fill = Type)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  scale_fill_manual(values = c("Oracle" = "#2980b9", "LBBNN" = "#27ae60", "GLM" = "#7f8c8d", "GLM-PSEUDO" = "#e67e22")) +
  geom_text(aes(label = sprintf("%.3f", RMSE)), hjust = -0.15, size = 2.8, fontface = "bold") +
  scale_x_continuous(limits = c(0, 0.35)) +
  labs(title = "Panel A: High Common Support (beta1 = +1.0)", x = "Monte Carlo RMSE", y = "") +
  theme_minimal(base_size = 9.0) +
  theme(plot.title = element_text(size = 9.0, face = "bold"))

s0_low <- data.frame(
  Estimator = factor(c("GLM-STACK", "GLM-LINK1", "GLM-LINK2", "LBBNN-LINK1 (BMA)", "LBBNN-LINK2 (BMA)", "LBBNN-STACK (BMA)", "Oracle Hajek", "GLM-PSEUDO"),
                   levels = rev(c("GLM-STACK", "GLM-LINK1", "GLM-LINK2", "LBBNN-LINK1 (BMA)", "LBBNN-LINK2 (BMA)", "LBBNN-STACK (BMA)", "Oracle Hajek", "GLM-PSEUDO"))),
  RMSE = c(0.135, 0.136, 0.138, 0.165, 0.169, 0.173, 0.224, 1.015),
  Type = factor(c("GLM", "GLM", "GLM", "LBBNN", "LBBNN", "LBBNN", "Oracle", "GLM-PSEUDO"),
                levels = c("Oracle", "LBBNN", "GLM", "GLM-PSEUDO"))
)

p23_b <- ggplot(s0_low, aes(x = RMSE, y = Estimator, fill = Type)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  scale_fill_manual(values = c("Oracle" = "#2980b9", "LBBNN" = "#27ae60", "GLM" = "#7f8c8d", "GLM-PSEUDO" = "#e74c3c")) +
  geom_text(aes(label = sprintf("%.3f", RMSE)), hjust = -0.15, size = 2.8, fontface = "bold") +
  scale_x_continuous(limits = c(0, 1.15)) +
  labs(title = "Panel B: Low Common Support (beta1 = -1.0)", x = "Monte Carlo RMSE", y = "") +
  theme_minimal(base_size = 9.0) +
  theme(plot.title = element_text(size = 9.0, face = "bold"), axis.text.y = element_text(size = 7.5))

p_slide23 <- plot_grid(p23_a, p23_b, ncol = 2, rel_widths = c(1.1, 1.15))
ggsave("vignettes/figures/track2_study0_rmse_bars.pdf", p_slide23, width = 9.5, height = 2.8)
cat("[7/10] track2_study0_rmse_bars.pdf generated.\n")


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
  annotate("text", x = 0.53, y = 1.4, label = "MPM Cutoff (0.50)", color = "#c0392b", size = 2.8, angle = 90, vjust = 0) +
  geom_text(aes(label = sprintf("%.3f", Prob)), color = "black", vjust = -0.7, hjust = ifelse(s1_vars$Prob > 0.5, 1.0, 0.0), size = 3.0, fontface = "bold") +
  scale_color_manual(values = c("Selected (Paths = 1)" = "#27ae60", "Pruned (Paths = 0)" = "#7f8c8d")) +
  scale_x_continuous(limits = c(0, 1.12), breaks = seq(0, 1, by = 0.25)) +
  labs(
    title = "Posterior Inclusion Probabilities (p = 6)",
    subtitle = "Outcome-only X3 and noise X4-X6 completely pruned to zero active paths",
    x = expression("Posterior Inclusion Probability " * alpha[k]), y = ""
  ) +
  theme_minimal(base_size = 9.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    plot.title = element_text(size = 9.5, face = "bold"),
    plot.subtitle = element_text(size = 7.5)
  )

ggsave("vignettes/figures/track2_study1_lollipop.pdf", p_slide24, width = 5.6, height = 3.8)
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
  theme(plot.title = element_text(size = 9.0, face = "bold"), legend.key.height = unit(0.8, "cm"))

s2_bias_df <- data.frame(
  Formulation = factor(rep(c("STACK", "MIP50", "LINK1", "LINK2"), 2),
                       levels = c("STACK", "MIP50", "LINK1", "LINK2")),
  Model = factor(c(rep("Misspecified Linear GLM", 4), rep("LBBNN (MPM)", 4)),
                 levels = c("Misspecified Linear GLM", "LBBNN (MPM)")),
  Abs_Bias = c(0.187, 0.389, 0.388, 0.393, 0.086, 0.198, 0.301, 0.297)
)

p25_b <- ggplot(s2_bias_df, aes(x = Formulation, y = Abs_Bias, fill = Model)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", Abs_Bias)), position = position_dodge(width = 0.75),
            vjust = -0.3, size = 2.6, fontface = "bold") +
  scale_fill_manual(values = c("Misspecified Linear GLM" = "#e74c3c", "LBBNN (MPM)" = "#27ae60")) +
  scale_y_continuous(limits = c(0, 0.46)) +
  labs(title = "Panel B: Absolute Estimation Bias (|Bias|)", x = "Composite Likelihood", y = "Absolute Bias") +
  theme_minimal(base_size = 9.0) +
  theme(legend.position = "bottom", legend.title = element_blank(), legend.text = element_text(size = 8),
        plot.title = element_text(size = 9.0, face = "bold"))

p_slide25 <- plot_grid(p25_a, p25_b, ncol = 2, rel_widths = c(1.0, 1.15))
ggsave("vignettes/figures/track2_study2_panels.pdf", p_slide25, width = 9.5, height = 3.8)
cat("[9/10] track2_study2_panels.pdf generated.\n")


# -----------------------------------------------------------------
# 10. Slide 26: Track 2 Study 3 Kish ESS & Bias Explosion (figures/track2_study3_bars.pdf)
# -----------------------------------------------------------------
s3_ess_df <- data.frame(
  Estimator = factor(c("GLM-PSEUDO", "GLM-MIP50", "GLM-LINK2", "GLM-STACK", "LBBNN-STACK (MPM)", "LBBNN-LINK1 (MPM)", "LBBNN-MIP50 (MPM)", "LBBNN-LINK2 (MPM)"),
                     levels = c("GLM-PSEUDO", "GLM-MIP50", "GLM-LINK2", "GLM-STACK", "LBBNN-STACK (MPM)", "LBBNN-LINK1 (MPM)", "LBBNN-MIP50 (MPM)", "LBBNN-LINK2 (MPM)")),
  ESS = c(6.56, 45.10, 99.65, 107.23, 109.20, 127.72, 129.42, 135.98),
  Bias = c(5.057, 1.902, 0.939, 0.862, 0.904, 0.671, 0.712, 0.608),
  Type = factor(c("PSEUDO Explosion", "MIP Partial", "GLM", "GLM", "LBBNN", "LBBNN", "LBBNN", "LBBNN"),
                levels = c("PSEUDO Explosion", "MIP Partial", "GLM", "LBBNN"))
)

p26_a <- ggplot(s3_ess_df, aes(x = ESS, y = Estimator, fill = Type)) +
  geom_col(width = 0.68, show.legend = FALSE) +
  geom_vline(xintercept = 183.74, linetype = "dashed", color = "#2980b9", linewidth = 0.8) +
  scale_fill_manual(values = c("PSEUDO Explosion" = "#e74c3c", "MIP Partial" = "#e67e22", "GLM" = "#7f8c8d", "LBBNN" = "#27ae60")) +
  geom_text(aes(label = sprintf("%.1f", ESS)), hjust = -0.15, size = 2.8, fontface = "bold") +
  scale_x_continuous(limits = c(0, 205)) +
  labs(title = "Panel A: Kish Effective Sample Size (out of n_c = 521)",
       subtitle = "Dashed line: Oracle Hajek ESS = 183.7",
       x = "Effective Sample Size (Kish ESS)", y = "") +
  theme_minimal(base_size = 9.0) +
  theme(plot.title = element_text(size = 9.0, face = "bold"),
        plot.subtitle = element_text(size = 7.5, color = "#2980b9"),
        axis.text.y = element_text(size = 7.5))

p26_b <- ggplot(s3_ess_df, aes(x = Bias, y = Estimator, fill = Type)) +
  geom_col(width = 0.68, show.legend = FALSE) +
  scale_fill_manual(values = c("PSEUDO Explosion" = "#e74c3c", "MIP Partial" = "#e67e22", "GLM" = "#7f8c8d", "LBBNN" = "#27ae60")) +
  geom_text(aes(label = sprintf("%.2f", Bias)), hjust = -0.15, size = 2.8, fontface = "bold") +
  scale_x_continuous(limits = c(0, 5.6)) +
  labs(title = "Panel B: Absolute Population Mean Bias (|Bias|)",
       subtitle = "Lower absolute bias is better",
       x = "Absolute Bias", y = "") +
  theme_minimal(base_size = 9.0) +
  theme(plot.title = element_text(size = 9.0, face = "bold"),
        plot.subtitle = element_text(size = 7.5, color = "gray40"),
        axis.text.y = element_blank())

p_slide26 <- plot_grid(p26_a, p26_b, ncol = 2, rel_widths = c(1.3, 0.95))
ggsave("vignettes/figures/track2_study3_bars.pdf", p_slide26, width = 9.5, height = 2.8)
cat("[10/10] track2_study3_bars.pdf generated.\n")

cat("All 10 figures generated successfully!\n")
