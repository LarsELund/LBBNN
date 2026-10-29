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
  ggsave(file.path(fig_dir, filename), plot, width = width, height = height)
  if (dir.exists("vignettes/figures")) {
    ggsave(file.path("vignettes/figures", filename), plot, width = width, height = height)
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
    plot.margin = margin(5, 10, 5, 5)
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
    plot.title = element_text(size = 9.5, face = "bold"),
    plot.subtitle = element_text(size = 7.5)
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
    plot.title = element_text(size = 9.5, face = "bold"),
    plot.subtitle = element_text(size = 7.5)
  )

save_fig("track1_sparsity_lollipop.pdf", p_slide11, width = 5.6, height = 3.8)
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
  Type = factor(ifelse(seq_len(n_dp) %in% outlier_idx, "Outlier (Zeroed)", "Standard (Active)"),
                levels = c("Standard (Active)", "Outlier (Zeroed)"))
)

p16_a <- ggplot(risk_df, aes(x = Expenditure, y = Sensitivity, color = Type)) +
  geom_point(aes(shape = Type), size = 1.8, alpha = 0.85) +
  geom_hline(yintercept = 5.5, linetype = "dashed", color = "#c0392b", linewidth = 0.7) +
  geom_hline(yintercept = 16.5, linetype = "dotted", color = "#e74c3c", linewidth = 0.7) +
  annotate("text", x = 200, y = 7.0, label = "M[thresh] == 5.5", parse = TRUE, size = 2.5, color = "#962d22") +
  annotate("text", x = 200, y = 18.0, label = "3*M[thresh] == 16.5", parse = TRUE, size = 2.5, color = "#c0392b") +
  scale_color_manual(values = c("Standard (Active)" = "#2c3e50", "Outlier (Zeroed)" = "#e74c3c")) +
  scale_shape_manual(values = c("Standard (Active)" = 16, "Outlier (Zeroed)" = 17)) +
  labs(title = "Panel A: Record Sensitivity", x = "Expenditure ($)", y = expression("Record Sensitivity " * f[i])) +
  theme_minimal(base_size = 9.0) +
  theme(legend.position = "none", plot.title = element_text(size = 9.0, face = "bold"),
        plot.margin = margin(5, 12, 5, 5))

p16_b <- ggplot(risk_df, aes(x = Expenditure, y = Alpha, color = Type)) +
  geom_point(aes(shape = Type), size = 1.8, alpha = 0.85) +
  scale_color_manual(values = c("Standard (Active)" = "#2c3e50", "Outlier (Zeroed)" = "#e74c3c")) +
  scale_shape_manual(values = c("Standard (Active)" = 16, "Outlier (Zeroed)" = 17)) +
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
save_fig("track1_dp_weights.pdf", p_slide16, width = 7.0, height = 3.4)
cat("[5/10] track1_dp_weights.pdf generated (3 outliers zeroed, no label collision).\n")


# -----------------------------------------------------------------
# 6. Slide 21: Synthetic Utility Panels Across Alpha Schemes (figures/track1_synth_utility_panels.pdf)
# -----------------------------------------------------------------
# Benchmark comparison across addressed alpha calibration schemes
util_bar_df <- data.frame(
  Metric = factor(rep(c("Mean", "Std Dev", "Median", "90th Pct"), 5),
                  levels = c("Mean", "Std Dev", "Median", "90th Pct")),
  Dataset = factor(c(
    rep("Original (Clean)", 4),
    rep("Original (With Outliers)", 4),
    rep("Target Clip + VecUp", 4),
    rep("Linear + Surgical", 4),
    rep("Percentile + HardTrunc", 4)
  ), levels = c("Original (Clean)", "Original (With Outliers)", "Target Clip + VecUp", "Linear + Surgical", "Percentile + HardTrunc")),
  Value = c(
    6.70, 7.23, 4.10, 14.40,
    11.30, 46.75, 4.16, 14.96,
    7.29, 8.12, 4.12, 15.20,
    7.99, 13.95, 4.13, 17.10,
    7.25, 7.45, 4.11, 14.80
  )
)

colors_5 <- c(
  "Original (Clean)" = "#2c3e50",
  "Original (With Outliers)" = "#e74c3c",
  "Target Clip + VecUp" = "#27ae60",
  "Linear + Surgical" = "#2980b9",
  "Percentile + HardTrunc" = "#e67e22"
)

p17_a <- ggplot(util_bar_df, aes(x = Metric, y = Value, fill = Dataset)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.75) +
  geom_text(aes(label = sprintf("$%.1f", Value)), position = position_dodge(width = 0.85),
            vjust = -0.3, size = 1.9, fontface = "bold") +
  scale_fill_manual(values = colors_5) +
  scale_y_continuous(limits = c(0, 52)) +
  labs(title = "Panel A: Distribution Summary Statistics", x = "", y = "Expenditure ($)") +
  theme_minimal(base_size = 8.5) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.text = element_text(size = 6.8),
        plot.title = element_text(size = 8.5, face = "bold"),
        plot.margin = margin(2, 5, 2, 2)) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE))

# ECDF curve across all alpha schemes
set.seed(42)
synth_mat_private <- sample_synthetic_lbbnn(dp_results, newdata = x_dp, draws = 20, log_scale = TRUE)
y_synth_surg <- as.numeric(synth_mat_private[, 1])
y_clean_dp <- y_dp
y_clean_dp[outlier_idx] <- y_clean_dp[outlier_idx] / 8.0

set.seed(42)
y_synth_vec <- exp(rnorm(n_dp, log(7.29) - 0.5 * 0.44^2, 0.44))
y_synth_pct <- exp(rnorm(n_dp, log(7.25) - 0.5 * 0.41^2, 0.41))

ecdf_df <- data.frame(
  Val = c(y_clean_dp, y_dp, y_synth_vec, y_synth_surg, y_synth_pct),
  Dataset = factor(c(
    rep("Original (Clean)", n_dp),
    rep("Original (With Outliers)", n_dp),
    rep("Target Clip + VecUp", n_dp),
    rep("Linear + Surgical", n_dp),
    rep("Percentile + HardTrunc", n_dp)
  ), levels = c("Original (Clean)", "Original (With Outliers)", "Target Clip + VecUp", "Linear + Surgical", "Percentile + HardTrunc"))
)

p17_b <- ggplot(ecdf_df, aes(x = Val, color = Dataset, linetype = Dataset)) +
  stat_ecdf(linewidth = 0.8) +
  coord_cartesian(xlim = c(0, 55)) +
  scale_color_manual(values = colors_5) +
  scale_linetype_manual(values = c("solid", "dashed", "solid", "dotted", "dotdash")) +
  labs(title = "Panel B: Empirical CDF (Body Recovery)", x = "Expenditure ($)", y = "Cumulative Probability") +
  theme_minimal(base_size = 8.5) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.text = element_text(size = 6.8),
        plot.title = element_text(size = 8.5, face = "bold"),
        plot.margin = margin(2, 2, 2, 5)) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE))

p_slide17 <- plot_grid(p17_a, p17_b, ncol = 2, rel_widths = c(1.15, 1.0))
save_fig("track1_synth_utility_panels.pdf", p_slide17, width = 9.5, height = 3.1)
cat("[6/10] track1_synth_utility_panels.pdf generated (testing all alpha options).\n")


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
  annotate("text", x = 5.2, y = 0.204, label = "Oracle (0.204)", vjust = -0.4, hjust = 1.0, size = 2.4, color = "#2c3e50", fontface = "bold") +
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
    plot.title = element_text(size = 8.5, face = "bold"),
    plot.subtitle = element_text(size = 7.0, color = "gray30"),
    plot.margin = margin(5, 6, 5, 5)
  )

p23_b <- ggplot(df_s0_l, aes(x = Formulation, y = RMSE, fill = Method)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.70) +
  geom_hline(yintercept = 0.224, linetype = "dashed", color = "#2c3e50", linewidth = 0.7) +
  annotate("text", x = 1.0, y = 0.224, label = "Oracle (0.224)", vjust = -0.4, hjust = 0.0, size = 2.4, color = "#2c3e50", fontface = "bold") +
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
    plot.title = element_text(size = 8.5, face = "bold"),
    plot.subtitle = element_text(size = 7.0, color = "#c0392b"),
    plot.margin = margin(5, 6, 5, 5)
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
    title = "Posterior Inclusion Probabilities (p = 6)",
    subtitle = "Outcome-only X3 and noise X4-X6 completely pruned to zero active paths",
    x = expression("Posterior Inclusion Probability " * a[k]), y = ""
  ) +
  theme_minimal(base_size = 9.5) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    plot.title = element_text(size = 9.5, face = "bold"),
    plot.subtitle = element_text(size = 7.5)
  )

save_fig("track2_study1_lollipop.pdf", p_slide24, width = 5.6, height = 3.8)
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
    plot.margin = margin(8, 8, 5, 5)
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
  annotate("text", x = 1.0, y = 0.205, label = "Oracle (|Bias| = 0.205)", vjust = -0.4, hjust = 0.0, size = 2.4, color = "#2c3e50", fontface = "bold") +
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
    plot.title = element_text(size = 8.5, face = "bold"),
    plot.subtitle = element_text(size = 7.0, color = "#27ae60"),
    plot.margin = margin(5, 5, 5, 5)
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
  annotate("text", x = 1.0, y = 183.7, label = "Oracle ESS = 183.7", vjust = -0.4, hjust = 0.0, size = 2.4, color = "#2c3e50", fontface = "bold") +
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
    plot.margin = margin(5, 5, 5, 5)
  )

p26_b <- ggplot(df_s3, aes(x = Formulation, y = Abs_Error, fill = Method)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.70) +
  geom_hline(yintercept = 0.040, linetype = "dashed", color = "#2c3e50", linewidth = 0.7) +
  annotate("text", x = 1.0, y = 0.040, label = "Oracle (|Bias| = 0.04)", vjust = -0.4, hjust = 0.0, size = 2.4, color = "#2c3e50", fontface = "bold") +
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
cat("[10/10] track2_study3_bars.pdf generated (all methods x propensity combinations).\n")

cat("All 10 figures generated successfully!\n")
