# LBBNN: Sparse & Explainable Bayesian Neural Networks for Official Statistics

[![R-CMD-check](https://img.shields.io/badge/R--CMD--check-passing-brightgreen.svg)](https://cran.r-project.org/package=LBBNN)
[![CRAN Status](https://www.r-pkg.org/badges/version/LBBNN)](https://cran.r-project.org/package=LBBNN)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**LBBNN** provides sparse, locally explainable Latent Binary Bayesian Neural Networks powered by a pure C++ `LibTorch` backend (`torch` for R) with zero Python dependencies.

This branch hosts the joint research and software collaboration between **NMBU / UiO Integreat** (Aliaksandr Hubin, Lars Skaaret-Lund) and the **U.S. Bureau of Labor Statistics (BLS)** (Terrance D. Savitsky, Matthew R. Williams, Vladislav Beresovsky, Julie Gershunskaya), organized around two core official statistics tracks:

1. **Project 1 (Track 1A & 1B): Survey-Weighted Variational Pseudo-Posteriors, Small-Domain Estimation, and Formal $(\epsilon, \delta)$-Differential Privacy Microdata Synthesis.**
2. **Project 2 (Track 2): Combining Non-Probability and Probability Samples via Deep Latent Binary Composite Likelihoods (`STACK`, `LINK1`, `LINK2`, `MIP`, `PSEUDO`).**

---

## Why LBBNN for Official Statistics?

* **Automated Confounder Selection & Noise Pruning**: Spike-and-slab variational edge selection ($\gamma_{k,p}^{(j)} \in \{0, 1\}$) automatically prunes pure noise and outcome-only covariates to zero active paths ($\text{Paths} = 0$), preventing weight variance inflation in Horvitz–Thompson and Hájek estimators.
* **Adaptive Depth via Input-Skip Architecture (ISLaB)**: Direct input skips allow the network to seamlessly reduce to a parsimonious linear GLM ($\text{Depth} = 1$) or expand into deep multi-way non-linear interactions ($\text{Depth} \ge 2$) only where warranted by the data.
* **Exact Local Polyhedral Explainability**: Piecewise-linear ReLU activations partition covariate space into polyhedral regions within which the network is an exact local GLM with closed-form respondent-level sensitivities and credible intervals.
* **True-Temperature ($T = 1.0$) Variational Calibration**: Explicit observation noise scaling (`sigma`) prevents artificial likelihood rescaling and calibrates both spike-and-slab KL shrinkage and differential privacy sensitivities.

---

## Project 1: Survey-Weighted Pseudo-Posteriors & Formal Differential Privacy

### Track 1A: Informative Survey Sampling, Domain Estimation & Imputation
Following Savitsky & Toth (2016), survey design weights $w_i$ are normalized to sample size ($\tilde{w}_i = w_i \frac{n}{\sum w_j}$, $\sum \tilde{w}_i = n$) to correct informative selection bias without overpowering the KL prior penalty. Epistemic parameter draws $\theta^{(s)} \sim q(\theta)$ are compounded with Dirichlet Bayesian bootstrap perturbations $\boldsymbol{\xi}^{(s)} \sim \text{Dirichlet}(1, \dots, 1)$ for calibrated design-cum-model uncertainty quantification.

```r
library(LBBNN)

# 1. Package survey design weights (normalized so sum(w_tilde) == n) & allow NA outcomes
dl_survey <- get_dataloaders(
  df_survey,
  weights = df_survey$w_design,
  normalize_weights = TRUE,
  allow_missing = TRUE,
  train_batch_size = 64
)

# 2. Initialize Input-Skip LBBNN with calibrated residual noise scale sigma
net_reg <- lbbnn_net(
  problem_type = "regression",
  sizes = c(p, 16, 1),
  sigma = 0.40,
  input_skip = TRUE
)

# 3. Fit variational pseudo-posterior
fit_reg <- train_lbbnn(LBBNN = net_reg, train_dl = dl_survey$train_loader, epochs = 100, lr = 0.01)

# 4. Small-domain estimation (Model-Assisted Conditional Expectation + Dirichlet UQ)
dom_summary <- predict_domains_lbbnn(
  net_reg,
  X = X_matrix,
  domain_matrix = G_domains,
  weights = df_survey$w_design,
  type = "conditional_mean",
  dirichlet_uq = TRUE,
  draws = 50
)

# 5. Multiple imputation for item non-response (y_i = NA)
imp_out <- impute_response_lbbnn(net_reg, newdata = df_survey, target_col = "y", draws = 50)
```

### Track 1B: Multi-Pass $(\epsilon_{\mathbf{y}}, \delta)$-Differential Privacy Microdata Synthesis
Implements the formal two-stage privacy weight calibration framework of Savitsky, Williams & Hu (2022) and Savitsky (2026) across the full $3 \times 2$ factorial design:
* **Stage 1 Initial Normalization**: `"linear"` ($c(1 - \tilde{f}_i)$, Remark 1), `"percentile"` (quantile deficit truncation), or `"target_clip"` ($\min(1, \Delta_{\text{goal}}/f_i)$, Proposition 1).
* **Stage 2 Threshold Enforcement ($\alpha_i^* f_i \le M_{\text{thresh}}$)**: `"hard_trunc"` or `"vector_upscale"` ($\min(1, M_{\text{thresh}}/f_i)$, which Loewner-dominates hard truncation in Fisher information by Proposition 2).
* **Automated Pass 3 Geometry Alignment**: When `"vector_upscale"` is selected, `fit_dp_lbbnn()` automatically executes Pass 3 to retrain the variational pseudo-posterior under the rescaled weight geometry and verifies $\Delta_{\alpha, \mathbf{y}} \le M_{\text{thresh}}$, guaranteeing $\epsilon_{\mathbf{y}} = 2\Delta_{\alpha, \mathbf{y}}$.

```r
# End-to-end multi-pass DP calibration & synthetic microdata generation
dp_fit <- fit_dp_lbbnn(
  model_spec = list(sizes = c(p, 16, 1), problem_type = "regression"),
  X = x_micro,
  y = y_expenditure,
  stage1_method = "target_clip",
  stage2_method = "vector_upscale",
  M_thresh = 5.5,
  epochs = 100,
  lr = 0.01
)

cat("Verified Empirical Privacy Budget (epsilon_y):", dp_fit$epsilon, "\n")
cat("Retained Likelihood Mass (ESS = sum(alpha*)):", sum(dp_fit$alpha), "/", length(y_expenditure), "\n")

# Generate (epsilon_y, delta)-PDP synthetic microdata via the Post-Processing Theorem
synth_replicates <- sample_synthetic_lbbnn(dp_fit, newdata = x_micro, draws = 50)
```

---

## Project 2: Non-Probability Sample Integration via Composite Propensities

Combines an opt-in convenience sample $S_c$ (unknown selection propensity $\pi_{ci} = \sigma(\eta_\theta(\mathbf{x}_i))$) with a reference probability survey $S_r$ (known design inclusion probability $\pi_{ri} > 0$) following Beresovsky, Gershunskaya & Savitsky (2024, 2026).

### Supported Propensity Estimators
| Method | Sample Structure | Composite Link $\pi_{zi}$ | Key Statistical Property |
| :--- | :--- | :--- | :--- |
| **`STACK`** | Unlinked multiset $S_c \oplus S_r$ | $\frac{\pi_{ci}}{\pi_{ci} + \pi_{ri}}$ | No record linkage required; bounded $[-1, 1]$ gradients. |
| **`LINK1`** | Deduplicated union $S_c \cup (S_r \setminus S_c)$ | $\frac{\pi_{ci}}{\pi_{ci} + \pi_{ri} - \pi_{ci}\pi_{ri}}$ | Exact overlap adjustment via autograd through $-\pi_{ci}\pi_{ri}$. |
| **`LINK2`** | Linked multiset $S_c \oplus S_r$ | $\frac{\pi_{ci}(1 + \pi_{ri})}{\pi_{ci} + \pi_{ri}}$ | Re-labels $I_{zi} = D_i + (1-D_i)I_{ci}$; maximizes Fisher information. |
| **`MIP`** | Probabilistic linkage $\tilde{\mathbf{p}} \in [0, 1]^{n_r}$ | Soft-label $\tilde{I}_{zi}^{\text{MIP}} = D_i + (1-D_i)\tilde{p}_i$ | Smoothly bridges `STACK` ($\tilde{p}_i \to 0$) and `LINK2` ($\tilde{p}_i \to 1$). |
| **`PSEUDO`** | Horvitz–Thompson on $S_r$ | Census pseudo-NLL (Chen et al., 2020) | Unbounded $\hat{\pi}_{ci}/\pi_{ri}$ gradients under low common support. |

```r
# 1. Build composite dataset for Missing Information Principle (MIP) with linkage vector p_vec_r
prep <- build_propensity_dataset(
  pop,
  method = "MIP",
  covar_cols = paste0("X", 1:p),
  p_link = p_vec_r
)

dl_prop <- get_dataloaders(
  prep$data,
  target_cols = prep$target_cols,
  train_batch_size = 64,
  test_batch_size = 32
)

# 2. Train sparse Input-Skip LBBNN with MIP composite Bernoulli loss
net_prop <- lbbnn_net(
  problem_type = "custom",
  sizes = c(p, 8, 1),
  raw_output = TRUE,
  input_skip = TRUE,
  nll = make_nll_mip()
)

train_lbbnn(LBBNN = net_prop, train_dl = dl_prop$train_loader, epochs = 150, lr = 0.02)

# 3. Predict convenience propensities using Median Probability Model (MPM) & compute Hajek mean
logits_c <- predict(net_prop, newdata = prep$data[prep$Sc_indices, 1:p], mpm = TRUE)
pi_c_hat <- as.numeric(torch::torch_sigmoid(torch::torch_tensor(logits_c)))

hajek_est <- hajek_estimate(
  y = pop$y[pop$I_c == 1],
  pi_hat = pi_c_hat,
  dirichlet_uq = TRUE
)
print(hajek_est)
```

---

## Reproducible Reports & Presentation Deck

All experimental benchmarks, Monte Carlo studies, and presentation materials are fully reproducible from source:

* **BLS Seminar Presentation**: [`bls_lbbnn_presentation.pdf`](bls_lbbnn_presentation.pdf) (Source: `bls_collaboration/presentation/bls_lbbnn_presentation.Rmd`)
* **Project 1 Technical Report**: [`project1_weighted_dp_report.pdf`](project1_weighted_dp_report.pdf) (Source: `bls_collaboration/track1_report/project1_weighted_dp_report.Rmd`)
* **Project 2 Technical Report**: [`project2_propensity_estimation_report.pdf`](project2_propensity_estimation_report.pdf) (Source: `bls_collaboration/track2_report/project2_propensity_estimation_report.Rmd`)

---

## References

1. **Beresovsky, V., Gershunskaya, J., & Savitsky, T. D.** (2024). *Review of quasi-randomization approaches for estimation from non-probability samples*. Statistical Science.
2. **Savitsky, T. D., Williams, M. R., & Hu, J.** (2022). *Bayesian pseudo posterior mechanism under asymptotic differential privacy*. Journal of Machine Learning Research, 23(232), 1–37.
3. **Savitsky, T. D., & Toth, D.** (2016). *Bayesian estimation under informative sampling*. Electronic Journal of Statistics, 10(1), 1677–1708.
4. **Savitsky, T. D.** (2026). *Privacy weight normalization and second-stage reweighting strategies in asymptotic differential privacy*. U.S. Bureau of Labor Statistics Technical Note.
5. **Hubin, A., & Storvik, G.** (2024). *Sparse Bayesian neural networks: Bridging model and parameter uncertainty through scalable variational learning*. Mathematics, 12(6), 788.
6. **Skaaret-Lund, L., Storvik, G., & Hubin, A.** (2024). *Sparsifying Bayesian neural networks with latent binary variables and normalizing flows*. Transactions on Machine Learning Research (TMLR).
7. **Høyheim, E., Skaaret-Lund, L., Sæbø, S., & Hubin, A.** (2026). *Explainable Bayesian deep learning through input-skip Latent Binary Bayesian Neural Networks*. Journal of Artificial Intelligence Research (JAIR), 86.
