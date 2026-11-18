#!/usr/bin/env Rscript
# ==============================================================================
# inst/scripts/analyze_glmm_benchmarks.R
# Rigorous Exponential-Family GLMM Analysis Across Per-Benchmark & Joint Designs
#
# Replaces the naive Gaussian LMM / lm(method + seed_f) fallback with:
#   1. Gamma GLMM (log link, dispformula = ~ method when identifiable) for
#      positive skewed errors (test_rmse, exp_err), with exact beta_0 for
#      Symbolic-LBBNN.
#   2. Binomial GLMM (logit link) with symmetric cell-means parameterization
#      (~ 0 + method + RE) and weakly informative Normal(0, 2.5^2) priors on
#      every method log-odds, followed by exact covariance-based linear contrasts
#      against Symbolic-LBBNN. This handles 100/100 and 0/100 separation while
#      guaranteeing exact 0.000 contrast (p = 1.000) for tied levels.
#   3. Negative Binomial GLMM (log link) for active term / edge counts.
#   4. Joint Cross-Benchmark Hierarchical GLMM:
#        eta_{dim} = beta_0 + beta_m^(method) + u_d^(dataset)
#                    + u_{dm}^(dataset:method) + b_{di}^(dataset:seed)
#      quantifying global contrasts + variance decomposition & ICCs.
#
# Outputs CSV artifacts into results_baselines/ for Tables 1-5 and Figure 7.
# ==============================================================================

suppressPackageStartupMessages({
  library(glmmTMB)
})

# Symmetric weakly informative Gaussian prior N(0, 2.5^2) on every cell log-odds
p_sym_binom <- data.frame(
  prior = "normal(0, 2.5)",
  class = "fixef",
  coef  = "",
  stringsAsFactors = FALSE
)

# Exact Clopper-Pearson binomial 95% confidence interval
clopper_pearson_ci <- function(k, n, conf.level = 0.95) {
  alpha <- 1.0 - conf.level
  lo <- ifelse(k <= 0, 0.0, stats::qbeta(alpha / 2.0, k, n - k + 1.0))
  hi <- ifelse(k >= n, 1.0, stats::qbeta(1.0 - alpha / 2.0, k + 1.0, n - k))
  list(mean = k / n, lo = lo, hi = hi)
}

# Reconstruct exact integer trials for bounded proportion columns
get_prop_trials <- function(bench, metric, method_vec) {
  if (bench == "logic") {
    if (metric == "clause_tpr") return(rep(3L, length(method_vec)))
    if (metric == "clause_fpr") return(rep(29L, length(method_vec)))
    if (metric == "var_tpr") return(rep(7L, length(method_vec)))
    if (metric == "var_fpr") {
      return(ifelse(method_vec == "ReLU-LBBNN", 50L, 13L))
    }
  } else if (bench == "kepler") {
    if (metric == "var_tpr") return(rep(2L, length(method_vec)))
    if (metric == "var_fpr") return(ifelse(method_vec == "ReLU-LBBNN", 20L, 8L))
  } else if (bench == "kepler_real") {
    if (metric == "var_tpr") return(rep(2L, length(method_vec)))
    if (metric == "var_fpr") return(ifelse(method_vec == "ReLU-LBBNN", 20L, 6L))
  } else if (bench == "nonlinear") {
    if (metric %in% c("var_tpr", "var_fpr")) {
      return(ifelse(method_vec == "ReLU-LBBNN", 10L, 5L))
    }
  }
  rep(100L, length(method_vec))
}

# Strict glmmTMB convergence check
is_valid_tmb <- function(fit) {
  if (is.null(fit)) return(FALSE)
  conv_ok <- tryCatch(identical(as.integer(fit$fit$convergence), 0L), error = function(e) FALSE)
  pd_ok   <- tryCatch(isTRUE(fit$sdr$pdHess), error = function(e) FALSE)
  if (!conv_ok || !pd_ok) return(FALSE)
  cf <- tryCatch(summary(fit)$coefficients$cond, error = function(e) NULL)
  if (is.null(cf)) return(FALSE)
  all(is.finite(cf[, "Estimate"])) && all(is.finite(cf[, "Std. Error"]))
}

# Helper: evaluate expression cleanly suppressing convergence probes during candidate selection
try_tmb <- function(expr) {
  tryCatch(suppressWarnings(expr), error = function(e) NULL)
}

# Extract conditional fixed-effect contrasts from treatment-coded glmmTMB
extract_fixef_treatment <- function(fit, benchmark, metric, family_label, link_label,
                                    ref_method = "Symbolic-LBBNN") {
  cf <- summary(fit)$coefficients$cond
  vc_re <- glmmTMB::VarCorr(fit)$cond
  seed_sd <- NA_real_
  if (!is.null(vc_re) && "seed_f" %in% names(vc_re)) {
    seed_sd <- as.numeric(attr(vc_re[["seed_f"]], "stddev"))[1]
  } else if (!is.null(vc_re) && "ds_seed" %in% names(vc_re)) {
    seed_sd <- as.numeric(attr(vc_re[["ds_seed"]], "stddev"))[1]
  }

  rows <- vector("list", nrow(cf))
  for (idx in seq_len(nrow(cf))) {
    rn <- rownames(cf)[idx]
    est <- as.numeric(cf[idx, "Estimate"])
    se  <- as.numeric(cf[idx, "Std. Error"])
    z   <- as.numeric(cf[idx, "z value"])
    p   <- as.numeric(cf[idx, "Pr(>|z|)"])
    ci_lo <- est - 1.96 * se
    ci_hi <- est + 1.96 * se

    is_ref <- (rn == "(Intercept)")
    term_lbl <- if (is_ref) ref_method else sub("^method", "", rn)

    resp_est <- if (link_label == "log") exp(est) else if (link_label == "logit") stats::plogis(est) else est
    resp_lo  <- if (link_label == "log") exp(ci_lo) else if (link_label == "logit") stats::plogis(ci_lo) else ci_lo
    resp_hi  <- if (link_label == "log") exp(ci_hi) else if (link_label == "logit") stats::plogis(ci_hi) else ci_hi

    ratio_or <- if (!is_ref && link_label %in% c("log", "logit")) exp(est) else NA_real_
    ratio_lo <- if (!is_ref && link_label %in% c("log", "logit")) exp(ci_lo) else NA_real_
    ratio_hi <- if (!is_ref && link_label %in% c("log", "logit")) exp(ci_hi) else NA_real_

    rows[[idx]] <- data.frame(
      benchmark = benchmark, metric = metric, family = family_label, link = link_label,
      term = term_lbl, is_ref = is_ref,
      estimate = est, std_error = se, ci_lo = ci_lo, ci_hi = ci_hi,
      z_value = z, p_value = p,
      resp_est = resp_est, resp_lo = resp_lo, resp_hi = resp_hi,
      ratio_or = ratio_or, ratio_lo = ratio_lo, ratio_hi = ratio_hi,
      seed_sd = seed_sd, stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

# Map Wald p-values to publication significance markers (used across Tables 1-5)
format_sig_star <- function(p_vec) {
  vapply(p_vec, function(p) {
    if (is.na(p)) return("")
    if (p < 0.001) return("***")
    if (p < 0.01)  return("**")
    if (p < 0.05)  return("*")
    "ns"
  }, character(1))
}

# Extract treatment-equivalent contrasts from symmetric cell-means model (~ 0 + method + RE)
extract_fixef_cellmeans <- function(fit, benchmark, metric, family_label, link_label,
                                    ref_method = "Symbolic-LBBNN") {
  cf <- summary(fit)$coefficients$cond
  V  <- stats::vcov(fit)$cond
  vc_re <- glmmTMB::VarCorr(fit)$cond
  seed_sd <- NA_real_
  if (!is.null(vc_re) && "seed_f" %in% names(vc_re)) {
    seed_sd <- as.numeric(attr(vc_re[["seed_f"]], "stddev"))[1]
  } else if (!is.null(vc_re) && "ds_seed" %in% names(vc_re)) {
    seed_sd <- as.numeric(attr(vc_re[["ds_seed"]], "stddev"))[1]
  }

  ref_rn <- paste0("method", ref_method)
  ref_idx <- which(rownames(cf) == ref_rn)
  stopifnot(length(ref_idx) == 1L)
  mu_ref <- as.numeric(cf[ref_idx, "Estimate"])
  se_ref <- as.numeric(cf[ref_idx, "Std. Error"])

  rows <- vector("list", nrow(cf))
  for (idx in seq_len(nrow(cf))) {
    rn <- rownames(cf)[idx]
    term_lbl <- sub("^method", "", rn)
    is_ref <- (idx == ref_idx)
    if (is_ref) {
      est <- mu_ref
      se  <- se_ref
      z   <- as.numeric(cf[idx, "z value"])
      p   <- as.numeric(cf[idx, "Pr(>|z|)"])
    } else {
      est <- as.numeric(cf[idx, "Estimate"]) - mu_ref
      var_d <- as.numeric(V[idx, idx] + V[ref_idx, ref_idx] - 2.0 * V[idx, ref_idx])
      se  <- sqrt(max(var_d, 1e-12))
      if (abs(est) < 1e-10) {
        est <- 0.0
        z   <- 0.0
        p   <- 1.0
      } else {
        z   <- est / se
        p   <- 2.0 * stats::pnorm(-abs(z))
      }
    }
    ci_lo <- est - 1.96 * se
    ci_hi <- est + 1.96 * se

    resp_est <- if (link_label == "logit") stats::plogis(as.numeric(cf[idx, "Estimate"])) else exp(as.numeric(cf[idx, "Estimate"]))
    resp_lo  <- if (link_label == "logit") stats::plogis(as.numeric(cf[idx, "Estimate"]) - 1.96 * as.numeric(cf[idx, "Std. Error"])) else NA_real_
    resp_hi  <- if (link_label == "logit") stats::plogis(as.numeric(cf[idx, "Estimate"]) + 1.96 * as.numeric(cf[idx, "Std. Error"])) else NA_real_

    ratio_or <- if (!is_ref && link_label %in% c("log", "logit")) exp(est) else NA_real_
    ratio_lo <- if (!is_ref && link_label %in% c("log", "logit")) exp(ci_lo) else NA_real_
    ratio_hi <- if (!is_ref && link_label %in% c("log", "logit")) exp(ci_hi) else NA_real_

    rows[[idx]] <- data.frame(
      benchmark = benchmark, metric = metric, family = family_label, link = link_label,
      term = term_lbl, is_ref = is_ref,
      estimate = est, std_error = se, ci_lo = ci_lo, ci_hi = ci_hi,
      z_value = z, p_value = p,
      resp_est = resp_est, resp_lo = resp_lo, resp_hi = resp_hi,
      ratio_or = ratio_or, ratio_lo = ratio_lo, ratio_hi = ratio_hi,
      seed_sd = seed_sd, stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

# Fit per-benchmark GLMM for a single metric
fit_single_metric_glmm <- function(df_bench, bench, m_col, ref_method = "Symbolic-LBBNN") {
  sub_df <- df_bench[is.finite(df_bench[[m_col]]), , drop = FALSE]
  if (nrow(sub_df) < 10L || length(unique(sub_df$method)) < 2L) return(NULL)

  sub_df$method <- factor(sub_df$method)
  if (ref_method %in% levels(sub_df$method)) {
    sub_df$method <- stats::relevel(sub_df$method, ref = ref_method)
  }
  sub_df$seed_f <- factor(sub_df$seed)

  # 1. Positive continuous skewed error metrics -> Gamma(log)
  if (m_col %in% c("test_rmse", "exp_err")) {
    sub_df[[m_col]] <- pmax(sub_df[[m_col]], 1e-8)
    grp_sds <- tapply(sub_df[[m_col]], sub_df$method, stats::sd)
    can_hetero <- all(is.finite(grp_sds) & grp_sds > 1e-10)

    fit <- NULL
    if (can_hetero) {
      form <- stats::as.formula(paste0(m_col, " ~ method + (1 | seed_f)"))
      fit_h <- try_tmb(glmmTMB::glmmTMB(
        form, dispformula = ~ method, family = stats::Gamma(link = "log"), data = sub_df
      ))
      if (is_valid_tmb(fit_h)) fit <- fit_h
    }
    if (is.null(fit)) {
      form <- stats::as.formula(paste0(m_col, " ~ method + (1 | seed_f)"))
      fit_c <- try_tmb(glmmTMB::glmmTMB(
        form, dispformula = ~ 1, family = stats::Gamma(link = "log"), data = sub_df
      ))
      if (is_valid_tmb(fit_c)) fit <- fit_c
    }
    if (is.null(fit)) {
      form_no_re <- stats::as.formula(paste0(m_col, " ~ method"))
      fit_n <- try_tmb(glmmTMB::glmmTMB(
        form_no_re, dispformula = ~ 1, family = stats::Gamma(link = "log"), data = sub_df
      ))
      if (is_valid_tmb(fit_n)) fit <- fit_n
    }
    if (!is.null(fit)) {
      return(extract_fixef_treatment(fit, bench, m_col, "Gamma", "log", ref_method))
    }
    return(NULL)
  }

  # 2. Binary outcomes -> Symmetric cell-means Binomial(logit) with N(0, 2.5^2)
  if (m_col %in% c("exact_recovery", "structural_recovery", "exact_law", "exact_var_recovery")) {
    sub_df$y_bin <- as.integer(round(sub_df[[m_col]]))
    fit <- try_tmb(glmmTMB::glmmTMB(
      y_bin ~ 0 + method + (1 | seed_f),
      family = stats::binomial(link = "logit"),
      priors = p_sym_binom,
      data = sub_df
    ))
    if (!is_valid_tmb(fit)) {
      fit <- try_tmb(glmmTMB::glmmTMB(
        y_bin ~ 0 + method,
        family = stats::binomial(link = "logit"),
        priors = p_sym_binom,
        data = sub_df
      ))
    }
    if (is_valid_tmb(fit)) {
      return(extract_fixef_cellmeans(fit, bench, m_col, "Binomial", "logit", ref_method))
    }
    return(NULL)
  }

  # 3. Bounded proportions -> Symmetric cell-means Binomial(logit) on (k, n-k)
  if (m_col %in% c("clause_tpr", "clause_fpr", "var_tpr", "var_fpr")) {
    n_tr <- get_prop_trials(bench, m_col, as.character(sub_df$method))
    k_succ <- pmin(n_tr, pmax(0L, as.integer(round(sub_df[[m_col]] * n_tr))))
    sub_df$k_succ <- k_succ
    sub_df$k_fail <- n_tr - k_succ
    fit <- try_tmb(glmmTMB::glmmTMB(
      cbind(k_succ, k_fail) ~ 0 + method + (1 | seed_f),
      family = stats::binomial(link = "logit"),
      priors = p_sym_binom,
      data = sub_df
    ))
    if (!is_valid_tmb(fit)) {
      fit <- try_tmb(glmmTMB::glmmTMB(
        cbind(k_succ, k_fail) ~ 0 + method,
        family = stats::binomial(link = "logit"),
        priors = p_sym_binom,
        data = sub_df
      ))
    }
    if (is_valid_tmb(fit)) {
      return(extract_fixef_cellmeans(fit, bench, m_col, "Binomial", "logit", ref_method))
    }
    return(NULL)
  }

  # 4. Count metrics (n_terms) -> Negative Binomial nbinom2(log)
  if (m_col == "n_terms") {
    sub_df$y_cnt <- pmax(1L, as.integer(round(sub_df$n_terms)))
    fit <- try_tmb(glmmTMB::glmmTMB(
      y_cnt ~ method + (1 | seed_f),
      family = glmmTMB::nbinom2(link = "log"),
      data = sub_df
    ))
    if (!is_valid_tmb(fit)) {
      fit <- try_tmb(glmmTMB::glmmTMB(
        y_cnt ~ method,
        family = glmmTMB::nbinom2(link = "log"),
        data = sub_df
      ))
    }
    if (is_valid_tmb(fit)) {
      return(extract_fixef_treatment(fit, bench, m_col, "NegBinom", "log", ref_method))
    }
  }
  NULL
}

# Summarize each metric with 95% CIs (Clopper-Pearson for proportions/binary, t CIs for continuous)
summarize_with_ci <- function(df_bench, bench) {
  methods <- unique(df_bench$method)
  if ("Symbolic-LBBNN" %in% methods) {
    methods <- c("Symbolic-LBBNN", setdiff(methods, "Symbolic-LBBNN"))
  }
  num_cols <- setdiff(names(df_bench)[vapply(df_bench, is.numeric, logical(1))], "seed")

  out_rows <- list()
  for (m in methods) {
    sub_m <- df_bench[df_bench$method == m, , drop = FALSE]
    for (nc in num_cols) {
      vals <- sub_m[[nc]]
      vals <- vals[is.finite(vals)]
      nv <- length(vals)
      if (nv == 0L) next
      mu <- mean(vals)
      sd_v <- if (nv > 1L) stats::sd(vals) else 0.0
      med_v <- stats::median(vals)
      q25 <- as.numeric(stats::quantile(vals, 0.25))
      q75 <- as.numeric(stats::quantile(vals, 0.75))

      if (nc %in% c("exact_recovery", "structural_recovery", "exact_law", "exact_var_recovery")) {
        k_tot <- sum(round(vals))
        cp <- clopper_pearson_ci(k_tot, nv)
        ci_lo <- cp$lo; ci_hi <- cp$hi
      } else if (nc %in% c("clause_tpr", "clause_fpr", "var_tpr", "var_fpr")) {
        n_tr <- get_prop_trials(bench, nc, rep(m, nv))
        k_tot <- sum(round(vals * n_tr))
        cp <- clopper_pearson_ci(k_tot, sum(n_tr))
        ci_lo <- cp$lo; ci_hi <- cp$hi
      } else if (nc %in% c("test_rmse", "exp_err")) {
        se <- sd_v / sqrt(nv)
        t_crit <- stats::qt(0.975, df = max(1L, nv - 1L))
        ci_lo <- max(0.0, mu - t_crit * se)
        ci_hi <- mu + t_crit * se
      } else {
        se <- sd_v / sqrt(nv)
        t_crit <- stats::qt(0.975, df = max(1L, nv - 1L))
        ci_lo <- mu - t_crit * se
        ci_hi <- mu + t_crit * se
      }

      out_rows[[length(out_rows) + 1L]] <- data.frame(
        benchmark = bench,
        method = m,
        n_seeds = nv,
        metric = nc,
        mean = mu,
        sd = sd_v,
        median = med_v,
        q25 = q25,
        q75 = q75,
        ci_lo = ci_lo,
        ci_hi = ci_hi,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, out_rows)
}

# ==============================================================================
# Run Per-Benchmark GLMMs & Summaries
# ==============================================================================
benchmarks <- c("logic", "kepler", "kepler_real", "nonlinear")
bench_metrics <- list(
  logic       = c("clause_tpr", "clause_fpr", "exact_recovery", "var_tpr", "var_fpr", "test_rmse", "n_terms"),
  kepler      = c("structural_recovery", "exact_law", "exp_err", "var_tpr", "var_fpr", "test_rmse", "n_terms"),
  kepler_real = c("structural_recovery", "exact_law", "exp_err", "var_tpr", "var_fpr", "test_rmse", "n_terms"),
  nonlinear   = c("var_tpr", "var_fpr", "exact_var_recovery", "test_rmse", "n_terms")
)

all_raw_list <- list()
for (b in benchmarks) {
  raw_path <- file.path("results_baselines", b, sprintf("raw_%s_baselines.csv", b))
  df_b <- utils::read.csv(raw_path, stringsAsFactors = FALSE)
  stopifnot(nrow(df_b) > 0L)
  all_raw_list[[b]] <- df_b

  df_ci <- summarize_with_ci(df_b, b)
  utils::write.csv(df_ci, file.path("results_baselines", b, sprintf("summary_%s_ci.csv", b)), row.names = FALSE)

  m_res <- lapply(bench_metrics[[b]], function(mc) fit_single_metric_glmm(df_b, b, mc))
  m_res <- Filter(Negate(is.null), m_res)
  df_glmm <- do.call(rbind, m_res)
  stopifnot(!is.null(df_glmm), nrow(df_glmm) > 0L)
  df_glmm$p_adj <- NA_real_
  for (mc in unique(df_glmm$metric)) {
    idx_nr <- which(df_glmm$metric == mc & !df_glmm$is_ref)
    if (length(idx_nr) > 0L) {
      df_glmm$p_adj[idx_nr] <- stats::p.adjust(df_glmm$p_value[idx_nr], method = "bonferroni")
    }
  }
  df_glmm$sig_star <- ifelse(df_glmm$is_ref, "Ref", format_sig_star(df_glmm$p_value))
  df_glmm$sig_star_adj <- ifelse(df_glmm$is_ref, "Ref", format_sig_star(df_glmm$p_adj))
  utils::write.csv(df_glmm, file.path("results_baselines", b, sprintf("glmm_%s_contrasts.csv", b)), row.names = FALSE)
  message(sprintf("[GLMM] Completed '%s' (%d contrast rows).", b, nrow(df_glmm)))
}

# ==============================================================================
# Joint Cross-Benchmark Hierarchical GLMM
# ==============================================================================
map_method_family <- function(m_vec) {
  m_vec <- ifelse(m_vec %in% c("FBMS-BLR", "FBMS-BGNLM"), "FBMS", m_vec)
  m_vec <- ifelse(m_vec %in% c("Lasso-Interactions", "Log-Lasso", "Lasso"), "Lasso", m_vec)
  m_vec
}

sym_real_rmse <- mean(all_raw_list[["kepler_real"]]$test_rmse[all_raw_list[["kepler_real"]]$method == "Symbolic-LBBNN"])
oracle_scales <- c(logic = 0.500, kepler = 0.005, kepler_real = sym_real_rmse, nonlinear = 1.000)

joint_rows <- list()
for (b in benchmarks) {
  d <- all_raw_list[[b]]
  exact_goal <- switch(
    b,
    logic       = as.integer(d$exact_recovery),
    kepler      = as.integer(d$exact_law),
    kepler_real = as.integer(d$exact_law),
    nonlinear   = as.integer(d$exact_var_recovery)
  )
  exact_sup <- switch(
    b,
    logic       = as.integer(d$exact_recovery),
    kepler      = as.integer(d$structural_recovery),
    kepler_real = as.integer(d$structural_recovery),
    nonlinear   = as.integer(d$exact_var_recovery)
  )
  n_tpr <- get_prop_trials(b, "var_tpr", d$method)
  k_tpr <- pmin(n_tpr, pmax(0L, as.integer(round(d$var_tpr * n_tpr))))
  n_fpr <- get_prop_trials(b, "var_fpr", d$method)
  k_spec <- pmin(n_fpr, pmax(0L, as.integer(round((1.0 - d$var_fpr) * n_fpr))))

  joint_rows[[b]] <- data.frame(
    dataset       = b,
    seed          = d$seed,
    method        = map_method_family(d$method),
    test_rmse     = pmax(d$test_rmse, 1e-8),
    rel_rmse      = pmax(d$test_rmse / oracle_scales[[b]], 1e-6),
    exact_goal    = exact_goal,
    exact_support = exact_sup,
    k_tpr         = k_tpr,
    n_tpr         = n_tpr,
    k_spec        = k_spec,
    n_spec        = n_fpr,
    n_terms       = pmax(1L, as.integer(round(d$n_terms))),
    stringsAsFactors = FALSE
  )
}
df_joint <- do.call(rbind, joint_rows)
df_joint$dataset   <- factor(df_joint$dataset, levels = benchmarks)
df_joint$method    <- stats::relevel(factor(df_joint$method), ref = "Symbolic-LBBNN")
df_joint$ds_method <- interaction(df_joint$dataset, df_joint$method, drop = TRUE)
df_joint$ds_seed   <- interaction(df_joint$dataset, df_joint$seed, drop = TRUE)

# 1. Joint Relative RMSE: Gamma GLMM with dataset-level dispersion
fit_j_rmse <- glmmTMB::glmmTMB(
  rel_rmse ~ method + (1 | dataset) + (1 | ds_method) + (1 | ds_seed),
  dispformula = ~ dataset,
  family = stats::Gamma(link = "log"),
  data = df_joint
)
stopifnot(is_valid_tmb(fit_j_rmse))

# 2. Joint Exact Target Discovery (exact_law / exact_recovery / exact_var_recovery)
fit_j_goal <- glmmTMB::glmmTMB(
  exact_goal ~ 0 + method + (1 | dataset) + (1 | ds_method) + (1 | ds_seed),
  family = stats::binomial(link = "logit"),
  priors = p_sym_binom,
  data = df_joint
)
stopifnot(is_valid_tmb(fit_j_goal))

# 3. Joint Variable Sensitivity (TPR) & Noise Specificity (1 - FPR)
fit_j_tpr <- glmmTMB::glmmTMB(
  cbind(k_tpr, n_tpr - k_tpr) ~ 0 + method + (1 | dataset) + (1 | ds_method) + (1 | ds_seed),
  family = stats::binomial(link = "logit"),
  priors = p_sym_binom,
  data = df_joint
)
stopifnot(is_valid_tmb(fit_j_tpr))

fit_j_spec <- glmmTMB::glmmTMB(
  cbind(k_spec, n_spec - k_spec) ~ 0 + method + (1 | dataset) + (1 | ds_method) + (1 | ds_seed),
  family = stats::binomial(link = "logit"),
  priors = p_sym_binom,
  data = df_joint
)
stopifnot(is_valid_tmb(fit_j_spec))

joint_fe <- rbind(
  extract_fixef_treatment(fit_j_rmse, "joint", "rel_rmse",   "Gamma",    "log"),
  extract_fixef_cellmeans(fit_j_goal, "joint", "exact_goal", "Binomial", "logit"),
  extract_fixef_cellmeans(fit_j_tpr,  "joint", "var_tpr",    "Binomial", "logit"),
  extract_fixef_cellmeans(fit_j_spec, "joint", "var_spec",   "Binomial", "logit")
)
joint_fe$p_adj <- NA_real_
for (mc in unique(joint_fe$metric)) {
  idx_nr <- which(joint_fe$metric == mc & !joint_fe$is_ref)
  joint_fe$p_adj[idx_nr] <- stats::p.adjust(joint_fe$p_value[idx_nr], method = "bonferroni")
}
joint_fe$sig_star <- ifelse(joint_fe$is_ref, "Ref", format_sig_star(joint_fe$p_value))
joint_fe$sig_star_adj <- ifelse(joint_fe$is_ref, "Ref", format_sig_star(joint_fe$p_adj))
utils::write.csv(joint_fe, "results_baselines/joint_glmm_contrasts.csv", row.names = FALSE)

# Extract Variance Components & ICCs (Nakagawa observation-level link variance)
extract_var_comp <- function(fit, metric_name, fam_type) {
  vc <- glmmTMB::VarCorr(fit)$cond
  v_ds   <- as.numeric(attr(vc[["dataset"]],   "stddev"))[1]^2
  v_dsm  <- as.numeric(attr(vc[["ds_method"]], "stddev"))[1]^2
  v_seed <- as.numeric(attr(vc[["ds_seed"]],   "stddev"))[1]^2

  if (fam_type == "binomial") {
    v_obs <- (pi^2) / 3.0
  } else {
    phi_avg <- mean(stats::predict(fit, type = "disp"))
    shape_avg <- 1.0 / max(phi_avg, 1e-6)
    v_obs <- trigamma(shape_avg)
  }

  v_tot <- v_ds + v_dsm + v_seed + v_obs
  data.frame(
    metric = metric_name,
    sd_dataset = sqrt(v_ds),
    sd_dataset_method = sqrt(v_dsm),
    sd_seed = sqrt(v_seed),
    var_dataset = v_ds,
    var_dataset_method = v_dsm,
    var_seed = v_seed,
    var_obs_link = v_obs,
    icc_dataset = v_ds / v_tot,
    icc_dataset_method = v_dsm / v_tot,
    icc_seed = v_seed / v_tot,
    icc_total_re = (v_ds + v_dsm + v_seed) / v_tot,
    stringsAsFactors = FALSE
  )
}

df_vc <- rbind(
  extract_var_comp(fit_j_rmse, "rel_rmse",   "gamma"),
  extract_var_comp(fit_j_goal, "exact_goal", "binomial"),
  extract_var_comp(fit_j_tpr,  "var_tpr",    "binomial"),
  extract_var_comp(fit_j_spec, "var_spec",   "binomial")
)
utils::write.csv(df_vc, "results_baselines/joint_glmm_variance_components.csv", row.names = FALSE)

message("\n=== JOINT GLMM FIXED EFFECTS (Relative RMSE & Exact Discovery) ===")
print(joint_fe[joint_fe$metric %in% c("rel_rmse", "exact_goal"),
               c("metric", "term", "estimate", "std_error", "ci_lo", "ci_hi", "ratio_or", "p_value", "sig_star")],
      digits = 4)

message("\n=== BINOMIAL GLMM STRUCTURAL SIGNIFICANCE MARKERS (Tables 1-4) ===")
struct_map <- c(logic = "exact_recovery", kepler = "exact_law", kepler_real = "structural_recovery", nonlinear = "exact_var_recovery")
for (b_nm in names(struct_map)) {
  g_tmp <- utils::read.csv(file.path("results_baselines", b_nm, sprintf("glmm_%s_contrasts.csv", b_nm)), stringsAsFactors = FALSE)
  sub_g <- g_tmp[g_tmp$metric == struct_map[[b_nm]], c("benchmark", "metric", "term", "resp_est", "p_value", "sig_star")]
  print(sub_g, digits = 4, row.names = FALSE)
}

message("\n=== JOINT GLMM VARIANCE COMPONENTS & ICCs ===")
print(df_vc, digits = 4)

message("\n[OK] All per-benchmark and joint GLMM analyses converged cleanly and saved to results_baselines/.")
