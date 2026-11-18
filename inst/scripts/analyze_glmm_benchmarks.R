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
  } else if (bench == "sindy_3d") {
    if (metric == "var_tpr") return(rep(7L, length(method_vec)))
    if (metric == "var_fpr") return(rep(11L, length(method_vec)))
  }
  rep(100L, length(method_vec))
}

# Compute/augment exact out-of-sample R^2 in (0, 1) using matched test-set variances
if (!exists("generate_logic_data", mode = "function")) source("inst/benchmarks/bench_logic.R")
if (!exists("generate_kepler_data", mode = "function")) source("inst/benchmarks/bench_kepler.R")
if (!exists("generate_friedman1_data", mode = "function")) source("inst/benchmarks/bench_nonlinear.R")

add_test_r2_column <- function(df_bench, bench) {
  if ("test_r2" %in% names(df_bench) && all(is.finite(df_bench$test_r2))) {
    return(df_bench)
  }
  seeds_u <- sort(unique(df_bench$seed))
  var_y_map <- stats::setNames(numeric(length(seeds_u)), as.character(seeds_u))
  for (s in seeds_u) {
    yt <- switch(
      bench,
      logic       = generate_logic_data(n = 1000, n_test = 500, p = 20, sigma = 0.5, seed = s)$y_test,
      kepler      = generate_kepler_data(n = 400, n_test = 200, p = 10, sigma = 0.005, seed = s)$y_test,
      kepler_real = load_exoplanet_benchmark(seed = s)$y_test,
      nonlinear   = generate_friedman1_data(n = 500, n_test = 300, p = 10, sigma = 1.0, seed = s)$y_test
    )
    var_y_map[as.character(s)] <- mean((yt - mean(yt))^2)
  }
  vy <- var_y_map[as.character(df_bench$seed)]
  r2_raw <- pmin(pmax(1.0 - (df_bench$test_rmse^2) / vy, 0.0), 1.0)
  n_obs <- nrow(df_bench)
  # Smithson-Verkuilen transformation mapping [0, 1] -> (0, 1)
  df_bench$test_r2 <- (r2_raw * (n_obs - 1.0) + 0.5) / n_obs
  df_bench
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

  # 4. Count metrics (n_terms) -> Negative Binomial nbinom2(log), Poisson(log) limit fallback
  if (m_col == "n_terms") {
    sub_df$y_cnt <- pmax(1L, as.integer(round(sub_df$n_terms)))
    fam_used <- "NegBinom"
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
    if (!is_valid_tmb(fit)) {
      fam_used <- "Poisson"
      fit <- try_tmb(glmmTMB::glmmTMB(
        y_cnt ~ method + (1 | seed_f),
        family = stats::poisson(link = "log"),
        data = sub_df
      ))
    }
    if (!is_valid_tmb(fit)) {
      fam_used <- "Poisson"
      fit <- try_tmb(glmmTMB::glmmTMB(
        y_cnt ~ method,
        family = stats::poisson(link = "log"),
        data = sub_df
      ))
    }
    if (is_valid_tmb(fit)) {
      return(extract_fixef_treatment(fit, bench, m_col, fam_used, "log", ref_method))
    }
  }

  # 5. Out-of-sample R^2 in (0, 1) -> Beta GLMM with logit link
  if (m_col == "test_r2") {
    n_obs <- nrow(sub_df)
    r2_v <- pmin(pmax(sub_df$test_r2, 0.0), 1.0)
    if (any(r2_v <= 1e-6 | r2_v >= (1.0 - 1e-6))) {
      r2_v <- (r2_v * (n_obs - 1.0) + 0.5) / n_obs
    }
    sub_df$r2_val <- pmin(pmax(r2_v, 1e-5), 1.0 - 1e-5)

    fit <- try_tmb(glmmTMB::glmmTMB(
      r2_val ~ method + (1 | seed_f),
      dispformula = ~ method,
      family = glmmTMB::beta_family(link = "logit"),
      data = sub_df
    ))
    if (!is_valid_tmb(fit)) {
      fit <- try_tmb(glmmTMB::glmmTMB(
        r2_val ~ method + (1 | seed_f),
        family = glmmTMB::beta_family(link = "logit"),
        data = sub_df
      ))
    }
    if (!is_valid_tmb(fit)) {
      fit <- try_tmb(glmmTMB::glmmTMB(
        r2_val ~ method,
        family = glmmTMB::beta_family(link = "logit"),
        data = sub_df
      ))
    }
    if (is_valid_tmb(fit)) {
      return(extract_fixef_treatment(fit, bench, m_col, "Beta", "logit", ref_method))
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
# Run Per-Benchmark GLMMs & Summaries (All 5 Benchmarks)
# ==============================================================================
benchmarks <- c("logic", "kepler", "kepler_real", "nonlinear", "sindy_3d")
bench_metrics <- list(
  logic       = c("clause_tpr", "clause_fpr", "exact_recovery", "var_tpr", "var_fpr", "test_rmse", "test_r2", "n_terms"),
  kepler      = c("structural_recovery", "exact_law", "exp_err", "var_tpr", "var_fpr", "n_terms", "test_rmse", "test_r2"),
  kepler_real = c("structural_recovery", "exact_law", "exp_err", "var_tpr", "var_fpr", "n_terms", "test_rmse", "test_r2"),
  nonlinear   = c("var_tpr", "var_fpr", "exact_var_recovery", "n_terms", "test_rmse", "test_r2"),
  sindy_3d    = c("structural_recovery", "exact_recovery", "var_tpr", "var_fpr", "n_terms", "test_rmse", "test_r2")
)

all_raw_list <- list()
all_glmm_list <- list()
for (b in benchmarks) {
  raw_path <- file.path("results_baselines", b, sprintf("raw_%s_baselines.csv", b))
  if (!file.exists(raw_path)) {
    stop("Missing benchmark CSV: ", raw_path)
  }
  df_b <- utils::read.csv(raw_path, stringsAsFactors = FALSE)
  stopifnot(nrow(df_b) > 0L)
  df_b <- df_b[df_b$method != "Symbolic-LBBNN (3-Layer)", , drop = FALSE]
  df_b <- add_test_r2_column(df_b, b)
  utils::write.csv(df_b, raw_path, row.names = FALSE)
  all_raw_list[[b]] <- df_b

  df_b_fit <- df_b
  f3_path <- "results_baselines/nonlinear/raw_friedman_3layer.csv"
  if (b == "nonlinear" && file.exists(f3_path)) {
    d_3l <- utils::read.csv(f3_path, stringsAsFactors = FALSE)
    d_3l_mapped <- data.frame(
      seed = d_3l$seed,
      benchmark = "nonlinear",
      method = "Symbolic-LBBNN (3-Layer)",
      var_tpr = d_3l$tpr,
      var_fpr = d_3l$fpr,
      exact_var_recovery = d_3l$exact_var_recovery,
      test_rmse = d_3l$test_rmse,
      n_terms = d_3l$active_edges,
      formula = d_3l$formula,
      test_r2 = d_3l$test_r2,
      stringsAsFactors = FALSE
    )
    df_b_fit <- rbind(df_b, d_3l_mapped)
  }

  df_ci <- summarize_with_ci(df_b_fit, b)
  utils::write.csv(df_ci, file.path("results_baselines", b, sprintf("summary_%s_ci.csv", b)), row.names = FALSE)

  m_res <- lapply(bench_metrics[[b]], function(mc) fit_single_metric_glmm(df_b_fit, b, mc))
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
  all_glmm_list[[b]] <- df_glmm
  message(sprintf("[GLMM] Completed '%s' (%d contrast rows).", b, nrow(df_glmm)))
}

# ==============================================================================
# Balanced Joint Cross-Benchmark Hierarchical GLMM (Panels A & B)
# ==============================================================================
map_method_family <- function(m_vec) {
  m_vec <- ifelse(m_vec %in% c("FBMS-BLR", "FBMS-BGNLM"), "FBMS", m_vec)
  m_vec <- ifelse(m_vec %in% c("Lasso-Interactions", "Log-Lasso", "Lasso"), "Lasso", m_vec)
  m_vec
}

sym_real_rmse <- mean(all_raw_list[["kepler_real"]]$test_rmse[all_raw_list[["kepler_real"]]$method == "Symbolic-LBBNN"])
oracle_scales <- c(logic = 0.500, kepler = 0.005, kepler_real = sym_real_rmse, nonlinear = 1.000, sindy_3d = 0.050)

build_joint_df <- function(bench_sub, allowed_methods = NULL) {
  joint_rows <- list()
  for (b in bench_sub) {
    d <- all_raw_list[[b]]
    exact_goal <- switch(
      b,
      logic       = as.integer(d$exact_recovery),
      kepler      = as.integer(d$exact_law),
      kepler_real = as.integer(d$exact_law),
      nonlinear   = as.integer(d$exact_var_recovery),
      sindy_3d    = as.integer(d$exact_recovery)
    )
    exact_sup <- switch(
      b,
      logic       = as.integer(d$exact_recovery),
      kepler      = as.integer(d$structural_recovery),
      kepler_real = as.integer(d$structural_recovery),
      nonlinear   = as.integer(d$exact_var_recovery),
      sindy_3d    = as.integer(d$structural_recovery)
    )
    n_tpr <- get_prop_trials(b, "var_tpr", d$method)
    k_tpr <- pmin(n_tpr, pmax(0L, as.integer(round(d$var_tpr * n_tpr))))
    n_fpr <- get_prop_trials(b, "var_fpr", d$method)
    k_spec <- pmin(n_fpr, pmax(0L, as.integer(round((1.0 - d$var_fpr) * n_fpr))))

    m_fam <- map_method_family(d$method)
    keep <- if (is.null(allowed_methods)) rep(TRUE, length(m_fam)) else (m_fam %in% allowed_methods)

    joint_rows[[b]] <- data.frame(
      dataset       = b,
      seed          = d$seed[keep],
      method        = m_fam[keep],
      test_rmse     = pmax(d$test_rmse[keep], 1e-8),
      rel_rmse      = pmax(d$test_rmse[keep] / oracle_scales[[b]], 1e-6),
      test_r2       = d$test_r2[keep],
      exact_goal    = exact_goal[keep],
      exact_support = exact_sup[keep],
      k_tpr         = k_tpr[keep],
      n_tpr         = n_tpr[keep],
      k_spec        = k_spec[keep],
      n_spec        = n_fpr[keep],
      n_terms       = pmax(1L, as.integer(round(d$n_terms[keep]))),
      stringsAsFactors = FALSE
    )
  }
  df_j <- do.call(rbind, joint_rows)
  df_j$dataset   <- factor(df_j$dataset, levels = bench_sub)
  df_j$method    <- stats::relevel(factor(df_j$method), ref = "Symbolic-LBBNN")
  df_j$ds_method <- interaction(df_j$dataset, df_j$method, drop = TRUE)
  df_j$ds_seed   <- interaction(df_j$dataset, df_j$seed, drop = TRUE)
  n_j <- nrow(df_j)
  r2_j <- pmin(pmax(df_j$test_r2, 0.0), 1.0)
  df_j$r2_sv <- (r2_j * (n_j - 1.0) + 0.5) / n_j
  df_j
}

extract_var_comp <- function(fit, metric_name, fam_type) {
  vc <- glmmTMB::VarCorr(fit)$cond
  v_ds   <- as.numeric(attr(vc[["dataset"]],   "stddev"))[1]^2
  v_dsm  <- as.numeric(attr(vc[["ds_method"]], "stddev"))[1]^2
  v_seed <- as.numeric(attr(vc[["ds_seed"]],   "stddev"))[1]^2
  if (fam_type == "binomial") {
    v_obs <- (pi^2) / 3.0
  } else if (fam_type == "beta") {
    v_obs <- 1.0
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

fit_balanced_joint_suite <- function(df_joint, label_prefix) {
  fit_j_rmse <- glmmTMB::glmmTMB(
    rel_rmse ~ method + (1 | dataset) + (1 | ds_method) + (1 | ds_seed),
    dispformula = ~ dataset,
    family = stats::Gamma(link = "log"),
    data = df_joint
  )
  stopifnot(is_valid_tmb(fit_j_rmse))

  fit_j_r2 <- try_tmb(glmmTMB::glmmTMB(
    r2_sv ~ method + (1 | dataset) + (1 | ds_method) + (1 | ds_seed),
    dispformula = ~ dataset,
    family = glmmTMB::beta_family(link = "logit"),
    data = df_joint
  ))
  if (!is_valid_tmb(fit_j_r2)) {
    fit_j_r2 <- glmmTMB::glmmTMB(
      r2_sv ~ method + (1 | dataset) + (1 | ds_method) + (1 | ds_seed),
      family = glmmTMB::beta_family(link = "logit"),
      data = df_joint
    )
  }
  stopifnot(is_valid_tmb(fit_j_r2))

  fit_j_goal <- glmmTMB::glmmTMB(
    exact_goal ~ 0 + method + (1 | dataset) + (1 | ds_method) + (1 | ds_seed),
    family = stats::binomial(link = "logit"),
    priors = p_sym_binom,
    data = df_joint
  )
  stopifnot(is_valid_tmb(fit_j_goal))

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

  fe <- rbind(
    extract_fixef_treatment(fit_j_rmse, label_prefix, "rel_rmse",   "Gamma",    "log"),
    extract_fixef_treatment(fit_j_r2,   label_prefix, "test_r2",    "Beta",     "logit"),
    extract_fixef_cellmeans(fit_j_goal, label_prefix, "exact_goal", "Binomial", "logit"),
    extract_fixef_cellmeans(fit_j_tpr,  label_prefix, "var_tpr",    "Binomial", "logit"),
    extract_fixef_cellmeans(fit_j_spec, label_prefix, "var_spec",   "Binomial", "logit")
  )
  fe$p_adj <- NA_real_
  for (mc in unique(fe$metric)) {
    idx_nr <- which(fe$metric == mc & !fe$is_ref)
    fe$p_adj[idx_nr] <- stats::p.adjust(fe$p_value[idx_nr], method = "bonferroni")
  }
  fe$sig_star <- ifelse(fe$is_ref, "Ref", format_sig_star(fe$p_value))
  fe$sig_star_adj <- ifelse(fe$is_ref, "Ref", format_sig_star(fe$p_adj))

  vc <- rbind(
    extract_var_comp(fit_j_rmse, "rel_rmse",   "gamma"),
    extract_var_comp(fit_j_r2,   "test_r2",    "beta"),
    extract_var_comp(fit_j_goal, "exact_goal", "binomial"),
    extract_var_comp(fit_j_tpr,  "var_tpr",    "binomial"),
    extract_var_comp(fit_j_spec, "var_spec",   "binomial")
  )
  list(fe = fe, vc = vc)
}

# Panel A: Strictly balanced 5 core methods across ALL 5 benchmarks (Tasks 1-5 x 100 seeds)
CORE_ALL_METHODS <- c("Symbolic-LBBNN", "FBMS", "Lasso", "ReLU-LBBNN", "RandomForest")
df_joint_all <- build_joint_df(benchmarks, allowed_methods = CORE_ALL_METHODS)
res_joint_all <- fit_balanced_joint_suite(df_joint_all, "joint_balanced_all5")
utils::write.csv(res_joint_all$fe, "results_baselines/joint_glmm_contrasts.csv", row.names = FALSE)
utils::write.csv(res_joint_all$vc, "results_baselines/joint_glmm_variance_components.csv", row.names = FALSE)

# Panel B: Strictly balanced 9 continuous methods across ALL 4 continuous benchmarks (Tasks 2-5 x 100 seeds)
CONT_BENCHMARKS <- c("kepler", "kepler_real", "nonlinear", "sindy_3d")
CONT_METHODS <- c("Symbolic-LBBNN", "FBMS", "MARS", "SR-Operon", "SR-PySR", "GAM", "Lasso", "ReLU-LBBNN", "RandomForest")
df_joint_cont <- build_joint_df(CONT_BENCHMARKS, allowed_methods = CONT_METHODS)
res_joint_cont <- fit_balanced_joint_suite(df_joint_cont, "joint_balanced_cont4")
utils::write.csv(res_joint_cont$fe, "results_baselines/joint_glmm_continuous_contrasts.csv", row.names = FALSE)
utils::write.csv(res_joint_cont$vc, "results_baselines/joint_glmm_continuous_variance_components.csv", row.names = FALSE)

# ==============================================================================
# Programmatic Appendix D: Full GLMM Regression Tables for ALL Metrics (Tables D.1-D.5)
# ==============================================================================
write_appendix_d_tex <- function(glmm_list, out_tex = "paper/appendix_d_glmm_tables.tex") {
  tex_esc <- function(s) {
    s <- gsub("_", "\\_", s, fixed = TRUE)
    s
  }
  fmt_p <- function(p) {
    ifelse(is.na(p), "---",
           ifelse(p < 0.001, "$<.001^{***}$",
                  ifelse(p < 0.01, sprintf("$%.3f^{**}$", p),
                         ifelse(p < 0.05, sprintf("$%.3f^{*}$", p), sprintf("$%.3f^{\\text{ns}}$", p)))))
  }
  fmt_ratio <- function(r, fam) {
    if (is.na(r)) return("---")
    if (r > 999) return("$>999$")
    if (r < 0.001) return("$<.001$")
    if (fam %in% c("Binomial", "Beta")) return(sprintf("%.3f", r))
    sprintf("$%.3f\\times$", r)
  }

  bench_titles <- list(
    logic       = c("D.1", "Benchmark 1 (Bayesian Logic Regression, $M = 100$ seeds)", "tab:glmm_app_logic"),
    kepler      = c("D.2", "Benchmark 2 (Synthetic Kepler's Third Law, $M = 100$ seeds)", "tab:glmm_app_kepler"),
    kepler_real = c("D.3", "Benchmark 3 (NASA Exoplanet Archive, $n = 939$, $M = 100$ splits)", "tab:glmm_app_exoplanet"),
    nonlinear   = c("D.4", "Benchmark 4 (Friedman \\#1 Nonlinear Benchmark, $M = 100$ seeds)", "tab:glmm_app_friedman"),
    sindy_3d    = c("D.5", "Benchmark 5 (Multivariate 3D R\\\"ossler--Lorenz SINDy, $M = 100$ seeds)", "tab:glmm_app_sindy")
  )

  metric_labels <- c(
    clause_tpr          = "Clause TPR (\\texttt{Binomial}, logit)",
    clause_fpr          = "Clause FPR (\\texttt{Binomial}, logit)",
    exact_recovery      = "Exact System Match (\\texttt{Binomial}, logit)",
    structural_recovery = "Support Recovery (\\texttt{Binomial}, logit)",
    exact_law           = "Exact Rational Law (\\texttt{Binomial}, logit)",
    exact_var_recovery  = "Exact Variable Recovery (\\texttt{Binomial}, logit)",
    var_tpr             = "Variable TPR (\\texttt{Binomial}, logit)",
    var_fpr             = "Variable FPR (\\texttt{Binomial}, logit)",
    exp_err             = "$L_1$ Exponent Error (\\texttt{Gamma}, log)",
    n_terms             = "Active Terms (\\texttt{NegBinom}, log)",
    test_rmse           = "Test RMSE (\\texttt{Gamma}, log)",
    test_r2             = "Out-of-Sample $R^2$ (\\texttt{Beta}, logit)"
  )

  lines <- c(
    "\\section{Complete Generalized Linear Mixed-Effects Regression Tables}",
    "\\label{app:glmm_full}",
    "",
    "This appendix reports the complete \\texttt{glmmTMB} fixed-effect estimates ($\\hat{\\beta}_0$ for \\texttt{Symbolic-LBBNN} as reference, contrast $\\hat{\\beta}_m$ with $95\\%$ Wald confidence intervals), exponentiated effect multipliers ($e^{\\hat{\\beta}_m}$ representing multiplicative mean ratios for \\texttt{Gamma}/\\texttt{NegBinom}/\\texttt{Poisson} and odds ratios for \\texttt{Binomial}/\\texttt{Beta}), Wald $z$-statistics, two-sided $p$-values, and matched seed random-effect standard deviations ($\\hat{\\sigma}_{\\text{seed}}$) across every metric and all five benchmarks in Tables~\\ref{tab:glmm_app_logic}--\\ref{tab:glmm_app_sindy}. Out-of-sample $R^2$ is modeled via Beta GLMM with logit link after Smithson--Verkuilen boundary regularization.",
    ""
  )

  for (b in names(glmm_list)) {
    df_g <- glmm_list[[b]]
    info <- bench_titles[[b]]
    need_compact <- (nrow(df_g) > 55L)
    very_tall <- (nrow(df_g) > 64L)
    resize_w <- if (very_tall) "0.88\\textwidth" else if (need_compact) "0.93\\textwidth" else "\\textwidth"
    arr_str  <- if (very_tall) "0.80" else if (need_compact) "0.85" else "1.0"
    lines <- c(
      lines,
      "\\begin{table}[htbp]",
      "\\centering",
      sprintf("\\caption{Full GLMM regression contrasts on %s.}", info[2]),
      sprintf("\\label{%s}", info[3]),
      if (need_compact) sprintf("{\\renewcommand{\\arraystretch}{%s}", arr_str) else "{",
      sprintf("\\resizebox{%s}{!}{%%", resize_w),
      "\\begin{tabular}{llcccccc}",
      "\\toprule",
      "Metric (Family, Link) & Method & $\\hat{\\beta}_0$ / $\\hat{\\beta}_m$ [95\\% CI] & $e^{\\hat{\\beta}_m}$ & Wald $z$ & $p$-value & $\\hat{\\sigma}_{\\text{seed}}$ \\\\",
      "\\midrule"
    )
    mets <- unique(df_g$metric)
    for (mi in seq_along(mets)) {
      mc <- mets[mi]
      sub_m <- df_g[df_g$metric == mc, , drop = FALSE]
      fam_tag <- as.character(sub_m$family[1])
      m_lbl <- if (mc == "n_terms") {
        sprintf("Active Terms (\\texttt{%s}, log)", fam_tag)
      } else if (mc %in% names(metric_labels)) {
        metric_labels[[mc]]
      } else {
        tex_esc(mc)
      }
      if (mi > 1L) lines <- c(lines, "\\midrule")
      for (r_i in seq_len(nrow(sub_m))) {
        row <- sub_m[r_i, ]
        col1 <- if (r_i == 1L) m_lbl else ""
        meth_s <- sprintf("\\texttt{%s}%s", tex_esc(row$term), ifelse(row$is_ref, " (Ref)", ""))
        beta_s <- sprintf("$%+.3f$ [$%+.3f, %+.3f$]", row$estimate, row$ci_lo, row$ci_hi)
        rat_s  <- if (row$is_ref) "Ref" else fmt_ratio(row$ratio_or, row$family)
        z_s    <- ifelse(is.na(row$z_value), "---", sprintf("$%+.2f$", row$z_value))
        p_s    <- ifelse(row$is_ref, "---", fmt_p(row$p_value))
        sd_s   <- if (r_i == 1L) ifelse(is.na(row$seed_sd), "$<0.001$", sprintf("%.3f", row$seed_sd)) else ""
        lines <- c(lines, sprintf("%s & %s & %s & %s & %s & %s & %s \\\\", col1, meth_s, beta_s, rat_s, z_s, p_s, sd_s))
      }
    }
    lines <- c(
      lines,
      "\\bottomrule",
      "\\end{tabular}%",
      "}",
      "}",
      "\\end{table}",
      ""
    )
  }
  writeLines(lines, out_tex)
  message(sprintf("[GLMM] Wrote Appendix D LaTeX tables (%d lines) to %s.", length(lines), out_tex))
}
write_appendix_d_tex(all_glmm_list)

message("\n=== BALANCED PANEL A: JOINT GLMM (ALL 5 BENCHMARKS x 100 SEEDS, 5 CORE METHODS) ===")
print(res_joint_all$fe[res_joint_all$fe$metric %in% c("rel_rmse", "test_r2", "exact_goal"),
                       c("metric", "term", "estimate", "ci_lo", "ci_hi", "ratio_or", "p_value", "sig_star")], digits = 4)

message("\n=== BALANCED PANEL B: JOINT GLMM (CONTINUOUS TASKS 2-5 x 100 SEEDS, 9 METHODS) ===")
print(res_joint_cont$fe[res_joint_cont$fe$metric %in% c("rel_rmse", "test_r2", "exact_goal"),
                        c("metric", "term", "estimate", "ci_lo", "ci_hi", "ratio_or", "p_value", "sig_star")], digits = 4)

message("\n[OK] All per-benchmark GLMMs (including Beta R2 & 3D SINDy), Balanced Panels A/B, and Appendix D tables complete!")
