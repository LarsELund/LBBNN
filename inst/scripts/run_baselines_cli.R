#!/usr/bin/env Rscript
# =============================================================================
# inst/scripts/run_baselines_cli.R
# Multi-Core CLI Runner & Mixed-Effects Statistical Evaluator for Baselines
# Usage:
#   Rscript inst/scripts/run_baselines_cli.R --benchmark all --seeds 1:100 --workers 8
# =============================================================================

suppressPackageStartupMessages({
  library(parallel)
})

# Zero-dependency base-R CLI parser (supports --key value and --key=value)
args <- commandArgs(trailingOnly = TRUE)
opt <- list(
  benchmark = "all",
  seeds = "1:100",
  workers = 8L,
  out_dir = "results_baselines",
  overnight_dir = "results_overnight",
  resume = TRUE
)
i <- 1L
while (i <= length(args)) {
  a <- args[i]
  if (grepl("^--", a)) {
    if (grepl("=", a, fixed = TRUE)) {
      kv <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1]]
      k <- kv[1]; v <- kv[2]
    } else {
      k <- sub("^--", "", a)
      v <- if (i < length(args)) args[i + 1L] else "TRUE"
      i <- i + 1L
    }
    if (k %in% c("b", "benchmark")) opt$benchmark <- v
    else if (k %in% c("s", "seeds")) opt$seeds <- v
    else if (k %in% c("w", "workers")) opt$workers <- as.integer(v)
    else if (k %in% c("o", "out_dir")) opt$out_dir <- v
    else if (k %in% c("d", "overnight_dir")) opt$overnight_dir <- v
    else if (k %in% c("r", "resume")) opt$resume <- as.logical(v)
  }
  i <- i + 1L
}

parse_seeds <- function(s_str) {
  if (grepl(":", s_str, fixed = TRUE)) {
    parts <- as.integer(strsplit(s_str, ":", fixed = TRUE)[[1]])
    seq.int(parts[1], parts[2])
  } else {
    as.integer(strsplit(s_str, ",", fixed = TRUE)[[1]])
  }
}

seeds <- parse_seeds(opt$seeds)
benchmarks <- if (opt$benchmark == "all") {
  c("logic", "kepler", "kepler_real", "nonlinear")
} else {
  strsplit(opt$benchmark, ",", fixed = TRUE)[[1]]
}

source("inst/benchmarks/bench_baselines.R")

fit_lmm_contrasts <- function(df_bench, metrics_vec, ref_method = "Symbolic-LBBNN") {
  df_bench$method <- factor(df_bench$method)
  if (ref_method %in% levels(df_bench$method)) {
    df_bench$method <- stats::relevel(df_bench$method, ref = ref_method)
  }
  df_bench$seed_f <- factor(df_bench$seed)

  out_rows <- list()
  for (m_col in metrics_vec) {
    if (!m_col %in% names(df_bench)) next
    sub_df <- df_bench[is.finite(df_bench[[m_col]]), , drop = FALSE]
    if (nrow(sub_df) < 10L || length(unique(sub_df$method)) < 2L) next
    fit <- NULL
    if (requireNamespace("lmerTest", quietly = TRUE)) {
      form <- stats::as.formula(paste0(m_col, " ~ method + (1 | seed_f)"))
      fit <- tryCatch(suppressWarnings(lmerTest::lmer(form, data = sub_df, REML = TRUE)), error = function(e) NULL)
    }
    if (is.null(fit)) {
      # Balanced repeated-measures block model (mathematically identical fixed-effect contrasts to random-intercept LMM)
      form_lm <- if (length(unique(sub_df$seed_f)) > 1L) {
        stats::as.formula(paste0(m_col, " ~ method + seed_f"))
      } else {
        stats::as.formula(paste0(m_col, " ~ method"))
      }
      fit <- tryCatch(stats::lm(form_lm, data = sub_df), error = function(e) NULL)
    }
    if (is.null(fit)) next
    coef_mat <- summary(fit)$coefficients
    for (r_nm in rownames(coef_mat)) {
      if (grepl("^seed_f", r_nm)) next
      if (r_nm == "(Intercept)") {
        m_label <- paste0(ref_method, " (Ref Intercept)")
      } else {
        m_label <- sub("^method", "", r_nm)
      }
      est <- coef_mat[r_nm, "Estimate"]
      se <- coef_mat[r_nm, "Std. Error"]
      t_val <- if ("t value" %in% colnames(coef_mat)) coef_mat[r_nm, "t value"] else NA_real_
      p_val <- if ("Pr(>|t|)" %in% colnames(coef_mat)) coef_mat[r_nm, "Pr(>|t|)"] else NA_real_
      out_rows[[length(out_rows) + 1L]] <- data.frame(
        benchmark = unique(df_bench$benchmark)[1],
        metric = m_col,
        term = m_label,
        estimate = est,
        std_error = se,
        t_value = t_val,
        p_value = p_val,
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(out_rows) == 0L) return(data.frame())
  do.call(rbind, out_rows)
}

summarize_benchmark_df <- function(df_bench) {
  num_cols <- setdiff(names(df_bench)[vapply(df_bench, is.numeric, logical(1))], "seed")
  methods <- unique(df_bench$method)
  sum_rows <- list()
  for (m in methods) {
    sub_m <- df_bench[df_bench$method == m, , drop = FALSE]
    row_l <- list(
      benchmark = unique(df_bench$benchmark)[1],
      method = m,
      n_seeds = nrow(sub_m)
    )
    for (nc in num_cols) {
      vals <- sub_m[[nc]]
      vals <- vals[is.finite(vals)]
      row_l[[paste0(nc, "_mean")]] <- if (length(vals) > 0L) mean(vals) else NA_real_
      row_l[[paste0(nc, "_sd")]] <- if (length(vals) > 1L) stats::sd(vals) else NA_real_
    }
    sum_rows[[length(sum_rows) + 1L]] <- as.data.frame(row_l, stringsAsFactors = FALSE)
  }
  do.call(rbind, sum_rows)
}

for (b in benchmarks) {
  b_dir <- file.path(opt$out_dir, b)
  ckpt_dir <- file.path(b_dir, "checkpoints")
  dir.create(ckpt_dir, recursive = TRUE, showWarnings = FALSE)

  message(sprintf("\n[BASELINES] Running benchmark '%s' across %d seeds (%d workers)...",
                  b, length(seeds), opt$workers))
  t0 <- Sys.time()

  res_list <- parallel::mclapply(seeds, function(s) {
    ckpt_file <- file.path(ckpt_dir, sprintf("seed_%04d.rds", s))
    if (isTRUE(opt$resume) && file.exists(ckpt_file)) {
      return(readRDS(ckpt_file))
    }
    res_s <- tryCatch({
      if (b == "logic") {
        run_baselines_logic_seed(seed = s, overnight_dir = opt$overnight_dir)
      } else if (b == "kepler") {
        run_baselines_kepler_seed(seed = s, mode = "synthetic", overnight_dir = opt$overnight_dir)
      } else if (b == "kepler_real") {
        run_baselines_kepler_seed(seed = s, mode = "real", overnight_dir = opt$overnight_dir)
      } else if (b == "nonlinear") {
        run_baselines_nonlinear_seed(seed = s, overnight_dir = opt$overnight_dir)
      } else {
        stop("Unknown benchmark: ", b)
      }
    }, error = function(e) {
      message(sprintf("[WARN] Benchmark %s seed %d failed: %s", b, s, conditionMessage(e)))
      NULL
    })
    if (!is.null(res_s)) {
      saveRDS(res_s, ckpt_file)
    }
    res_s
  }, mc.cores = max(1L, opt$workers))

  valid_res <- Filter(Negate(is.null), res_list)
  if (length(valid_res) > 0L) {
    df_all <- do.call(rbind, valid_res)
    raw_csv <- file.path(b_dir, paste0("raw_", b, "_baselines.csv"))
    utils::write.csv(df_all, raw_csv, row.names = FALSE)

    df_sum <- summarize_benchmark_df(df_all)
    sum_csv <- file.path(b_dir, paste0("summary_", b, "_baselines.csv"))
    utils::write.csv(df_sum, sum_csv, row.names = FALSE)

    metrics_b <- switch(
      b,
      "logic" = c("clause_tpr", "clause_fpr", "exact_recovery", "var_tpr", "var_fpr", "test_rmse", "n_terms"),
      "kepler" = c("structural_recovery", "exact_law", "exp_err", "var_tpr", "var_fpr", "test_rmse", "n_terms"),
      "kepler_real" = c("structural_recovery", "exact_law", "exp_err", "var_tpr", "var_fpr", "test_rmse", "n_terms"),
      "nonlinear" = c("var_tpr", "var_fpr", "exact_var_recovery", "test_rmse", "n_terms")
    )
    df_lmm <- fit_lmm_contrasts(df_all, metrics_b, ref_method = "Symbolic-LBBNN")
    if (nrow(df_lmm) > 0L) {
      lmm_csv <- file.path(b_dir, paste0("lmm_", b, "_contrasts.csv"))
      utils::write.csv(df_lmm, lmm_csv, row.names = FALSE)
    }

    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    message(sprintf("[OK] Completed '%s' (%d seeds) in %.1f sec -> %s", b, length(valid_res), elapsed, sum_csv))
    print(df_sum)
  }
}
