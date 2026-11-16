#!/usr/bin/env Rscript
# =============================================================================
# inst/scripts/run_remote_fbms_update.R
# Fast Multicore JCS-Protocol Runner & Prior Explorer for FBMS-BLR / FBMS-BGNLM.
# Uses native BMA out-of-sample predictions (predict() + predmean()) and
# gmjmcmc.parallel across cluster cores.
#
# Examples:
#   # Update main baseline CSVs using a single prior (default or explicit key):
#   Rscript inst/scripts/run_remote_fbms_update.R \
#     --benchmark all --seeds 1:100 --fbms_cores 40 --fbms_runs 40 --fbms_P 25
#
#   # Explore all Gaussian beta priors (g-prior, Jeffreys-BIC known/unknown var,
#   # ZS-null, ZS-full, EB-local, hyper-g-laplace, robust, intrinsic) on cluster:
#   Rscript inst/scripts/run_remote_fbms_update.R \
#     --benchmark all --seeds 1:100 --priors explore --fbms_cores 40 --fbms_runs 40 --fbms_P 30
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)

if (any(args %in% c("-h", "--help"))) {
  cat(paste0(
    "Usage: Rscript inst/scripts/run_remote_fbms_update.R [OPTIONS]\n",
    "Options:\n",
    "  --benchmark    all | comma list of logic,kepler,kepler_real,nonlinear (default: all)\n",
    "  --seeds        range or comma list, e.g. 1:100 or 1:20 (default: 1:100)\n",
    "  --priors       default | explore | comma list of prior keys:\n",
    "                   g_prior_def, g_prior_n, jbic_unknown, jbic_known_var,\n",
    "                   zs_null, zs_full, eb_local, hyper_g_laplace, robust, intrinsic\n",
    "  --merge_prior  prior key to merge into raw_<bench>_baselines.csv (or 'best_rmse' / 'none')\n",
    "                 default: 'default' when --priors=default, 'none' when exploring multiple priors\n",
    "  --fbms_cores     cores passed to gmjmcmc.parallel per seed (default: min(40, detectCores - 1))\n",
    "  --fbms_runs      independent parallel chains per seed (default: 40)\n",
    "  --fbms_P         population count/budget P for GMJMCMC (default: 25)\n",
    "  --fbms_N         iterations N per run (default: NULL -> task default 200/250/150)\n",
    "  --outer_workers  concurrent seeds to run in parallel (default: 1)\n",
    "  --out_dir        output root directory (default: results_baselines)\n",
    "  --resume         TRUE/FALSE resume from existing checkpoints (default: TRUE)\n"
  ))
  quit(save = "no", status = 0L)
}

default_cores <- tryCatch(
  max(1L, min(40L, parallel::detectCores(logical = TRUE) - 1L)),
  error = function(e) 1L
)

opt <- list(
  benchmark = "all",
  seeds = "1:100",
  priors = "default",
  merge_prior = NULL,
  fbms_cores = default_cores,
  fbms_runs = 40L,
  fbms_P = 25L,
  fbms_N = NULL,
  outer_workers = 1L,
  out_dir = "results_baselines",
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
    else if (k %in% c("p", "priors")) opt$priors <- v
    else if (k == "merge_prior") opt$merge_prior <- v
    else if (k == "fbms_cores") opt$fbms_cores <- as.integer(v)
    else if (k == "fbms_runs") opt$fbms_runs <- as.integer(v)
    else if (k == "fbms_P") opt$fbms_P <- as.integer(v)
    else if (k == "fbms_N") opt$fbms_N <- if (toupper(v) %in% c("NULL", "NA", "")) NULL else as.integer(v)
    else if (k %in% c("w", "outer_workers")) opt$outer_workers <- max(1L, as.integer(v))
    else if (k %in% c("o", "out_dir")) opt$out_dir <- v
    else if (k %in% c("r", "resume")) opt$resume <- as.logical(v)
  }
  i <- i + 1L
}

if (is.null(opt$merge_prior)) {
  opt$merge_prior <- if (opt$priors == "default") "default" else "none"
}

parse_seeds <- function(s_str) {
  out <- integer(0)
  for (part in strsplit(s_str, ",", fixed = TRUE)[[1]]) {
    part <- trimws(part)
    if (!nzchar(part)) next
    if (grepl(":", part, fixed = TRUE)) {
      bounds <- as.integer(strsplit(part, ":", fixed = TRUE)[[1]])
      out <- c(out, seq.int(bounds[1], bounds[2]))
    } else {
      out <- c(out, as.integer(part))
    }
  }
  unique(out)
}

seeds <- parse_seeds(opt$seeds)
benchmarks <- if (opt$benchmark == "all") {
  c("logic", "kepler", "kepler_real", "nonlinear")
} else {
  trimws(strsplit(opt$benchmark, ",", fixed = TRUE)[[1]])
}

# Match run_baselines_cli.R summary schema (grouped by group_col: "method" or "prior_key")
summarize_grouped_df <- function(df_bench, group_col = "method") {
  groups <- unique(df_bench[[group_col]])
  num_cols <- names(df_bench)[vapply(df_bench, is.numeric, logical(1))]
  num_cols <- setdiff(num_cols, "seed")
  sum_rows <- list()
  for (g in groups) {
    sub_g <- df_bench[df_bench[[group_col]] == g, , drop = FALSE]
    row_l <- list(
      benchmark = unique(df_bench$benchmark)[1],
      method = unique(sub_g$method)[1]
    )
    row_l[[group_col]] <- g
    row_l$n_seeds <- nrow(sub_g)
    for (nc in num_cols) {
      vals <- sub_g[[nc]]
      vals <- vals[is.finite(vals)]
      row_l[[paste0(nc, "_mean")]] <- if (length(vals) > 0L) mean(vals) else NA_real_
      row_l[[paste0(nc, "_sd")]] <- if (length(vals) > 1L) stats::sd(vals) else NA_real_
    }
    df_row <- as.data.frame(row_l, stringsAsFactors = FALSE)
    # Remove duplicate method column when group_col == "method"
    df_row <- df_row[, !duplicated(names(df_row)), drop = FALSE]
    sum_rows[[length(sum_rows) + 1L]] <- df_row
  }
  do.call(rbind, sum_rows)
}

write_csv_atomic <- function(df, path) {
  tmp <- paste0(path, ".tmp_", Sys.getpid())
  utils::write.csv(df, tmp, row.names = FALSE)
  if (!file.rename(tmp, path)) {
    file.copy(tmp, path, overwrite = TRUE)
    unlink(tmp)
  }
}

save_rds_atomic <- function(obj, path) {
  tmp <- paste0(path, ".tmp_", Sys.getpid())
  saveRDS(obj, tmp)
  if (!file.rename(tmp, path)) {
    file.copy(tmp, path, overwrite = TRUE)
    unlink(tmp)
  }
}

replace_fbms_in_df <- function(df_existing, df_fbms, fbms_method) {
  if (is.null(df_existing) || nrow(df_existing) == 0L) {
    return(df_fbms)
  }
  orig_order <- unique(df_existing$method)
  if (!(fbms_method %in% orig_order)) orig_order <- c(orig_order, fbms_method)

  missing_cols <- setdiff(names(df_existing), names(df_fbms))
  for (mc in missing_cols) df_fbms[[mc]] <- NA
  df_fbms <- df_fbms[, names(df_existing), drop = FALSE]

  keep_mask <- !(df_existing$method == fbms_method & df_existing$seed %in% df_fbms$seed)
  merged <- rbind(df_existing[keep_mask, , drop = FALSE], df_fbms)
  m_ord <- match(merged$method, orig_order)
  m_ord[is.na(m_ord)] <- length(orig_order) + 1L
  merged <- merged[order(merged$seed, m_ord), , drop = FALSE]
  rownames(merged) <- NULL
  merged
}

merge_fbms_rows_into_baselines <- function(b_dir, b, fbms_method, df_fbms_all) {
  base_ckpt_dir <- file.path(b_dir, "checkpoints")
  for (r_i in seq_len(nrow(df_fbms_all))) {
    row_s <- df_fbms_all[r_i, , drop = FALSE]
    s <- row_s$seed[1]
    base_ckpt_file <- file.path(base_ckpt_dir, sprintf("seed_%04d.rds", s))
    if (file.exists(base_ckpt_file)) {
      df_seed <- tryCatch(readRDS(base_ckpt_file), error = function(e) NULL)
      if (!is.null(df_seed) && is.data.frame(df_seed)) {
        df_seed_updated <- replace_fbms_in_df(df_seed, row_s, fbms_method)
        save_rds_atomic(df_seed_updated, base_ckpt_file)
      }
    }
  }

  raw_csv <- file.path(b_dir, paste0("raw_", b, "_baselines.csv"))
  if (file.exists(raw_csv)) {
    bak_csv <- paste0(raw_csv, ".pre_fbms_jcs.bak")
    if (!file.exists(bak_csv)) {
      file.copy(raw_csv, bak_csv)
      message("[BACKUP] Saved original raw CSV to ", bak_csv)
    }
    df_existing <- utils::read.csv(raw_csv, stringsAsFactors = FALSE)
    df_merged <- replace_fbms_in_df(df_existing, df_fbms_all, fbms_method)
  } else {
    df_merged <- df_fbms_all
  }

  write_csv_atomic(df_merged, raw_csv)
  df_sum <- summarize_grouped_df(df_merged, group_col = "method")
  sum_csv <- file.path(b_dir, paste0("summary_", b, "_baselines.csv"))
  write_csv_atomic(df_sum, sum_csv)
  list(raw_csv = raw_csv, sum_csv = sum_csv, df_sum = df_sum)
}

source("inst/benchmarks/bench_baselines.R")

message(sprintf(
  "[FBMS-UPDATE] benchmarks=%s | n_seeds=%d | priors=%s | merge_prior=%s | outer_workers=%d | fbms_cores=%d | runs=%d | P=%d | N=%s",
  paste(benchmarks, collapse = ","), length(seeds), opt$priors, opt$merge_prior,
  opt$outer_workers, opt$fbms_cores, opt$fbms_runs, opt$fbms_P,
  if (is.null(opt$fbms_N)) "task-default" else as.character(opt$fbms_N)
))

for (b in benchmarks) {
  b_dir <- file.path(opt$out_dir, b)
  dir.create(b_dir, recursive = TRUE, showWarnings = FALSE)
  fbms_method <- if (b == "logic") "FBMS-BLR" else "FBMS-BGNLM"
  pcat <- get_fbms_prior_catalog(b)

  prior_keys <- if (opt$priors == "default") {
    "default"
  } else if (opt$priors == "explore") {
    names(pcat)
  } else {
    req <- trimws(strsplit(opt$priors, ",", fixed = TRUE)[[1]])
    intersect(req, c("default", names(pcat)))
  }

  if (length(prior_keys) == 0L) {
    message("[WARN] No matching priors for benchmark ", b, "; skipping.")
    next
  }

  message(sprintf("\n[FBMS-UPDATE] === Benchmark '%s' (%s) | Priors: %s ===",
                  b, fbms_method, paste(prior_keys, collapse = ", ")))

  all_prior_dfs <- list()

  for (pk in prior_keys) {
    ckpt_dir <- if (pk == "default" && opt$priors == "default") {
      file.path(b_dir, "fbms_checkpoints")
    } else {
      file.path(b_dir, "fbms_priors", pk)
    }
    dir.create(ckpt_dir, recursive = TRUE, showWarnings = FALSE)
    t_p0 <- Sys.time()
    cat(sprintf("\n  [PRIOR '%s'] running %d seeds (outer_workers=%d, fbms_cores=%d, runs=%d, P=%d)...\n",
                pk, length(seeds), opt$outer_workers, opt$fbms_cores, opt$fbms_runs, opt$fbms_P))
    flush.console()

    run_one_seed <- function(idx) {
      s <- seeds[idx]
      ckpt_file <- file.path(ckpt_dir, sprintf("seed_%04d.rds", s))

      row_s <- NULL
      if (isTRUE(opt$resume) && file.exists(ckpt_file)) {
        row_s <- tryCatch(readRDS(ckpt_file), error = function(e) NULL)
      }
      if (is.null(row_s)) {
        t_s0 <- Sys.time()
        row_s <- tryCatch(
          run_fbms_task_seed(
            benchmark = b,
            seed = s,
            cores = opt$fbms_cores,
            runs = opt$fbms_runs,
            P = opt$fbms_P,
            N = opt$fbms_N,
            prior_key = if (pk == "default") NULL else pk
          ),
          error = function(e) {
            cat(sprintf("  [WARN] %s (%s) seed %d failed: %s\n", b, pk, s, conditionMessage(e)))
            flush.console()
            NULL
          }
        )
        if (!is.null(row_s)) {
          dt_s <- as.numeric(difftime(Sys.time(), t_s0, units = "secs"))
          row_s$prior_key <- pk
          row_s$elapsed_secs <- dt_s
          save_rds_atomic(row_s, ckpt_file)
          cat(sprintf("    [%s | %-15s] seed %3d (%3d/%d) | %5.1fs | RMSE=%.5f | terms=%s | %s\n",
                      b, pk, s, idx, length(seeds), dt_s,
                      row_s$test_rmse[1], row_s$n_terms[1], row_s$formula[1]))
          flush.console()
        }
      }
      row_s
    }

    if (opt$outer_workers > 1L && .Platform$OS.type == "unix") {
      invisible(parallel::mclapply(seq_along(seeds), run_one_seed, mc.cores = opt$outer_workers, mc.preschedule = FALSE))
    } else {
      for (idx in seq_along(seeds)) run_one_seed(idx)
    }

    ckpt_files <- sort(list.files(ckpt_dir, pattern = "^seed_[0-9]+\\.rds$", full.names = TRUE))
    pk_list <- Filter(Negate(is.null), lapply(ckpt_files, function(f) {
      d <- tryCatch(readRDS(f), error = function(e) NULL)
      if (!is.null(d)) {
        if (!("prior_key" %in% names(d))) d$prior_key <- pk
        if (!("elapsed_secs" %in% names(d))) d$elapsed_secs <- NA_real_
      }
      d
    }))
    if (length(pk_list) > 0L) {
      all_prior_dfs[[pk]] <- do.call(rbind, pk_list)
      dt_p <- as.numeric(difftime(Sys.time(), t_p0, units = "secs"))
      message(sprintf("  [OK] Prior '%s': %d completed seeds (%.1fs)", pk, nrow(all_prior_dfs[[pk]]), dt_p))
    }
  }

  # Save combined prior comparison table if running prior exploration (or >1 prior)
  if (length(all_prior_dfs) > 0L && (opt$priors != "default")) {
    df_priors_raw <- do.call(rbind, all_prior_dfs)
    rownames(df_priors_raw) <- NULL
    priors_raw_csv <- file.path(b_dir, paste0("raw_", b, "_fbms_priors.csv"))
    priors_sum_csv <- file.path(b_dir, paste0("summary_", b, "_fbms_priors.csv"))
    if (file.exists(priors_raw_csv) && file.info(priors_raw_csv)$size > 0) {
      prev_p <- tryCatch(utils::read.csv(priors_raw_csv, stringsAsFactors = FALSE), error = function(e) NULL)
      if (!is.null(prev_p) && nrow(prev_p) > 0L && "prior_key" %in% names(prev_p)) {
        keep_p <- prev_p[!(prev_p$prior_key %in% names(all_prior_dfs)), , drop = FALSE]
        if (nrow(keep_p) > 0L) {
          df_priors_raw <- rbind(keep_p[, names(df_priors_raw), drop = FALSE], df_priors_raw)
        }
      }
    }
    write_csv_atomic(df_priors_raw, priors_raw_csv)
    df_priors_sum <- summarize_grouped_df(df_priors_raw, group_col = "prior_key")
    write_csv_atomic(df_priors_sum, priors_sum_csv)
    message(sprintf("\n[OK] Prior comparison summary written to %s:", priors_sum_csv))
    print(df_priors_sum)
  }

  # Merge selected prior into main baselines CSV if requested
  if (opt$merge_prior != "none" && length(all_prior_dfs) > 0L) {
    sel_pk <- opt$merge_prior
    if (sel_pk == "best_rmse") {
      means_rmse <- vapply(names(all_prior_dfs), function(pk) {
        mean(all_prior_dfs[[pk]]$test_rmse, na.rm = TRUE)
      }, numeric(1))
      sel_pk <- names(which.min(means_rmse))[1]
      message(sprintf("[SELECT] Best mean-RMSE prior for '%s': %s (%.5f)", b, sel_pk, min(means_rmse)))
    }
    if (sel_pk %in% names(all_prior_dfs)) {
      df_merge_src <- all_prior_dfs[[sel_pk]]
      df_merge_src$prior_key <- NULL
      df_merge_src$elapsed_secs <- NULL
      m_res <- merge_fbms_rows_into_baselines(b_dir, b, fbms_method, df_merge_src)
      message(sprintf("[OK] Merged '%s' (%d rows) -> %s", sel_pk, nrow(df_merge_src), m_res$raw_csv))
      print(m_res$df_sum[m_res$df_sum$method == fbms_method, , drop = FALSE])
    } else {
      message(sprintf("[WARN] Requested merge_prior '%s' not among completed priors (%s); skipped merge.",
                      sel_pk, paste(names(all_prior_dfs), collapse = ", ")))
    }
  }
}

message("\n[OK] FBMS job finished cleanly.")
