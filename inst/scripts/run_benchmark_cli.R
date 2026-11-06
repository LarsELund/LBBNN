#!/usr/bin/env Rscript
# =============================================================================
# CLI Multi-Seed Benchmark Runner with Per-Seed Checkpointing
# Usage:
#   Rscript inst/scripts/run_benchmark_cli.R --benchmark logic --seeds 1:100 \
#     --workers 8 --device cpu --out_dir results/
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(flag, default) {
  idx <- which(args == flag)
  if (length(idx) == 0L || idx[1] >= length(args)) return(default)
  args[idx[1] + 1L]
}

benchmark <- parse_arg("--benchmark", "logic")
seeds_str <- parse_arg("--seeds", "1:3")
workers <- as.integer(parse_arg("--workers", "1"))
device <- parse_arg("--device", "cpu")
out_dir <- parse_arg("--out_dir", "results")
epochs <- as.integer(parse_arg("--epochs", "300"))
debias_epochs <- as.integer(parse_arg("--debias_epochs", "100"))

parse_seeds <- function(s) {
  if (grepl(":", s, fixed = TRUE)) {
    parts <- as.integer(strsplit(s, ":", fixed = TRUE)[[1]])
    return(seq.int(parts[1], parts[2]))
  }
  as.integer(strsplit(s, ",", fixed = TRUE)[[1]])
}
seeds <- parse_seeds(seeds_str)

ckpt_dir <- file.path(out_dir, "checkpoints")
dir.create(ckpt_dir, recursive = TRUE, showWarnings = FALSE)

# Note: Do NOT load `torch` or `LBBNN` in the parent process before `mclapply`,
# so forked worker processes initialize their own isolated LibTorch/MPS runtime!
.worker_initialized <- FALSE
init_worker_env <- function() {
  if (!.worker_initialized) {
    if (requireNamespace("pkgload", quietly = TRUE) && file.exists("DESCRIPTION")) {
      pkgload::load_all(".", quiet = TRUE)
    } else {
      suppressPackageStartupMessages(library(LBBNN))
    }
    bench_dir <- if (dir.exists("inst/benchmarks")) {
      "inst/benchmarks"
    } else {
      system.file("benchmarks", package = "LBBNN")
    }
    source(file.path(bench_dir, "bench_logic.R"), local = FALSE)
    source(file.path(bench_dir, "bench_kepler.R"), local = FALSE)
    source(file.path(bench_dir, "bench_nonlinear.R"), local = FALSE)
    torch::torch_set_num_threads(1L)
    .worker_initialized <<- TRUE
  }
}

run_one_seed <- function(seed_val, bench_name) {
  ckpt_file <- file.path(ckpt_dir, sprintf("%s_seed_%04d.rds", bench_name, seed_val))
  if (file.exists(ckpt_file)) {
    message(sprintf("[%s] Seed %d already checkpointed -> loading %s", bench_name, seed_val, ckpt_file))
    return(readRDS(ckpt_file))
  }

  init_worker_env()
  set.seed(seed_val)
  torch::torch_manual_seed(seed_val)

  res <- switch(
    bench_name,
    "logic" = run_single_logic_seed(
      seed = seed_val, epochs = epochs, debias_epochs = debias_epochs, device = device
    ),
    "kepler" = run_single_kepler_seed(
      seed = seed_val, mode = "synthetic", epochs = epochs, debias_epochs = debias_epochs, device = device
    ),
    "kepler_real" = run_single_kepler_seed(
      seed = seed_val, mode = "real", epochs = epochs, debias_epochs = debias_epochs, device = device
    ),
    "nonlinear" = run_single_nonlinear_seed(
      seed = seed_val, epochs = epochs, debias_epochs = debias_epochs, device = device
    ),
    stop(sprintf("Unsupported --benchmark '%s'. Use logic, kepler, kepler_real, or nonlinear.", bench_name))
  )

  tmp_file <- paste0(ckpt_file, ".tmp")
  saveRDS(res, file = tmp_file)
  file.rename(tmp_file, ckpt_file)
  message(sprintf("[%s] Completed seed %d -> %s", bench_name, seed_val, res$formula))
  res
}

bench_list <- if (benchmark == "all") c("logic", "kepler", "nonlinear") else benchmark

for (b_name in bench_list) {
  message(sprintf(
    "=== Starting Benchmark '%s' | Seeds: %s (%d total) | Workers: %d | Device: %s ===",
    b_name, seeds_str, length(seeds), workers, device
  ))

  if (workers > 1L && .Platform$OS.type == "unix" && device == "cpu") {
    res_list <- parallel::mclapply(
      seeds,
      function(s) run_one_seed(s, b_name),
      mc.cores = workers,
      mc.preschedule = TRUE
    )
  } else {
    res_list <- lapply(seeds, function(s) run_one_seed(s, b_name))
  }

  df_all <- do.call(rbind, res_list)
  csv_path <- file.path(out_dir, sprintf("summary_%s.csv", b_name))
  utils::write.csv(df_all, file = csv_path, row.names = FALSE)

  num_cols <- names(df_all)[vapply(df_all, is.numeric, logical(1))]
  num_cols <- setdiff(num_cols, "seed")
  tex_lines <- c(
    "\\begin{table}[ht]",
    "\\centering",
    sprintf("\\caption{Monte Carlo Summary for Benchmark: %s ($M = %d$ seeds)}", b_name, nrow(df_all)),
    "\\begin{tabular}{lcc}",
    "\\hline",
    "Metric & Mean & SD \\\\",
    "\\hline"
  )
  for (col in num_cols) {
    vals <- df_all[[col]]
    tex_lines <- c(
      tex_lines,
      sprintf("%s & %.4f & %.4f \\\\", gsub("_", "\\\\_", col), mean(vals, na.rm = TRUE), stats::sd(vals, na.rm = TRUE))
    )
  }
  tex_lines <- c(tex_lines, "\\hline", "\\end{tabular}", "\\end{table}")
  tex_path <- file.path(out_dir, sprintf("summary_%s.tex", b_name))
  writeLines(tex_lines, con = tex_path)
  message(sprintf("=== Saved summary CSV (%s) and LaTeX table (%s) ===", csv_path, tex_path))
}
