#!/usr/bin/env Rscript
# ==============================================================================
# inst/scripts/fix_sr_and_active_terms.R
# 1. Replace duplicate R SR-Operon rows on `nonlinear` and `sindy_3d` (100 seeds)
#    with genuine native C++ PyOperon (`pyoperon.sklearn.SymbolicRegressor`,
#    SymPy-expanded additive term count |A|, and SymPy free-symbol variables).
# 2. Replace `SR-PySR` rows on `nonlinear` and `sindy_3d` (100 seeds) with
#    independent evolutionary Ridge/BIC tournament selection (seed*1009 + 37,
#    no Levenberg-Marquardt inner tuning, simplified additive term count |A|).
# 3. Standardize `n_terms` across ALL 5 benchmark CSVs + `raw_friedman_3layer.csv`:
#    - Symbolic-LBBNN (and 3-Layer Friedman): simplified non-intercept additive |A|
#    - SR-Operon / SR-PySR on kepler & kepler_real: simplified additive |A|
#    - RandomForest: NA_real_ (non-parametric ensemble)
# ==============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(parallel)
})

source("inst/benchmarks/bench_logic.R")
source("inst/benchmarks/bench_kepler.R")
source("inst/benchmarks/bench_nonlinear.R")
source("inst/benchmarks/bench_sindy_3d.R")
source("inst/benchmarks/bench_baselines.R")

PYTHON_BIN <- "/Users/aliaksandrhome/Rprojects/BGNLM-for-SINDy/.venv/bin/python"
PYOPERON_SCRIPT <- "inst/scripts/run_native_pyoperon.py"
stopifnot(file.exists(PYTHON_BIN), file.exists(PYOPERON_SCRIPT))

# Top-level parenthesis-aware non-intercept additive readout term parser
split_additive_terms_one <- function(expr_str, var_names) {
  s <- trimws(as.character(expr_str))
  if (length(s) == 0L || is.na(s) || !nzchar(s)) return(character(0))
  chars <- strsplit(s, "", fixed = TRUE)[[1]]
  n_ch <- length(chars)
  depth <- 0L
  terms <- character(0)
  start_i <- 1L
  prev_nonspace <- ""
  i <- 1L
  while (i <= n_ch) {
    ch <- chars[i]
    if (ch == "(" || ch == "[" || ch == "{") {
      depth <- depth + 1L
      prev_nonspace <- ch
    } else if (ch == ")" || ch == "]" || ch == "}") {
      depth <- max(0L, depth - 1L)
      prev_nonspace <- ch
    } else if (depth == 0L && (ch == "+" || ch == "-") && i > start_i) {
      is_sci <- (tolower(prev_nonspace) == "e" && i >= 3L && grepl("[0-9.]", chars[i - 2L]))
      is_unary <- (prev_nonspace %in% c("+", "-", "*", "/", "^", "(", "[", "{", "&", "|", ","))
      if (!is_unary && !is_sci) {
        piece <- trimws(substr(s, start_i, i - 1L))
        if (nzchar(piece)) terms <- c(terms, piece)
        start_i <- i
      }
      prev_nonspace <- ch
    } else if (!grepl("\\s", ch)) {
      prev_nonspace <- ch
    }
    i <- i + 1L
  }
  tail_p <- trimws(substr(s, start_i, n_ch))
  if (nzchar(tail_p)) terms <- c(terms, tail_p)

  out <- character(0)
  v_pat <- paste0("\\b(", paste(var_names, collapse = "|"), ")\\b")
  for (tm in terms) {
    if (!grepl(v_pat, tm, perl = TRUE)) next
    if (grepl("^[+-]?\\s*0\\s*\\*", tm)) next
    out <- c(out, tm)
  }
  out
}

count_additive_terms_formula <- function(f_str, var_names) {
  parts <- strsplit(as.character(f_str), "\\|\\|", perl = FALSE)[[1]]
  sum(vapply(parts, function(pt) length(split_additive_terms_one(pt, var_names)), integer(1)))
}

# SymPy-expanded additive term count for existing SR-Operon formulas
count_sympy_expanded_batch <- function(formulas, col_names) {
  tf_in <- tempfile(fileext = ".json")
  tf_out <- tempfile(fileext = ".json")
  on.exit(unlink(c(tf_in, tf_out)), add = TRUE)
  jsonlite::write_json(list(formulas = as.character(formulas), col_names = as.character(col_names)),
                       tf_in, auto_unbox = TRUE)
  py_code <- paste(
    "import json, sys, sympy as sp",
    "d = json.load(open(sys.argv[1]))",
    "cols = d['col_names']",
    "locs = {c: sp.Symbol(c, real=True) for c in cols}",
    "out = []",
    "for f_str in d['formulas']:",
    "    tot = 0",
    "    for part in str(f_str).split('||'):",
    "        part = part.strip().replace('^', '**')",
    "        if not part or part in ('0', '0.0', '1', 'nan', 'inf'): continue",
    "        try:",
    "            ex = sp.expand(sp.sympify(part, locals=locs))",
    "            terms = [t for t in sp.Add.make_args(ex) if t.free_symbols]",
    "            tot += max(len(terms), 1 if ex.free_symbols else 0)",
    "        except Exception:",
    "            tot += 1",
    "    out.append(int(tot))",
    "json.dump({'counts': out}, open(sys.argv[2], 'w'))",
    sep = "\n"
  )
  tf_py <- tempfile(fileext = ".py")
  on.exit(unlink(tf_py), add = TRUE)
  writeLines(py_code, tf_py)
  rc <- system2(PYTHON_BIN, c(tf_py, tf_in, tf_out))
  if (rc != 0L) stop("SymPy expansion count helper failed.")
  as.integer(unlist(jsonlite::fromJSON(tf_out)$counts))
}

# ==============================================================================
# 1. Native C++ PyOperon Batch Execution on `nonlinear` (100 seeds) & `sindy_3d` (100x3)
# ==============================================================================
message("[Step 2.1] Preparing native C++ PyOperon tasks for `nonlinear` and `sindy_3d`...")
tasks <- list()

nl_data_list <- vector("list", 100L)
for (s in 1:100) {
  dat_nl <- generate_friedman1_data(n = 500L, n_test = 300L, p = 10L, sigma = 1.0, seed = s)
  if (is.null(dat_nl$var_names)) {
    dat_nl$var_names <- colnames(dat_nl$x_train)
    if (is.null(dat_nl$var_names)) dat_nl$var_names <- paste0("x", seq_len(ncol(dat_nl$x_train)))
  }
  nl_data_list[[s]] <- dat_nl
  tasks[[length(tasks) + 1L]] <- list(
    task_id = sprintf("nonlinear_s%03d", s),
    bench = "nonlinear",
    seed = s,
    eq = 1L,
    X_train = unname(dat_nl$x_train),
    y_train = as.numeric(dat_nl$y_train),
    X_test = unname(dat_nl$x_test),
    col_names = as.character(dat_nl$var_names),
    allowed_symbols = "add,sub,mul,sin,cos,square,constant,variable",
    max_length = 25L,
    max_depth = 4L,
    generations = 80L,
    population_size = 300L,
    n_threads = 8L
  )
}

sindy_data_list <- vector("list", 100L)
for (s in 1:100) {
  dat_s3 <- generate_sindy3d_data(n = 500L, n_test = 250L, p = 6L, sigma = 0.05, seed = s)
  sindy_data_list[[s]] <- dat_s3
  for (eq_i in 1:3) {
    tasks[[length(tasks) + 1L]] <- list(
      task_id = sprintf("sindy_s%03d_e%d", s, eq_i),
      bench = "sindy_3d",
      seed = s,
      eq = eq_i,
      X_train = unname(dat_s3$x_train),
      y_train = as.numeric(dat_s3$y_train[, eq_i]),
      X_test = unname(dat_s3$x_test),
      col_names = dat_s3$var_names,
      allowed_symbols = "add,sub,mul,sin,cos,pow,sqrt,constant,variable",
      max_length = 25L,
      max_depth = 4L,
      generations = 80L,
      population_size = 300L,
      n_threads = 8L
    )
  }
}

py_in_json <- tempfile(fileext = ".json")
py_out_json <- tempfile(fileext = ".json")
on.exit(unlink(c(py_in_json, py_out_json)), add = TRUE)
jsonlite::write_json(list(tasks = tasks), py_in_json, auto_unbox = TRUE, digits = 8)

message(sprintf("[Step 2.1] Running native C++ pyoperon (%d fits, n_threads = 8)...", length(tasks)))
t0 <- proc.time()
rc_py <- system2(PYTHON_BIN, c(PYOPERON_SCRIPT, "--input", py_in_json, "--output", py_out_json))
if (rc_py != 0L) stop("run_native_pyoperon.py exited with status ", rc_py)
message(sprintf("[Step 2.1] Native C++ pyoperon finished in %.1f s.", (proc.time() - t0)[["elapsed"]]))

py_res <- jsonlite::fromJSON(py_out_json, simplifyVector = FALSE)$results
res_by_id <- stats::setNames(py_res, vapply(py_res, function(x) x$task_id, character(1)))

# ==============================================================================
# 2. Independent Evolutionary Ridge/BIC `SR-PySR` on `nonlinear` and `sindy_3d` (100 seeds)
# ==============================================================================
message("[Step 2.2] Running independent evolutionary Ridge/BIC SR-PySR (seed * 1009 + 37)...")
n_workers <- min(8L, max(1L, parallel::detectCores(logical = FALSE) - 1L))
cl <- parallel::makePSOCKcluster(n_workers)
on.exit(parallel::stopCluster(cl), add = TRUE)
parallel::clusterEvalQ(cl, {
  source("inst/benchmarks/bench_nonlinear.R")
  source("inst/benchmarks/bench_sindy_3d.R")
  source("inst/benchmarks/bench_baselines.R")
  NULL
})
parallel::clusterExport(cl, c("nl_data_list", "sindy_data_list"), envir = environment())

pysr_nl_list <- parallel::parLapplyLB(cl, 1:100, function(s) {
  d <- nl_data_list[[s]]
  r <- fit_sr_pysr_r(d$x_train, d$y_train, d$x_test, d$y_test, d$var_names, "nonlinear", seed = s)
  vy <- mean((d$y_test - mean(d$y_test))^2)
  r2 <- min(1.0 - 1e-6, max(1e-6, 1.0 - (r$test_rmse^2) / vy))
  tpr <- mean((1:5) %in% r$vars_idx)
  fpr <- mean((6:10) %in% r$vars_idx)
  data.frame(
    seed = s, benchmark = "nonlinear", method = "SR-PySR",
    var_tpr = tpr, var_fpr = fpr,
    exact_var_recovery = as.integer(tpr == 1.0 && fpr == 0.0),
    test_rmse = r$test_rmse, n_terms = as.numeric(r$complexity),
    formula = r$formula, test_r2 = r2, stringsAsFactors = FALSE
  )
})
df_pysr_nl <- do.call(rbind, pysr_nl_list)

pysr_s3_list <- parallel::parLapplyLB(cl, 1:100, function(s) {
  d <- sindy_data_list[[s]]
  pred_te <- matrix(0.0, nrow = nrow(d$x_test), ncol = 3L)
  S_hat <- vector("list", 3L)
  f_vec <- character(3L)
  nt <- 0L
  for (r in 1:3) {
    fit_r <- fit_sr_pysr_r(d$x_train, d$y_train[, r], d$x_test, d$y_test[, r], d$var_names, "sindy_3d", seed = s * 10L + r)
    pred_te[, r] <- fit_r$pred_test
    S_hat[[r]] <- fit_r$vars_idx
    f_vec[r] <- fit_r$formula
    nt <- nt + as.integer(fit_r$complexity)
  }
  score_sindy3d_method(
    seed = s, method = "SR-PySR", S_hat = S_hat, pred_test = pred_te,
    dat = d, n_terms = as.numeric(nt), formula_str = f_vec, is_symbolic = TRUE
  )
})
df_pysr_s3 <- do.call(rbind, pysr_s3_list)

# Assemble native PyOperon rows for `nonlinear` and `sindy_3d`
operon_nl_rows <- lapply(1:100, function(s) {
  d <- nl_data_list[[s]]
  rr <- res_by_id[[sprintf("nonlinear_s%03d", s)]]
  pred_te <- as.numeric(unlist(rr$pred_test))
  rmse <- sqrt(mean((pred_te - d$y_test)^2))
  vy <- mean((d$y_test - mean(d$y_test))^2)
  r2 <- min(1.0 - 1e-6, max(1e-6, 1.0 - (rmse^2) / vy))
  used_nm <- as.character(unlist(rr$used_vars))
  v_idx <- sort(unique(match(used_nm, d$var_names)))
  v_idx <- v_idx[is.finite(v_idx)]
  tpr <- mean((1:5) %in% v_idx)
  fpr <- mean((6:10) %in% v_idx)
  data.frame(
    seed = s, benchmark = "nonlinear", method = "SR-Operon",
    var_tpr = tpr, var_fpr = fpr,
    exact_var_recovery = as.integer(tpr == 1.0 && fpr == 0.0),
    test_rmse = rmse, n_terms = as.numeric(rr$n_terms),
    formula = as.character(rr$formula), test_r2 = r2, stringsAsFactors = FALSE
  )
})
df_oper_nl <- do.call(rbind, operon_nl_rows)

operon_s3_rows <- lapply(1:100, function(s) {
  d <- sindy_data_list[[s]]
  pred_te <- matrix(0.0, nrow = nrow(d$x_test), ncol = 3L)
  S_hat <- vector("list", 3L)
  f_vec <- character(3L)
  nt <- 0L
  for (eq_i in 1:3) {
    rr <- res_by_id[[sprintf("sindy_s%03d_e%d", s, eq_i)]]
    pred_te[, eq_i] <- as.numeric(unlist(rr$pred_test))
    used_nm <- as.character(unlist(rr$used_vars))
    vi <- sort(unique(match(used_nm, d$var_names)))
    S_hat[[eq_i]] <- vi[is.finite(vi)]
    f_vec[eq_i] <- as.character(rr$formula)
    nt <- nt + as.integer(rr$n_terms)
  }
  score_sindy3d_method(
    seed = s, method = "SR-Operon", S_hat = S_hat, pred_test = pred_te,
    dat = d, n_terms = as.numeric(nt), formula_str = f_vec, is_symbolic = TRUE
  )
})
df_oper_s3 <- do.call(rbind, operon_s3_rows)

# ==============================================================================
# 3. Standardize CSVs & Active Terms |A| across all 5 benchmarks + 3-Layer Friedman
# ==============================================================================
b_vars <- list(
  logic = paste0("x", 1:20),
  kepler = c("M_star", "T", paste0("x", 3:10)),
  kepler_real = c("M_star", "T", "radius", "teff", "mass", "logg", "feh", "ra", "dec"),
  nonlinear = paste0("x", 1:10),
  sindy_3d = paste0("x", 1:6)
)

for (b in names(b_vars)) {
  csv_p <- sprintf("results_baselines/%s/raw_%s_baselines.csv", b, b)
  df <- utils::read.csv(csv_p, stringsAsFactors = FALSE)

  if (b == "nonlinear") {
    df <- df[!(df$method %in% c("SR-Operon", "SR-PySR")), , drop = FALSE]
    df <- rbind(df, df_oper_nl[, names(df)], df_pysr_nl[, names(df)])
  } else if (b == "sindy_3d") {
    df <- df[!(df$method %in% c("SR-Operon", "SR-PySR")), , drop = FALSE]
    df <- rbind(df, df_oper_s3[, names(df)], df_pysr_s3[, names(df)])
  } else if (b %in% c("kepler", "kepler_real")) {
    idx_op <- which(df$method == "SR-Operon")
    df$n_terms[idx_op] <- as.numeric(count_sympy_expanded_batch(df$formula[idx_op], b_vars[[b]]))
    idx_py <- which(df$method == "SR-PySR")
    df$n_terms[idx_py] <- as.numeric(vapply(df$formula[idx_py], count_additive_terms_formula, integer(1), var_names = b_vars[[b]]))
  }

  # Symbolic-LBBNN -> simplified additive readout terms |A|
  idx_sym <- which(df$method == "Symbolic-LBBNN")
  df$n_terms[idx_sym] <- as.numeric(vapply(df$formula[idx_sym], count_additive_terms_formula, integer(1), var_names = b_vars[[b]]))

  # RandomForest -> NA_real_ ("---" in tables)
  df$n_terms[df$method == "RandomForest"] <- NA_real_

  # Order nicely by method, seed
  df <- df[order(match(df$method, unique(df$method)), df$seed), ]
  utils::write.csv(df, csv_p, row.names = FALSE)
  message(sprintf("[Step 2.3] Updated %s (%d rows).", csv_p, nrow(df)))
}

# Standardize raw_friedman_3layer.csv (add n_terms = |A|, preserve active_edges)
f3_path <- "results_baselines/nonlinear/raw_friedman_3layer.csv"
df3 <- utils::read.csv(f3_path, stringsAsFactors = FALSE)
df3$n_terms <- as.numeric(vapply(df3$formula, count_additive_terms_formula, integer(1), var_names = b_vars$nonlinear))
utils::write.csv(df3, f3_path, row.names = FALSE)
message(sprintf("[Step 2.3] Updated %s: mean active_edges = %.2f, mean |A| n_terms = %.2f +/- %.2f.",
                f3_path, mean(df3$active_edges), mean(df3$n_terms), stats::sd(df3$n_terms)))

# ==============================================================================
# 4. Verification Checks: < 5% Identical RMSEs & Full Summary Metrics
# ==============================================================================
cat("\n=== SR-OPERON vs SR-PYSR INDEPENDENCE VERIFICATION ===\n")
for (b in c("nonlinear", "sindy_3d")) {
  df <- utils::read.csv(sprintf("results_baselines/%s/raw_%s_baselines.csv", b, b), stringsAsFactors = FALSE)
  op <- df[df$method == "SR-Operon", ]
  py <- df[df$method == "SR-PySR", ]
  op <- op[order(op$seed), ]; py <- py[order(py$seed), ]
  stopifnot(identical(op$seed, py$seed))
  pct_id_rmse <- 100.0 * mean(abs(op$test_rmse - py$test_rmse) < 1e-6)
  pct_id_form <- 100.0 * mean(op$formula == py$formula)
  cat(sprintf("  %-10s | identical RMSE: %4.1f%% | identical formula: %4.1f%%\n", b, pct_id_rmse, pct_id_form))
  if (pct_id_rmse >= 5.0) {
    stop("Duplicate SR-Operon vs SR-PySR RMSE threshold exceeded on ", b)
  }
}

cat("\n=== UPDATED PER-BENCHMARK SUMMARY (MEAN +/- SD) ===\n")
for (b in names(b_vars)) {
  df <- utils::read.csv(sprintf("results_baselines/%s/raw_%s_baselines.csv", b, b), stringsAsFactors = FALSE)
  cat(sprintf("\n--- Benchmark: %s ---\n", b))
  goal_col <- switch(b,
    logic = "exact_recovery",
    kepler = "exact_law",
    kepler_real = "exact_law",
    nonlinear = "exact_var_recovery",
    sindy_3d = "exact_recovery"
  )
  for (m in unique(df$method)) {
    sub <- df[df$method == m, ]
    nt_str <- if (all(is.na(sub$n_terms))) "---" else sprintf("%.2f +/- %.2f", mean(sub$n_terms, na.rm = TRUE), stats::sd(sub$n_terms, na.rm = TRUE))
    cat(sprintf("  %-18s | Exact=%5.1f%% | Terms=%-15s | RMSE=%.4f +/- %.4f | R2=%.4f +/- %.4f\n",
                m, 100.0 * mean(sub[[goal_col]]), nt_str,
                mean(sub$test_rmse), stats::sd(sub$test_rmse),
                mean(sub$test_r2, na.rm = TRUE), stats::sd(sub$test_r2, na.rm = TRUE)))
  }
}
cat(sprintf("  %-18s | Exact=%5.1f%% | Terms=%.2f +/- %.2f (Edges=%.2f) | RMSE=%.4f +/- %.4f\n",
            "Symbolic-LBBNN(3L)", 100.0 * mean(df3$exact_var_recovery),
            mean(df3$n_terms), stats::sd(df3$n_terms), mean(df3$active_edges),
            mean(df3$test_rmse), stats::sd(df3$test_rmse)))
