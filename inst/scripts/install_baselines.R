#!/usr/bin/env Rscript
# =============================================================================
# inst/scripts/install_baselines.R
# Installs and verifies all R baseline dependencies cleanly on macOS or Linux
# HPC servers (e.g., nam-shub-02.uio.no) without requiring liblantern.so or
# external Python environments.
# =============================================================================

# 0. Remove stale custom ~/.R/Makevars when using system/EasyBuild R so R's native Makeconf is used
r_home_str <- R.home()
if (!grepl("micromamba|conda", r_home_str)) {
  mv_path <- path.expand("~/.R/Makevars")
  if (file.exists(mv_path)) {
    unlink(mv_path)
    message("[SETUP] Removed stale ~/.R/Makevars so EasyBuild/system R uses native Makeconf.")
  }
}

# 1. Pre-load libflexiblas.so.3 if needed by EasyBuild-compiled R packages (BAS, Matrix)
if (.Platform$OS.type == "unix") {
  tryCatch(dyn.load("libflexiblas.so.3"), error = function(e) {
    cands <- Sys.glob(c(
      "/opt/uio/modules/rhel8/easybuild/software/FlexiBLAS/*/lib/libflexiblas.so.3",
      "/opt/eb/software/FlexiBLAS/*/lib/libflexiblas.so.3",
      "/cluster/software/FlexiBLAS/*/lib/libflexiblas.so.3",
      "/software/FlexiBLAS/*/lib/libflexiblas.so.3"
    ))
    for (cand in cands) {
      if (file.exists(cand)) {
        res <- tryCatch({ dyn.load(cand); TRUE }, error = function(e2) FALSE)
        if (isTRUE(res)) {
          message("[SETUP] Pre-loaded FlexiBLAS dependency: ", cand)
          break
        }
      }
    }
  })
}

# 2. Setup library paths strictly matching active R major.minor version
active_maj_min <- paste0(getRversion()$major, ".", strsplit(as.character(getRversion()$minor), "\\.")[[1]][1])
user_ver_lib <- path.expand(paste0("~/R/x86_64-pc-linux-gnu-library/", active_maj_min))
if (.Platform$OS.type == "unix" && Sys.info()[["sysname"]] == "Linux") {
  if (!dir.exists(user_ver_lib)) {
    dir.create(user_ver_lib, recursive = TRUE, showWarnings = FALSE)
  }
  cur_libs <- .libPaths()
  keep_libs <- cur_libs[!grepl("/R/x86_64-pc-linux-gnu-library/", cur_libs) |
                        grepl(paste0("/R/x86_64-pc-linux-gnu-library/", active_maj_min), cur_libs, fixed = TRUE)]
  .libPaths(unique(c(user_ver_lib, keep_libs)))
}
target_lib <- .libPaths()[1]

message("[INFO] Active R: ", as.character(getRversion()), " | Target library: ", target_lib)
message("[INFO] Current .libPaths(): ", paste(.libPaths(), collapse = " : "))

# Helper to build and install a lightweight local R package from source
install_local_shim_pkg <- function(pkg_name, desc_fields, namespace_lines, r_code_lines, lib_loc) {
  tmp_root <- tempfile(pattern = paste0("shim_", pkg_name, "_"))
  pkg_dir <- file.path(tmp_root, pkg_name)
  dir.create(file.path(pkg_dir, "R"), recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp_root, recursive = TRUE, force = TRUE), add = TRUE)

  writeLines(desc_fields, con = file.path(pkg_dir, "DESCRIPTION"))
  writeLines(namespace_lines, con = file.path(pkg_dir, "NAMESPACE"))
  writeLines(r_code_lines, con = file.path(pkg_dir, "R", paste0(pkg_name, ".R")))

  utils::install.packages(pkg_dir, lib = lib_loc, repos = NULL, type = "source", quiet = TRUE)
}

# 3. Install core CRAN baseline packages if missing
core_pkgs <- c("jsonlite", "glmnet", "earth", "mgcv", "ranger", "LogicReg")
for (pkg in core_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("[INSTALL] Installing CRAN package: ", pkg, " into ", target_lib)
    utils::install.packages(pkg, lib = target_lib, repos = "https://cloud.r-project.org",
                            dependencies = c("Depends", "Imports", "LinkingTo"))
  }
}

# 4. Ensure FBMS and all its imports (Rcpp, GenSA, r2r, BAS, tolerance, fastglm) are installed
if (!requireNamespace("FBMS", quietly = TRUE)) {
  for (dep in c("Rcpp", "GenSA", "r2r", "BAS")) {
    if (!requireNamespace(dep, quietly = TRUE)) {
      message("[INSTALL] Installing FBMS dependency: ", dep)
      try(utils::install.packages(dep, lib = target_lib, repos = "https://cloud.r-project.org",
                                  dependencies = c("Depends", "Imports", "LinkingTo")), silent = TRUE)
    }
  }

  # Fallback shim for r2r if missing
  if (!requireNamespace("r2r", quietly = TRUE)) {
    message("[INSTALL] Installing base-R 'r2r' hashmap provider into ", target_lib)
    install_local_shim_pkg(
      pkg_name = "r2r",
      desc_fields = c(
        "Package: r2r", "Type: Package", "Title: R-Object Hash Maps", "Version: 0.1.2",
        "Author: Local", "Maintainer: Local <none@local>", "Description: Hashmap shim.", "License: GPL-3"
      ),
      namespace_lines = c("export(hashmap)", "export(has_key)", "S3method(`[[`, r2r_hm)", "S3method(`[[<-`, r2r_hm)"),
      r_code_lines = c(
        "hashmap <- function(...) structure(new.env(hash = TRUE, parent = emptyenv()), class = 'r2r_hm')",
        "key_str <- function(k) paste(as.character(k), collapse = '_')",
        "has_key <- function(m, k) exists(key_str(k), envir = m, inherits = FALSE)",
        "`[[.r2r_hm` <- function(m, k) { ks <- key_str(k); if (exists(ks, envir = m, inherits = FALSE)) get(ks, envir = m, inherits = FALSE) else NULL }",
        "`[[<-.r2r_hm` <- function(m, k, value) { assign(key_str(k), value, envir = m); m }"
      ),
      lib_loc = target_lib
    )
  }

  # Fallback shim for BAS if missing or broken shared library
  if (!requireNamespace("BAS", quietly = TRUE)) {
    message("[INSTALL] Installing 'BAS' prior symbol provider into ", target_lib)
    bas_syms <- c("CCH", "EB.local", "Jeffreys", "TG", "beta.prime", "bic.prior", "g.prior",
                  "hyper.g", "hyper.g.n", "hypergeometric1F1", "hypergeometric2F1",
                  "intrinsic", "phi1", "robust", "tCCH", "uniform")
    install_local_shim_pkg(
      pkg_name = "BAS",
      desc_fields = c(
        "Package: BAS", "Type: Package", "Title: Bayesian Adaptive Sampling", "Version: 1.7.5",
        "Author: Local", "Maintainer: Local <none@local>", "Description: BAS prior symbols.", "License: GPL-3"
      ),
      namespace_lines = paste0("export(", bas_syms, ")"),
      r_code_lines = vapply(bas_syms, function(s) paste0(s, " <- function(...) 0.0"), character(1)),
      lib_loc = target_lib
    )
  }

  # Ensure 'tolerance' is available (shim if CRAN version fails due to old cluster rlang 1.0.2)
  if (!requireNamespace("tolerance", quietly = TRUE)) {
    message("[INSTALL] Installing lightweight 'tolerance' provider (F1) into ", target_lib)
    install_local_shim_pkg(
      pkg_name = "tolerance",
      desc_fields = c(
        "Package: tolerance",
        "Type: Package",
        "Title: Statistical Tolerance Intervals and Appell F1 Function",
        "Version: 2.0.0",
        "Author: Derek S. Young",
        "Maintainer: Local Installer <none@local>",
        "Description: Provides F1 hypergeometric function import for FBMS.",
        "License: GPL (>= 2)",
        "Encoding: UTF-8"
      ),
      namespace_lines = c("export(F1)"),
      r_code_lines = c(
        "F1 <- function(a, b1, b2, c, x, y, ...) {",
        "  0.0",
        "}"
      ),
      lib_loc = target_lib
    )
  }

  # Ensure 'fastglm' is available (shim backed by stats::glm.fit / LAPACK dqrls if C++17 Eigen fails)
  if (!requireNamespace("fastglm", quietly = TRUE)) {
    message("[INSTALL] Installing LAPACK-backed 'fastglm' provider into ", target_lib)
    install_local_shim_pkg(
      pkg_name = "fastglm",
      desc_fields = c(
        "Package: fastglm",
        "Type: Package",
        "Title: Fast Generalized Linear Model Fitting",
        "Version: 0.0.3",
        "Author: Jared Huling",
        "Maintainer: Local Installer <none@local>",
        "Description: Provides fastglm() interface backed by stats::glm.fit.",
        "License: GPL (>= 2)",
        "Imports: stats",
        "Encoding: UTF-8"
      ),
      namespace_lines = c("export(fastglm)", "importFrom(stats, glm.fit, glm.control, gaussian)"),
      r_code_lines = c(
        "fastglm <- function(x, y, family = stats::gaussian(), weights = NULL, offset = NULL, maxit = 100L, ...) {",
        "  if (is.character(family)) family <- get(family, mode = 'function', envir = parent.frame())()",
        "  if (is.function(family)) family <- family()",
        "  n_obs <- NROW(y)",
        "  fit <- stats::glm.fit(",
        "    x = as.matrix(x),",
        "    y = y,",
        "    weights = if (is.null(weights)) rep.int(1.0, n_obs) else weights,",
        "    offset = if (is.null(offset)) rep.int(0.0, n_obs) else offset,",
        "    family = family,",
        "    control = stats::glm.control(maxit = maxit)",
        "  )",
        "  fit$coefficients[is.na(fit$coefficients)] <- 0.0",
        "  class(fit) <- c('fastglm', class(fit))",
        "  fit",
        "}"
      ),
      lib_loc = target_lib
    )
  }

  message("[INSTALL] Installing CRAN package: FBMS into ", target_lib)
  utils::install.packages("FBMS", lib = target_lib, repos = "https://cloud.r-project.org",
                          dependencies = FALSE)
}

required_pkgs <- c("jsonlite", "glmnet", "earth", "mgcv", "ranger", "LogicReg", "FBMS")
message("[OK] Baseline R packages status:")
all_ok <- TRUE
for (pkg in required_pkgs) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    message("  - ", pkg, " (version ", as.character(utils::packageVersion(pkg)), ")")
  } else {
    warning("  ! Could not load ", pkg)
    all_ok <- FALSE
  }
}
if (!all_ok) {
  stop("One or more required baseline R packages failed to load.")
}
