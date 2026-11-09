#!/usr/bin/env Rscript
# =============================================================================
# inst/scripts/install_baselines.R
# Installs all R and Python baseline dependencies cleanly on macOS or Linux HPC
# servers (e.g., nam-shub-02.uio.no) without requiring liblantern.so.
# =============================================================================

# 0. Pre-load libflexiblas.so.3 if needed by EasyBuild-compiled R packages (BAS, Matrix)
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

# 1. Ensure ~/.R/Makevars directs R to system gcc/g++ if conda compiler wrappers are missing
r_dir <- path.expand("~/.R")
if (!dir.exists(r_dir)) dir.create(r_dir, recursive = TRUE, showWarnings = FALSE)
makevars_file <- file.path(r_dir, "Makevars")

gcc_bin <- Sys.which("gcc")
gxx_bin <- Sys.which("g++")
gfort_bin <- Sys.which("gfortran")

if (nzchar(gcc_bin)) {
  mv_lines <- c(
    paste0("CC = ", gcc_bin, " -fPIC"),
    paste0("CXX = ", gxx_bin, " -fPIC"),
    paste0("CXX11 = ", gxx_bin, " -fPIC"),
    paste0("CXX14 = ", gxx_bin, " -fPIC"),
    paste0("CXX17 = ", gxx_bin, " -fPIC")
  )
  if (nzchar(gfort_bin)) {
    mv_lines <- c(mv_lines,
      paste0("FC = ", gfort_bin, " -fPIC"),
      paste0("F77 = ", gfort_bin, " -fPIC")
    )
  }
  writeLines(mv_lines, con = makevars_file)
  message("[SETUP] Configured ~/.R/Makevars to use host compilers (", gcc_bin, ", ", gxx_bin, ").")
}

# 2. Symlink conda wrapper names in active conda/micromamba bin if they don't exist
r_home <- R.home("bin")
conda_bin <- dirname(r_home)
if (dir.exists(file.path(conda_bin, "bin"))) conda_bin <- file.path(conda_bin, "bin")

for (wrapper in c("x86_64-conda-linux-gnu-cc", "x86_64-conda-linux-gnu-gcc")) {
  tgt <- file.path(conda_bin, wrapper)
  if (nzchar(gcc_bin) && !file.exists(tgt)) {
    try(file.symlink(gcc_bin, tgt), silent = TRUE)
  }
}
for (wrapper in c("x86_64-conda-linux-gnu-c++", "x86_64-conda-linux-gnu-g++")) {
  tgt <- file.path(conda_bin, wrapper)
  if (nzchar(gxx_bin) && !file.exists(tgt)) {
    try(file.symlink(gxx_bin, tgt), silent = TRUE)
  }
}
for (wrapper in c("x86_64-conda-linux-gnu-gfortran")) {
  tgt <- file.path(conda_bin, wrapper)
  if (nzchar(gfort_bin) && !file.exists(tgt)) {
    try(file.symlink(gfort_bin, tgt), silent = TRUE)
  }
}

# 3. Setup library paths matching active R version
active_maj_min <- paste0(getRversion()$major, ".", getRversion()$minor)
user_ver_lib <- path.expand(paste0("~/R/x86_64-pc-linux-gnu-library/", active_maj_min))
if (!dir.exists(user_ver_lib)) {
  dir.create(user_ver_lib, recursive = TRUE, showWarnings = FALSE)
}

.libPaths(unique(c(user_ver_lib, .libPaths())))
target_lib <- user_ver_lib

message("[INFO] Active R: ", as.character(getRversion()), " | Target library: ", target_lib)
message("[INFO] Current .libPaths(): ", paste(.libPaths(), collapse = " : "))

# 4. Only install strictly necessary packages with Minimal dependencies
required_pkgs <- c("jsonlite", "glmnet", "earth", "mgcv", "ranger", "LogicReg", "FBMS")

for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("[INSTALL] Installing CRAN package: ", pkg, " into ", target_lib)
    install.packages(pkg, lib = target_lib, repos = "https://cloud.r-project.org",
                     dependencies = c("Depends", "Imports", "LinkingTo"))
  }
}

message("[OK] Baseline R packages status:")
for (pkg in required_pkgs) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    message("  - ", pkg, " (version ", as.character(packageVersion(pkg)), ")")
  } else {
    warning("  ! Could not load ", pkg)
  }
}

# 5. Check Python numpy + scipy for SR-PySR / SR-Operon symbolic regression bridge
cands_py <- c(
  file.path(conda_bin, "python"),
  file.path(conda_bin, "python3"),
  Sys.which("python3"),
  Sys.which("python")
)
py_bin <- ""
for (cp in unique(cands_py)) {
  if (nzchar(cp) && file.exists(cp)) {
    chk <- tryCatch(
      system2(cp, c("-c", shQuote("import numpy, scipy; print('Python SR bridge OK')")),
              stdout = TRUE, stderr = TRUE),
      error = function(e) character(0)
    )
    if (any(grepl("Python SR bridge OK", chk))) {
      py_bin <- cp
      message("[OK] Found working Python interpreter with numpy and scipy: ", py_bin)
      break
    }
  }
}
if (!nzchar(py_bin)) {
  message("[WARN] No Python interpreter with numpy and scipy found on standard paths.")
}
