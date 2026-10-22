# ==============================================================================
# Population and Sample Generation for Probability and Non-Probability Data
# Preserves Vladislav Beresovsky's baseline generator and adds non-linear extensions
# ==============================================================================

if(!require(rootSolve)){ install.packages("rootSolve"); library(rootSolve) }
if(!require(nleqslv)){ install.packages("nleqslv"); library(nleqslv) }
if(!require(survey)){ install.packages("survey"); library(survey) }
if(!require(MASS)){ install.packages("MASS"); library(MASS) }
if(!require(sampling)){ install.packages("sampling"); library(sampling) }
if(!require(dplyr)){ install.packages("dplyr"); library(dplyr) }
if(!require(Matrix)){ install.packages("Matrix"); library(Matrix) }

# ------------------------------------------------------------------------------
# 1. Vladislav's Baseline Functions (100% Backward Compatible)
# ------------------------------------------------------------------------------

# Estimates intercept a_0 of logit(pi_c) = a_0 + a_1*x_1 + a_2*x_2 + ...
# to guarantee the given size of non-prob sample
approx.int <- function(Xc, beta0, n){
  f <- function(x) {
    beta0x <- c(x, beta0[2: length(beta0)])
    p <- plogis(Xc %*% beta0x)
    sum(p) - n
  }
  tt <- uniroot(f, interval = c(-15, 15))
  tt$root
}

# Generate population with given linear characteristics
gen.Pop <- function (
    N ,      # Pop size 
    mnX ,    # vector of means
    SigmaX , # COV matrix
    betaY,   # beta-coefficients
    sigmaY,  # variance 
    n.r,     # S.r sample size
    beta.pr, # size measure coefficients
    n.c,     # S.c sample size
    beta.pc  # propensity score coefficients
) {
  X <- cbind(rep(1, N), mvrnorm(N, mnX, SigmaX))
  colnames(X) <- c("X0", paste0("X", (1 : length(mnX))))
  
  y <- rnorm(N, X %*% betaY, sigmaY)
  
  sz.pr <- plogis(X %*% beta.pr)
  pi.r <- inclusionprobabilities(sz.pr, n.r)
  
  Pop <- as_tibble(X)
  Pop <- Pop |> mutate(y = y, pi.r = pi.r)
  
  beta.pc[1] <- round(approx.int(model.matrix(~X1 + X2 + X3, Pop), beta.pc, n.c), 2)
  Pop$pi.c <- as.vector(plogis(model.matrix(~X1 + X2 + X3, Pop) %*% beta.pc))
  
  return(Pop)
}

# Draw samples from the population
draw.samples <- function (pop){
  pop$I.r <- 0
  pop$I.r[which(UPrandomsystematic(pop$pi.r) == 1)] <- 1
  pop$I.c <- rbinom(nrow(pop), 1, pop$pi.c)
  return(pop)
}

# ------------------------------------------------------------------------------
# 2. Extended Non-Linear & High-Dimensional Generator
# ------------------------------------------------------------------------------

gen.Pop.Nonlinear <- function(
    N = 10000,
    p = 10,
    nonlinear_y = FALSE,
    nonlinear_propensity = FALSE,
    common_support = c("high", "low"),
    b2 = NULL,
    n.r = 1000,
    n.c = 1000,
    seed = 100
) {
  common_support <- match.arg(common_support)
  if (!is.null(seed)) set.seed(seed)

  # Covariance matrix: Vladislav's 3x3 block for X1, X2, X3; noise for X4..Xp
  SigmaX <- diag(1, p)
  SigmaX[1, 2] <- SigmaX[2, 1] <- 0.5
  SigmaX[1, 3] <- SigmaX[3, 1] <- 0.3
  SigmaX[2, 3] <- SigmaX[3, 2] <- 0.3

  X <- MASS::mvrnorm(n = N, mu = rep(0, p), Sigma = SigmaX)
  colnames(X) <- paste0("X", 1:p)

  # Outcome Y
  if (!nonlinear_y) {
    mu_y <- 1.0 + X[, "X1"] + X[, "X2"] + X[, "X3"]
  } else {
    mu_y <- 1.0 + X[, "X1"] + X[, "X2"] + X[, "X3"] +
      0.5 * (X[, "X1"]^2) - 0.5 * sin(pi * X[, "X2"])
  }
  y <- rnorm(N, mean = mu_y, sd = 1.0)

  # Probability Reference Sample Sr Inclusion Probability pi.r
  sz.pr <- plogis(X[, "X1"] + X[, "X2"])
  pi.r <- sampling::inclusionprobabilities(sz.pr, n.r)

  # Convenience Sample Sc Response Propensity pi.c
  b1 <- if (common_support == "high") 1.0 else -1.0
  if (is.null(b2)) {
    b2 <- if (common_support == "high") -0.5 else -0.3
  }

  if (!nonlinear_propensity) {
    eta_c_raw <- b1 * X[, "X1"] + b2 * X[, "X2"]
  } else {
    eta_c_raw <- b1 * X[, "X1"] + b2 * X[, "X2"] +
      0.6 * X[, "X1"] * X[, "X2"] +
      0.5 * cos(pi * X[, "X1"]) -
      0.4 * (X[, "X2"]^2)
  }

  f_root <- function(b0) {
    sum(plogis(b0 + eta_c_raw)) - n.c
  }
  b0 <- uniroot(f_root, interval = c(-15, 15))$root
  pi.c <- plogis(b0 + eta_c_raw)

  pop_df <- as.data.frame(X)
  pop_df$y <- y
  pop_df$pi.r <- pi.r
  pop_df$pi.c <- pi.c
  pop_df$id <- 1:N

  return(pop_df)
}
