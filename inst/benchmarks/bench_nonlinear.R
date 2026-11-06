# =============================================================================
# Benchmark C: Friedman #1 Nonlinear Symbolic Benchmark
# Y = 10 * sin(pi * X1 * X2) + 20 * (X3 - 0.5)^2 + 10 * X4 + 5 * X5 + eps
# =============================================================================

generate_friedman1_data <- function(n = 500, n_test = 300, p = 10, sigma = 1.0, seed = 1) {
  set.seed(seed)
  n_tot <- n + n_test
  X_all <- matrix(runif(n_tot * p, min = 0.05, max = 1.0), nrow = n_tot, ncol = p)
  colnames(X_all) <- paste0("x", seq_len(p))

  y_clean <- 10.0 * sin(pi * X_all[, 1] * X_all[, 2]) +
    20.0 * (X_all[, 3] - 0.5)^2 +
    10.0 * X_all[, 4] +
    5.0 * X_all[, 5]
  y_all <- y_clean + rnorm(n_tot, mean = 0.0, sd = sigma)

  idx_train <- seq_len(n)
  idx_test <- seq.int(n + 1L, n_tot)
  list(
    x_train = X_all[idx_train, , drop = FALSE],
    y_train = y_all[idx_train],
    x_test = X_all[idx_test, , drop = FALSE],
    y_test = y_all[idx_test],
    true_vars = paste0("x", 1:5)
  )
}

run_single_nonlinear_seed <- function(seed = 1,
                                      n = 500,
                                      n_test = 300,
                                      p = 10,
                                      sigma = 1.0,
                                      epochs = 300,
                                      debias_epochs = 150,
                                      lr = 0.02,
                                      device = get_compute_device(),
                                      verbose = FALSE) {
  set.seed(seed)
  torch::torch_manual_seed(seed)

  dat <- generate_friedman1_data(n = n, n_test = n_test, p = p, sigma = sigma, seed = seed)
  x_tr <- torch::torch_tensor(dat$x_train, dtype = torch::torch_float32(), device = device)
  y_tr <- torch::torch_tensor(dat$y_train, dtype = torch::torch_float32(), device = device)
  x_te <- torch::torch_tensor(dat$x_test, dtype = torch::torch_float32(), device = device)

  train_ds <- torch::tensor_dataset(x_tr, y_tr)
  train_dl <- torch::dataloader(train_ds, batch_size = 128L, shuffle = TRUE)

  sizes <- c(p, p + 6L, 12L, 1L)
  layer_acts <- list(
    list(clog = p, id = 4L, sin = 2L),
    list(exp = 4L, sin = 4L, square = 2L, id = 2L)
  )
  priors <- compute_symbolic_priors(
    sizes = sizes,
    layer_activations_list = layer_acts,
    base_prior = 0.3,
    depth_decay = 0.3,
    complexity_weight = 0.25,
    input_skip = TRUE
  )
  stds <- c(1.0, 1.0, 1.0)
  inclusion_inits <- matrix(rep(c(0.0, 2.0), 3), nrow = 2, ncol = 3)

  model <- lbbnn_net(
    problem_type = "regression",
    sizes = sizes,
    prior = priors,
    std = stds,
    inclusion_inits = inclusion_inits,
    input_skip = TRUE,
    flow = FALSE,
    device = device,
    activations = layer_acts
  )

  train_symbolic_lbbnn(
    epochs = epochs,
    LBBNN = model,
    lr = lr,
    train_dl = train_dl,
    device = device,
    debias_epochs = debias_epochs,
    snap_rationals = TRUE,
    prune_bic = FALSE,
    verbose = verbose
  )

  sym <- extract_symbolic_mpm(model, digits = 3, simplify = TRUE)
  torch::with_no_grad({
    pred_test <- as.numeric(model(x_te, MPM = TRUE)$squeeze()$cpu())
  })
  test_rmse <- sqrt(mean((pred_test - dat$y_test)^2))

  # Check variable inclusion across active paths
  f_str <- sym$formula_str
  inc_vars <- vapply(paste0("x", seq_len(p)), function(v) {
    grepl(paste0("\\b", v, "\\b"), f_str)
  }, logical(1))
  tpr <- mean(inc_vars[1:5])
  fpr <- mean(inc_vars[6:p])

  data.frame(
    seed = seed,
    benchmark = "friedman1",
    tpr = tpr,
    fpr = fpr,
    test_rmse = test_rmse,
    active_density = model$density_active_path(),
    active_edges = nrow(sym$active_paths_df),
    formula = sym$formula_str,
    stringsAsFactors = FALSE
  )
}

if (sys.nframe() == 0L) {
  pkgload::load_all(".", quiet = TRUE)
  message("Running 3-seed smoke test for Benchmark C (Friedman #1)...")
  res <- do.call(rbind, lapply(1:3, function(s) run_single_nonlinear_seed(seed = s, epochs = 200)))
  print(res[, c("seed", "tpr", "fpr", "test_rmse", "active_density", "active_edges")])
}

