test_that("Protected symbolic operators are numerically stable on extreme inputs", {
  skip_if_not(torch_available())
  extreme_vals <- c(-100.0, -20.0, -1e-8, 0.0, 1e-8, 20.0, 100.0)

  # Test plog, clog, pexp, psqrt, pinv, psquare
  for (fn in list(plog, clog, pexp, psqrt, pinv, psquare)) {
    z <- torch::torch_tensor(extreme_vals, dtype = torch::torch_float32(), requires_grad = TRUE)
    y <- fn(z)
    expect_true(all(as.logical(torch::torch_isfinite(y)$cpu())))
    loss <- torch::torch_sum(y)
    loss$backward()
    expect_true(all(as.logical(torch::torch_isfinite(z$grad)$cpu())))
  }

  # Test logic_and and logic_or in both product_tnorm and lukasiewicz modes
  for (mode in c("product_tnorm", "lukasiewicz")) {
    x <- torch::torch_tensor(
      matrix(c(-100, 0, 1, 100, 0, 1, 0, 1), nrow = 2, ncol = 4),
      dtype = torch::torch_float32(), requires_grad = TRUE
    )
    w <- torch::torch_tensor(
      matrix(c(1.0, -1.0, 0.5, -2.0, -1.0, 1.0, 0.0, 1.0), nrow = 2, ncol = 4),
      dtype = torch::torch_float32(), requires_grad = TRUE
    )
    g <- torch::torch_tensor(
      matrix(c(0.9, 0.8, 0.1, 0.5, 0.7, 0.9, 0.2, 0.8), nrow = 2, ncol = 4),
      dtype = torch::torch_float32(), requires_grad = TRUE
    )

    out_and <- logic_and(x, weight = w, gate = g, MPM = FALSE, mode = mode)
    expect_true(all(as.logical(torch::torch_isfinite(out_and)$cpu())))
    torch::torch_sum(out_and)$backward()
    expect_true(all(as.logical(torch::torch_isfinite(x$grad)$cpu())))
    expect_true(all(as.logical(torch::torch_isfinite(w$grad)$cpu())))
    expect_true(all(as.logical(torch::torch_isfinite(g$grad)$cpu())))

    out_or <- logic_or(x$detach(), weight = w$detach(), gate = g$detach(), MPM = TRUE, mode = mode)
    expect_true(all(as.logical(torch::torch_isfinite(out_or)$cpu())))
  }
})

test_that("Backward compatibility is 100% preserved when activations = NULL", {
  skip_if_not(torch_available())
  set.seed(42)
  torch::torch_manual_seed(42)
  x <- torch::torch_randn(16, 4)
  y <- torch::torch_randn(16)
  dl <- torch::dataloader(torch::tensor_dataset(x, y), batch_size = 8, shuffle = FALSE)

  sizes <- c(4, 6, 1)
  priors <- c(0.5, 0.5)
  stds <- c(1.0, 1.0)
  inits <- matrix(rep(c(-1, 1), 2), nrow = 2, ncol = 2)

  net <- lbbnn_net(
    problem_type = "regression",
    sizes = sizes,
    prior = priors,
    std = stds,
    inclusion_inits = inits,
    input_skip = TRUE,
    flow = FALSE,
    device = "cpu",
    activations = NULL
  )
  expect_false(net$is_symbolic)
  res_tr <- suppressMessages(train_lbbnn(epochs = 2, LBBNN = net, lr = 0.01, train_dl = dl))
  expect_length(res_tr$loss, 2)
  res_val <- validate_lbbnn(LBBNN = net, num_samples = 2, test_dl = dl)
  expect_true(is.finite(res_val$validation_error))
})

test_that("Toy 2-variable Kepler recovery outputs x1^(1/3) * x2^(2/3)", {
  skip_if_not(torch_available())
  set.seed(123)
  torch::torch_manual_seed(123)

  n <- 300
  x_mat <- matrix(runif(n * 2, min = 0.5, max = 2.5), nrow = n, ncol = 2)
  colnames(x_mat) <- c("x1", "x2")
  y_vec <- (x_mat[, 1]^(1 / 3)) * (x_mat[, 2]^(2 / 3))

  x_ten <- torch::torch_tensor(x_mat, dtype = torch::torch_float32())
  y_ten <- torch::torch_tensor(y_vec, dtype = torch::torch_float32())
  dl <- torch::dataloader(torch::tensor_dataset(x_ten, y_ten), batch_size = 100, shuffle = TRUE)

  sizes <- c(2, 4, 3, 1)
  layer_acts <- list(
    list(clog = 2, id = 2),
    list(exp = 2, id = 1)
  )
  priors <- compute_symbolic_priors(sizes, layer_acts, base_prior = 0.3, input_skip = TRUE)
  inits <- matrix(rep(c(0, 2), 3), nrow = 2, ncol = 3)

  model <- lbbnn_net(
    problem_type = "regression",
    sizes = sizes,
    prior = priors,
    std = c(1.0, 1.0, 1.0),
    inclusion_inits = inits,
    input_skip = TRUE,
    flow = FALSE,
    device = "cpu",
    activations = layer_acts
  )

  train_symbolic_lbbnn(
    epochs = 400,
    LBBNN = model,
    lr = 0.02,
    train_dl = dl,
    debias_epochs = 100,
    snap_rationals = TRUE,
    verbose = FALSE
  )

  sym <- extract_symbolic_mpm(model, var_names = c("x1", "x2"))
  expect_s3_class(sym, "lbbnn_symbolic")
  expect_equal(sym$formula_str, "x1^(1/3) * x2^(2/3)")

  # Verify evaluable R expression matches true values
  newdata <- as.data.frame(x_mat[1:10, , drop = FALSE])
  y_eval <- eval(sym$r_expr, envir = newdata)
  expect_lt(max(abs(y_eval - y_vec[1:10])), 1e-5)
})

test_that("Toy 4-variable Logic recovery outputs (x1 & !x2)", {
  skip_if_not(torch_available())
  set.seed(123)
  torch::torch_manual_seed(123)

  n <- 300
  x_mat <- matrix(rbinom(n * 4, size = 1, prob = 0.5), nrow = n, ncol = 4)
  colnames(x_mat) <- c("x1", "x2", "x3", "x4")
  y_vec <- 2.0 * (x_mat[, 1] * (1.0 - x_mat[, 2]))

  x_ten <- torch::torch_tensor(x_mat, dtype = torch::torch_float32())
  y_ten <- torch::torch_tensor(y_vec, dtype = torch::torch_float32())
  dl <- torch::dataloader(torch::tensor_dataset(x_ten, y_ten), batch_size = 100, shuffle = TRUE)

  sizes <- c(4, 5, 1)
  layer_acts <- list(list(logic_and = 4, id = 1))
  priors <- compute_symbolic_priors(sizes, layer_acts, base_prior = 0.25, input_skip = TRUE)
  inits <- matrix(rep(c(-0.5, 1.5), 2), nrow = 2, ncol = 2)

  model <- lbbnn_net(
    problem_type = "regression",
    sizes = sizes,
    prior = priors,
    std = c(1.0, 1.0),
    inclusion_inits = inits,
    input_skip = TRUE,
    flow = FALSE,
    device = "cpu",
    activations = layer_acts
  )

  train_symbolic_lbbnn(
    epochs = 400,
    LBBNN = model,
    lr = 0.025,
    train_dl = dl,
    debias_epochs = 100,
    snap_rationals = TRUE,
    verbose = FALSE
  )

  sym <- extract_symbolic_mpm(model, var_names = c("x1", "x2", "x3", "x4"))
  expect_s3_class(sym, "lbbnn_symbolic")
  expect_true(grepl("\\(x1 & !x2\\)", sym$formula_str))
  expect_equal(sym$formula_str, "2 * (x1 & !x2)")

  newdata <- as.data.frame(x_mat[1:10, , drop = FALSE])
  y_eval <- eval(sym$r_expr, envir = newdata)
  expect_lt(max(abs(y_eval - y_vec[1:10])), 1e-5)
})

