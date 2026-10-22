test_that("gen_pop_nonlinear generates valid population and sample structures", {
  pop <- gen_pop_nonlinear(N = 500, p = 6, nonlinear_y = TRUE,
                           nonlinear_propensity = TRUE, common_support = "high",
                           n_r = 100, n_c = 100, seed = 42)
  expect_equal(nrow(pop), 500)
  expect_true(all(c("X1", "X2", "X3", "X4", "X5", "X6", "y", "pi.r", "pi.c") %in% colnames(pop)))
  expect_true(all(pop$pi.r > 0 & pop$pi.r < 1))
  expect_true(all(pop$pi.c > 0 & pop$pi.c < 1))

  # Drawing samples
  pop_sampled <- draw_combined_samples(pop, seed = 42)
  expect_true(all(c("I.r", "I.c") %in% colnames(pop_sampled)))
  expect_true(sum(pop_sampled$I.r) > 0)
  expect_true(sum(pop_sampled$I.c) > 0)
})

test_that("Composite loss functions evaluate and compute finite gradients", {
  batch_size <- 20
  input <- torch::torch_randn(c(batch_size, 1), requires_grad = TRUE)
  target2 <- torch::torch_tensor(matrix(c(rep(1, 10), rep(0, 10), rep(0.1, 20)), ncol = 2))
  target3 <- torch::torch_tensor(matrix(c(rep(1, 10), rep(0, 10), rep(1, 5), rep(0, 15), rep(0.1, 20)), ncol = 3))

  # 1. nll_stack
  l_stack <- nll_stack(input, target2, reduction = "sum")
  expect_true(is.numeric(l_stack$item()))
  expect_true(l_stack$item() > 0)
  l_stack$backward()
  expect_false(any(as.logical(torch::torch_isnan(input$grad))))

  # 2. nll_link1
  input$grad$zero_()
  l_link1 <- nll_link1(input, target2, reduction = "sum")
  expect_true(l_link1$item() > 0)
  l_link1$backward()
  expect_false(any(as.logical(torch::torch_isnan(input$grad))))

  # 3. nll_link2
  input$grad$zero_()
  l_link2 <- nll_link2(input, target2, reduction = "sum")
  expect_true(l_link2$item() > 0)
  l_link2$backward()
  expect_false(any(as.logical(torch::torch_isnan(input$grad))))

  # 4. nll_mip
  input$grad$zero_()
  mip_loss <- make_nll_mip(p_link = 0.5)
  l_mip <- mip_loss(input, target3, reduction = "sum")
  expect_true(l_mip$item() > 0)
  l_mip$backward()
  expect_false(any(as.logical(torch::torch_isnan(input$grad))))

  # 5. nll_pseudo
  input$grad$zero_()
  l_pseudo <- nll_pseudo(input, target2, reduction = "sum")
  expect_true(is.numeric(l_pseudo$item()))
  l_pseudo$backward()
  expect_false(any(as.logical(torch::torch_isnan(input$grad))))
})

test_that("build_propensity_dataset formats data correctly across all 5 methods", {
  pop <- gen_pop_nonlinear(N = 300, p = 4, n_r = 50, n_c = 50, seed = 123)
  pop <- draw_combined_samples(pop, seed = 123)

  for (m in c("STACK", "LINK1", "LINK2", "MIP", "PSEUDO")) {
    res <- build_propensity_dataset(pop, method = m, covar_cols = paste0("X", 1:4))
    expect_true(is.data.frame(res$data))
    expect_true(length(res$Sc_indices) > 0)
    if (m == "MIP") {
      expect_equal(res$target_cols, 3)
    } else {
      expect_equal(res$target_cols, 2)
    }
  }
})

test_that("Hajek estimator correctly computes population mean", {
  y <- c(10, 20, 30, 40)
  pi_hat <- c(0.1, 0.2, 0.2, 0.5) # weights: 10, 5, 5, 2
  # Weighted mean: (10*10 + 20*5 + 30*5 + 40*2) / (10 + 5 + 5 + 2) = (100 + 100 + 150 + 80) / 22 = 430 / 22 = 19.545
  h_est <- hajek_estimate(y, pi_hat)
  expect_equal(round(h_est, 3), round(430 / 22, 3))
})

test_that("LBBNN trains successfully with composite loss function", {
  set.seed(42)
  torch::torch_manual_seed(42)
  pop <- gen_pop_nonlinear(N = 400, p = 4, n_r = 80, n_c = 80, seed = 42)
  pop <- draw_combined_samples(pop, seed = 42)

  ds <- build_propensity_dataset(pop, method = "STACK", covar_cols = paste0("X", 1:4))
  dl <- get_dataloaders(ds$data, train_proportion = 1.0, train_batch_size = 64,
                        test_batch_size = 64, standardize = FALSE,
                        shuffle_train = TRUE, target_cols = ds$target_cols, seed = 42)

  model <- lbbnn_net(
    problem_type = "custom",
    sizes = c(4, 8, 1),
    prior = c(0.2, 0.2),
    std = c(1.0, 1.0),
    inclusion_inits = "balanced",
    input_skip = TRUE,
    flow = FALSE,
    raw_output = TRUE,
    nll = nll_stack,
    device = "cpu"
  )

  expect_error(
    train_lbbnn(epochs = 3, LBBNN = model, lr = 0.01, train_dl = dl$train_loader, verbose = FALSE),
    NA
  )

  # Check prediction
  preds <- predict(model, newdata = as.matrix(pop[pop$I.c == 1, paste0("X", 1:4)]),
                   draws = 10, link = torch::torch_sigmoid)
  expect_equal(dim(preds)[2], sum(pop$I.c == 1))
  pi_hat <- apply(as.matrix(preds[, , 1]$cpu()), 2, mean)
  expect_true(all(pi_hat >= 0 & pi_hat <= 1))
})
