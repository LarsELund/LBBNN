polarized <- readRDS("tests_current/results/raisin_polarized.rds")
polarized_mild <- readRDS("tests_current/results/raisin_polarized_mild.rds")
polarized_dense <- readRDS("tests_current/results/raisin_polarized_dense.rds")
polarized_sparse <- readRDS("tests_current/results/raisin_polarized_sparse.rds")


balanced <- readRDS("tests_current/results/raisin_balanced.rds")
dense <- readRDS("tests_current/results/raisin_dense.rds")
sparse <- readRDS("tests_current/results/raisin_sparse.rds")

polarized_df <- data.frame(
  epoch = seq_along(polarized$density),
  density = polarized$density,
  type = "polarized"
)
polarized_mild_df <- data.frame(
  epoch = seq_along(polarized_mild$density),
  density = polarized_mild$density,
  type = "polarized_mild"
)

polarized_dense_df <- data.frame(
  epoch = seq_along(polarized_dense$density),
  density = polarized_dense$density,
  type = "polarized_dense"
)

polarized_sparse_df <- data.frame(
  epoch = seq_along(polarized_sparse$density),
  density = polarized_sparse$density,
  type = "polarized_sparse"
)


balanced_df <- data.frame(
  epoch = seq_along(balanced$density),
  density = balanced$density,
  type = "balanced"
)

dense_df <- data.frame(
  epoch = seq_along(dense$density),
  density = dense$density,
  type = "dense"
)

sparse_df <- data.frame(
  epoch = seq_along(sparse$density),
  density = sparse$density,
  type = "sparse"
)

plot_df <- rbind(balanced_df, dense_df,sparse_df,polarized_df,polarized_mild_df,
                 polarized_dense_df,polarized_sparse_df)

N <- dim(balanced_df)[1]

library(ggplot2)

ggplot(plot_df, aes(x = epoch, y = density, color = type)) +
  geom_line(linewidth = 0.5) 
  theme_bw(base_size = 14)



library(knitr)

results_df <- data.frame(
  Initialization = c("balanced", "dense", "polarized", "polarized_dense", 
                     "polarized_mild", "polarized_sparse", "sparse"),
  Accuracy = c(balanced$accuracy[N], dense$accuracy[N], polarized$accuracy[N],
               polarized_dense$accuracy[N], polarized_mild$accuracy[N], 
               polarized_sparse$accuracy[N], sparse$accuracy[N]),
  `Sparse accuracy` = c(balanced$sparse_accuracy[N], dense$sparse_accuracy[N],
                        polarized$sparse_accuracy[N],
                        polarized_dense$sparse_accuracy[N], 
                        polarized_mild$sparse_accuracy[N], 
                        polarized_sparse$sparse_accuracy[N],sparse$sparse_accuracy[N]),
  `Density (all paths)` = c(balanced$density[N], dense$density[N],polarized$density[N],
                            polarized_dense$density[N],polarized_mild$density[N],
                            polarized_sparse$density[N],sparse$density[N]),
  `Density (active paths)` = c(balanced$density_active_path[N],
                               dense$density_active_path[N],
                               polarized$density_active_path[N],
                               polarized_dense$density_active_path[N],
                               polarized_mild$density_active_path[N],
                               polarized_sparse$density_active_path[N],
                               sparse$density_active_path[N])
)
