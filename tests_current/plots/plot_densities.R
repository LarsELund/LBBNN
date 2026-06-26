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
  loss = polarized$loss,
  type = "polarized"
)
polarized_mild_df <- data.frame(
  epoch = seq_along(polarized_mild$density),
  density = polarized_mild$density,
  loss = polarized_mild$loss,
  type = "polarized_mild"
)

polarized_dense_df <- data.frame(
  epoch = seq_along(polarized_dense$density),
  density = polarized_dense$density,
  loss = polarized_dense$loss,
  type = "polarized_dense"
)

polarized_sparse_df <- data.frame(
  epoch = seq_along(polarized_sparse$density),
  density = polarized_sparse$density,
  loss = polarized_sparse$loss,
  type = "polarized_sparse"
)


balanced_df <- data.frame(
  epoch = seq_along(balanced$density),
  density = balanced$density,
  loss = balanced$loss,
  type = "balanced"
)

dense_df <- data.frame(
  epoch = seq_along(dense$density),
  density = dense$density,
  loss = dense$loss,
  type = "dense"
)

sparse_df <- data.frame(
  epoch = seq_along(sparse$density),
  density = sparse$density,
  loss = sparse$loss,
  type = "sparse"
)

plot_df <- rbind(balanced_df, dense_df,sparse_df,polarized_df,polarized_mild_df,
                 polarized_dense_df,polarized_sparse_df)

N <- dim(balanced_df)[1]

library(ggplot2)

ggplot(plot_df, aes(x = epoch, y = density, color = type)) +
  geom_line(linewidth = 0.5) +
  theme_bw(base_size = 14)
  
ggplot(plot_df, aes(x = epoch, y = loss, color = type)) +
  geom_line(linewidth = 0.5) +
  theme_bw(base_size = 14)  

  
ggplot(plot_df, aes(x = epoch, y = density, color = type)) +
  geom_line(linewidth = 0.5) +
  theme_bw(base_size = 14) +
  coord_cartesian(xlim = c(1000, N), ylim = c(0, 0.1))



