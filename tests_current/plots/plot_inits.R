library(ggplot2)

sigmoid <- function(x) 1/(1 + exp(-x))

configs <- data.frame(
  type = c(
    "polarized",
    "polarized_mild",
    "polarized_sparse",
    "polarized_dense",
    "dense",
    "sparse",
    "balanced"
  ),
  a = c(-10, -3, -10, -5, 1.5, -2.5, -1),
  b = c(10,  3,   5, 10, 2.5, -1.5, 1)
)

set.seed(42)

n <- 100000

samples <- do.call(rbind,
                   lapply(seq_len(nrow(configs)), function(i) {
                     
                     z <- runif(n, configs$a[i], configs$b[i])
                     
                     data.frame(
                       type = configs$type[i],
                       p = sigmoid(z)
                     )
                   })
)

ggplot(samples, aes(x = p)) +
  geom_histogram(
    bins = 50,
    color = "black",
    fill = "grey75"
  ) +
  facet_wrap(~type, ncol = 4) +
  labs(
    x = expression(p),
    y = "Count"
  ) +
  theme_bw(base_size = 12)