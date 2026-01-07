### Tutorial 2: Simulation study with non-linear effects

#check if the package is already installed.
if (!requireNamespace("LBBNN", quietly = TRUE)) {
  install.packages("LBBNN")
}
library(LBBNN)

i <- 1000
j <- 15

set.seed(42)
torch::torch_manual_seed(42)
#generate some data
X <- matrix(runif(i * j, 0, 0.5), ncol = j)

#make some X relevant for prediction
y_base <- (- 3 +  0.1 * log(abs(X[, 1])) + 3 * cos(X[, 2]) + 2 * X[, 3] * X[, 4]
           + X[, 5] -  X[, 6] ** 2 + rnorm(i, sd = 0.1))
hist(y_base)
y <- c()
# change y to 0 and 1
y[y_base > median(y_base)] <- 1
y[y_base <= median(y_base)] <- 0

sim_data <- as.data.frame(X)
sim_data <- cbind(sim_data, y)

loaders <- get_dataloaders(sim_data, train_proportion = 0.9,
                           train_batch_size = 450, test_batch_size = 100,
                           standardize = FALSE)
train_loader <- loaders$train_loader
test_loader  <- loaders$test_loader

problem <- "binary classification"
sizes <- c(j, 5, 5, 1) # 2 hidden layers, 5 neurons in each
incl_priors <- c(0.5, 0.5, 0.5) #prior inclusion probs for each weight matrix
stds <- c(1, 1, 1) #prior distribution for the standard deviation of the weights
incl_inits <- matrix(rep(c(-10, 10), 3), nrow = 2, ncol = 3) #initializations for inclusion params
device <- "cpu" #can also be 'gpu' or 'mps'


model_input_skip <- lbbnn_net(problem_type = problem, sizes = sizes,
                              prior = incl_priors,
                              inclusion_inits = incl_inits, input_skip = TRUE,
                              std = stds, flow = TRUE, device = device,
                              bias_inclusion_prob = FALSE)

train_lbbnn(epochs = 1500, LBBNN = model_input_skip,
            lr = 0.005, train_dl = train_loader, device = device)

validate_lbbnn(LBBNN = model_input_skip, num_samples = 100, test_dl = test_loader,
               device)

plot(model_input_skip, type = "global", vertex_size = 10,
     edge_width = 0.4, label_size = 0.3)
