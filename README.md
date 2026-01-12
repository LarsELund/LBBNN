
<!-- README.md is generated from README.Rmd. Please edit that file -->

# LBBNN

<!-- badges: start -->

[![](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![](https://img.shields.io/github/last-commit/LarsELund/LBBNN.svg)](https://github.com/LarsELund/LBBNN/commits/main)
[![](https://img.shields.io/github/languages/code-size/LarsELund/LBBNN.svg)](https://github.com/LarsELund/LBBNN)
[![R-CMD-check](https://github.com/LarsELund/LBBNN/workflows/R-CMD-check/badge.svg)](https://github.com/LarsELund/LBBNN/actions)
[![License:
MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<!-- badges: end -->

The goal of LBBNN is to implement Latent Bayesian Binary Neural Networks
(<https://openreview.net/pdf?id=d6kqUKzG3V>) in R, using the torch for R
package. Currently, standard LBBNNs are implemented. In the future, we
will also implement LBBNNs with input-skip (see
<https://arxiv.org/abs/2503.10496>).

## Installation

You can install the development version of LBBNN from
[GitHub](https://github.com/LarsELund/LBBNN) with:

``` r
# install.packages("pak")
pak::pak("LarsELund/LBBNN")
```

## Example

This example demonstrates how to implement a simple feed forward LBBNN
on the raisin dataset, using both the mean-field posterior and
normalizing flows. We start by demonstrating how to preprocess the data
so it can be used within the torch ecosystem.

``` r
library(LBBNN)
library(ggplot2)
library(torch)

#the get_dataloaders function takes a data.frame dataset as input, then splits the data
#in a training and validation set based on train_proportion, and returns torch dataloader 
#objects.
loaders <- get_dataloaders(raisin_dataset, train_proportion = 0.8, 
                           train_batch_size = 720, test_batch_size = 180)
train_loader <- loaders$train_loader
test_loader <- loaders$test_loader
```

To initialize the LBBNN, we need to define several hyperparameters.
First, the user must define what type of problem they are facing. This
could be either binary classification (as in this case), multiclass
classification (more than two classes), or regression (continuous
output). Next, the user needs to define a size vector. The first element
in the vector is the number of variables in the dataset (7 in this
case), the last element is the number of output neurons (1 here), and
the elements in between represent the number of neurons in the hidden
layer(s). Then, the user must define the prior inclusion probability for
each weight matrix (all weights share the same prior). This parameter is
important as it reflects prior beliefs about how dense the network
should be. The user also needs to define the prior standard deviation
for the weight and bias parameters. Lastly, the user must define the
initialization of the inclusion parameters.

``` r
problem <- 'binary classification'
sizes <- c(7, 5, 5, 1) #7 input variables, one hidden layer of 100 neurons, 1 output neuron.
inclusion_priors <-c(0.5, 0.5, 0.5) #one prior probability per weight matrix.
stds <- c(1, 1, 1) #prior standard deviation for each layer.
inclusion_inits <- matrix(rep(c(-10, 15), 3),nrow = 2, ncol = 3) #one low and high for each layer
device <- 'cpu' #can also be mps or gpu.
```

We are now ready to define the models. Here we define two models: one
with the mean-field posterior and one with normalizing flows:

``` r
torch_manual_seed(0)
model_input_skip <- lbbnn_net(problem_type = problem,sizes = sizes,prior = inclusion_priors,
                      inclusion_inits = inclusion_inits,input_skip = TRUE,std = stds,
                   flow = FALSE,device = device)
model_LBBNN <- lbbnn_net(problem_type = problem,sizes = sizes,prior = inclusion_priors,
                   inclusion_inits = inclusion_inits,input_skip = FALSE,std = stds,
                   flow = FALSE,device = device)
```

To train the models, one can use the function train_lbbnn. The function
takes number of epochs, model to train, learning rate, and training data
as arguments:

``` r
#model_input_skip$local_explanation = TRUE #to make sure we are using RELU
results_input_skip <- suppressMessages(train_lbbnn(epochs = 800,LBBNN = model_input_skip, lr = 0.01,train_dl = train_loader,device = device))
#save the model 
#torch::torch_save(model_input_skip$state_dict(), 
#paste(getwd(),'/R/saved_models/README_input_skip_example_model.pth',sep = ''))
```

To evaluate performance on the validation data, one can use the function
Validate_LBBNN. This function takes a model, number of samples for model
averaging, and the validation data as input.

``` r
validate_lbbnn(LBBNN = model_input_skip,num_samples = 100,test_dl = test_loader,device)
#> $accuracy_full_model
#> [1] 0.8666667
#> 
#> $accuracy_sparse
#> [1] 0.8888889
#> 
#> $density
#> [1] 0.1308411
#> 
#> $density_active_path
#> [1] 0.1028037
#validate_lbbnn(LBBNN = model_flows,num_samples = 1000,test_dl = test_loader,device)
```

Plot the global structure of the given model:

``` r
plot(model_input_skip,type = 'global',vertex_size = 13,edge_width = 0.6,label_size = 0.6)
```

<img src="man/figures/README-unnamed-chunk-6-1.png" width="100%" />

Note that only 4 of the 7 input variables are used.

This can also be seen using the summary function:

``` r
summary(model_input_skip)
#> Summary of lbbnn_net object:
#> -----------------------------------
#> Shows the number of times each variable was included from each layer
#> -----------------------------------
#> Then the average inclusion probability for each input from each layer
#> -----------------------------------
#> The final column shows the average inclusion probability across all layers
#> -----------------------------------
#>    L0 L1 L2    a0    a1    a2 a_avg
#> x0  0  0  0 0.138 0.069 0.036 0.097
#> x1  0  0  0 0.326 0.059 0.152 0.189
#> x2  0  1  0 0.077 0.335 0.079 0.195
#> x3  1  0  1 0.427 0.267 0.994 0.406
#> x4  1  1  0 0.327 0.351 0.058 0.313
#> x5  0  0  0 0.143 0.250 0.218 0.198
#> x6  1  0  1 0.232 0.202 0.986 0.287
#> The model took 10.89 seconds to train, using cpu
```

The user can also plot local explanations for each input variable
related to a prediction using plot():

``` r
x <- train_loader$dataset$tensors[[1]] #grab the dataset
ind <- 42
data <- x[ind,] #plot this specific data-point
plot(model_input_skip,type = 'local',data = data,num_samples = 100)
```

<img src="man/figures/README-unnamed-chunk-8-1.png" width="100%" />

Compute residuals: y_true - y_predicted

``` r
residuals(model_input_skip)[1:10]
#>  [1] -0.42256758  0.25760156 -0.06397691  0.41818994 -0.04891150  0.37335026
#>  [7] -0.07111212 -0.06033467 -0.12792712 -0.03229851
```

Get local explanations from some training data:

``` r
coef(model_input_skip,dataset = train_loader,inds = c(2,3,4,5,6))
#>         lower        mean      upper
#> x0  0.0000000  0.00000000  0.0000000
#> x1  0.0000000  0.00000000  0.0000000
#> x2 -1.1450108 -0.85363343 -0.1005854
#> x3 -2.7260612 -1.36945037 -0.4463915
#> x4 -0.7780398 -0.04011143  0.8547966
#> x5  0.0000000  0.00000000  0.0000000
#> x6 -2.4800578 -1.11293317 -0.5610269
```

Get predictions from the posterior:

``` r
predictions <- predict(model_input_skip,mpm = TRUE,newdata = test_loader,draws = 100)
dim(predictions) #shape is (draws,samples,classes)
#> [1] 100 180   1
```

Print the model:

``` r
print(model_input_skip)
#> 
#> ========================================
#>           LBBNN Model Summary           
#> ========================================
#> 
#> Module Overview:
#>   - An `nn_module` containing 343 parameters.
#> 
#> ---------------- Submodules ----------------
#>   - layers               : nn_module_list  # 305 parameters
#>   - layers.0             : lbbnn_linear    # 115 parameters
#>   - layers.1             : lbbnn_linear    # 190 parameters
#>   - act                  : nn_leaky_relu   # 0 parameters
#>   - out_layer            : lbbnn_linear    # 38 parameters
#>   - out                  : nn_sigmoid      # 0 parameters
#>   - loss_fn              : nn_bce_loss     # 0 parameters
#> 
#> Model Configuration:
#>   - LBBNN with input-skip 
#>   - Optimized using variational inference without normalizing flows 
#> 
#> Priors:
#>   - Prior inclusion probabilities per layer:  0.5, 0.5, 0.5 
#>   - Prior std dev for weights per layer:     1, 1, 1 
#> 
#> =================================================================
```
