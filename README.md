
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
loaders <- get_dataloaders(Raisin_Dataset,train_proportion = 0.8,train_batch_size = 720,test_batch_size = 180)
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
sizes <- c(7,5,5,1) #7 input variables, one hidden layer of 100 neurons, 1 output neuron.
inclusion_priors <-c(0.5,0.5,0.5) #one prior probability per weight matrix.
stds <- c(1,1,1) #prior standard deviation for each layer.
inclusion_inits <- matrix(rep(c(-10,15),3),nrow = 2,ncol = 3) #one low and high for each layer
device <- 'cpu' #can also be mps or gpu.
```

We are now ready to define the models. Here we define two models: one
with the mean-field posterior and one with normalizing flows:

``` r
torch_manual_seed(0)
model_input_skip <- LBBNN_Net(problem_type = problem,sizes = sizes,prior = inclusion_priors,
                      inclusion_inits = inclusion_inits,input_skip = TRUE,std = stds,
                   flow = FALSE,device = device)
model_LBBNN <- LBBNN_Net(problem_type = problem,sizes = sizes,prior = inclusion_priors,
                   inclusion_inits = inclusion_inits,input_skip = FALSE,std = stds,
                   flow = FALSE,device = device)
```

To train the models, one can use the function train_LBBNN. The function
takes number of epochs, model to train, learning rate, and training data
as arguments:

``` r
#model_input_skip$local_explanation = TRUE #to make sure we are using RELU
results_input_skip <- train_LBBNN(epochs = 800,LBBNN = model_input_skip, lr = 0.005,train_dl = train_loader,device = device)
#> 
#> Epoch 1, training: loss = 1217.83057, acc = 0.49167, density = 0.59813
#> 
#> Epoch 2, training: loss = 1213.35803, acc = 0.49167, density = 0.59813
#> 
#> Epoch 3, training: loss = 1208.90503, acc = 0.49167, density = 0.59813
#> 
#> Epoch 4, training: loss = 1204.64014, acc = 0.49583, density = 0.59813
#> 
#> Epoch 5, training: loss = 1200.25708, acc = 0.51944, density = 0.59813
#> 
#> Epoch 6, training: loss = 1195.87708, acc = 0.54583, density = 0.59813
#> 
#> Epoch 7, training: loss = 1191.60315, acc = 0.58611, density = 0.59813
#> 
#> Epoch 8, training: loss = 1187.16003, acc = 0.62222, density = 0.59813
#> 
#> Epoch 9, training: loss = 1182.94495, acc = 0.64583, density = 0.59813
#> 
#> Epoch 10, training: loss = 1178.86328, acc = 0.66667, density = 0.59813
#> 
#> Epoch 11, training: loss = 1174.29993, acc = 0.69861, density = 0.59813
#> 
#> Epoch 12, training: loss = 1169.62549, acc = 0.73333, density = 0.59813
#> 
#> Epoch 13, training: loss = 1165.11609, acc = 0.75139, density = 0.59813
#> 
#> Epoch 14, training: loss = 1160.61450, acc = 0.76667, density = 0.59813
#> 
#> Epoch 15, training: loss = 1155.02344, acc = 0.77917, density = 0.59813
#> 
#> Epoch 16, training: loss = 1151.27515, acc = 0.79861, density = 0.59813
#> 
#> Epoch 17, training: loss = 1145.92651, acc = 0.80278, density = 0.59813
#> 
#> Epoch 18, training: loss = 1141.70276, acc = 0.80972, density = 0.59813
#> 
#> Epoch 19, training: loss = 1136.54761, acc = 0.83194, density = 0.59813
#> 
#> Epoch 20, training: loss = 1131.27686, acc = 0.82778, density = 0.59813
#> 
#> Epoch 21, training: loss = 1127.13098, acc = 0.83750, density = 0.59813
#> 
#> Epoch 22, training: loss = 1121.74976, acc = 0.84444, density = 0.59813
#> 
#> Epoch 23, training: loss = 1116.72534, acc = 0.84028, density = 0.59813
#> 
#> Epoch 24, training: loss = 1111.70557, acc = 0.85000, density = 0.59813
#> 
#> Epoch 25, training: loss = 1106.40210, acc = 0.85417, density = 0.59813
#> 
#> Epoch 26, training: loss = 1100.99597, acc = 0.85000, density = 0.59813
#> 
#> Epoch 27, training: loss = 1094.46741, acc = 0.85417, density = 0.59813
#> 
#> Epoch 28, training: loss = 1090.57764, acc = 0.85417, density = 0.59813
#> 
#> Epoch 29, training: loss = 1085.01843, acc = 0.85833, density = 0.59813
#> 
#> Epoch 30, training: loss = 1081.05823, acc = 0.85972, density = 0.59813
#> 
#> Epoch 31, training: loss = 1074.84412, acc = 0.86250, density = 0.59813
#> 
#> Epoch 32, training: loss = 1068.24377, acc = 0.85972, density = 0.59813
#> 
#> Epoch 33, training: loss = 1063.71094, acc = 0.86389, density = 0.59813
#> 
#> Epoch 34, training: loss = 1058.33643, acc = 0.86250, density = 0.59813
#> 
#> Epoch 35, training: loss = 1053.96436, acc = 0.85972, density = 0.59813
#> 
#> Epoch 36, training: loss = 1047.27380, acc = 0.86528, density = 0.59813
#> 
#> Epoch 37, training: loss = 1043.09009, acc = 0.86111, density = 0.59813
#> 
#> Epoch 38, training: loss = 1036.86267, acc = 0.86250, density = 0.59813
#> 
#> Epoch 39, training: loss = 1033.08557, acc = 0.85833, density = 0.59813
#> 
#> Epoch 40, training: loss = 1027.79126, acc = 0.86528, density = 0.59813
#> 
#> Epoch 41, training: loss = 1020.79791, acc = 0.86806, density = 0.59813
#> 
#> Epoch 42, training: loss = 1018.23596, acc = 0.86389, density = 0.59813
#> 
#> Epoch 43, training: loss = 1011.64429, acc = 0.86389, density = 0.59813
#> 
#> Epoch 44, training: loss = 1012.06305, acc = 0.86250, density = 0.59813
#> 
#> Epoch 45, training: loss = 1001.86487, acc = 0.86667, density = 0.59813
#> 
#> Epoch 46, training: loss = 1001.38794, acc = 0.86250, density = 0.59813
#> 
#> Epoch 47, training: loss = 997.34381, acc = 0.86944, density = 0.59813
#> 
#> Epoch 48, training: loss = 992.00641, acc = 0.86667, density = 0.59813
#> 
#> Epoch 49, training: loss = 987.55682, acc = 0.86667, density = 0.59813
#> 
#> Epoch 50, training: loss = 987.88110, acc = 0.86528, density = 0.59813
#> 
#> Epoch 51, training: loss = 981.78839, acc = 0.86806, density = 0.59813
#> 
#> Epoch 52, training: loss = 978.59338, acc = 0.86806, density = 0.59813
#> 
#> Epoch 53, training: loss = 975.25745, acc = 0.87083, density = 0.59813
#> 
#> Epoch 54, training: loss = 972.38757, acc = 0.87083, density = 0.59813
#> 
#> Epoch 55, training: loss = 975.27960, acc = 0.86528, density = 0.59813
#> 
#> Epoch 56, training: loss = 970.52563, acc = 0.86667, density = 0.59813
#> 
#> Epoch 57, training: loss = 968.90393, acc = 0.87083, density = 0.59813
#> 
#> Epoch 58, training: loss = 965.97736, acc = 0.86667, density = 0.59813
#> 
#> Epoch 59, training: loss = 964.91687, acc = 0.86528, density = 0.59813
#> 
#> Epoch 60, training: loss = 965.47498, acc = 0.86667, density = 0.59813
#> 
#> Epoch 61, training: loss = 960.12000, acc = 0.86111, density = 0.59813
#> 
#> Epoch 62, training: loss = 957.58582, acc = 0.87083, density = 0.59813
#> 
#> Epoch 63, training: loss = 955.92627, acc = 0.86667, density = 0.59813
#> 
#> Epoch 64, training: loss = 956.63867, acc = 0.85972, density = 0.59813
#> 
#> Epoch 65, training: loss = 953.88269, acc = 0.86389, density = 0.59813
#> 
#> Epoch 66, training: loss = 946.78766, acc = 0.86250, density = 0.59813
#> 
#> Epoch 67, training: loss = 951.04144, acc = 0.86667, density = 0.59813
#> 
#> Epoch 68, training: loss = 945.19073, acc = 0.86111, density = 0.59813
#> 
#> Epoch 69, training: loss = 952.78766, acc = 0.86111, density = 0.59813
#> 
#> Epoch 70, training: loss = 953.89301, acc = 0.85833, density = 0.59813
#> 
#> Epoch 71, training: loss = 946.99835, acc = 0.86528, density = 0.59813
#> 
#> Epoch 72, training: loss = 959.12268, acc = 0.86389, density = 0.59813
#> 
#> Epoch 73, training: loss = 945.20117, acc = 0.86528, density = 0.59813
#> 
#> Epoch 74, training: loss = 949.96045, acc = 0.86111, density = 0.59813
#> 
#> Epoch 75, training: loss = 949.29065, acc = 0.86250, density = 0.59813
#> 
#> Epoch 76, training: loss = 947.97632, acc = 0.86250, density = 0.59813
#> 
#> Epoch 77, training: loss = 950.06024, acc = 0.86667, density = 0.59813
#> 
#> Epoch 78, training: loss = 951.10254, acc = 0.86806, density = 0.59813
#> 
#> Epoch 79, training: loss = 941.79468, acc = 0.86389, density = 0.59813
#> 
#> Epoch 80, training: loss = 949.14355, acc = 0.86111, density = 0.59813
#> 
#> Epoch 81, training: loss = 942.48682, acc = 0.86111, density = 0.59813
#> 
#> Epoch 82, training: loss = 943.67029, acc = 0.86667, density = 0.59813
#> 
#> Epoch 83, training: loss = 941.57019, acc = 0.86111, density = 0.59813
#> 
#> Epoch 84, training: loss = 938.74487, acc = 0.86389, density = 0.59813
#> 
#> Epoch 85, training: loss = 944.89099, acc = 0.87361, density = 0.59813
#> 
#> Epoch 86, training: loss = 945.87891, acc = 0.86389, density = 0.59813
#> 
#> Epoch 87, training: loss = 936.35754, acc = 0.86250, density = 0.59813
#> 
#> Epoch 88, training: loss = 941.03217, acc = 0.86806, density = 0.59813
#> 
#> Epoch 89, training: loss = 937.42065, acc = 0.86528, density = 0.59813
#> 
#> Epoch 90, training: loss = 937.74500, acc = 0.87083, density = 0.59813
#> 
#> Epoch 91, training: loss = 937.97900, acc = 0.86944, density = 0.59813
#> 
#> Epoch 92, training: loss = 937.32812, acc = 0.86944, density = 0.59813
#> 
#> Epoch 93, training: loss = 937.13348, acc = 0.86667, density = 0.59813
#> 
#> Epoch 94, training: loss = 930.89429, acc = 0.86528, density = 0.59813
#> 
#> Epoch 95, training: loss = 935.48352, acc = 0.86806, density = 0.59813
#> 
#> Epoch 96, training: loss = 933.60327, acc = 0.86528, density = 0.59813
#> 
#> Epoch 97, training: loss = 940.79486, acc = 0.86944, density = 0.59813
#> 
#> Epoch 98, training: loss = 933.69684, acc = 0.86389, density = 0.59813
#> 
#> Epoch 99, training: loss = 934.48340, acc = 0.86111, density = 0.59813
#> 
#> Epoch 100, training: loss = 936.25232, acc = 0.86528, density = 0.59813
#> 
#> Epoch 101, training: loss = 928.34497, acc = 0.86806, density = 0.59813
#> 
#> Epoch 102, training: loss = 928.11969, acc = 0.86806, density = 0.59813
#> 
#> Epoch 103, training: loss = 932.05627, acc = 0.86528, density = 0.59813
#> 
#> Epoch 104, training: loss = 930.64929, acc = 0.86528, density = 0.59813
#> 
#> Epoch 105, training: loss = 929.68750, acc = 0.87083, density = 0.59813
#> 
#> Epoch 106, training: loss = 929.55432, acc = 0.86667, density = 0.59813
#> 
#> Epoch 107, training: loss = 929.51440, acc = 0.87083, density = 0.59813
#> 
#> Epoch 108, training: loss = 928.47791, acc = 0.86389, density = 0.59813
#> 
#> Epoch 109, training: loss = 928.42310, acc = 0.86667, density = 0.59813
#> 
#> Epoch 110, training: loss = 920.26794, acc = 0.86944, density = 0.59813
#> 
#> Epoch 111, training: loss = 924.19989, acc = 0.86806, density = 0.59813
#> 
#> Epoch 112, training: loss = 925.79016, acc = 0.87222, density = 0.59813
#> 
#> Epoch 113, training: loss = 923.87488, acc = 0.86944, density = 0.59813
#> 
#> Epoch 114, training: loss = 924.18591, acc = 0.86389, density = 0.59813
#> 
#> Epoch 115, training: loss = 926.11719, acc = 0.86806, density = 0.59813
#> 
#> Epoch 116, training: loss = 921.79651, acc = 0.86667, density = 0.59813
#> 
#> Epoch 117, training: loss = 918.90503, acc = 0.86528, density = 0.58879
#> 
#> Epoch 118, training: loss = 924.58118, acc = 0.86944, density = 0.58879
#> 
#> Epoch 119, training: loss = 919.15295, acc = 0.86528, density = 0.58879
#> 
#> Epoch 120, training: loss = 922.65735, acc = 0.86944, density = 0.58879
#> 
#> Epoch 121, training: loss = 918.52600, acc = 0.87222, density = 0.58879
#> 
#> Epoch 122, training: loss = 920.97070, acc = 0.86944, density = 0.58879
#> 
#> Epoch 123, training: loss = 922.44257, acc = 0.86250, density = 0.58879
#> 
#> Epoch 124, training: loss = 921.20532, acc = 0.87083, density = 0.58879
#> 
#> Epoch 125, training: loss = 916.46564, acc = 0.86806, density = 0.58879
#> 
#> Epoch 126, training: loss = 919.85254, acc = 0.87083, density = 0.58879
#> 
#> Epoch 127, training: loss = 911.04523, acc = 0.86806, density = 0.58879
#> 
#> Epoch 128, training: loss = 914.51416, acc = 0.86528, density = 0.58879
#> 
#> Epoch 129, training: loss = 916.79401, acc = 0.86667, density = 0.58879
#> 
#> Epoch 130, training: loss = 912.52942, acc = 0.87083, density = 0.58879
#> 
#> Epoch 131, training: loss = 916.96021, acc = 0.87083, density = 0.58879
#> 
#> Epoch 132, training: loss = 915.90625, acc = 0.87083, density = 0.58879
#> 
#> Epoch 133, training: loss = 913.52478, acc = 0.86944, density = 0.58879
#> 
#> Epoch 134, training: loss = 909.83887, acc = 0.86944, density = 0.58879
#> 
#> Epoch 135, training: loss = 910.01233, acc = 0.86667, density = 0.58879
#> 
#> Epoch 136, training: loss = 912.50110, acc = 0.86667, density = 0.58879
#> 
#> Epoch 137, training: loss = 906.50580, acc = 0.87083, density = 0.58879
#> 
#> Epoch 138, training: loss = 912.10364, acc = 0.86389, density = 0.58879
#> 
#> Epoch 139, training: loss = 908.59930, acc = 0.87361, density = 0.58879
#> 
#> Epoch 140, training: loss = 908.25360, acc = 0.86806, density = 0.58879
#> 
#> Epoch 141, training: loss = 908.67993, acc = 0.86528, density = 0.58879
#> 
#> Epoch 142, training: loss = 905.46613, acc = 0.87222, density = 0.58879
#> 
#> Epoch 143, training: loss = 905.63818, acc = 0.86528, density = 0.58879
#> 
#> Epoch 144, training: loss = 907.32788, acc = 0.86806, density = 0.58879
#> 
#> Epoch 145, training: loss = 907.91309, acc = 0.86389, density = 0.58879
#> 
#> Epoch 146, training: loss = 901.65430, acc = 0.87083, density = 0.58879
#> 
#> Epoch 147, training: loss = 905.77808, acc = 0.86667, density = 0.58879
#> 
#> Epoch 148, training: loss = 903.09070, acc = 0.86250, density = 0.58879
#> 
#> Epoch 149, training: loss = 905.50812, acc = 0.86250, density = 0.58879
#> 
#> Epoch 150, training: loss = 901.41199, acc = 0.86111, density = 0.58879
#> 
#> Epoch 151, training: loss = 902.52246, acc = 0.86667, density = 0.58879
#> 
#> Epoch 152, training: loss = 901.09680, acc = 0.86806, density = 0.58879
#> 
#> Epoch 153, training: loss = 903.23840, acc = 0.86944, density = 0.58879
#> 
#> Epoch 154, training: loss = 901.75037, acc = 0.86111, density = 0.58879
#> 
#> Epoch 155, training: loss = 899.20886, acc = 0.86528, density = 0.58879
#> 
#> Epoch 156, training: loss = 897.94116, acc = 0.86944, density = 0.58879
#> 
#> Epoch 157, training: loss = 900.19659, acc = 0.86667, density = 0.58879
#> 
#> Epoch 158, training: loss = 901.89185, acc = 0.86528, density = 0.58879
#> 
#> Epoch 159, training: loss = 896.16626, acc = 0.87361, density = 0.58879
#> 
#> Epoch 160, training: loss = 895.67456, acc = 0.86667, density = 0.58879
#> 
#> Epoch 161, training: loss = 896.62695, acc = 0.86667, density = 0.58879
#> 
#> Epoch 162, training: loss = 894.71686, acc = 0.87083, density = 0.58879
#> 
#> Epoch 163, training: loss = 895.78772, acc = 0.86528, density = 0.58879
#> 
#> Epoch 164, training: loss = 894.29218, acc = 0.86111, density = 0.58879
#> 
#> Epoch 165, training: loss = 896.80450, acc = 0.86944, density = 0.58879
#> 
#> Epoch 166, training: loss = 892.42126, acc = 0.87222, density = 0.58879
#> 
#> Epoch 167, training: loss = 891.14197, acc = 0.86111, density = 0.58879
#> 
#> Epoch 168, training: loss = 890.57397, acc = 0.86667, density = 0.58879
#> 
#> Epoch 169, training: loss = 889.16150, acc = 0.87083, density = 0.58879
#> 
#> Epoch 170, training: loss = 890.35645, acc = 0.86667, density = 0.58879
#> 
#> Epoch 171, training: loss = 888.41577, acc = 0.86528, density = 0.58879
#> 
#> Epoch 172, training: loss = 888.77716, acc = 0.86250, density = 0.58879
#> 
#> Epoch 173, training: loss = 890.80731, acc = 0.86667, density = 0.58879
#> 
#> Epoch 174, training: loss = 888.44934, acc = 0.87083, density = 0.58879
#> 
#> Epoch 175, training: loss = 888.30359, acc = 0.86944, density = 0.58879
#> 
#> Epoch 176, training: loss = 885.78186, acc = 0.86528, density = 0.58879
#> 
#> Epoch 177, training: loss = 887.78552, acc = 0.86528, density = 0.58879
#> 
#> Epoch 178, training: loss = 883.60498, acc = 0.86389, density = 0.58879
#> 
#> Epoch 179, training: loss = 887.79053, acc = 0.87222, density = 0.58879
#> 
#> Epoch 180, training: loss = 885.29675, acc = 0.86806, density = 0.58879
#> 
#> Epoch 181, training: loss = 886.56360, acc = 0.86250, density = 0.58879
#> 
#> Epoch 182, training: loss = 881.45441, acc = 0.86944, density = 0.58879
#> 
#> Epoch 183, training: loss = 884.16296, acc = 0.86528, density = 0.58879
#> 
#> Epoch 184, training: loss = 881.19641, acc = 0.86806, density = 0.58879
#> 
#> Epoch 185, training: loss = 885.45844, acc = 0.86389, density = 0.57944
#> 
#> Epoch 186, training: loss = 885.06927, acc = 0.86250, density = 0.57944
#> 
#> Epoch 187, training: loss = 882.11273, acc = 0.86667, density = 0.57944
#> 
#> Epoch 188, training: loss = 880.62103, acc = 0.86389, density = 0.57944
#> 
#> Epoch 189, training: loss = 882.40112, acc = 0.86389, density = 0.57944
#> 
#> Epoch 190, training: loss = 882.25427, acc = 0.87361, density = 0.57944
#> 
#> Epoch 191, training: loss = 880.14703, acc = 0.86250, density = 0.57944
#> 
#> Epoch 192, training: loss = 881.27856, acc = 0.86389, density = 0.57944
#> 
#> Epoch 193, training: loss = 878.29492, acc = 0.86944, density = 0.57944
#> 
#> Epoch 194, training: loss = 879.52698, acc = 0.87222, density = 0.57944
#> 
#> Epoch 195, training: loss = 880.32214, acc = 0.86389, density = 0.57944
#> 
#> Epoch 196, training: loss = 875.58862, acc = 0.86667, density = 0.57944
#> 
#> Epoch 197, training: loss = 873.09766, acc = 0.86389, density = 0.57944
#> 
#> Epoch 198, training: loss = 876.15973, acc = 0.86944, density = 0.57944
#> 
#> Epoch 199, training: loss = 873.07562, acc = 0.86111, density = 0.57944
#> 
#> Epoch 200, training: loss = 873.75037, acc = 0.85417, density = 0.57944
#> 
#> Epoch 201, training: loss = 872.05585, acc = 0.86389, density = 0.57944
#> 
#> Epoch 202, training: loss = 870.66577, acc = 0.87083, density = 0.57944
#> 
#> Epoch 203, training: loss = 871.93860, acc = 0.86667, density = 0.57944
#> 
#> Epoch 204, training: loss = 871.43982, acc = 0.85833, density = 0.57944
#> 
#> Epoch 205, training: loss = 870.75604, acc = 0.87361, density = 0.57944
#> 
#> Epoch 206, training: loss = 871.49817, acc = 0.86806, density = 0.57944
#> 
#> Epoch 207, training: loss = 875.81030, acc = 0.86389, density = 0.57944
#> 
#> Epoch 208, training: loss = 869.83807, acc = 0.86667, density = 0.57944
#> 
#> Epoch 209, training: loss = 868.64581, acc = 0.86806, density = 0.57944
#> 
#> Epoch 210, training: loss = 870.35187, acc = 0.87222, density = 0.57944
#> 
#> Epoch 211, training: loss = 865.49927, acc = 0.86528, density = 0.57944
#> 
#> Epoch 212, training: loss = 865.57141, acc = 0.86667, density = 0.57944
#> 
#> Epoch 213, training: loss = 865.90991, acc = 0.87083, density = 0.57944
#> 
#> Epoch 214, training: loss = 867.25720, acc = 0.87083, density = 0.57944
#> 
#> Epoch 215, training: loss = 868.81921, acc = 0.86528, density = 0.57944
#> 
#> Epoch 216, training: loss = 863.98938, acc = 0.86528, density = 0.57944
#> 
#> Epoch 217, training: loss = 864.63892, acc = 0.87083, density = 0.57944
#> 
#> Epoch 218, training: loss = 862.76611, acc = 0.87222, density = 0.57944
#> 
#> Epoch 219, training: loss = 862.69385, acc = 0.86111, density = 0.57944
#> 
#> Epoch 220, training: loss = 859.30762, acc = 0.86111, density = 0.57944
#> 
#> Epoch 221, training: loss = 863.48499, acc = 0.86667, density = 0.57944
#> 
#> Epoch 222, training: loss = 859.21069, acc = 0.86528, density = 0.57944
#> 
#> Epoch 223, training: loss = 862.64880, acc = 0.86667, density = 0.57944
#> 
#> Epoch 224, training: loss = 859.46783, acc = 0.87083, density = 0.57944
#> 
#> Epoch 225, training: loss = 859.00385, acc = 0.86250, density = 0.57944
#> 
#> Epoch 226, training: loss = 856.97437, acc = 0.87361, density = 0.57944
#> 
#> Epoch 227, training: loss = 859.57593, acc = 0.86944, density = 0.57944
#> 
#> Epoch 228, training: loss = 856.34058, acc = 0.86944, density = 0.57944
#> 
#> Epoch 229, training: loss = 856.82422, acc = 0.85972, density = 0.57944
#> 
#> Epoch 230, training: loss = 857.10315, acc = 0.86944, density = 0.57944
#> 
#> Epoch 231, training: loss = 861.94098, acc = 0.87083, density = 0.57944
#> 
#> Epoch 232, training: loss = 858.09979, acc = 0.86806, density = 0.57944
#> 
#> Epoch 233, training: loss = 859.02399, acc = 0.86806, density = 0.57944
#> 
#> Epoch 234, training: loss = 855.15619, acc = 0.87083, density = 0.57944
#> 
#> Epoch 235, training: loss = 852.30170, acc = 0.86944, density = 0.57944
#> 
#> Epoch 236, training: loss = 850.16650, acc = 0.86528, density = 0.57009
#> 
#> Epoch 237, training: loss = 852.29901, acc = 0.86667, density = 0.57009
#> 
#> Epoch 238, training: loss = 850.48560, acc = 0.86806, density = 0.57009
#> 
#> Epoch 239, training: loss = 853.49310, acc = 0.85972, density = 0.57009
#> 
#> Epoch 240, training: loss = 850.77100, acc = 0.86944, density = 0.57009
#> 
#> Epoch 241, training: loss = 849.37878, acc = 0.86528, density = 0.57009
#> 
#> Epoch 242, training: loss = 849.78540, acc = 0.86806, density = 0.57009
#> 
#> Epoch 243, training: loss = 846.97644, acc = 0.87361, density = 0.57009
#> 
#> Epoch 244, training: loss = 845.71814, acc = 0.87361, density = 0.57009
#> 
#> Epoch 245, training: loss = 855.10547, acc = 0.86667, density = 0.57009
#> 
#> Epoch 246, training: loss = 846.65289, acc = 0.86250, density = 0.57009
#> 
#> Epoch 247, training: loss = 845.80438, acc = 0.86944, density = 0.57009
#> 
#> Epoch 248, training: loss = 846.03851, acc = 0.85972, density = 0.56075
#> 
#> Epoch 249, training: loss = 846.61383, acc = 0.86528, density = 0.56075
#> 
#> Epoch 250, training: loss = 846.17279, acc = 0.86111, density = 0.56075
#> 
#> Epoch 251, training: loss = 843.54773, acc = 0.86528, density = 0.56075
#> 
#> Epoch 252, training: loss = 843.92047, acc = 0.87083, density = 0.56075
#> 
#> Epoch 253, training: loss = 841.19568, acc = 0.86806, density = 0.56075
#> 
#> Epoch 254, training: loss = 843.30530, acc = 0.86944, density = 0.56075
#> 
#> Epoch 255, training: loss = 840.88000, acc = 0.86528, density = 0.56075
#> 
#> Epoch 256, training: loss = 846.76367, acc = 0.87361, density = 0.56075
#> 
#> Epoch 257, training: loss = 844.30353, acc = 0.86667, density = 0.56075
#> 
#> Epoch 258, training: loss = 839.37183, acc = 0.86944, density = 0.56075
#> 
#> Epoch 259, training: loss = 844.72632, acc = 0.87222, density = 0.56075
#> 
#> Epoch 260, training: loss = 836.62964, acc = 0.87639, density = 0.56075
#> 
#> Epoch 261, training: loss = 839.08606, acc = 0.87083, density = 0.56075
#> 
#> Epoch 262, training: loss = 838.85236, acc = 0.86806, density = 0.56075
#> 
#> Epoch 263, training: loss = 841.16101, acc = 0.86806, density = 0.56075
#> 
#> Epoch 264, training: loss = 840.49133, acc = 0.86667, density = 0.56075
#> 
#> Epoch 265, training: loss = 838.30212, acc = 0.86944, density = 0.56075
#> 
#> Epoch 266, training: loss = 834.26294, acc = 0.86667, density = 0.56075
#> 
#> Epoch 267, training: loss = 836.87903, acc = 0.87361, density = 0.56075
#> 
#> Epoch 268, training: loss = 832.12231, acc = 0.86944, density = 0.56075
#> 
#> Epoch 269, training: loss = 833.80786, acc = 0.86528, density = 0.56075
#> 
#> Epoch 270, training: loss = 836.75269, acc = 0.86528, density = 0.56075
#> 
#> Epoch 271, training: loss = 832.46985, acc = 0.86944, density = 0.56075
#> 
#> Epoch 272, training: loss = 829.39502, acc = 0.87222, density = 0.56075
#> 
#> Epoch 273, training: loss = 830.63318, acc = 0.86250, density = 0.56075
#> 
#> Epoch 274, training: loss = 833.49890, acc = 0.86806, density = 0.56075
#> 
#> Epoch 275, training: loss = 829.39288, acc = 0.86667, density = 0.56075
#> 
#> Epoch 276, training: loss = 830.21027, acc = 0.86250, density = 0.56075
#> 
#> Epoch 277, training: loss = 827.22168, acc = 0.86667, density = 0.56075
#> 
#> Epoch 278, training: loss = 827.56274, acc = 0.86944, density = 0.55140
#> 
#> Epoch 279, training: loss = 827.22009, acc = 0.86806, density = 0.55140
#> 
#> Epoch 280, training: loss = 830.64435, acc = 0.87222, density = 0.55140
#> 
#> Epoch 281, training: loss = 827.01050, acc = 0.87222, density = 0.55140
#> 
#> Epoch 282, training: loss = 824.07117, acc = 0.86806, density = 0.55140
#> 
#> Epoch 283, training: loss = 825.96820, acc = 0.86667, density = 0.55140
#> 
#> Epoch 284, training: loss = 821.66467, acc = 0.86806, density = 0.55140
#> 
#> Epoch 285, training: loss = 826.01257, acc = 0.86528, density = 0.55140
#> 
#> Epoch 286, training: loss = 825.63171, acc = 0.86806, density = 0.55140
#> 
#> Epoch 287, training: loss = 819.79645, acc = 0.86806, density = 0.55140
#> 
#> Epoch 288, training: loss = 823.31384, acc = 0.86667, density = 0.55140
#> 
#> Epoch 289, training: loss = 825.63843, acc = 0.86111, density = 0.55140
#> 
#> Epoch 290, training: loss = 820.64172, acc = 0.86667, density = 0.55140
#> 
#> Epoch 291, training: loss = 823.21741, acc = 0.86389, density = 0.55140
#> 
#> Epoch 292, training: loss = 822.23999, acc = 0.86806, density = 0.55140
#> 
#> Epoch 293, training: loss = 822.49005, acc = 0.86111, density = 0.55140
#> 
#> Epoch 294, training: loss = 821.69104, acc = 0.86944, density = 0.55140
#> 
#> Epoch 295, training: loss = 821.21033, acc = 0.86528, density = 0.55140
#> 
#> Epoch 296, training: loss = 815.45227, acc = 0.86389, density = 0.55140
#> 
#> Epoch 297, training: loss = 817.39978, acc = 0.87361, density = 0.55140
#> 
#> Epoch 298, training: loss = 818.42908, acc = 0.87222, density = 0.55140
#> 
#> Epoch 299, training: loss = 819.00940, acc = 0.86389, density = 0.55140
#> 
#> Epoch 300, training: loss = 816.11456, acc = 0.86944, density = 0.55140
#> 
#> Epoch 301, training: loss = 816.72894, acc = 0.87361, density = 0.55140
#> 
#> Epoch 302, training: loss = 814.09369, acc = 0.86111, density = 0.55140
#> 
#> Epoch 303, training: loss = 821.56494, acc = 0.86389, density = 0.55140
#> 
#> Epoch 304, training: loss = 810.78638, acc = 0.86528, density = 0.55140
#> 
#> Epoch 305, training: loss = 817.08820, acc = 0.87083, density = 0.55140
#> 
#> Epoch 306, training: loss = 813.83466, acc = 0.87361, density = 0.55140
#> 
#> Epoch 307, training: loss = 814.53741, acc = 0.86806, density = 0.55140
#> 
#> Epoch 308, training: loss = 813.12622, acc = 0.86111, density = 0.55140
#> 
#> Epoch 309, training: loss = 808.14563, acc = 0.86944, density = 0.55140
#> 
#> Epoch 310, training: loss = 811.66931, acc = 0.86528, density = 0.55140
#> 
#> Epoch 311, training: loss = 808.89893, acc = 0.87083, density = 0.55140
#> 
#> Epoch 312, training: loss = 811.06567, acc = 0.86528, density = 0.55140
#> 
#> Epoch 313, training: loss = 809.19769, acc = 0.86528, density = 0.55140
#> 
#> Epoch 314, training: loss = 807.91980, acc = 0.86806, density = 0.55140
#> 
#> Epoch 315, training: loss = 806.07233, acc = 0.86944, density = 0.55140
#> 
#> Epoch 316, training: loss = 807.50464, acc = 0.87222, density = 0.55140
#> 
#> Epoch 317, training: loss = 806.89771, acc = 0.87083, density = 0.55140
#> 
#> Epoch 318, training: loss = 803.35071, acc = 0.86944, density = 0.55140
#> 
#> Epoch 319, training: loss = 806.86279, acc = 0.86944, density = 0.55140
#> 
#> Epoch 320, training: loss = 801.78217, acc = 0.86667, density = 0.55140
#> 
#> Epoch 321, training: loss = 803.25470, acc = 0.86667, density = 0.55140
#> 
#> Epoch 322, training: loss = 805.22894, acc = 0.86389, density = 0.55140
#> 
#> Epoch 323, training: loss = 799.85608, acc = 0.86389, density = 0.55140
#> 
#> Epoch 324, training: loss = 803.91925, acc = 0.86944, density = 0.55140
#> 
#> Epoch 325, training: loss = 803.12622, acc = 0.87222, density = 0.55140
#> 
#> Epoch 326, training: loss = 800.93085, acc = 0.87222, density = 0.55140
#> 
#> Epoch 327, training: loss = 802.90704, acc = 0.86667, density = 0.55140
#> 
#> Epoch 328, training: loss = 800.19946, acc = 0.87083, density = 0.55140
#> 
#> Epoch 329, training: loss = 803.29205, acc = 0.87083, density = 0.55140
#> 
#> Epoch 330, training: loss = 801.14856, acc = 0.86389, density = 0.55140
#> 
#> Epoch 331, training: loss = 803.22888, acc = 0.86250, density = 0.54206
#> 
#> Epoch 332, training: loss = 797.85004, acc = 0.86806, density = 0.54206
#> 
#> Epoch 333, training: loss = 799.64227, acc = 0.86389, density = 0.54206
#> 
#> Epoch 334, training: loss = 798.71344, acc = 0.86667, density = 0.54206
#> 
#> Epoch 335, training: loss = 796.17651, acc = 0.86667, density = 0.53271
#> 
#> Epoch 336, training: loss = 791.60309, acc = 0.86111, density = 0.53271
#> 
#> Epoch 337, training: loss = 798.08722, acc = 0.86250, density = 0.53271
#> 
#> Epoch 338, training: loss = 794.89587, acc = 0.86806, density = 0.53271
#> 
#> Epoch 339, training: loss = 792.70013, acc = 0.86806, density = 0.53271
#> 
#> Epoch 340, training: loss = 793.59528, acc = 0.86389, density = 0.53271
#> 
#> Epoch 341, training: loss = 792.48120, acc = 0.86806, density = 0.53271
#> 
#> Epoch 342, training: loss = 795.43805, acc = 0.86667, density = 0.53271
#> 
#> Epoch 343, training: loss = 791.47327, acc = 0.86389, density = 0.53271
#> 
#> Epoch 344, training: loss = 792.08301, acc = 0.86806, density = 0.53271
#> 
#> Epoch 345, training: loss = 791.65906, acc = 0.87222, density = 0.53271
#> 
#> Epoch 346, training: loss = 787.69971, acc = 0.86667, density = 0.53271
#> 
#> Epoch 347, training: loss = 788.02258, acc = 0.86111, density = 0.53271
#> 
#> Epoch 348, training: loss = 788.58752, acc = 0.86806, density = 0.53271
#> 
#> Epoch 349, training: loss = 790.73303, acc = 0.86667, density = 0.52336
#> 
#> Epoch 350, training: loss = 784.25934, acc = 0.87500, density = 0.52336
#> 
#> Epoch 351, training: loss = 789.12378, acc = 0.87222, density = 0.52336
#> 
#> Epoch 352, training: loss = 789.61060, acc = 0.86667, density = 0.52336
#> 
#> Epoch 353, training: loss = 784.01740, acc = 0.86944, density = 0.52336
#> 
#> Epoch 354, training: loss = 784.14893, acc = 0.86389, density = 0.52336
#> 
#> Epoch 355, training: loss = 783.71631, acc = 0.86667, density = 0.52336
#> 
#> Epoch 356, training: loss = 783.30298, acc = 0.86111, density = 0.52336
#> 
#> Epoch 357, training: loss = 780.86414, acc = 0.87361, density = 0.52336
#> 
#> Epoch 358, training: loss = 781.25061, acc = 0.86806, density = 0.52336
#> 
#> Epoch 359, training: loss = 788.53387, acc = 0.86528, density = 0.52336
#> 
#> Epoch 360, training: loss = 783.57776, acc = 0.86111, density = 0.52336
#> 
#> Epoch 361, training: loss = 779.21655, acc = 0.86389, density = 0.52336
#> 
#> Epoch 362, training: loss = 780.24347, acc = 0.86944, density = 0.52336
#> 
#> Epoch 363, training: loss = 780.16431, acc = 0.86528, density = 0.52336
#> 
#> Epoch 364, training: loss = 778.12183, acc = 0.86667, density = 0.52336
#> 
#> Epoch 365, training: loss = 771.40381, acc = 0.86667, density = 0.52336
#> 
#> Epoch 366, training: loss = 781.78723, acc = 0.86806, density = 0.52336
#> 
#> Epoch 367, training: loss = 778.60754, acc = 0.87222, density = 0.52336
#> 
#> Epoch 368, training: loss = 777.24011, acc = 0.86528, density = 0.52336
#> 
#> Epoch 369, training: loss = 776.21326, acc = 0.86389, density = 0.52336
#> 
#> Epoch 370, training: loss = 776.65735, acc = 0.86528, density = 0.52336
#> 
#> Epoch 371, training: loss = 774.83044, acc = 0.86806, density = 0.52336
#> 
#> Epoch 372, training: loss = 773.88977, acc = 0.87083, density = 0.52336
#> 
#> Epoch 373, training: loss = 775.51624, acc = 0.86111, density = 0.52336
#> 
#> Epoch 374, training: loss = 775.38721, acc = 0.86806, density = 0.52336
#> 
#> Epoch 375, training: loss = 773.58264, acc = 0.86806, density = 0.52336
#> 
#> Epoch 376, training: loss = 767.21350, acc = 0.86806, density = 0.52336
#> 
#> Epoch 377, training: loss = 768.81860, acc = 0.86389, density = 0.52336
#> 
#> Epoch 378, training: loss = 771.94489, acc = 0.87083, density = 0.52336
#> 
#> Epoch 379, training: loss = 767.75250, acc = 0.86806, density = 0.52336
#> 
#> Epoch 380, training: loss = 770.41150, acc = 0.87083, density = 0.52336
#> 
#> Epoch 381, training: loss = 770.26953, acc = 0.86944, density = 0.52336
#> 
#> Epoch 382, training: loss = 770.36829, acc = 0.86250, density = 0.52336
#> 
#> Epoch 383, training: loss = 766.79193, acc = 0.86528, density = 0.52336
#> 
#> Epoch 384, training: loss = 765.83307, acc = 0.86944, density = 0.52336
#> 
#> Epoch 385, training: loss = 765.12439, acc = 0.86667, density = 0.52336
#> 
#> Epoch 386, training: loss = 768.22900, acc = 0.87083, density = 0.51402
#> 
#> Epoch 387, training: loss = 765.44116, acc = 0.86111, density = 0.51402
#> 
#> Epoch 388, training: loss = 761.44836, acc = 0.87222, density = 0.51402
#> 
#> Epoch 389, training: loss = 768.61646, acc = 0.86667, density = 0.51402
#> 
#> Epoch 390, training: loss = 760.35583, acc = 0.86528, density = 0.51402
#> 
#> Epoch 391, training: loss = 768.99506, acc = 0.86667, density = 0.51402
#> 
#> Epoch 392, training: loss = 768.64490, acc = 0.86389, density = 0.51402
#> 
#> Epoch 393, training: loss = 763.61707, acc = 0.86111, density = 0.50467
#> 
#> Epoch 394, training: loss = 765.13916, acc = 0.87361, density = 0.50467
#> 
#> Epoch 395, training: loss = 764.05493, acc = 0.86389, density = 0.50467
#> 
#> Epoch 396, training: loss = 757.41589, acc = 0.86528, density = 0.50467
#> 
#> Epoch 397, training: loss = 759.25562, acc = 0.86111, density = 0.50467
#> 
#> Epoch 398, training: loss = 760.72351, acc = 0.86944, density = 0.50467
#> 
#> Epoch 399, training: loss = 759.79047, acc = 0.86944, density = 0.50467
#> 
#> Epoch 400, training: loss = 757.58167, acc = 0.86250, density = 0.50467
#> 
#> Epoch 401, training: loss = 757.82056, acc = 0.86944, density = 0.50467
#> 
#> Epoch 402, training: loss = 753.30359, acc = 0.87222, density = 0.50467
#> 
#> Epoch 403, training: loss = 753.71680, acc = 0.87083, density = 0.50467
#> 
#> Epoch 404, training: loss = 757.71832, acc = 0.86528, density = 0.50467
#> 
#> Epoch 405, training: loss = 759.04211, acc = 0.86250, density = 0.50467
#> 
#> Epoch 406, training: loss = 753.77698, acc = 0.86667, density = 0.50467
#> 
#> Epoch 407, training: loss = 753.92297, acc = 0.87500, density = 0.50467
#> 
#> Epoch 408, training: loss = 754.59192, acc = 0.85972, density = 0.50467
#> 
#> Epoch 409, training: loss = 750.41394, acc = 0.86667, density = 0.50467
#> 
#> Epoch 410, training: loss = 753.11584, acc = 0.86528, density = 0.50467
#> 
#> Epoch 411, training: loss = 748.76086, acc = 0.85972, density = 0.50467
#> 
#> Epoch 412, training: loss = 749.46997, acc = 0.86389, density = 0.50467
#> 
#> Epoch 413, training: loss = 751.23511, acc = 0.86528, density = 0.50467
#> 
#> Epoch 414, training: loss = 749.68750, acc = 0.86806, density = 0.50467
#> 
#> Epoch 415, training: loss = 751.36298, acc = 0.85833, density = 0.50467
#> 
#> Epoch 416, training: loss = 751.99530, acc = 0.86806, density = 0.50467
#> 
#> Epoch 417, training: loss = 744.59631, acc = 0.86944, density = 0.50467
#> 
#> Epoch 418, training: loss = 745.04419, acc = 0.87361, density = 0.50467
#> 
#> Epoch 419, training: loss = 750.43158, acc = 0.86806, density = 0.50467
#> 
#> Epoch 420, training: loss = 749.52344, acc = 0.86528, density = 0.50467
#> 
#> Epoch 421, training: loss = 750.98572, acc = 0.86667, density = 0.50467
#> 
#> Epoch 422, training: loss = 745.18835, acc = 0.87778, density = 0.50467
#> 
#> Epoch 423, training: loss = 743.73523, acc = 0.87083, density = 0.50467
#> 
#> Epoch 424, training: loss = 741.93591, acc = 0.86250, density = 0.50467
#> 
#> Epoch 425, training: loss = 738.26624, acc = 0.87222, density = 0.50467
#> 
#> Epoch 426, training: loss = 742.62061, acc = 0.87500, density = 0.50467
#> 
#> Epoch 427, training: loss = 741.11682, acc = 0.86528, density = 0.50467
#> 
#> Epoch 428, training: loss = 737.69495, acc = 0.86250, density = 0.50467
#> 
#> Epoch 429, training: loss = 740.81238, acc = 0.86528, density = 0.50467
#> 
#> Epoch 430, training: loss = 741.58411, acc = 0.86250, density = 0.50467
#> 
#> Epoch 431, training: loss = 738.75586, acc = 0.87083, density = 0.50467
#> 
#> Epoch 432, training: loss = 739.81262, acc = 0.86667, density = 0.50467
#> 
#> Epoch 433, training: loss = 736.38953, acc = 0.87083, density = 0.50467
#> 
#> Epoch 434, training: loss = 740.04285, acc = 0.86667, density = 0.50467
#> 
#> Epoch 435, training: loss = 740.33722, acc = 0.85833, density = 0.49533
#> 
#> Epoch 436, training: loss = 740.48096, acc = 0.86111, density = 0.49533
#> 
#> Epoch 437, training: loss = 730.33679, acc = 0.86667, density = 0.49533
#> 
#> Epoch 438, training: loss = 733.99670, acc = 0.86528, density = 0.49533
#> 
#> Epoch 439, training: loss = 733.56372, acc = 0.86806, density = 0.49533
#> 
#> Epoch 440, training: loss = 736.17126, acc = 0.86806, density = 0.49533
#> 
#> Epoch 441, training: loss = 734.27783, acc = 0.87083, density = 0.49533
#> 
#> Epoch 442, training: loss = 733.96887, acc = 0.86528, density = 0.49533
#> 
#> Epoch 443, training: loss = 725.47961, acc = 0.87500, density = 0.49533
#> 
#> Epoch 444, training: loss = 732.08789, acc = 0.87361, density = 0.49533
#> 
#> Epoch 445, training: loss = 728.37134, acc = 0.86250, density = 0.49533
#> 
#> Epoch 446, training: loss = 731.88525, acc = 0.86250, density = 0.49533
#> 
#> Epoch 447, training: loss = 731.62048, acc = 0.86667, density = 0.49533
#> 
#> Epoch 448, training: loss = 728.00409, acc = 0.86806, density = 0.49533
#> 
#> Epoch 449, training: loss = 729.26660, acc = 0.85972, density = 0.49533
#> 
#> Epoch 450, training: loss = 730.69702, acc = 0.86389, density = 0.49533
#> 
#> Epoch 451, training: loss = 723.70569, acc = 0.86389, density = 0.49533
#> 
#> Epoch 452, training: loss = 731.41101, acc = 0.85833, density = 0.49533
#> 
#> Epoch 453, training: loss = 727.44653, acc = 0.85972, density = 0.49533
#> 
#> Epoch 454, training: loss = 718.58447, acc = 0.86528, density = 0.49533
#> 
#> Epoch 455, training: loss = 728.79645, acc = 0.86528, density = 0.49533
#> 
#> Epoch 456, training: loss = 729.10931, acc = 0.87361, density = 0.49533
#> 
#> Epoch 457, training: loss = 724.90955, acc = 0.86944, density = 0.49533
#> 
#> Epoch 458, training: loss = 726.91443, acc = 0.86111, density = 0.49533
#> 
#> Epoch 459, training: loss = 720.13574, acc = 0.87083, density = 0.48598
#> 
#> Epoch 460, training: loss = 722.18896, acc = 0.86111, density = 0.48598
#> 
#> Epoch 461, training: loss = 716.81049, acc = 0.87361, density = 0.48598
#> 
#> Epoch 462, training: loss = 721.62561, acc = 0.86667, density = 0.48598
#> 
#> Epoch 463, training: loss = 713.82458, acc = 0.86528, density = 0.48598
#> 
#> Epoch 464, training: loss = 717.18237, acc = 0.87222, density = 0.48598
#> 
#> Epoch 465, training: loss = 722.55017, acc = 0.86528, density = 0.48598
#> 
#> Epoch 466, training: loss = 718.66040, acc = 0.86528, density = 0.48598
#> 
#> Epoch 467, training: loss = 716.49158, acc = 0.86528, density = 0.48598
#> 
#> Epoch 468, training: loss = 716.63794, acc = 0.86389, density = 0.48598
#> 
#> Epoch 469, training: loss = 712.19330, acc = 0.87361, density = 0.48598
#> 
#> Epoch 470, training: loss = 718.75281, acc = 0.86944, density = 0.48598
#> 
#> Epoch 471, training: loss = 719.89014, acc = 0.85972, density = 0.48598
#> 
#> Epoch 472, training: loss = 711.20160, acc = 0.85972, density = 0.48598
#> 
#> Epoch 473, training: loss = 712.01733, acc = 0.86806, density = 0.48598
#> 
#> Epoch 474, training: loss = 712.42639, acc = 0.87222, density = 0.47664
#> 
#> Epoch 475, training: loss = 713.79919, acc = 0.86806, density = 0.47664
#> 
#> Epoch 476, training: loss = 707.34204, acc = 0.86806, density = 0.47664
#> 
#> Epoch 477, training: loss = 709.99268, acc = 0.86667, density = 0.47664
#> 
#> Epoch 478, training: loss = 714.15015, acc = 0.86111, density = 0.47664
#> 
#> Epoch 479, training: loss = 711.64935, acc = 0.86389, density = 0.47664
#> 
#> Epoch 480, training: loss = 704.77002, acc = 0.86250, density = 0.47664
#> 
#> Epoch 481, training: loss = 707.15344, acc = 0.86667, density = 0.47664
#> 
#> Epoch 482, training: loss = 709.14264, acc = 0.87222, density = 0.47664
#> 
#> Epoch 483, training: loss = 706.66296, acc = 0.86667, density = 0.47664
#> 
#> Epoch 484, training: loss = 706.95483, acc = 0.87361, density = 0.47664
#> 
#> Epoch 485, training: loss = 702.37476, acc = 0.86528, density = 0.47664
#> 
#> Epoch 486, training: loss = 704.91357, acc = 0.86389, density = 0.47664
#> 
#> Epoch 487, training: loss = 706.13940, acc = 0.86667, density = 0.47664
#> 
#> Epoch 488, training: loss = 702.55298, acc = 0.87222, density = 0.46729
#> 
#> Epoch 489, training: loss = 701.61218, acc = 0.86667, density = 0.46729
#> 
#> Epoch 490, training: loss = 703.41968, acc = 0.86806, density = 0.46729
#> 
#> Epoch 491, training: loss = 702.87048, acc = 0.86250, density = 0.46729
#> 
#> Epoch 492, training: loss = 702.58020, acc = 0.86944, density = 0.46729
#> 
#> Epoch 493, training: loss = 706.96655, acc = 0.86389, density = 0.46729
#> 
#> Epoch 494, training: loss = 701.07959, acc = 0.86667, density = 0.46729
#> 
#> Epoch 495, training: loss = 700.04517, acc = 0.87083, density = 0.46729
#> 
#> Epoch 496, training: loss = 702.08868, acc = 0.85972, density = 0.46729
#> 
#> Epoch 497, training: loss = 700.64136, acc = 0.86806, density = 0.46729
#> 
#> Epoch 498, training: loss = 698.49902, acc = 0.86667, density = 0.46729
#> 
#> Epoch 499, training: loss = 699.84137, acc = 0.86806, density = 0.46729
#> 
#> Epoch 500, training: loss = 693.33069, acc = 0.86528, density = 0.46729
#> 
#> Epoch 501, training: loss = 696.19727, acc = 0.86667, density = 0.46729
#> 
#> Epoch 502, training: loss = 697.23425, acc = 0.85833, density = 0.45794
#> 
#> Epoch 503, training: loss = 695.17310, acc = 0.86806, density = 0.45794
#> 
#> Epoch 504, training: loss = 699.11658, acc = 0.85972, density = 0.44860
#> 
#> Epoch 505, training: loss = 697.65991, acc = 0.86528, density = 0.44860
#> 
#> Epoch 506, training: loss = 691.35956, acc = 0.86389, density = 0.43925
#> 
#> Epoch 507, training: loss = 699.36792, acc = 0.85972, density = 0.43925
#> 
#> Epoch 508, training: loss = 693.38843, acc = 0.86528, density = 0.43925
#> 
#> Epoch 509, training: loss = 697.09424, acc = 0.85694, density = 0.43925
#> 
#> Epoch 510, training: loss = 697.19897, acc = 0.86528, density = 0.43925
#> 
#> Epoch 511, training: loss = 697.31824, acc = 0.86111, density = 0.43925
#> 
#> Epoch 512, training: loss = 686.82983, acc = 0.86667, density = 0.43925
#> 
#> Epoch 513, training: loss = 689.57623, acc = 0.86111, density = 0.43925
#> 
#> Epoch 514, training: loss = 688.27148, acc = 0.86111, density = 0.43925
#> 
#> Epoch 515, training: loss = 689.39343, acc = 0.86250, density = 0.43925
#> 
#> Epoch 516, training: loss = 684.79333, acc = 0.86528, density = 0.43925
#> 
#> Epoch 517, training: loss = 683.19904, acc = 0.87083, density = 0.43925
#> 
#> Epoch 518, training: loss = 684.86121, acc = 0.86806, density = 0.43925
#> 
#> Epoch 519, training: loss = 687.55597, acc = 0.86111, density = 0.43925
#> 
#> Epoch 520, training: loss = 685.53217, acc = 0.86528, density = 0.43925
#> 
#> Epoch 521, training: loss = 677.91852, acc = 0.86389, density = 0.43925
#> 
#> Epoch 522, training: loss = 683.36475, acc = 0.86944, density = 0.43925
#> 
#> Epoch 523, training: loss = 690.98328, acc = 0.86389, density = 0.43925
#> 
#> Epoch 524, training: loss = 678.72681, acc = 0.86528, density = 0.43925
#> 
#> Epoch 525, training: loss = 686.28992, acc = 0.86806, density = 0.43925
#> 
#> Epoch 526, training: loss = 681.19659, acc = 0.86667, density = 0.43925
#> 
#> Epoch 527, training: loss = 682.41479, acc = 0.86528, density = 0.43925
#> 
#> Epoch 528, training: loss = 680.24292, acc = 0.85972, density = 0.43925
#> 
#> Epoch 529, training: loss = 679.13153, acc = 0.85556, density = 0.43925
#> 
#> Epoch 530, training: loss = 682.29938, acc = 0.86250, density = 0.43925
#> 
#> Epoch 531, training: loss = 682.01581, acc = 0.86528, density = 0.43925
#> 
#> Epoch 532, training: loss = 676.97937, acc = 0.87083, density = 0.43925
#> 
#> Epoch 533, training: loss = 674.03809, acc = 0.86250, density = 0.43925
#> 
#> Epoch 534, training: loss = 676.79333, acc = 0.86389, density = 0.43925
#> 
#> Epoch 535, training: loss = 675.70056, acc = 0.86528, density = 0.43925
#> 
#> Epoch 536, training: loss = 676.76550, acc = 0.85833, density = 0.43925
#> 
#> Epoch 537, training: loss = 674.68103, acc = 0.86528, density = 0.43925
#> 
#> Epoch 538, training: loss = 668.45642, acc = 0.86944, density = 0.43925
#> 
#> Epoch 539, training: loss = 670.59753, acc = 0.86667, density = 0.43925
#> 
#> Epoch 540, training: loss = 676.55273, acc = 0.86528, density = 0.43925
#> 
#> Epoch 541, training: loss = 678.44604, acc = 0.86389, density = 0.43925
#> 
#> Epoch 542, training: loss = 669.04846, acc = 0.86389, density = 0.43925
#> 
#> Epoch 543, training: loss = 671.89954, acc = 0.85972, density = 0.43925
#> 
#> Epoch 544, training: loss = 673.08765, acc = 0.86667, density = 0.43925
#> 
#> Epoch 545, training: loss = 669.04724, acc = 0.86389, density = 0.43925
#> 
#> Epoch 546, training: loss = 667.78674, acc = 0.86528, density = 0.43925
#> 
#> Epoch 547, training: loss = 667.37659, acc = 0.86111, density = 0.43925
#> 
#> Epoch 548, training: loss = 666.61243, acc = 0.86528, density = 0.43925
#> 
#> Epoch 549, training: loss = 667.04565, acc = 0.86528, density = 0.43925
#> 
#> Epoch 550, training: loss = 664.03479, acc = 0.86250, density = 0.43925
#> 
#> Epoch 551, training: loss = 662.76788, acc = 0.86528, density = 0.43925
#> 
#> Epoch 552, training: loss = 660.36914, acc = 0.87083, density = 0.43925
#> 
#> Epoch 553, training: loss = 673.74866, acc = 0.85833, density = 0.43925
#> 
#> Epoch 554, training: loss = 662.33472, acc = 0.87083, density = 0.43925
#> 
#> Epoch 555, training: loss = 670.67584, acc = 0.86944, density = 0.43925
#> 
#> Epoch 556, training: loss = 663.81519, acc = 0.86250, density = 0.43925
#> 
#> Epoch 557, training: loss = 663.33167, acc = 0.86250, density = 0.43925
#> 
#> Epoch 558, training: loss = 658.49622, acc = 0.86250, density = 0.43925
#> 
#> Epoch 559, training: loss = 667.41394, acc = 0.86111, density = 0.43925
#> 
#> Epoch 560, training: loss = 657.30212, acc = 0.86806, density = 0.43925
#> 
#> Epoch 561, training: loss = 662.92480, acc = 0.85556, density = 0.43925
#> 
#> Epoch 562, training: loss = 655.59918, acc = 0.86667, density = 0.43925
#> 
#> Epoch 563, training: loss = 658.28491, acc = 0.86389, density = 0.43925
#> 
#> Epoch 564, training: loss = 665.83301, acc = 0.86389, density = 0.43925
#> 
#> Epoch 565, training: loss = 656.54675, acc = 0.86528, density = 0.43925
#> 
#> Epoch 566, training: loss = 661.73657, acc = 0.86528, density = 0.43925
#> 
#> Epoch 567, training: loss = 655.17181, acc = 0.86806, density = 0.43925
#> 
#> Epoch 568, training: loss = 655.22687, acc = 0.86389, density = 0.43925
#> 
#> Epoch 569, training: loss = 649.14166, acc = 0.87778, density = 0.41121
#> 
#> Epoch 570, training: loss = 655.00568, acc = 0.87361, density = 0.41121
#> 
#> Epoch 571, training: loss = 652.87659, acc = 0.86528, density = 0.41121
#> 
#> Epoch 572, training: loss = 654.07874, acc = 0.85694, density = 0.41121
#> 
#> Epoch 573, training: loss = 658.03699, acc = 0.85972, density = 0.41121
#> 
#> Epoch 574, training: loss = 652.92285, acc = 0.85972, density = 0.41121
#> 
#> Epoch 575, training: loss = 649.44098, acc = 0.87361, density = 0.41121
#> 
#> Epoch 576, training: loss = 652.77051, acc = 0.87083, density = 0.41121
#> 
#> Epoch 577, training: loss = 653.45044, acc = 0.86111, density = 0.41121
#> 
#> Epoch 578, training: loss = 649.72705, acc = 0.86250, density = 0.41121
#> 
#> Epoch 579, training: loss = 649.48975, acc = 0.86250, density = 0.41121
#> 
#> Epoch 580, training: loss = 648.53931, acc = 0.86389, density = 0.41121
#> 
#> Epoch 581, training: loss = 649.93164, acc = 0.86944, density = 0.41121
#> 
#> Epoch 582, training: loss = 647.20587, acc = 0.86667, density = 0.41121
#> 
#> Epoch 583, training: loss = 653.56281, acc = 0.86944, density = 0.41121
#> 
#> Epoch 584, training: loss = 642.49567, acc = 0.86667, density = 0.41121
#> 
#> Epoch 585, training: loss = 643.69104, acc = 0.86806, density = 0.40187
#> 
#> Epoch 586, training: loss = 641.77716, acc = 0.87222, density = 0.40187
#> 
#> Epoch 587, training: loss = 645.48462, acc = 0.86389, density = 0.40187
#> 
#> Epoch 588, training: loss = 649.84955, acc = 0.86944, density = 0.40187
#> 
#> Epoch 589, training: loss = 645.93774, acc = 0.86528, density = 0.40187
#> 
#> Epoch 590, training: loss = 644.46100, acc = 0.86667, density = 0.40187
#> 
#> Epoch 591, training: loss = 639.85278, acc = 0.86250, density = 0.39252
#> 
#> Epoch 592, training: loss = 638.00720, acc = 0.85972, density = 0.39252
#> 
#> Epoch 593, training: loss = 640.23499, acc = 0.86389, density = 0.39252
#> 
#> Epoch 594, training: loss = 643.46240, acc = 0.85972, density = 0.39252
#> 
#> Epoch 595, training: loss = 633.28619, acc = 0.86806, density = 0.39252
#> 
#> Epoch 596, training: loss = 638.84009, acc = 0.86250, density = 0.39252
#> 
#> Epoch 597, training: loss = 637.13123, acc = 0.86806, density = 0.39252
#> 
#> Epoch 598, training: loss = 636.66736, acc = 0.86111, density = 0.39252
#> 
#> Epoch 599, training: loss = 637.17004, acc = 0.86667, density = 0.39252
#> 
#> Epoch 600, training: loss = 646.21161, acc = 0.86389, density = 0.39252
#> 
#> Epoch 601, training: loss = 635.46301, acc = 0.86806, density = 0.39252
#> 
#> Epoch 602, training: loss = 639.62842, acc = 0.86806, density = 0.39252
#> 
#> Epoch 603, training: loss = 637.56384, acc = 0.86528, density = 0.39252
#> 
#> Epoch 604, training: loss = 639.25677, acc = 0.85972, density = 0.39252
#> 
#> Epoch 605, training: loss = 635.50708, acc = 0.86250, density = 0.39252
#> 
#> Epoch 606, training: loss = 624.65674, acc = 0.86667, density = 0.39252
#> 
#> Epoch 607, training: loss = 632.29614, acc = 0.86528, density = 0.39252
#> 
#> Epoch 608, training: loss = 628.27881, acc = 0.86389, density = 0.39252
#> 
#> Epoch 609, training: loss = 630.03973, acc = 0.86667, density = 0.38318
#> 
#> Epoch 610, training: loss = 636.06433, acc = 0.85972, density = 0.38318
#> 
#> Epoch 611, training: loss = 628.86652, acc = 0.86528, density = 0.38318
#> 
#> Epoch 612, training: loss = 626.83240, acc = 0.86250, density = 0.38318
#> 
#> Epoch 613, training: loss = 632.74231, acc = 0.86528, density = 0.38318
#> 
#> Epoch 614, training: loss = 629.23578, acc = 0.86250, density = 0.38318
#> 
#> Epoch 615, training: loss = 629.56824, acc = 0.86806, density = 0.38318
#> 
#> Epoch 616, training: loss = 627.10254, acc = 0.86528, density = 0.38318
#> 
#> Epoch 617, training: loss = 626.48914, acc = 0.86389, density = 0.38318
#> 
#> Epoch 618, training: loss = 624.13391, acc = 0.85972, density = 0.38318
#> 
#> Epoch 619, training: loss = 623.24731, acc = 0.86528, density = 0.38318
#> 
#> Epoch 620, training: loss = 622.12927, acc = 0.86389, density = 0.38318
#> 
#> Epoch 621, training: loss = 618.41675, acc = 0.86528, density = 0.38318
#> 
#> Epoch 622, training: loss = 625.15515, acc = 0.85694, density = 0.38318
#> 
#> Epoch 623, training: loss = 623.13110, acc = 0.85972, density = 0.38318
#> 
#> Epoch 624, training: loss = 622.35132, acc = 0.86944, density = 0.38318
#> 
#> Epoch 625, training: loss = 621.40179, acc = 0.87222, density = 0.38318
#> 
#> Epoch 626, training: loss = 628.12427, acc = 0.86667, density = 0.38318
#> 
#> Epoch 627, training: loss = 619.17664, acc = 0.86667, density = 0.38318
#> 
#> Epoch 628, training: loss = 618.16064, acc = 0.85833, density = 0.38318
#> 
#> Epoch 629, training: loss = 623.69006, acc = 0.86250, density = 0.38318
#> 
#> Epoch 630, training: loss = 614.17310, acc = 0.86250, density = 0.38318
#> 
#> Epoch 631, training: loss = 616.65173, acc = 0.86528, density = 0.38318
#> 
#> Epoch 632, training: loss = 627.69373, acc = 0.86528, density = 0.38318
#> 
#> Epoch 633, training: loss = 620.34399, acc = 0.86250, density = 0.38318
#> 
#> Epoch 634, training: loss = 615.84821, acc = 0.85556, density = 0.37383
#> 
#> Epoch 635, training: loss = 614.14209, acc = 0.86389, density = 0.37383
#> 
#> Epoch 636, training: loss = 620.02283, acc = 0.86111, density = 0.37383
#> 
#> Epoch 637, training: loss = 618.98706, acc = 0.85972, density = 0.37383
#> 
#> Epoch 638, training: loss = 611.82166, acc = 0.85694, density = 0.37383
#> 
#> Epoch 639, training: loss = 615.30731, acc = 0.85972, density = 0.37383
#> 
#> Epoch 640, training: loss = 608.57660, acc = 0.86389, density = 0.37383
#> 
#> Epoch 641, training: loss = 614.79236, acc = 0.85972, density = 0.37383
#> 
#> Epoch 642, training: loss = 606.16638, acc = 0.86111, density = 0.37383
#> 
#> Epoch 643, training: loss = 613.13428, acc = 0.86389, density = 0.36449
#> 
#> Epoch 644, training: loss = 608.46912, acc = 0.85694, density = 0.36449
#> 
#> Epoch 645, training: loss = 605.74976, acc = 0.86250, density = 0.36449
#> 
#> Epoch 646, training: loss = 608.56274, acc = 0.86528, density = 0.35514
#> 
#> Epoch 647, training: loss = 613.26660, acc = 0.86111, density = 0.35514
#> 
#> Epoch 648, training: loss = 611.51025, acc = 0.85556, density = 0.35514
#> 
#> Epoch 649, training: loss = 603.81421, acc = 0.86389, density = 0.35514
#> 
#> Epoch 650, training: loss = 611.17615, acc = 0.86389, density = 0.35514
#> 
#> Epoch 651, training: loss = 603.43750, acc = 0.86389, density = 0.35514
#> 
#> Epoch 652, training: loss = 601.68750, acc = 0.86250, density = 0.35514
#> 
#> Epoch 653, training: loss = 600.05872, acc = 0.86111, density = 0.35514
#> 
#> Epoch 654, training: loss = 607.85449, acc = 0.86528, density = 0.35514
#> 
#> Epoch 655, training: loss = 608.50610, acc = 0.85694, density = 0.35514
#> 
#> Epoch 656, training: loss = 604.92126, acc = 0.86528, density = 0.35514
#> 
#> Epoch 657, training: loss = 604.38831, acc = 0.86528, density = 0.35514
#> 
#> Epoch 658, training: loss = 598.08130, acc = 0.86806, density = 0.35514
#> 
#> Epoch 659, training: loss = 603.60339, acc = 0.86111, density = 0.35514
#> 
#> Epoch 660, training: loss = 602.55200, acc = 0.86389, density = 0.35514
#> 
#> Epoch 661, training: loss = 599.25903, acc = 0.85833, density = 0.35514
#> 
#> Epoch 662, training: loss = 602.91296, acc = 0.86250, density = 0.35514
#> 
#> Epoch 663, training: loss = 598.36377, acc = 0.85694, density = 0.35514
#> 
#> Epoch 664, training: loss = 595.75012, acc = 0.86944, density = 0.35514
#> 
#> Epoch 665, training: loss = 603.91785, acc = 0.86944, density = 0.35514
#> 
#> Epoch 666, training: loss = 599.72540, acc = 0.85833, density = 0.35514
#> 
#> Epoch 667, training: loss = 595.08405, acc = 0.86806, density = 0.35514
#> 
#> Epoch 668, training: loss = 594.85522, acc = 0.85833, density = 0.35514
#> 
#> Epoch 669, training: loss = 593.95129, acc = 0.85972, density = 0.35514
#> 
#> Epoch 670, training: loss = 598.92072, acc = 0.86250, density = 0.35514
#> 
#> Epoch 671, training: loss = 593.94727, acc = 0.86250, density = 0.35514
#> 
#> Epoch 672, training: loss = 588.30927, acc = 0.86806, density = 0.35514
#> 
#> Epoch 673, training: loss = 589.93524, acc = 0.86528, density = 0.35514
#> 
#> Epoch 674, training: loss = 585.49866, acc = 0.86111, density = 0.35514
#> 
#> Epoch 675, training: loss = 593.22113, acc = 0.86528, density = 0.35514
#> 
#> Epoch 676, training: loss = 585.54065, acc = 0.86250, density = 0.35514
#> 
#> Epoch 677, training: loss = 588.12396, acc = 0.86806, density = 0.35514
#> 
#> Epoch 678, training: loss = 590.32214, acc = 0.86250, density = 0.35514
#> 
#> Epoch 679, training: loss = 595.00793, acc = 0.86389, density = 0.35514
#> 
#> Epoch 680, training: loss = 582.47125, acc = 0.85833, density = 0.35514
#> 
#> Epoch 681, training: loss = 585.99951, acc = 0.86250, density = 0.35514
#> 
#> Epoch 682, training: loss = 589.22729, acc = 0.86667, density = 0.35514
#> 
#> Epoch 683, training: loss = 584.68030, acc = 0.86528, density = 0.34579
#> 
#> Epoch 684, training: loss = 589.23755, acc = 0.86250, density = 0.34579
#> 
#> Epoch 685, training: loss = 584.11322, acc = 0.86389, density = 0.34579
#> 
#> Epoch 686, training: loss = 584.76794, acc = 0.86806, density = 0.34579
#> 
#> Epoch 687, training: loss = 589.21265, acc = 0.86944, density = 0.34579
#> 
#> Epoch 688, training: loss = 587.37201, acc = 0.86111, density = 0.34579
#> 
#> Epoch 689, training: loss = 582.85938, acc = 0.86389, density = 0.34579
#> 
#> Epoch 690, training: loss = 583.00256, acc = 0.86389, density = 0.34579
#> 
#> Epoch 691, training: loss = 578.60382, acc = 0.86250, density = 0.34579
#> 
#> Epoch 692, training: loss = 579.36902, acc = 0.87083, density = 0.34579
#> 
#> Epoch 693, training: loss = 580.53772, acc = 0.85833, density = 0.34579
#> 
#> Epoch 694, training: loss = 584.02478, acc = 0.86111, density = 0.34579
#> 
#> Epoch 695, training: loss = 575.48499, acc = 0.86389, density = 0.34579
#> 
#> Epoch 696, training: loss = 570.92969, acc = 0.85972, density = 0.34579
#> 
#> Epoch 697, training: loss = 578.20874, acc = 0.85972, density = 0.33645
#> 
#> Epoch 698, training: loss = 580.05652, acc = 0.85694, density = 0.33645
#> 
#> Epoch 699, training: loss = 579.14667, acc = 0.85833, density = 0.33645
#> 
#> Epoch 700, training: loss = 585.22729, acc = 0.86389, density = 0.33645
#> 
#> Epoch 701, training: loss = 572.95422, acc = 0.86528, density = 0.33645
#> 
#> Epoch 702, training: loss = 571.92456, acc = 0.86250, density = 0.33645
#> 
#> Epoch 703, training: loss = 575.00348, acc = 0.86389, density = 0.33645
#> 
#> Epoch 704, training: loss = 573.75293, acc = 0.86389, density = 0.33645
#> 
#> Epoch 705, training: loss = 576.18225, acc = 0.86667, density = 0.33645
#> 
#> Epoch 706, training: loss = 578.16327, acc = 0.86944, density = 0.33645
#> 
#> Epoch 707, training: loss = 573.87433, acc = 0.85833, density = 0.33645
#> 
#> Epoch 708, training: loss = 568.87292, acc = 0.86389, density = 0.33645
#> 
#> Epoch 709, training: loss = 573.88544, acc = 0.86250, density = 0.33645
#> 
#> Epoch 710, training: loss = 567.69586, acc = 0.86389, density = 0.33645
#> 
#> Epoch 711, training: loss = 572.58789, acc = 0.85694, density = 0.33645
#> 
#> Epoch 712, training: loss = 568.39209, acc = 0.86250, density = 0.32710
#> 
#> Epoch 713, training: loss = 563.50287, acc = 0.86389, density = 0.32710
#> 
#> Epoch 714, training: loss = 568.26221, acc = 0.86528, density = 0.32710
#> 
#> Epoch 715, training: loss = 562.95911, acc = 0.85972, density = 0.32710
#> 
#> Epoch 716, training: loss = 560.92371, acc = 0.86111, density = 0.32710
#> 
#> Epoch 717, training: loss = 570.88989, acc = 0.86806, density = 0.32710
#> 
#> Epoch 718, training: loss = 565.33179, acc = 0.86667, density = 0.32710
#> 
#> Epoch 719, training: loss = 559.97766, acc = 0.86389, density = 0.32710
#> 
#> Epoch 720, training: loss = 563.79150, acc = 0.86250, density = 0.32710
#> 
#> Epoch 721, training: loss = 567.20886, acc = 0.86111, density = 0.32710
#> 
#> Epoch 722, training: loss = 559.22742, acc = 0.85972, density = 0.32710
#> 
#> Epoch 723, training: loss = 564.65167, acc = 0.87361, density = 0.32710
#> 
#> Epoch 724, training: loss = 553.83777, acc = 0.85833, density = 0.32710
#> 
#> Epoch 725, training: loss = 567.07178, acc = 0.86389, density = 0.32710
#> 
#> Epoch 726, training: loss = 558.75208, acc = 0.86806, density = 0.32710
#> 
#> Epoch 727, training: loss = 557.76306, acc = 0.85972, density = 0.32710
#> 
#> Epoch 728, training: loss = 558.73987, acc = 0.86667, density = 0.32710
#> 
#> Epoch 729, training: loss = 557.11304, acc = 0.86111, density = 0.32710
#> 
#> Epoch 730, training: loss = 558.41925, acc = 0.86667, density = 0.32710
#> 
#> Epoch 731, training: loss = 558.75671, acc = 0.86528, density = 0.32710
#> 
#> Epoch 732, training: loss = 555.15002, acc = 0.87222, density = 0.32710
#> 
#> Epoch 733, training: loss = 562.50092, acc = 0.86389, density = 0.32710
#> 
#> Epoch 734, training: loss = 559.52173, acc = 0.86250, density = 0.32710
#> 
#> Epoch 735, training: loss = 554.85608, acc = 0.86389, density = 0.31776
#> 
#> Epoch 736, training: loss = 551.95062, acc = 0.86944, density = 0.31776
#> 
#> Epoch 737, training: loss = 556.58777, acc = 0.86667, density = 0.31776
#> 
#> Epoch 738, training: loss = 558.31750, acc = 0.86389, density = 0.31776
#> 
#> Epoch 739, training: loss = 551.35968, acc = 0.86528, density = 0.30841
#> 
#> Epoch 740, training: loss = 556.81104, acc = 0.86667, density = 0.30841
#> 
#> Epoch 741, training: loss = 559.37781, acc = 0.86111, density = 0.30841
#> 
#> Epoch 742, training: loss = 556.75592, acc = 0.86944, density = 0.30841
#> 
#> Epoch 743, training: loss = 554.43268, acc = 0.87361, density = 0.30841
#> 
#> Epoch 744, training: loss = 553.81464, acc = 0.86250, density = 0.30841
#> 
#> Epoch 745, training: loss = 550.36365, acc = 0.85694, density = 0.30841
#> 
#> Epoch 746, training: loss = 544.08899, acc = 0.86389, density = 0.30841
#> 
#> Epoch 747, training: loss = 545.25037, acc = 0.87222, density = 0.30841
#> 
#> Epoch 748, training: loss = 542.13416, acc = 0.86806, density = 0.30841
#> 
#> Epoch 749, training: loss = 552.24255, acc = 0.86528, density = 0.30841
#> 
#> Epoch 750, training: loss = 544.79602, acc = 0.86111, density = 0.30841
#> 
#> Epoch 751, training: loss = 543.01233, acc = 0.86806, density = 0.29907
#> 
#> Epoch 752, training: loss = 548.10071, acc = 0.87222, density = 0.29907
#> 
#> Epoch 753, training: loss = 546.49548, acc = 0.85972, density = 0.29907
#> 
#> Epoch 754, training: loss = 551.43658, acc = 0.86806, density = 0.29907
#> 
#> Epoch 755, training: loss = 540.95581, acc = 0.86111, density = 0.29907
#> 
#> Epoch 756, training: loss = 542.22876, acc = 0.85417, density = 0.28972
#> 
#> Epoch 757, training: loss = 543.03436, acc = 0.86667, density = 0.28972
#> 
#> Epoch 758, training: loss = 544.14038, acc = 0.85694, density = 0.28972
#> 
#> Epoch 759, training: loss = 544.30872, acc = 0.86528, density = 0.28037
#> 
#> Epoch 760, training: loss = 534.52966, acc = 0.85972, density = 0.28037
#> 
#> Epoch 761, training: loss = 543.37793, acc = 0.86667, density = 0.28037
#> 
#> Epoch 762, training: loss = 532.45490, acc = 0.85972, density = 0.28037
#> 
#> Epoch 763, training: loss = 536.56696, acc = 0.86389, density = 0.28037
#> 
#> Epoch 764, training: loss = 538.48962, acc = 0.86528, density = 0.28037
#> 
#> Epoch 765, training: loss = 539.36353, acc = 0.86528, density = 0.28037
#> 
#> Epoch 766, training: loss = 540.88989, acc = 0.85972, density = 0.28037
#> 
#> Epoch 767, training: loss = 535.16248, acc = 0.86389, density = 0.28037
#> 
#> Epoch 768, training: loss = 540.85577, acc = 0.86389, density = 0.28037
#> 
#> Epoch 769, training: loss = 537.16016, acc = 0.85694, density = 0.27103
#> 
#> Epoch 770, training: loss = 545.17432, acc = 0.86806, density = 0.27103
#> 
#> Epoch 771, training: loss = 531.90320, acc = 0.86250, density = 0.27103
#> 
#> Epoch 772, training: loss = 531.38184, acc = 0.86389, density = 0.27103
#> 
#> Epoch 773, training: loss = 534.52930, acc = 0.86389, density = 0.27103
#> 
#> Epoch 774, training: loss = 536.22217, acc = 0.86528, density = 0.27103
#> 
#> Epoch 775, training: loss = 535.19281, acc = 0.86111, density = 0.27103
#> 
#> Epoch 776, training: loss = 530.11597, acc = 0.86389, density = 0.27103
#> 
#> Epoch 777, training: loss = 535.13147, acc = 0.87083, density = 0.27103
#> 
#> Epoch 778, training: loss = 535.96466, acc = 0.86111, density = 0.26168
#> 
#> Epoch 779, training: loss = 531.57275, acc = 0.86389, density = 0.26168
#> 
#> Epoch 780, training: loss = 533.37799, acc = 0.86111, density = 0.26168
#> 
#> Epoch 781, training: loss = 529.36658, acc = 0.86250, density = 0.26168
#> 
#> Epoch 782, training: loss = 531.50787, acc = 0.86111, density = 0.26168
#> 
#> Epoch 783, training: loss = 532.17358, acc = 0.85556, density = 0.26168
#> 
#> Epoch 784, training: loss = 529.77124, acc = 0.85556, density = 0.26168
#> 
#> Epoch 785, training: loss = 521.33356, acc = 0.86806, density = 0.26168
#> 
#> Epoch 786, training: loss = 523.58514, acc = 0.86389, density = 0.26168
#> 
#> Epoch 787, training: loss = 525.90479, acc = 0.86250, density = 0.26168
#> 
#> Epoch 788, training: loss = 522.30298, acc = 0.86528, density = 0.26168
#> 
#> Epoch 789, training: loss = 518.09509, acc = 0.86250, density = 0.26168
#> 
#> Epoch 790, training: loss = 523.52411, acc = 0.86250, density = 0.26168
#> 
#> Epoch 791, training: loss = 527.68658, acc = 0.86667, density = 0.26168
#> 
#> Epoch 792, training: loss = 518.56104, acc = 0.86389, density = 0.25234
#> 
#> Epoch 793, training: loss = 523.73364, acc = 0.86528, density = 0.25234
#> 
#> Epoch 794, training: loss = 521.29004, acc = 0.86389, density = 0.25234
#> 
#> Epoch 795, training: loss = 530.77838, acc = 0.85694, density = 0.25234
#> 
#> Epoch 796, training: loss = 519.41101, acc = 0.86111, density = 0.25234
#> 
#> Epoch 797, training: loss = 516.42957, acc = 0.86806, density = 0.25234
#> 
#> Epoch 798, training: loss = 518.16736, acc = 0.85000, density = 0.25234
#> 
#> Epoch 799, training: loss = 522.14008, acc = 0.86389, density = 0.25234
#> 
#> Epoch 800, training: loss = 515.26746, acc = 0.86389, density = 0.25234
#save the model 
#torch::torch_save(model_input_skip$state_dict(), 
#paste(getwd(),'/R/saved_models/README_input_skip_example_model.pth',sep = ''))
```

To evaluate performance on the validation data, one can use the function
Validate_LBBNN. This function takes a model, number of samples for model
averaging, and the validation data as input.

``` r
validate_LBBNN(LBBNN = model_input_skip,num_samples = 100,test_dl = test_loader,device)
#> $accuracy_full_model
#> [1] 0.8722222
#> 
#> $accuracy_sparse
#> [1] 0.8666667
#> 
#> $density
#> [1] 0.2523364
#> 
#> $density_active_path
#> [1] 0.09345794
#validate_LBBNN(LBBNN = model_flows,num_samples = 1000,test_dl = test_loader,device)
```

Plot the global structure of the given model:

``` r
plot(model_input_skip,type = 'global',vertex_size = 13,edge_width = 0.6,label_size = 0.6)
```

<img src="man/figures/README-unnamed-chunk-6-1.png" width="100%" />

Note that only 3 of the 7 input variables are used, where one of them
only has a linear connection.

This can also be seen using the summary function:

``` r
summary(model_input_skip)
#> Summary of LBBNN_Net object:
#> -----------------------------------
#> Shows the number of times each variable was included from each layer
#> -----------------------------------
#> Then the average inclusion probability for each input from each layer
#> -----------------------------------
#> The final column shows the average inclusion probability across all layers
#> -----------------------------------
#>    L0 L1 L2    a0    a1    a2 a_avg
#> x0  0  0  0 0.175 0.219 0.006 0.180
#> x1  0  0  0 0.343 0.109 0.005 0.206
#> x2  0  0  1 0.135 0.344 0.863 0.296
#> x3  1  1  1 0.539 0.330 0.991 0.485
#> x4  0  0  0 0.321 0.575 0.124 0.419
#> x5  0  0  0 0.098 0.438 0.137 0.256
#> x6  1  0  1 0.300 0.244 0.997 0.338
#> -----------------------------------
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
#>  [1] -0.47491318  0.46904647 -0.14563666  0.40983611 -0.05854724  0.19214028
#>  [7] -0.08140449 -0.28510988 -0.12890983 -0.04447413
```

Get local explanations from some training data:

``` r
coef(model_input_skip,dataset = train_loader,inds = c(2,3,4,5,6))
#>         lower       mean      upper
#> x0  0.0000000  0.0000000  0.0000000
#> x1  0.0000000  0.0000000  0.0000000
#> x2 -0.1718342 -0.1708280 -0.1694423
#> x3 -0.6384742 -0.6348897 -0.6288301
#> x4  0.0000000  0.0000000  0.0000000
#> x5  0.0000000  0.0000000  0.0000000
#> x6 -2.2649529 -2.2543384 -2.2471136
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
#>   - layers.0             : LBBNN_Linear    # 115 parameters
#>   - layers.1             : LBBNN_Linear    # 190 parameters
#>   - act                  : nn_leaky_relu   # 0 parameters
#>   - out_layer            : LBBNN_Linear    # 38 parameters
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
