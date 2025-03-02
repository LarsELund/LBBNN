% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Models.R
\name{LBBNN_Net}
\alias{LBBNN_Net}
\title{Class to generate a LBBNN network}
\usage{
LBBNN_Net(problem_type, sizes, prior, device = "cpu", link = NULL, nll = NULL)
}
\arguments{
\item{problem_type}{'binary classification', 'multiclass classification' or 'regression'.}

\item{sizes}{a vector containing the sizes of layers of the network, where the first element is the input size, and the last the output size.}

\item{prior}{a vector containing the inclusion probabilities for each layer in the network. Length must be ONE less than sizes.}

\item{device}{the device to be trained on. Can be 'cpu', 'gpu' or 'mps'. Default is cpu.}
}
\description{
Generates a LBBNN composed of feed forward layers defined by LBBNN_Linear
e.g sizes = c(20,200,200,5) generates an LBBNN with 20 input variables,
two hidden layers with 200 neurons each, and an output layer of 5 neurons.
LBBNN_net also contains functions to compute kl-divergence and the density of the entire network.
}
\examples{
layers <- c(20,200,200,5) #Two hidden layers 
alpha <- c(0.3,0.5,0.9)  # One prior inclusion probability for each weight matrix 
prob <- 'multiclass classification'
net <- LBBNN_Net(problem_type = prob, sizes = layers, prior = alpha,device = 'cpu')
print(net)

x <- torch::torch_rand(100,20,requires_grad = FALSE) #generate some dummy data
output <- net(x) #forward pass
net$kl_div()$item() #get KL-divergence
net$density() #get the density of the network
}
