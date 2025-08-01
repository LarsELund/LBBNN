% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/local_explanations.R
\name{plot_local_explanations_gradient}
\alias{plot_local_explanations_gradient}
\title{Function to plot the local explanations}
\usage{
plot_local_explanations_gradient(
  model,
  input_data,
  num_samples,
  device = "cpu"
)
}
\arguments{
\item{model}{Input-skip model to explain. Not implemented for regular LBBNNs.}

\item{input_data}{The data to be explained (one sample).}

\item{num_samples}{How many sample to use to produce credible intervals.}

\item{device}{the device to be trained on. Default is cpu.}
}
\value{
a list containing the losses and accuracies (if classification) and density for each epoch during training.
For comparisons sake we show the density with and without active paths.
}
\description{
Plots the contribution of each covariate, and the prediction, with error bars.
}
