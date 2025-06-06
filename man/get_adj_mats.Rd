% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plotting_graphs.R
\name{get_adj_mats}
\alias{get_adj_mats}
\title{Function to obtain adjacency matrices to be used with igraph plotting}
\usage{
get_adj_mats(model)
}
\arguments{
\item{model}{A trained LBBNN model with input-skip.}
}
\value{
The adjacency matrices.
}
\description{
Given a trained LBBNN model with input-skip, this
function takes the alpha active path matrices for each layer and converts
them to adjacency matrices so that they can be plotted with igraph.
}
