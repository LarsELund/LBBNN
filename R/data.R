#' Raisins Dataset
#'
#' @description Ilkay Cinar, Murat Kokl and Sakir Tasdemi(2020)
#' provide a dataset consisting of 2 varieties of Turkish raisins, with 450
#' samples of each type. The dataset contains 7 morphological features,
#' extracted from images taken of the Raisins. The goal is to classify to one of
#' the two types of Raisins.
#' @format this data frame has 900 rows and the following 8 columns:
#' \describe{
#'   \item{Area}{Number of pixels within the boundary}
#'   \item{MajorAxisLength}{Length of the main axis}
#'   \item{MinorAxisLength}{Length of the small axis}
#'   \item{Eccentricity}{Measure of the eccentricity of the ellipse}
#'   \item{ConvexArea}{The number of pixels of the
#'   smallest convex shell of the region formed by the raisin grain}
#'   \item{Extent}{Ratio of the region formed by the
#'   raisin grain to the total pixels in the bounding box}
#'   \item{Perimeter}{distance between the boundaries of the
#'   raisin grain and the pixels around it}
#'   \item{Class}{Kecimen or Besni raisin.}
#' }
#' @source \url{https://archive.ics.uci.edu/dataset/850/raisin}
"raisin_dataset"

#' Gallstone Dataset
#' @description Taken from the UCI machine learning repository. The task is
#' to classify whether the patient had gallstones or not. It contains a mix
#' of demographic data and bioimpedance data.
#' @format  This dataset has 319 rows and  38 columns.
#' @source \url{https://pmc.ncbi.nlm.nih.gov/articles/PMC11309733/#T2}
"gallstone_dataset"
