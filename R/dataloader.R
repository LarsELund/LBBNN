#' @title Wrapper around \code{torch::dataloader}
#' @description  Avoids users having to manually define their own dataloaders.
#' @param dataset A \code{data.frame}.
#' The last \code{target_cols} column(s) are assumed to be the dependent variable(s).
#' @param train_proportion numeric, between 0 and 1.
#' Proportion of data to be used for training.
#' @param train_batch_size integer, samples per batch in the train dataloader.
#' @param test_batch_size integer, samples per batch in the test dataloader.
#' @param standardize logical, standardize input-features, default is TRUE.
#' @param shuffle_train logical, shuffle training data each epoch. default TRUE
#' @param shuffle_test  logical, shuffle test data, default is FALSE.
#' @param seed integer. Used for reproducibility in the train/test split.
#' @param weights optional numeric vector of length \code{nrow(dataset)} specifying observation weights.
#' @param target_cols integer, number of columns at the end of \code{dataset} representing the target (default 1).
#' @param normalize_weights logical, whether to normalize weights to sum to the sample size (recommended for survey weights w_i per Savitsky & Toth 2016). Default is FALSE.
#' @return A list containing:
#'   \describe{
#'     \item{train_loader}{A \code{torch::dataloader} for the training data.}
#'     \item{test_loader}{A \code{torch::dataloader} for the test data (or NULL if \code{train_proportion = 1}).}
#'   }
#'@export
get_dataloaders <- function(dataset, train_proportion, train_batch_size,
                            test_batch_size, standardize = TRUE,
                            shuffle_train = TRUE, shuffle_test = FALSE,
                            seed = 1, weights = NULL, target_cols = 1,
                            normalize_weights = FALSE) {
  if (!inherits(dataset, "data.frame"))
    stop(paste("dataset must be a data.frame object"))
  if (!inherits(train_proportion, "numeric"))
    stop(paste(train_proportion, "must be a numeric value"))
  if (train_proportion > 1 || train_proportion <= 0)
    stop(paste(train_proportion, "must be a numeric between 0 and 1"))

  if (!is.null(weights)) {
    if (!is.numeric(weights) || length(weights) != nrow(dataset)) {
      stop(sprintf("weights must be a numeric vector of length nrow(dataset) (%d)", nrow(dataset)))
    }
    if (normalize_weights) {
      weights <- weights * (length(weights) / sum(weights))
    }
  }

  p <- ncol(dataset) - target_cols
  if (p < 1) {
    stop("ncol(dataset) must be strictly greater than target_cols")
  }

  set.seed(seed)
  if (train_proportion == 1) {
    train <- dataset
    test <- NULL
    w_train <- if (!is.null(weights)) as.numeric(weights) else NULL
    w_test <- NULL
  } else {
    sample <- sample.int(n = nrow(dataset),
                         size = floor(train_proportion * nrow(dataset)),
                         replace = FALSE)
    train  <- dataset[sample, , drop = FALSE]
    test   <- dataset[-sample, , drop = FALSE]
    if (!is.null(weights)) {
      w_train <- as.numeric(weights[sample])
      w_test  <- as.numeric(weights[-sample])
      if (normalize_weights) {
        if (sum(w_train) > 0) w_train <- w_train * (length(w_train) / sum(w_train))
        if (!is.null(w_test) && length(w_test) > 0 && sum(w_test) > 0) {
          w_test <- w_test * (length(w_test) / sum(w_test))
        }
      }
    } else {
      w_train <- NULL
      w_test  <- NULL
    }
  }

  if (target_cols == 1) {
    y_train <- as.numeric(train[, ncol(train)])
    y_test  <- if (!is.null(test)) as.numeric(test[, ncol(test)]) else NULL
    if (min(y_train) == 0 && max(y_train) > 1) {
      y_train <- y_train + 1
      if (!is.null(y_test)) y_test <- y_test + 1
    }
  } else {
    y_train <- as.matrix(train[, (p + 1):ncol(train), drop = FALSE])
    y_test  <- if (!is.null(test)) as.matrix(test[, (p + 1):ncol(test), drop = FALSE]) else NULL
  }

  x_train <- as.matrix(train[, 1:p, drop = FALSE])
  x_test  <- if (!is.null(test)) as.matrix(test[, 1:p, drop = FALSE]) else NULL

  if (standardize) {
    x_train <- scale(x_train)
    if (!is.null(x_test) && nrow(x_test) > 0) {
      x_test <- scale(x_test, center = attr(x_train, "scaled:center"),
                      scale = attr(x_train, "scaled:scale"))
    }
  }

  if (!is.null(weights)) {
    train_data <- torch::tensor_dataset(
      torch::torch_tensor(x_train, dtype = torch::torch_float32()),
      torch::torch_tensor(y_train, dtype = torch::torch_float32()),
      torch::torch_tensor(w_train, dtype = torch::torch_float32())
    )
    if (!is.null(test) && nrow(test) > 0) {
      test_data <- torch::tensor_dataset(
        torch::torch_tensor(x_test, dtype = torch::torch_float32()),
        torch::torch_tensor(y_test, dtype = torch::torch_float32()),
        torch::torch_tensor(w_test, dtype = torch::torch_float32())
      )
    } else {
      test_data <- NULL
    }
  } else {
    train_data <- torch::tensor_dataset(
      torch::torch_tensor(x_train, dtype = torch::torch_float32()),
      torch::torch_tensor(y_train, dtype = torch::torch_float32())
    )
    if (!is.null(test) && nrow(test) > 0) {
      test_data <- torch::tensor_dataset(
        torch::torch_tensor(x_test, dtype = torch::torch_float32()),
        torch::torch_tensor(y_test, dtype = torch::torch_float32())
      )
    } else {
      test_data <- NULL
    }
  }

  if (train_batch_size > length(train_data))
    (stop("Can not have larger batch size than the amount of training data"))

  train_loader <- torch::dataloader(train_data, batch_size = train_batch_size,
                                    shuffle = shuffle_train)

  if (!is.null(test_data)) {
    if (test_batch_size > length(test_data))
      (stop("Can not have larger test batch size than the amount of test data"))
    test_loader <- torch::dataloader(test_data, batch_size = test_batch_size,
                                     shuffle = shuffle_test)
  } else {
    test_loader <- NULL
  }

  l <- list("train_loader" = train_loader, "test_loader" = test_loader)
  return(l)
}
