#' @title Feed-forward Latent Binary Bayesian Neural Network (LBBNN)
#' @description
#' Each layer is defined by \code{lbbnn_linear}.
#' For example, \code{sizes = c(20, 200, 200, 5)} generates a network with:
#' \itemize{
#'   \item 20 input features,
#'   \item two hidden layers of 200 neurons each,
#'   \item an output layer with 5 neurons.
#'}
#' @param problem_type character, one of:
#' \code{'binary classification'}, \code{'multiclass classification'},
#' \code{'regression'}, or \code{'custom'}.
#' @param sizes Integer vector specifying the layer sizes of the network.
#' The first element is the input size,
#' the last is the output size,
#' and the intermediate integers represent hidden layers.
#' @param prior numeric vector of prior inclusion probabilities for
#' each weight matrix. length must be \code{length(sizes) - 1}.
#' @param std numeric vector of prior standard deviation for each weight matrix.
#' length must be \code{length(sizes) - 1}.
#' @param inclusion_inits numeric matrix of shape (2, number of weight matrices)
#' specifying the lower and upper bounds for initializations
#' of the inclusion parameters.
#' @param input_skip logical, whether to include input_skip.
#' @param flow logical, whether to use normalizing flows.
#' @param num_transforms integer, how many transformations to use in the flow.
#' @param dims numeric vector, hidden dimension for the neural network
#' in the RNVP transform.
#' @param device the device to be trained on. Can be 'cpu', 'gpu' or 'mps'.
#' Default is cpu.
#' @param raw_output logical, whether the network skips the last sigmoid/softmax
#'  layer to compute local explanations.
#' @param custom_act Allows the user to submit their own customized
#' activation function.
#' @param link User can define their own link function (not implemented).
#' @param nll User can define their own likelihood function (not implemented).
#' @param bias_inclusion_prob logical, determines whether the bias should
#' be as associated with inclusion probabilities.
#' @examples
#' \donttest{
#' layers <- c(10,2,5)
#' alpha <- c(0.3,0.9)
#' stds <- c(1.0,1.0)
#' inclusion_inits <- matrix(rep(c(-10,10),2),nrow = 2,ncol = 2)
#' prob <- 'multiclass classification'
#' net <- lbbnn_net(problem_type = prob, sizes = layers, prior = alpha,
#' std = stds, inclusion_inits = inclusion_inits,input_skip = FALSE,
#' flow = FALSE,device = 'cpu')
#' x <- torch::torch_rand(20,10,requires_grad = FALSE)
#' output <- net(x)
#' net$kl_div()$item()
#' net$density()
#' }
#' @return A \code{torch::nn_module} object representing the LBBNN.
#'   It includes the following methods:
#'   \itemize{
#'     \item \code{forward(x, MPM = FALSE)}: Performs a forward pass
#'     through the whole network.
#'     \item \code{kl_div()}: Returns the KL divergence of the network.
#'     \item \code{density()}: Returns the density of the whole network,
#'     i.e. the proportion of weights
#'     with inclusion probabilities greater than 0.5.
#'     \item \code{compute_paths()}: Computes active paths through the network
#'     without input-skip.
#'     \item \code{compute_paths_input_skip()}: Computes active paths with
#'     input-skip enabled.
#'     \item \code{density_active_path()}: Returns network density after
#'     removing inactive paths.
#'   }
#'
#' @export
lbbnn_net <- torch::nn_module(
  "lbbnn_net",

  initialize = function(problem_type, sizes, prior, std, inclusion_inits,
                        input_skip = FALSE, flow = FALSE, num_transforms = 2,
                        dims = c(200, 200), device = "cpu", raw_output = FALSE,
                        custom_act = NULL, link = NULL, nll = NULL,
                        bias_inclusion_prob = FALSE) {
    self$device <- device
    self$layers <- torch::nn_module_list()
    self$problem_type <- problem_type
    self$input_skip <- input_skip
    self$flow <- flow
    self$bias_inclusion_prob <- bias_inclusion_prob
    self$num_transforms <- num_transforms
    self$dims <- dims
    self$sizes <- sizes
    self$prior_inclusion <- prior
    self$prior_std <- std
    self$elapsed_time <- 0 #to check how much time the model takes to train
    self$raw_output <- raw_output # TRUE when we want local explanations
    self$act <- torch::nn_leaky_relu(0.00)
    self$computed_paths <- FALSE
    if (! is.null(custom_act)) {
      self$act <- custom_act
      }
    if (length(prior) != length(sizes) - 1)
    (stop("Must have one prior inclusion probability per weight matrix"))

     #define the layers of the network
     #for the first layer input-skip and LBBNNs are the same,
     #but for all subsequent layers, input-skip will have shape (dim + p,dim)
     #where p is the number of variables. (i.e. the first element of sizes)
     for (i in 1:(length(sizes) - 2)) {
      if (i == 1 || input_skip == FALSE) {
        in_shape <- sizes[i]
        }
      else (in_shape <- sizes[i] + sizes[1])
      self$layers$append(lbbnn_linear(
        in_shape,
        sizes[i + 1],
        prior_inclusion = prior[i],
        standard_prior = std[i],
        density_init = inclusion_inits[, i],
        flow = self$flow,
        num_transforms = self$num_transforms,
        hidden_dims = self$dims,
        device = self$device,
        bias_inclusion_prob = self$bias_inclusion_prob))
    }
    if (input_skip) {
      out_size <- sizes[length(sizes) - 1] + sizes[1]
    } else {
      out_size <- sizes[length(sizes) - 1]
      }
    self$out_layer <- (lbbnn_linear(
      out_size,
      sizes[length(sizes)],
      prior_inclusion = prior[length(prior)],
      standard_prior = std[length(std)],
      density_init = inclusion_inits[, ncol(inclusion_inits)],
      flow = self$flow,
      num_transforms = self$num_transforms,
      hidden_dims = self$dims,
      device = self$device,
      bias_inclusion_prob = self$bias_inclusion_prob))

    if (problem_type == "binary classification") {
      self$out <- torch::nn_sigmoid()
      self$loss_fn <- torch::nn_bce_loss(reduction = "sum")
    }
    else if (problem_type == "multiclass classification" | problem_type == "MNIST") {
      self$out <- torch::nn_log_softmax(dim = 2)
      self$loss_fn <- torch::nn_nll_loss(reduction = "sum")
    }
    else if (problem_type == "regression") {
      self$out <- torch::nn_identity()
      self$loss_fn <- torch::nn_mse_loss(reduction = "sum")
    }else if (problem_type == "custom") {
      if (length(link) == 0 | length(nll) == 0)
        stop("Under custom problem, link function and the negative log likelihood
             must be provided as torch functions")
      self$out <- link()
      self$loss_fn <- nll(reduction = "sum")
    }
    else(stop("the type of problem must either be: \'binary classification\',
              \'multiclass classification\', \'regression\' or \'custom\'"))
  },
  forward = function(x, MPM = FALSE) {
    if (self$problem_type == "MNIST")(x <- x$view(c(-1, 28 * 28)))
    #regular LBBNN
    if (!self$input_skip) {
      x <- x$view(c(-1, self$sizes[1]))
      for (l in self$layers$children){
       x <- self$act(l(x, MPM)) #iterate over hidden layers
      }
      x <- self$out(self$out_layer(x, MPM))
    }
    else {x_input <- x$view(c(-1, self$sizes[1]))
    x <- self$layers$children$`0`(x_input, MPM) #first layer
    j <- 1
    for (l in self$layers$children) {
      if (j > 1) {#skip the first layer when iterating.
        x <- l(torch::torch_cat(c(x, x_input), dim = 2), MPM)
      }
      j <- j + 1
    }

    #if we only want the raw output, skip sigmoid/softmax
    if (self$raw_output) {
      x <- self$out_layer(torch::torch_cat(c(x, x_input), dim = 2), MPM)
    }
    else (x <- self$out(self$out_layer(torch::torch_cat(c(x, x_input), dim = 2), MPM)))

    }

    return(x)
    },
  kl_div = function() {
    kl <- 0
    for (l in self$layers$children)(kl <- kl + l$kl_div())
    kl <- kl + self$out_layer$kl_div()
    return(kl)
  },
  compute_paths = function() {

    if (self$input_skip == TRUE) {
      stop("self$input_skip must be FALSE to use this function")
    }
    self$computed_paths <- TRUE
    #sending a random input through the network of alpha matrices (0 and 1)
    #and then backpropagating to find active paths
    a <- rnorm(n = self$layers$children$`0`$alpha$shape[2])
    x0 <- torch::torch_tensor(a, dtype = torch::torch_float32(), 
                              device = self$device)
    alpha_mats <- list() #initialize empty list to append network alphas

    for (l in self$layers$children) {
      lamd <- l$lambda_l$detach()
      alpha <- (torch::torch_sigmoid(lamd) > 0.5) * 1
      alpha$requires_grad <- TRUE
      alpha_mats <- append(alpha_mats, alpha)
      x0 <- torch::torch_matmul(x0, torch::torch_t(alpha))
    }
    #output layer
    lamd_out <- self$out_layer$lambda_l$detach()
    alpha_out <- (torch::torch_sigmoid(lamd_out) > 0.5) * 1
    alpha_out <- alpha_out$detach()
    alpha_out$requires_grad <- TRUE
    alpha_mats <- append(alpha_mats, alpha_out)
    x0 <- torch::torch_matmul(x0, torch::torch_t(alpha_out))
    L <- torch::torch_sum(x0) #summing in case more than 1 output. This is
    #equivalent to backpropagate for each output node.
    L$backward() #compute derivatives to get active paths
                  #any alpha preceding an alpha with value 0 will also become
                  #zero when gradients are passed backwards, and thus we will
                  #be left with the active paths.
    i <- 1
    alpha_mats_out <- list()
    for (j in self$layers$children) {
      alp <- alpha_mats[[i]] * alpha_mats[[i]]$grad
      alp[alp != 0] <- 1
      alpha_mats_out <- append(alpha_mats_out, alp$detach())
      j$alpha_active_path <- alp$detach()
      i <- i + 1

    }
    alp_out <- alpha_mats[[length(alpha_mats)]] * alpha_mats[[length(alpha_mats)]]$grad #last alpha
    alp_out[alp_out != 0] <- 1
    alpha_mats_out <- append(alpha_mats_out, alp_out$detach())
    self$out_layer$alpha_active_path <- alp_out$detach()
    return(alpha_mats_out)
  },
  compute_paths_input_skip = function() {
    if (self$input_skip == FALSE) {
      stop("self$input_skip must be TRUE to use this funciton")
    }
    self$computed_paths <- TRUE
    #sending a random input through the network of alpha matrices (0 and 1)
    #and then backpropagating to find active paths
    a <- rnorm(n = self$layers$children$`0`$alpha$shape[2])
    x0 <- torch::torch_tensor(a, dtype = torch::torch_float32(),
                              device = self$device)$unsqueeze(dim = 1)
    alpha_mats <- list() #initialize empty list to append network alphas
    lamd_input <- self$layers$children$`0`$lambda_l
    alpha_input <- (torch::torch_sigmoid(lamd_input) > 0.5) * 1
    alpha_input$requires_grad <- TRUE
    alpha_mats <- append(alpha_mats, alpha_input)
    x <- torch::torch_matmul(x0, torch::torch_t(alpha_input))
    j <- 1
    for (l in self$layers$children) {
      if (j > 1) {#skip the first layer when iterating.
        x <- (torch::torch_cat(c(x, x0), dim = 2))
        lamd <- l$lambda_l$detach()
        alpha <- (torch::torch_sigmoid(lamd) > 0.5) * 1
        alpha$requires_grad <- TRUE
        alpha_mats <- append(alpha_mats, alpha)
        x <- torch::torch_matmul(x, torch::torch_t(alpha))
      }
      j <- j + 1
    }
    #output layer
    lamd_out <- self$out_layer$lambda_l$detach()
    alpha_out <- (torch::torch_sigmoid(lamd_out) > 0.5) * 1
    alpha_out <- alpha_out$detach()
    alpha_out$requires_grad <- TRUE
    alpha_mats <- append(alpha_mats, alpha_out)
    x_out <- (torch::torch_cat(c(x, x0), dim = 2))
    x_out <- torch::torch_matmul(x_out, torch::torch_t(alpha_out))
    L <- torch::torch_sum(x_out) #summing in case more than 1 output. This is
    #equivalent to backpropagate for each output node.
    L$backward() #compute derivatives to get active paths
    #any alpha preceding an alpha with value 0 will also become
    #zero when gradients are passed backwards, and thus we will
    #be left with the active paths.
    i <- 1
    alpha_mats_out <- list()
    for (j in self$layers$children) {
      alp <- alpha_mats[[i]] * alpha_mats[[i]]$grad
      alp[alp != 0] <- 1
      alpha_mats_out <- append(alpha_mats_out, alp$detach())
      j$alpha_active_path <- alp$detach()
      i <- i + 1
    }
    alp_out <- alpha_mats[[length(alpha_mats)]] * alpha_mats[[length(alpha_mats)]]$grad #last alpha
    alp_out[alp_out != 0] <- 1
    alpha_mats_out <- append(alpha_mats_out, alp_out$detach())
    self$out_layer$alpha_active_path <- alp_out$detach()
    return(alpha_mats_out)
  },

  density = function() { #the standard density, without active paths
    alphas <- NULL
    for (l in self$layers$children) {
      alphas <- c(alphas, as.numeric(l$alpha$clone()$detach()))
      }
    alphas <- c(alphas, as.numeric(self$out_layer$alpha$clone()$detach()))
    return(mean(alphas > 0.5))


  },
  density_active_path = function() {#density when removing connections not within active paths
    num_incl <- 0
    tot <- 0
    paths <- c() #intialize empty vector to compute path depths (only for input-skip)
    p <- self$sizes[1]
    for (l in self$layers$children) {
      tot <- tot +  l$alpha_active_path$numel() #total number of alphas
      num_incl <- num_incl + torch::torch_sum(l$alpha_active_path) #number of 1s
      if (self$input_skip) {
        alp <- l$alpha_active_path
        count <- (alp[, (dim(alp)[2] - p + 1):dim(alp)[2]] != 0)$flatten()$sum()$item() #count number non-zero elements in the p last entries of alpha matrix. this gives active paths
        paths <- append(paths, count)
      }
    }
    num_incl <- num_incl + torch::torch_sum(self$out_layer$alpha_active_path) #output layer
    tot <- tot +  self$out_layer$alpha_active_path$numel()
  
    #if no input skip then avg and max depth is always equal to the number of layers
    avg_path_length <- length(self$layers) + 1
    max_path_length <- length(self$layers) + 1

    if (self$input_skip) {
      alp_out <- self$out_layer$alpha_active_path
      out_count <- (alp_out[, (dim(alp_out)[2] - p + 1):dim(alp_out)[2]] != 0)$flatten()$sum()$item()
      paths <- append(paths, out_count)
      count_vector <- c((length(self$layers) + 1):1) #from number of layers -> 1. number of layers is the maximum possible path length.
      avg_path_length <- (torch::torch_dot(paths, count_vector) / sum(paths))$item()
      max_path_length <- count_vector[torch::torch_argmax((paths != 0) * 1)$item()] #get the index of of the first non-zero item in count vector, corresponding to the maximum path
    }

    return(num_incl$item() / tot)

  }

)
