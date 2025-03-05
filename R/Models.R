library(torch)

#' Class to generate a LBBNN network
#' @description Generates a LBBNN composed of feed forward layers defined by LBBNN_Linear
#' e.g sizes = c(20,200,200,5) generates an LBBNN with 20 input variables,
#' two hidden layers with 200 neurons each, and an output layer of 5 neurons.
#' LBBNN_net also contains functions to compute kl-divergence and the density of the entire network.
#' @param problem_type 'binary classification', 'multiclass classification' or 'regression'. 
#' @param sizes a vector containing the sizes of layers of the network, where the first element is the input size, and the last the output size.
#' @param prior a vector containing the prior inclusion probabilities for each layer in the network. Length must be ONE less than sizes.
#' @param std  a vector containing the prior standard deviation for each layer in the network. Length must be ONE less than sizes.
#' @param inclusion_inits a matrix of size (2,number of weight matrices). One upper and one lower bound for each layer.
#' @param device the device to be trained on. Can be 'cpu', 'gpu' or 'mps'. Default is cpu.
#' @examples
#' layers <- c(20,200,200,5) #Two hidden layers 
#' alpha <- c(0.3,0.5,0.9)  # One prior inclusion probability for each weight matrix 
#' stds <- c(1.0,1.0,1.0)  # One prior inclusion probability for each weight matrix 
#' inclusion_inits <- matrix(rep(c(-10,10),3),nrow = 2,ncol = 3)
#' prob <- 'multiclass classification'
#' net <- LBBNN_Net(problem_type = prob, sizes = layers, prior = alpha,std = stds,inclusion_inits = inclusion_inits,device = 'cpu')
#' print(net)
#' x <- torch::torch_rand(100,20,requires_grad = FALSE) #generate some dummy data
#' output <- net(x) #forward pass
#' net$kl_div()$item() #get KL-divergence
#' net$density() #get the density of the network
#' @export
LBBNN_Net <- torch::nn_module(
  "LBBNN_Net",
  

  initialize = function(problem_type,sizes,prior,std,inclusion_inits,device = 'cpu',link = NULL, nll = NULL) {

    self$layers <- torch::nn_module_list()
    self$problem_type = problem_type
    if(length(prior) != length(sizes) - 1)(stop('Must have one prior inclusion probability per weight matrix'))
    for(i in 1:(length(sizes)-2)){

      self$layers$append(LBBNN_Linear(sizes[i],sizes[i+1],prior_inclusion = prior[i],
                        standard_prior = std[i],density_init = inclusion_inits[,i],device))
    }
    self$out_layer <- (LBBNN_Linear(sizes[length(sizes)-1],sizes[length(sizes)]
                      ,prior_inclusion = prior[length(prior)],standard_prior = std[length(std)],
                      density_init = inclusion_inits[,ncol(inclusion_inits)],device))

    if(problem_type == 'binary classification'){
      self$out <- torch::nn_sigmoid()
      self$loss_fn <- torch::nn_bce_loss(reduction='sum')
    }
    else if(problem_type == 'multiclass classification' | problem_type == 'MNIST'){
      self$out <- torch::nn_log_softmax(dim = 2)
      self$loss_fn <- torch::nn_nll_loss(reduction='sum')
    }
    else if(problem_type == 'regression'){
      self$out <- torch::nn_identity()
      self$loss_fn <- torch::nn_mse_loss(reduction='sum')
    }else if(problem_type == 'custom')
    {
      if(length(link) == 0 | length(nll) == 0)
        stop("Under custom problem, link function and the negative log likelihood must be provided as torch functions")
      self$out <- link()
      self$loss_fn <- nll(reduction='sum')
    }
    else(stop('the type of problem must either be: \'binary classification\', 
              \'multiclass classification\', \'regression\' or \'custom\''))
  },
  forward = function(x,MPM=FALSE) {
    if(self$problem_type == 'MNIST')(x <- x$view(c(-1,28*28)))
    for(l in self$layers$children){
      
      x <- torch::nnf_leaky_relu(l(x,MPM)) #iterate over hidden layers
      
    }
    x <- self$out(self$out_layer(x,MPM))
    return(x)
    
  },
  kl_div = function(){
    kl <- 0
    for(l in self$layers$children)(kl <- kl + l$kl_div()) 
    kl <- kl + self$out_layer$kl_div()
    return(kl)
  },
  density = function(){
    alphas <- NULL
    for(l in self$layers$children)(alphas <- c(alphas,as.numeric(l$alpha$detach()))) #as.numeric flattens the matrix
    alphas<-c(alphas,as.numeric(self$out_layer$alpha$detach()))
    return(mean(alphas > 0.5))
    
    
  }
)



