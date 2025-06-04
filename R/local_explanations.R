library(torch)
#' @export
get_local_explanations_gradient <- function(model,input_data, 
                                            num_samples = 1,magnitude = TRUE,
                                            include_potential_contribution = FALSE,device = 'cpu'){
  if(model$input_skip == FALSE)(stop('This function is only implemented for input-skip'))
  #need to make sure input_data comes in shape (1,p) where p is #input variables
  num_classes <- model$sizes[length(model$sizes)]
  if(input_data$dim() == 4){ #in the case of MNIST or other image data
    (input_data <- input_data$view(c(-1,dim(input_data)[3]*dim(input_data)[4]))) #reshape to(1,p)
    
  }
  else{ #in case shape is either (p) or (1,p), both cases returns (1,p)
    input_data <- input_data$view((c(-1,length(input_data))))
  }
  
  p <- input_data$shape[2] #number of variables
  explanations <- torch::torch_zeros(num_samples,p,num_classes)
  predictions <- torch::torch_zeros(num_samples,num_classes)
  
  model$local_explanation = TRUE #to skip last sigmoid/softmax layer
  for( b in 1:num_samples){#for each sample, get explanations for each class
    input_data$requires_grad = TRUE
    input_data <- input_data$to(device = device)
    model$zero_grad()
    output = model(input_data,MPM = TRUE) #forward pass, using MPM
    for(k in 1:num_classes){
      output_value <- output[1,k]
      grad = torch::autograd_grad(outputs = output_value,inputs = input_data,
                                  grad_outputs = torch::torch_ones_like(output_value),
                                  retain_graph =TRUE)
      
      explanations[b,,k] <- grad[[1]]
    
      predictions[b,k] <- output[1,k]
      
    }
    
  }

  inds <- torch::torch_nonzero(input_data == 0)[,2] #find index of 0s in input data
  if(include_potential_contribution){
      explanations[,inds] <- - explanations[,inds]
  }
  else{#remove variables that do not contribute to predictions
    explanations[,inds] <- 0
  }
  if(! magnitude){
    explanations <- explanations * input_data$view(c(1,-1,1))
  }
  predictions <- predictions$detach()$cpu()
  outputs = list('explanations' = explanations,'p' = p,'predictions' = predictions)
  return(outputs)
}



problem <- 'MNIST'
sizes <- c(28*28,3,3,10) 
inclusion_priors <-c(0.1,0.1,0.1) #one prior probability per weight matrix.
std_priors <-c(1.0,1.0,1.0) #one prior probability per weight matrix.
inclusion_inits <- matrix(rep(c(-10,10),3),nrow = 2,ncol = 3)
device <- 'mps'
torch_manual_seed(0)
model <- LBBNN_Net(problem_type = problem,sizes = sizes,
                   prior = inclusion_priors,inclusion_inits =inclusion_inits ,input_skip = TRUE,
                   std = std_priors,flow = FALSE,num_transforms = 2,dims = c(200,200),device = device)




dat <- torch_rand(1,28*28)
outputs <- get_local_explanations_gradient(model,dat,device = device)







