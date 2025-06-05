library(torch)
library(ggplot2)


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



problem <- 'multiclass classification'
sizes <- c(7,3,3,10) 
inclusion_priors <-c(0.1,0.1,0.1) #one prior probability per weight matrix.
std_priors <-c(1.0,1.0,1.0) #one prior probability per weight matrix.
inclusion_inits <- matrix(rep(c(-10,10),3),nrow = 2,ncol = 3)
device <- 'mps'
torch_manual_seed(0)
model <- LBBNN_Net(problem_type = problem,sizes = sizes,
                   prior = inclusion_priors,inclusion_inits =inclusion_inits ,input_skip = TRUE,
                   std = std_priors,flow = FALSE,num_transforms = 2,dims = c(200,200),device = device)



### next is to plot the contributions with error bars

#' @export
quants <- function(x){
  return(quantile(x,probs = c(0.025,0.5,0.975))) #95% CI and median
}



#' @export
plot_local_explanations_gradient <- function(model,input_data,num_samples,device){
  outputs <- get_local_explanations_gradient(model = model,input_data = input_data,num_samples
                                             =num_samples,device = device)
  
  preds<- as.matrix(outputs$predictions) #shape (num_samples,num_classes)
  expl <- as.array(outputs$explanations) #shape (num_samples,p,num_classes)
  
  for(cls in 1:model$sizes[length(model$sizes)]){ #loop over each class and compute quantiles
    expl_class <- expl[,,cls]
    expl_quantiles <-apply(expl_class,2,quants)
   
    pred_quantiles <-apply(preds,2,quants)
    
    names <- c()
    median <- expl_quantiles[2,]
    pred_median<- pred_quantiles[,cls][2] #add the median of the prediction to the end
    contribution <- c(median,pred_median)
    min <- c(expl_quantiles[1,],pred_quantiles[,cls][1])
    max <- c(expl_quantiles[3,],pred_quantiles[,cls][3])
    
    
    for(x in 1:model$sizes[1]){#get names for x-axis
      name <- paste('x',x-1,sep = '')
      names <- c(names,name)
      
    }
    names <- c(names,'prediction') 
  

    data <- data.frame(
      name=names,
      contribution = contribution,
      min = min,
      max = max
    )
    #add a row for the prediction
    #data<-rbind(data,c('prediction',pred_median,pred_quantiles[,cls][1],pred_quantiles[,cls][3]))
   
    print(ggplot(data <- data,aes(x=factor(name,levels = name),
                            y=contribution,
                            fill=factor(ifelse(name=="prediction","prediction","input variables")))) +
      geom_bar(stat="identity") +
      scale_fill_manual(name = "", values=c("#D5E8D4",'#F8CECC')) +
      geom_errorbar( aes(x=name, ymin=min, ymax=max), width=0.6, colour="black", alpha=0.9, size=0.5) +
      xlab("")) + ylab('Contribution')
    
    
    
    

    
  
   
    
    
  } 



}





