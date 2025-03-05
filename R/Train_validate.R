library(torch)

#' Function to train an instance of LBBNN_Net
#' @description Function that for each epoch itereates through each mini-batch, computing
#' the loss and using backpropagation to update the network parameters.
#' @param epochs Total number of epochs to train for, where one epoch is a pass through the entire training dataset (all minibatches).
#' @param LBBNN An instance of the LBBNN_Net class, to be trained.
#' @param lr The learning rate to be used in the Adam optimizer.
#' @param train_dl An instance of torch dataloader, containing the data to be trained.
#' @param device the device to be trained on. Default is cpu.
#' @return a list containing the losses and accuracies (if classification) and density for each epoch during training.
#' @examples 
#' x<-torch::torch_randn(1000,10) #generate some data
#'b <- torch::torch_rand(10)
#'y <- torch::torch_matmul(x,b)
#'med <- torch::torch_median(y)
#'y[y > med] = 1 #change it into binary classification
#'y[y <= med] = 0
#'train_data <- torch::tensor_dataset(x,y)
#'train_loader <- torch::dataloader(train_data,batch_size = 100,shuffle=TRUE)
#'problem<-'binary classification'
#'sizes <- c(10,50,1) #10 input variables, one hidden layer of 50 neurons, 1 output
#'inclusion_priors <-c(0.3,0.9) #one prior probability per weight matrix
#'inclusion_inits <- matrix(rep(c(-10,10),2),nrow = 2,ncol = 2)
#'stds <- c(1.0,1.0)
#'model <- LBBNN_Net(problem,sizes,inclusion_priors,stds,inclusion_inits,flow = FALSE)
#'output <- train_LBBNN(epochs = 10,LBBNN = model, lr = 0.01,train_dl = train_loader)
#'@export
train_LBBNN <- function(epochs,LBBNN,lr,train_dl,device = 'cpu'){
  opt <- torch::optim_adam(LBBNN$parameters,lr = lr)
  accs <- c()
  losses <-c()
  density <- c()
  out_layer_density <- c()
  for (epoch in 1:epochs) {
    LBBNN$train()
    corrects <- 0
    totals <- 0
    train_loss <- c()
    # use coro::loop() for stability and performance
    coro::loop(for (b in train_dl) {

      opt$zero_grad()
      data <- b[[1]]$to(device = device)
      output <- LBBNN(data,MPM=FALSE)
      target <- b[[2]]$to(device=device)
  
      if(LBBNN$problem_type == 'multiclass classification'| LBBNN$problem_type == 'MNIST'){ #nll loss needs float tensors but bce loss needs long tensors 
        target <- torch::torch_tensor(target,dtype = torch::torch_long())
      }
      else(output <- output$squeeze()) #remove last dimension from binary classifiction or regression
      loss <- LBBNN$loss_fn(output, target) + LBBNN$kl_div() / length(train_dl)
      loss$backward()
      opt$step()
      if(LBBNN$problem_type == 'multiclass classification' | LBBNN$problem_type == 'MNIST'){
        prediction <-max.col(output)
        corrects <- corrects + sum(prediction == target)
        totals <- totals + length(target)
        train_loss <- c(train_loss,loss$item())
      }
      else if(LBBNN$problem_type == 'binary classification'){
        corrects<-corrects + sum((output > 0.5) == target)
        totals <- totals + length(target)
        train_loss <- c(train_loss,loss$item())
        
      }
      else if(LBBNN$problem_type == 'custom')
      {
        train_loss <- c(train_loss,loss$item())
      }
      else{#for regression
        train_loss <- c(train_loss,loss$item())
        
      }
      
      
    })
    
    train_acc <- corrects / totals
    if(LBBNN$problem_type != 'regression'){
      cat(sprintf(
        "\nEpoch %d, training: loss = %3.5f, acc = %3.5f,, density = %3.5f \n",
        epoch, mean(train_loss), train_acc,LBBNN$density()
      ))
      
      
      accs <- c(accs,train_acc$item())
      losses <- c(losses,mean(train_loss))
    }
    if(LBBNN$problem_type == 'regression'){
      cat(sprintf(
        "\nEpoch %d, training: loss = %3.5f \n",
        epoch, mean(train_loss)
      ))
      
      losses <- c(losses,mean(train_loss))
    }
    density <- c(density,LBBNN$density())
    
    
  }
  l = list('accs' = accs,'loss' = losses,'density' = density)
  return(l)
}





#'Function to validate a trained LBBNN model
#' @description Computing metrics for each-mini batch in the validation dataset,
#' without computing gradients. Model averaging (i.e. num samples > 1) is encouraged 
#' for improved performance. 
#'@param LBBNN An instance of a trained LBBNN_net to be validated
#'@param num_samples The number of samples from the variational posterior to be used for model averaging
#'@param test_dl An instance of torch dataloader, containing the validation data
#'@param device The device to to validate on. Default is cpu
#'@return A list containing accuracy if classification, or loss if regression. In both cases
#'results are returned for both the full model, and the sparse model, only using weights with
#'a posterior inclusion probability larger than 0.5. The density is also returned.
#'@export
validate_LBBNN <- function(LBBNN,num_samples,test_dl,device = 'cpu'){
  LBBNN$eval
  corrects <- 0
  corrects_sparse <-0
  totals <- 0 
  val_loss <- c()
  val_loss_mpm <-c()
  out_shape <- 1 #if binary classification or regression
  with_no_grad({ 
    coro::loop(for (b in test_dl){
      target <- b[[2]]$to(device=device)
      if(LBBNN$problem_type == 'multiclass classification'| LBBNN$problem_type == 'MNIST'){ #nll loss needs float tensors but bce loss needs long tensors 
        target <- torch::torch_tensor(target,dtype = torch::torch_long())
        out_shape <- max(target)$item()
      }
      outputs <- torch::torch_zeros(num_samples,dim(b[[1]])[1],out_shape)$to(device=device)
      output_mpm <- torch::torch_zeros_like(outputs)
      for(i in 1:num_samples){
        data <- b[[1]]$to(device = device)
        outputs[i]<- LBBNN(data,MPM=FALSE)
        output_mpm[i]<- LBBNN(data,MPM=TRUE)
        
      }
      out_full <-outputs$mean(1) #average over num_samples dimension
      out_mpm <-output_mpm$mean(1)
      
      if(LBBNN$problem_type == 'multiclass classification' | LBBNN$problem_type == 'MNIST'){
        prediction <-max.col(out_full)
        prediction_mpm <- max.col(out_mpm)
        corrects <- corrects + sum(prediction == target)
        corrects_sparse <- corrects_sparse + sum(prediction_mpm == target)
        totals <- totals + length(target)
        
        
      }
      
      else if(LBBNN$problem_type == 'binary classification'){
        out_full <- out_full$squeeze()
        out_mpm <-out_mpm$squeeze()
        corrects<-corrects + sum((out_full > 0.5) == target)
        corrects_sparse<-corrects_sparse + sum((out_mpm > 0.5) == target)
        totals <- totals + length(target)
      }
      else{#for regression
        out_full <- out_full$squeeze()
        out_mpm <-out_mpm$squeeze()
        
        loss <- torch::torch_sqrt(torch::nnf_mse_loss(out_full, target))
        loss_mpm <- torch::torch_sqrt(torch::nnf_mse_loss(out_mpm, target))
        val_loss <- c(val_loss,loss$item())
        val_loss_mpm <- c(val_loss_mpm,loss_mpm$item())
      }
      
      
      
      
    })  
  })
  acc_full<- corrects / totals
  acc_sparse <- corrects_sparse / totals
  density <- LBBNN$density()
  if(LBBNN$problem_type!='regression'){
    l = list('accuracy_full_model' = acc_full$item(),'accuracy_sparse' = acc_sparse$item(),'density'=density)
  }
  else{
    l = list('validation_error'=mean(val_loss),'validation_error_sparse' = mean(val_loss_mpm),'density'=density)
  }
  return(l)
}


#example

#torch_manual_seed(0)
#x<-torch_randn(200,10) #generate some data
#b <- torch_rand(10)
#y <- torch_matmul(x,b)
#med <- torch_median(y)
#y[y > med] = 1
#y[y <= med] = 0
#train_data <- tensor_dataset(x,y)
#train_loader <- dataloader(train_data,batch_size = 100,shuffle=TRUE)
#problem<-'binary classification'
#sizes <- c(10,50,50,1) #10 input variables, one hidden layer of 50 neurons, 1 output
#inclusion_priors <-c(0.3,0.5,0.9) #one prior probability per weight matrix
#model <- LBBNN_Net(problem,sizes,inclusion_priors)
#output <- train_LBBNN(epochs = 100,LBBNN = model, lr = 0.01,train_dl = train_loader)

#generate test data
#x<-torch_randn(200,10) #generate some data
#y <- torch_matmul(x,b)
#med <- torch_median(y)
#y[y > med] = 1
#y[y <= med] = 0
#test_data <- tensor_dataset(x,y)
#test_loader <- dataloader(test_data,batch_size = 100,shuffle=FALSE)
#val_out <- validate_LBBNN(model,10,test_loader)






