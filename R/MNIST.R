library(torchvision)

dir <- "./mnist"

train_ds <- mnist_dataset(
  dir,
  download = TRUE,
  transform = transform_to_tensor
)

test_ds <- mnist_dataset(
  dir,
  train = FALSE,
  transform = transform_to_tensor
)

train_dl <- dataloader(train_ds, batch_size = 1000, shuffle = TRUE)
test_dl <- dataloader(test_ds, batch_size = 1000)

torch_manual_seed(0)
#define hyperparameters
problem<-'MNIST'
sizes <- c(28*28,600,600,10) #10 input variables, one hidden layer of 50 neurons, 1 output
inclusion_priors <-c(0.25,0.25,0.25) #one prior probability per weight matrix
device <- 'cpu' #other possibilities are 'gpu' or 'mps'
#model <- LBBNN_Net(problem,sizes,inclusion_priors,device)
#output <- train_LBBNN(epochs = 100,LBBNN = model, lr = 0.001,train_dl = train_dl,device)
#validate <-validate_LBBNN(LBBNN = model,num_samples = 10,test_dl = test_dl,device)

