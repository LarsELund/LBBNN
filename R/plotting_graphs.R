library(igraph)
library(Matrix)
require(graphics)

torch_manual_seed(0)
problem <- 'binary classification'
sizes <- c(4,2,2,1) 
inclusion_priors <-c(0.1,0.1,0.1) #one prior probability per weight matrix.
std_priors <-c(1.0,1.0,1.0) #one prior probability per weight matrix.
inclusion_inits <- matrix(rep(c(-3,3),3),nrow = 2,ncol = 3)
device <- 'cpu'

model <- LBBNN_Net(problem_type = problem,sizes = sizes,
                   prior = inclusion_priors,inclusion_inits =inclusion_inits ,input_skip = TRUE,
                   std = std_priors,flow = FALSE,num_transforms = 2,dims = c(200,200),device = device)


#create a function that returns adjacency matricies



mat <- matrix(data = c(1,1,1,0,0,1,0,1),ncol = 2,nrow = 4) #input layer data




ccc <- matrix(0,nrow = sum(dim(mat)),ncol = sum(dim(mat))) ## adjacency matrix

first_dim <- 1:dim(mat)[1]
second_dim <- (dim(mat)[1] +1):sum(dim(mat))
ccc[first_dim,second_dim] <- mat
colnames(ccc) <- c('x11','x21','x31','x41','u1','u2')



mat2 <- matrix(0,nrow = 8,ncol = 8)
colnames(mat2) <- c('u1','u2','x12','x22','x32','x42','v_1','v_2')
mat2[1,8] = 1
mat2[1,7] = 1
mat2[2,8] = 1
mat2[3,7] = 1
mat2 <- graph_from_adjacency_matrix(mat2,mode = 'directed')

mat3<- matrix(0,nrow = 7,ncol = 7)
colnames(mat3) <- c('v_1','v_2','x13','x23','x33','x43','y')
mat3[1,7] = 1
mat3[2,7] = 1
mat3[3,7] = 1
mat3 <- graph_from_adjacency_matrix(mat3,mode = 'directed')

#aa <- graph_from_adjacency_matrix(A)
cc <- graph_from_adjacency_matrix(ccc,mode = 'directed')
#l <- layout_as_tree(cc,flip.y = FALSE)
#plot(cc,layout =  l)

g <- cc + mat2 + mat3
tr <- layout_as_tree(g,flip.y = T)
node_dist <- abs(tr[1,1]  -tr[2,1])
inp_layer_center <- mean(tr[1:4,1])

inp_size <- 4
h1_size <- 2 + inp_size
h2_size <- 2
tr2 <- matrix(0,nrow = length(g), ncol = 2) #specifies x and y position for each node in graph
tr2[1:inp_size,2] <- 1 #all nodes of input layer at same height
tr2[(inp_size+1):(inp_size + h1_size),2] <- 0.5 #hidden layer one
tr2[(inp_size + h1_size+1):(inp_size +h1_size + h2_size + inp_size),2] <- 0.0 #hidden layer two
tr2[17,2] <- -0.5

#now adjust the positions of the nodes within each layer
start_pos <- 0
end_pos <- 1.5
input_pos <- seq(from = start_pos, to = end_pos, length.out = 4)
u_pos <- seq(from = start_pos + 0.5,end_pos - 0.5,length.out = 6)
v_pos <- seq(from = start_pos + 0.5,end_pos - 0.5,length.out = 6)
y_pos <- 0.75

tr2[1:inp_size,1] <- input_pos
tr2[(inp_size+1):(inp_size + h1_size),1] <- u_pos
tr2[(inp_size + h1_size+1):(inp_size +h1_size + h2_size + inp_size),1] <- v_pos
tr2[17,1] <- y_pos
plot(g,vertex.size = 14,vertex.color = 'lightblue',
     edge.width = 1, layout = -tr2[,2:1],edge.arrow.mode = '-')



###try to generalize with input layer first:

N <- 4 #num input neurons
start <- 0 #always start at the coordinate 0
neuron_spacing = 0.5 #how much whitespace between neurons in the plot
input_positions <- seq(from = start,length.out = N,by = neuron_spacing)
N_u <-  4 + 2 #number of neurons in hidden layer

if(N %% 2 == 0 & N_u %% 2 == 0){ #if both layers have even number of neurons
  N_u_center <- median(input_positions)
  N_u_start_pos <- N_u_center + neuron_spacing / 2 - (N_u /2 * neuron_spacing) #add the half space, then subtract half of the array to get to start point
  N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)
} 

if(N %% 2 != 0 & N_u %% 2 != 0){ #if both layers have odd number of neurons
  N_u_center <- median(input_positions)
  N_u_start_pos <- N_u_center - ((N_u - 1) / 2) * neuron_spacing #just need to figure out how many neurons to the left of the median one
  N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)
} 

if((N + N_u) %% 2 != 0){ #in the case of even and odd number of neurons. Even + odd = odd
  if(N > N_u){ #in this case, N_u is odd
    N_u_center <- median(input_positions)
    N_u_start_pos <- N_u_center + neuron_spacing / 2 - (N_u /2 * neuron_spacing) 
    N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)
  }
  if(N < N_u){ #in this case, N_u is even
    N_u_center <- median(input_positions)
    N_u_start_pos <- N_u_center - ((N_u - 1) / 2) * neuron_spacing 
    N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)
  }
}

print(input_positions)
print(N_u_positions)

assign_plot_pts <- function(spacing){
  
}
#similar to the example above
a <- matrix(rnorm(36),nrow = 6,ncol=6)
b <- matrix(rnorm(64),nrow = 8,ncol = 8)
cc <- matrix(rnorm(49),nrow = 7,ncol = 7)
n_inp <- 4
sizes <- c(4,2,2,1) #4 input, two hidden layers of two each, one output

alphas <- list(a,b,cc) #a list with adjacency matrices of the original alpha matrices
for(i in 1:length(alphas)){
  mat_names <- c()
  if(i == 1){ #for the input layer
    for(j in 1:n_inp){ #first the x_i
      name <- paste('x',j,'_',i-1,sep = '') #i-1 because x belongs to the first (input layer)
      mat_names <- c(mat_names,name)
    }
    for(j in 1:sizes[i + 1]){#then the u
      name <- paste('u',j,'_',i,sep = '')
      mat_names <- c(mat_names,name)
    }
    
    colnames(alphas[[i]]) <- mat_names
  }
  else if(i < length(alphas)){#all other layers except the last
   
    mat_names <- c()
    for(j in 1:sizes[i]){#N - n_input is the number of neurons in the hidden layer
      name <- paste('u',j,'_',i-1,sep = '')
      mat_names <- c(mat_names,name)
    }
    for(j in 1:n_inp){#the input skip x
      name <- paste('x',j,'_',i-1,sep = '')
      mat_names <- c(mat_names,name)
    }
    for(j in 1:sizes[i + 1]){#the hidden neurons for the next layer
      name <- paste('u',j,'_',i,sep = '')
      mat_names <- c(mat_names,name)
    }
    colnames(alphas[[i]]) <- mat_names
    
  }
  else{#the last layer note: this is almost the same as above, could join them together??
    mat_names <- c()
    for(j in 1:sizes[i]){#N - n_input is the number of neurons in the hidden layer
      name <- paste('u',j,'_',i-1,sep = '')
      mat_names <- c(mat_names,name)
    }
    for(j in 1:n_inp){#the input skip x
      name <- paste('x',j,'_',i-1,sep = '')
      mat_names <- c(mat_names,name)
    }
    for(j in 1:sizes[i + 1]){#the hidden neurons for the next layer
      name <- paste('y',j,sep = '')
      mat_names <- c(mat_names,name)
    }
    colnames(alphas[[i]]) <- mat_names
    
  }
  
}



##need to generalize so that we can have a function that takes a list of L layers of alphas
## and returns the plot
## need to automatically give names to the inputs and the hidden neurons # x1... xn, u1,..un etc

