library(igraph)
library(Matrix)
g <- make_graph(edges = c(1, 2, 1, 5,3,4,6,7), n = 10, directed = TRUE)
plot(g)
b <- as_adjacency_matrix(g)
plot(graph_from_adjacency_matrix(b))

mat <- matrix(data = c(1,1,1,0,0,1,0,1),ncol = 2,nrow = 4)


i <- c(1,3:10); j <- c(2,9,6:12); x <- 7 * (1:9)
(A <- sparseMatrix(i, j, x = x))    

bbb <- matrix((rnorm(16) > 0) * 1,nrow = 4,ncol = 4)
ccc <- matrix(0,nrow = sum(dim(mat)),ncol = sum(dim(mat))) ## adjacency matrix

first_dim <- 1:dim(mat)[1]
second_dim <- (dim(mat)[1] +1):sum(dim(mat))
ccc[first_dim,second_dim] <- mat
colnames(ccc) <- c('x1','x2','x3','x4','u1','u2')
#rownames(bbb) <- c('x1','x2','x3','x4')
colnames(bbb) <- c('x1','x2','x3','x4')

mat2 <- matrix(0,nrow = 4,ncol = 4)
colnames(mat2) <- c('u1','u2','v1','v2')
mat2[1,4] = 1
mat2[1,3] = 1
mat2[2,4] = 1
mat2 <- graph_from_adjacency_matrix(mat2,mode = 'directed')

#aa <- graph_from_adjacency_matrix(A)
cc <- graph_from_adjacency_matrix(ccc,mode = 'directed')
#l <- layout_as_tree(cc,flip.y = FALSE)
#plot(cc,layout =  l)

g <- cc + mat2
tr <- layout_as_tree(g,flip.y = T)
node_dist <- abs(tr[1,1]  -tr[2,1])
inp_layer_center <- mean(tr[1:4,1])

inp_size <- 4
h1_size <- 2
h2_size <- 2
tr2 <- matrix(0,nrow = length(g), ncol = 2) #specifies x and y position for each node in graph
tr2[1:inp_size,2] <- 1 #all nodes of input layer at same height
tr2[(inp_size+1):(inp_size + h1_size),2] <- 0.5 #hidden layer one
tr2[(inp_size + h1_size+1):(inp_size +h1_size + h2_size),2] <- 0 #hidden layer two

#now adjust the positions of the nodes within each layer
start_pos <- 0
end_pos <- 1.5
input_pos <- seq(from = start_pos, to = end_pos, length.out = 4)
u_pos <- seq(from = start_pos + 0.5,end_pos - 0.5,length.out = 2)
v_pos <- seq(from = start_pos + 0.5,end_pos - 0.5,length.out = 2)

tr2[1:inp_size,1] <- input_pos
tr2[(inp_size+1):(inp_size + h1_size),1] <- u_pos
tr2[(inp_size + h1_size+1):(inp_size +h1_size + h2_size),1] <- v_pos
plot(g,vertex.size = 15,vertex.color = 'lightblue',
     edge.width = 1, layout = -tr2[,2:1],edge.arrow.mode = '-')



###try to generalize with input layer first:

N <- 5 #num input neurons
start <- 0 #always start at the coordinate 0
neuron_spacing = 0.5 #how much whitespace between neurons in the plot
input_positions <- seq(from = start,length.out = N,by = neuron_spacing)
N_u <- 4 #number of neurons in hidden layer

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

##need to generalize so that we can have a function that takes a list of L layers of alphas
## and returns the plot
