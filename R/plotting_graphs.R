library(igraph)
library(Matrix)
require(graphics)

#' Function that checks which inputs are included, and from which layer
#' @description Useful when the number of inputs and/or hidden neurons are very
#' large, and direct visualization of the network is difficult. 
#' @param model A trained LBBNN model with input_skip. 
#' @return A matrix of shape (p, L-1) where p is the number of input variables
#' and L the total number of layers (including input and output), with each element being 1 if the variable is included
#' or 0 if not included. 
#' @export
get_input_inclusions <- function(model){
  if(model$input_skip == FALSE)(stop('This function is currently only implemented for input-skip'))
  x_names <- c()
  layer_names <- c()
  for(k in 1:model$sizes[1]){
    x_names<- c(x_names,paste('x',k-1,sep = ''))
  }
  for(l in 1:(length(model$sizes)-1)){
    layer_names <- c(layer_names,paste('L',l-1,sep = ''))
  }
  
  
  inclusion_matrix <- matrix(0,nrow = model$sizes[1],ncol = length(model$sizes) - 1)
  #add the names
  colnames(inclusion_matrix) <- layer_names
  rownames(inclusion_matrix) <- x_names
  
  
  inp_size <- model$sizes[1]
  i <- 1
  for(l in model$layers$children){
    alp <- l$alpha_active_path
    incl<- as.numeric(torch::torch_max(alp[,-inp_size:dim(alp)[2]],dim = 1)[[1]])
    inclusion_matrix[,i] <- incl 
    i <- i + 1
  }
  alp_out <- model$out_layer$alpha_active_path
  incl<- as.numeric(torch::torch_max(alp_out[,-inp_size:dim(alp_out)[2]],dim = 1)[[1]])
  inclusion_matrix[,i] <- incl 
  i <- i + 1
 
 return(inclusion_matrix) 
}


#' Function to obtain adjacency matrices to be used with igraph plotting
#' @description Given a trained LBBNN model with input-skip, this 
#' function takes the alpha active path matrices for each layer and converts
#' them to adjacency matrices so that they can be plotted with igraph.
#' @param model A trained LBBNN model with input-skip. 
#' @return The adjacency matrices. 
#' @export
get_adj_mats <- function(model){
  mats_out <-list()
  i <- 1
  for(l in model$layers$children){
    alp <- t(as.matrix(l$alpha_active_path))
    adj_mat <- matrix(0,nrow = sum(dim(alp)),ncol = sum(dim(alp))) #initialize empty matrix
    first_dim <- 1:dim(alp)[1]
    second_dim <- (dim(alp)[1] +1):sum(dim(alp))
    adj_mat[first_dim,second_dim] <- alp
    mats_out[[i]] <- adj_mat
    i <- i + 1
  } #do the same for the output layer
  alp_out <- t(as.matrix(model$out_layer$alpha_active_path))
  adj_mat_out <- matrix(0,nrow = sum(dim(alp_out)),ncol = sum(dim(alp_out))) #
  first_dim <- 1:dim(alp_out)[1]
  second_dim <- (dim(alp_out)[1] +1):sum(dim(alp_out))
  adj_mat_out[first_dim,second_dim] <- alp_out
  mats_out[[i]] <- adj_mat_out
  
  
  return(mats_out)
}

#' Function for plotting nodes in the network between two layers. 
#' @description Takes care of the three possible cases. Both layers have even
#' number of neurons, both layers have odd numbers, or one of each. 
#' @param N Number of neurons in the first layer.
#' @param N_u Number of neurons in the second layer.
#' @param input_positions Positions of the neurons in the input layer.
#' @param neuron_spacing How much space between the neurons. (do we need this?, seems like igraph overrides or smth)
#' @return Positions of the second layer. 
assign_within_layer_pos<- function(N,N_u,input_positions,neuron_spacing){
  if(N %% 2 == 0 & N_u %% 2 == 0){ #if both layers have even number of neurons
    N_u_center <- stats::median(input_positions)
    N_u_start_pos <- N_u_center + neuron_spacing / 2 - (N_u /2 * neuron_spacing) #add the half space, then subtract half of the array to get to start point
    N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)

  } 
  
  if(N %% 2 != 0 & N_u %% 2 != 0){ #if both layers have odd number of neurons
    N_u_center <- stats::median(input_positions)
    N_u_start_pos <- N_u_center - ((N_u - 1) / 2) * neuron_spacing #just need to figure out how many neurons to the left of the median one
    N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)
  } 
  
  if((N + N_u) %% 2 != 0){ #in the case of even and odd number of neurons. Even + odd = odd
    if(N > N_u){ #in this case, N_u is odd
      N_u_center <- stats::median(input_positions)
      N_u_start_pos <- N_u_center + neuron_spacing / 2 - (N_u /2 * neuron_spacing) 
      N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)
    }
    if(N < N_u){ #in this case, N_u is even
      N_u_center <- stats::median(input_positions)
      N_u_start_pos <- N_u_center - ((N_u - 1) / 2) * neuron_spacing 
      N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)
    }
  }
    return(N_u_positions)
}



assign_names<- function(model){#assign names to the nodes before plotting
  alphas <- get_adj_mats(model)
  sizes <- model$sizes
  for(i in 1:length(alphas)){
    mat_names <- c()
    if(i == 1){ #for the input layer
      for(j in 1:sizes[1]){ #first the x_i
        name <- paste('x',j-1,'_',i-1,sep = '') #i-1 because x belongs to the first (input layer)
        mat_names <- c(mat_names,name)
      }
      for(j in 1:sizes[i + 1]){#then the u
        name <- paste('u',j-1,'_',i,sep = '')
        mat_names <- c(mat_names,name)
      }
      
      colnames(alphas[[i]]) <- mat_names
    }
    else if(i < length(alphas)){#all other layers except the last
      
      mat_names <- c()
      for(j in 1:sizes[i]){#N - n_input is the number of neurons in the hidden layer
        name <- paste('u',j-1,'_',i-1,sep = '')
        mat_names <- c(mat_names,name)
      }
      for(j in 1:sizes[1]){#the input skip x
        name <- paste('x',j-1,'_',i-1,sep = '')
        mat_names <- c(mat_names,name)
      }
      for(j in 1:sizes[i + 1]){#the hidden neurons for the next layer
        name <- paste('u',j-1,'_',i,sep = '')
        mat_names <- c(mat_names,name)
      }
      colnames(alphas[[i]]) <- mat_names
      
    }
    else{#the last layer note: this is almost the same as above, could join them together??
      mat_names <- c()
      for(j in 1:sizes[i]){#N - n_input is the number of neurons in the hidden layer
        name <- paste('u',j-1,'_',i-1,sep = '')
        mat_names <- c(mat_names,name)
      }
      for(j in 1:sizes[1]){#the input skip x
        name <- paste('x',j-1,'_',i-1,sep = '')
        mat_names <- c(mat_names,name)
      }
      for(j in 1:sizes[i + 1]){#the hidden neurons for the next layer
        name <- paste('y',j-1,sep = '')
        mat_names <- c(mat_names,name)
      }
      colnames(alphas[[i]]) <- mat_names
      
    }
    
  }
  return(alphas)
  
}




#' @export
LBBNN_plot <- function(model,layer_spacing,neuron_spacing,vertex_size,edge_width){
  if(model$input_skip == FALSE)(stop('Plotting currently only implemented for input-skip'))
  graph <- assign_names(model) #the graph with names neurons, given some model with alpha matrices
  g <- igraph::make_empty_graph(n = 0) #initialize empty graph
  for(L in 1:length(graph)){
    g <- g +  igraph::graph_from_adjacency_matrix(graph[[L]],mode = 'directed')
  }
  plot_points <- matrix(0,nrow = length(g),ncol = 2) #x,y coordinates for all neurons in g
  layer_positions <- seq(from = 0,by = - layer_spacing,length.out = length(model$sizes)) #position for each layer
  index_start <- 0 
  dim_1_pos <- 0
  i <- 1
  for(s in model$sizes){
    
    if(i == 1){
      plot_points[1:model$sizes[i],2] <- layer_positions[i] #input layer coords
      index_start <- model$sizes[i] + 1 #where to start next layer
      dim_1_pos <- seq(from = 0,length.out = model$sizes[i],by = neuron_spacing)#coords within input layer
      plot_points[1:model$sizes[i],1] <- dim_1_pos 
      
    }
    else if(i < length(model$sizes)){#all other layers except the last
      
      plot_points[(index_start:(index_start + model$sizes[1] + model$sizes[i]-1)),2] <- layer_positions[i]
      #N = size of prev layer #N_u size of current layer
      dim_1_pos <- assign_within_layer_pos(N = length(dim_1_pos),N_u = model$sizes[1] + model$sizes[i],
                                           input_positions = dim_1_pos,neuron_spacing = neuron_spacing)
      
      
      
      plot_points[(index_start:(index_start + model$sizes[1] + model$sizes[i]-1)),1] <- dim_1_pos
      index_start <- index_start + model$sizes[1] + model$sizes[i] 
      
      
      
    }
    else{ #output layer
      dim_1_pos <- assign_within_layer_pos(N = length(dim_1_pos),N_u = model$sizes[length(model$sizes)],
                                           input_positions = dim_1_pos,neuron_spacing = neuron_spacing)
      plot_points[(index_start:(dim(plot_points)[1])),1] <- dim_1_pos
      plot_points[(index_start:(dim(plot_points)[1])),2] <- layer_positions[i]
      
      
    }
    i <- i + 1
    
  }
  #assign colors based on what type of neuron it is
  for(z in 1:length(igraph::V(g))){ 
    string <- igraph::V(g)[z]
    if(grepl('x',string$name)){ #for input neurons
      igraph::V(g)[z]$color <- '#D5E8D4'
      igraph::V(g)[z]$frame.color <- '#D5E8D4'#change color of boundary too
    }
    else if(grepl('u',string$name)){ #hidden neurons
      igraph::V(g)[z]$color <- '#ADD8E6'
      igraph::V(g)[z]$frame.color <- '#ADD8E6'
    }
    else{
      igraph::V(g)[z]$color <- '#F8CECC' #output neurons
      igraph::V(g)[z]$frame.color <- '#F8CECC'
    }
    
    
  }
  
  plot(g,vertex.size = vertex_size,vertex.label.cex = 0.5, 
       edge.color = 'black',vertex.label.color='black',
       edge.width = edge_width, layout = -plot_points[,2:1],edge.arrow.mode = '-',margin = 0.0)
 

  
}
#can use the below as an example
sizes <- c(2,3,3,2) 
problem <- 'multiclass classification'
inclusion_priors <-c(0.1,0.1,0.1) #one prior probability per weight matrix.
std_priors <-c(1.0,1.0,1.0) #one prior probability per weight matrix.
inclusion_inits <- matrix(rep(c(-10,10),3),nrow = 2,ncol = 3)
device <- 'cpu'
torch_manual_seed(0)
model <- LBBNN_Net(problem_type = problem,sizes = sizes,
                   prior = inclusion_priors,inclusion_inits =inclusion_inits ,input_skip = TRUE,
                   std = std_priors,flow = FALSE,num_transforms = 2,dims = c(200,200),device = device)

LBBNN_plot(model,1,1,14,1)
