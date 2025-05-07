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
plot(g,vertex.size = 30,vertex.color = 'lightblue',
     edge.width = 1, layout = -tr[,2:1],edge.arrow.mode = '-')







