library(ucimlrepo)
mgp<- fetch_ucirepo(id=9) 

mgp_dataset <- as.data.frame(mgp$data$features)



#replace the nan in the horsepower column
mean_x <- mean(mgp_dataset$horsepower, na.rm = TRUE)
mgp_dataset$horsepower[is.nan(mgp_dataset$horsepower)] <- mean_x


N<- dim(mgp_dataset)[1]
#one hot the cylinder variable
cyl4<- numeric(N)
cyl6 <- numeric(N)
cyl8 <- numeric(N)
cyl4[mgp_dataset$cylinders == 3] = 1 #3 has very few so combine it with 4
cyl4[mgp_dataset$cylinders == 4] = 1
cyl6[mgp_dataset$cylinders == 5] = 1 #5 has very few so combine it with 6
cyl6[mgp_dataset$cylinders == 6] = 1
cyl8[mgp_dataset$cylinders == 8] = 1
mgp_dataset$cylinders <- NULL
mgp_dataset$cyl4 <- cyl4
mgp_dataset$cyl6 <- cyl6
mgp_dataset$cyl8 <- cyl8

#one hot the model year variable
y70 <- numeric(N)
y71 <- numeric(N)
y72 <- numeric(N)
y73 <- numeric(N)
y74 <- numeric(N)
y75 <- numeric(N)
y76 <- numeric(N)
y77 <- numeric(N)
y78 <- numeric(N)
y79 <- numeric(N)
y80 <- numeric(N)
y81 <- numeric(N)
y82 <- numeric(N)

y70[mgp_dataset$model_year == 70] = 1
y71[mgp_dataset$model_year == 71] = 1
y72[mgp_dataset$model_year == 72] = 1
y73[mgp_dataset$model_year == 73] = 1
y74[mgp_dataset$model_year == 74] = 1
y75[mgp_dataset$model_year == 75] = 1
y76[mgp_dataset$model_year == 76] = 1
y77[mgp_dataset$model_year == 77] = 1
y78[mgp_dataset$model_year == 78] = 1
y79[mgp_dataset$model_year == 79] = 1
y80[mgp_dataset$model_year == 80] = 1
y81[mgp_dataset$model_year == 81] = 1
y82[mgp_dataset$model_year == 82] = 1

mgp_dataset$model_year <- NULL

mgp_dataset$y70 <- y70
mgp_dataset$y71 <- y71
mgp_dataset$y72 <- y72
mgp_dataset$y73 <- y73
mgp_dataset$y74 <- y74
mgp_dataset$y75 <- y75
mgp_dataset$y76 <- y76
mgp_dataset$y77 <- y77
mgp_dataset$y78 <- y78
mgp_dataset$y79 <- y79
mgp_dataset$y80 <- y80
mgp_dataset$y81 <- y81
mgp_dataset$y82 <- y82



#one hot the origin variable
o1 <- numeric(N)
o2 <- numeric(N)
o3 <- numeric(N)
o1[mgp_dataset$origin == 1] = 1
o2[mgp_dataset$origin == 2] = 1
o3[mgp_dataset$origin == 3] = 1

mgp_dataset$origin <- NULL

mgp_dataset$o1 <- o1
mgp_dataset$o2 <- o2
mgp_dataset$o3 <- o3





#add the outcome
mgp_dataset$outcome <- mgp$data$targets$mpg
usethis::use_data(mgp_dataset, overwrite = TRUE)