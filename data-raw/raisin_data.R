library(readxl)
if (!require("pacman")) install.packages("pacman") 
pacman::p_load(magrittr, dplyr, usethis, data.table, here)
Raisin_Dataset <- read_excel(here::here("data-raw","Raisin_Dataset.xlsx"))
Raisin_Dataset<-as.data.frame(Raisin_Dataset)
usethis::use_data(Raisin_Dataset, overwrite = TRUE)
