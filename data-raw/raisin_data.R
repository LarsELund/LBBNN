library(readxl)
if (!require("pacman")) install.packages("pacman") 
pacman::p_load(magrittr, dplyr, usethis, data.table, here)
Raisin_Dataset <- read_excel(here::here("data-raw","Raisin_Dataset.xlsx"))
Raisin_Dataset<-as.data.frame(Raisin_Dataset)
Raisin_Dataset$Class[Raisin_Dataset$Class == 'Kecimen'] = 1
Raisin_Dataset$Class[Raisin_Dataset$Class == 'Besni'] = 0
usethis::use_data(Raisin_Dataset, overwrite = TRUE)
