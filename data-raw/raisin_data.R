library(readxl)
if (!require("pacman")) install.packages("pacman")
pacman::p_load(magrittr, dplyr, usethis, data.table, here)
raisin_dataset <- read_excel(here::here("data-raw", "Raisin_Dataset.xlsx"))
raisin_dataset <- as.data.frame(raisin_dataset)
raisin_dataset$Class[raisin_dataset$Class == "Kecimen"] <- 1
raisin_dataset$Class[raisin_dataset$Class == "Besni"] <- 0
usethis::use_data(raisin_dataset, overwrite = TRUE)
