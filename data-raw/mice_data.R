Mice_Dataset <- read.csv(here::here("data-raw","mice_data.csv"))
Mice_Dataset<-as.data.frame(Mice_Dataset)
usethis::use_data(Mice_Dataset, overwrite = TRUE)