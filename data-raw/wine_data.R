library(ucimlrepo)
wine_quality <- fetch_ucirepo(id=186) 

Wine_quality_dataset <- as.data.frame(wine_quality$data$features)
Wine_quality_dataset$outcome <- wine_quality$data$targets$quality
usethis::use_data(Wine_quality_dataset, overwrite = TRUE)