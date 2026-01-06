library(readxl)
if (!require("pacman")) install.packages("pacman")
pacman::p_load(magrittr, dplyr, usethis, data.table, here)
gallstone_dataset <- read_excel(here::here("data-raw", "gallstone_data.xlsx"))
gallstone_dataset <- as.data.frame(gallstone_dataset)



n_rows <- dim(gallstone_dataset)[1]

#need to process as bit
y <- gallstone_dataset$`Gallstone Status`
gallstone_dataset <- gallstone_dataset[, -1] #remove y from the first column

hist(gallstone_dataset$Age)
age_bins_custom <- cut(gallstone_dataset$Age,
                       breaks = c(0, 40, 55, Inf), # Breakpoints
                       labels = c("Young Adult", "Adult", "Senior"),
                       right = FALSE)

young_adult <- numeric(N)
adult <- numeric(N)
senior <- numeric(N)
#create dummy vars
young_adult[age_bins_custom == "Young Adult"] <- 1
adult[age_bins_custom == "Adult"] <- 1
senior[age_bins_custom == "Senior"] <- 1

#add these to the end of the data.frame
gallstone_dataset$young_adult <- young_adult
gallstone_dataset$adult <- adult
gallstone_dataset$senior <- senior

#Comorbidity has two 3's and one 2. change these to 1.
gallstone_dataset$Comorbidity[gallstone_dataset$Comorbidity == 3] <- 1
gallstone_dataset$Comorbidity[gallstone_dataset$Comorbidity == 2] <- 1

#CAD only has 12 non zero values
#Hypothyroidism has 9 non zero
#hyperlipidemia has 8 non zero
#DM has 43 non zero
#combine some or all of the above into one variable? and remove them after?
cad_ht_hl_dm <- numeric(n_rows)
cad_ht_hl_dm[gallstone_dataset$`Coronary Artery Disease (CAD)` == 1] <- 1
cad_ht_hl_dm[gallstone_dataset$Hypothyroidism == 1] <- 1
cad_ht_hl_dm[gallstone_dataset$Hyperlipidemia == 1] <- 1
cad_ht_hl_dm[gallstone_dataset$`Diabetes Mellitus (DM)` == 1] <- 1
gallstone_dataset$CAD_HT_HL_DM <- cad_ht_hl_dm

#Obesity % has one outlier, change this to be the mean
gallstone_dataset$`Obesity (%)`[gallstone_dataset$`Obesity (%)` == 1954
] <- mean(gallstone_dataset$`Obesity (%)`)


#Create dummy variable for CRP

crp_zero <- numeric(n_rows)
crp_small <- numeric(n_rows)
crp_large <- numeric(n_rows)
#create dummy vars
crp_zero[gallstone_dataset$`C-Reactive Protein (CRP)` == 0] <- 1
crp_small[gallstone_dataset$`C-Reactive Protein (CRP)` > 0 &
            gallstone_dataset$`C-Reactive Protein (CRP)` < 3] <- 1
crp_large[gallstone_dataset$`C-Reactive Protein (CRP)` > 3] <- 1

gallstone_dataset$CRP_ZERO <- crp_zero
gallstone_dataset$CRP_SMALL <- crp_small
gallstone_dataset$CRP_LARGE <- crp_large
gallstone_dataset$`C-Reactive Protein (CRP)` <- NULL

#change HFA to be 0 or 1
hfa_0_1 <- numeric(n_rows)
hfa_0_1[gallstone_dataset$`Hepatic Fat Accumulation (HFA)` == 0] <- 0
hfa_0_1[gallstone_dataset$`Hepatic Fat Accumulation (HFA)` != 0] <- 1
gallstone_dataset$HFA_0_1 <- hfa_0_1

#remove some variables
gallstone_dataset$`Hepatic Fat Accumulation (HFA)` <- NULL
gallstone_dataset$`Coronary Artery Disease (CAD)` <- NULL
gallstone_dataset$Hypothyroidism <- NULL
gallstone_dataset$Hyperlipidemia <- NULL
gallstone_dataset$`Diabetes Mellitus (DM)` <- NULL



#do something with viceral fat rating?
#could probably do more than this still


#one hot BMI into regular, overweight and obese
bmi_bins_custom <- cut(gallstone_dataset$`Body Mass Index (BMI)`,
                       breaks = c(0, 25, 30, Inf), # Breakpoints
                       labels = c("Regular", "Overweight", "Obese"),
                       right = FALSE)

regular_weight <- numeric(n_rows)
overweight <- numeric(n_rows)
obese <- numeric(n_rows)
#create dummy vars
regular_weight[bmi_bins_custom == "Regular"] <- 1
overweight[bmi_bins_custom == "Overweight"] <- 1
obese[bmi_bins_custom == "Obese"] <- 1

gallstone_dataset$Regular_weight <- regular_weight
gallstone_dataset$Overweight <- overweight
gallstone_dataset$Obese <- obese
#remove BMI
gallstone_dataset$`Body Mass Index (BMI)` <- NULL
#add the outcome back


#bin vitamin levels

vitd_deficient <- numeric(n_rows)
vitd_low <- numeric(n_rows)
vitd_sufficient <- numeric(n_rows)
vitd_optimal <- numeric(n_rows)
vitd_deficient[gallstone_dataset$`Vitamin D` < 12] <- 1
vitd_low[gallstone_dataset$`Vitamin D` > 12 &
           gallstone_dataset$`Vitamin D` < 20] <- 1
vitd_sufficient[gallstone_dataset$`Vitamin D` > 20 &
                  gallstone_dataset$`Vitamin D` < 30] <- 1
vitd_optimal[gallstone_dataset$`Vitamin D` > 30] <- 1

gallstone_dataset$VitD_deficient <- vitd_deficient
gallstone_dataset$VitD_low <- vitd_low
gallstone_dataset$VitD_sufficient <- vitd_sufficient
gallstone_dataset$VitD_optimal <- vitd_optimal

gallstone_dataset$`Vitamin D` <- NULL



#remove some features based on the article
gallstone_dataset$Age <- NULL
gallstone_dataset$`Alanin Aminotransferaz (ALT)` <- NULL
gallstone_dataset$`Glomerular Filtration Rate (GFR)` <- NULL
gallstone_dataset$`Total Cholesterol (TC)` <- NULL
gallstone_dataset$`Glomerular Filtration Rate (GFR)` <- NULL
gallstone_dataset$Glucose <- NULL
gallstone_dataset$outcome <- y
usethis::use_data(gallstone_dataset, overwrite = TRUE)
