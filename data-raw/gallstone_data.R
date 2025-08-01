library(readxl)
if (!require("pacman")) install.packages("pacman") 
pacman::p_load(magrittr, dplyr, usethis, data.table, here)
Gallstone_Dataset <- read_excel(here::here("data-raw","gallstone_data.xlsx"))
Gallstone_Dataset<-as.data.frame(Gallstone_Dataset)



N <- dim(Gallstone_Dataset)[1]


#need to process as bit 
y<- Gallstone_Dataset$`Gallstone Status`
Gallstone_Dataset <- Gallstone_Dataset[,-1] #remove y from the first column

hist(Gallstone_Dataset$Age)
age_bins_custom <- cut(Gallstone_Dataset$Age,
                       breaks = c(0, 40, 55, Inf), # Breakpoints
                       labels = c("Young Adult", "Adult", "Senior"), # Labels for bins
                       right = FALSE)

young_adult <- numeric(N)
adult <- numeric(N)
senior <- numeric(N)
#create dummy vars
young_adult[age_bins_custom == 'Young Adult'] = 1
adult[age_bins_custom == 'Adult'] = 1
senior[age_bins_custom == 'Senior'] = 1

#add these to the end of the data.frame
Gallstone_Dataset$young_adult <- young_adult
Gallstone_Dataset$adult <-adult
Gallstone_Dataset$senior <-senior

#Comorbidity has two 3's and one 2. change these to 1. 
Gallstone_Dataset$Comorbidity[Gallstone_Dataset$Comorbidity == 3] = 1
Gallstone_Dataset$Comorbidity[Gallstone_Dataset$Comorbidity == 2] = 1

#CAD only has 12 non zero values
#Hypothyroidism has 9 non zero
#hyperlipidemia has 8 non zero
#DM has 43 non zero
#combine some or all of the above into one variable? and remove them after?
CAD_HT_HL_DM <- numeric(N)
CAD_HT_HL_DM[Gallstone_Dataset$`Coronary Artery Disease (CAD)` == 1] = 1
CAD_HT_HL_DM[Gallstone_Dataset$Hypothyroidism == 1] = 1 
CAD_HT_HL_DM[Gallstone_Dataset$Hyperlipidemia == 1] = 1 
CAD_HT_HL_DM[Gallstone_Dataset$`Diabetes Mellitus (DM)`==1] = 1 
Gallstone_Dataset$CAD_HT_HL_DM <- CAD_HT_HL_DM

#Obesity % has one outlier, change this to be the mean
Gallstone_Dataset$`Obesity (%)`[Gallstone_Dataset$`Obesity (%)` == 1954] = mean(Gallstone_Dataset$`Obesity (%)`)


#Create dummy variable for CRP

CRP_ZERO <- numeric(N)
CRP_SMALL <- numeric(N)
CRP_LARGE <- numeric(N)
#create dummy vars
CRP_ZERO[Gallstone_Dataset$`C-Reactive Protein (CRP)` == 0] = 1
CRP_SMALL[Gallstone_Dataset$`C-Reactive Protein (CRP)` > 0 & Gallstone_Dataset$`C-Reactive Protein (CRP)` <3] = 1
CRP_LARGE[Gallstone_Dataset$`C-Reactive Protein (CRP)` > 3] = 1

Gallstone_Dataset$CRP_ZERO <- CRP_ZERO
Gallstone_Dataset$CRP_SMALL <- CRP_SMALL
Gallstone_Dataset$CRP_LARGE <- CRP_LARGE
Gallstone_Dataset$`C-Reactive Protein (CRP)` <- NULL

#change HFA to be 0 or 1
HFA_0_1 <- numeric(N)
HFA_0_1[Gallstone_Dataset$`Hepatic Fat Accumulation (HFA)` == 0] = 0
HFA_0_1[Gallstone_Dataset$`Hepatic Fat Accumulation (HFA)` != 0] = 1
Gallstone_Dataset$HFA_0_1 <- HFA_0_1

#remove some variables
Gallstone_Dataset$`Hepatic Fat Accumulation (HFA)` <- NULL
Gallstone_Dataset$`Coronary Artery Disease (CAD)` <- NULL
Gallstone_Dataset$Hypothyroidism <- NULL
Gallstone_Dataset$Hyperlipidemia <- NULL
Gallstone_Dataset$`Diabetes Mellitus (DM)` <- NULL



#do something with viceral fat rating?
#could probably do more than this still


#one hot BMI into regular, overweight and obese
BMI_bins_custom <- cut(Gallstone_Dataset$`Body Mass Index (BMI)`,
                       breaks = c(0, 25, 30, Inf), # Breakpoints
                       labels = c("Regular", "Overweight", "Obese"), # Labels for bins
                       right = FALSE)

Regular_weight <- numeric(N)
Overweight <- numeric(N)
Obese <- numeric(N)
#create dummy vars
Regular_weight[BMI_bins_custom == 'Regular'] = 1
Overweight[BMI_bins_custom == 'Overweight'] = 1
Obese[BMI_bins_custom == 'Obese'] = 1

Gallstone_Dataset$Regular_weight <- Regular_weight
Gallstone_Dataset$Overweight <-Overweight
Gallstone_Dataset$Obese <-Obese
#remove BMI
Gallstone_Dataset$`Body Mass Index (BMI)` <- NULL
#add the outcome back


#bin vitamin levels

VitD_deficient <- numeric(N)
VitD_low <- numeric(N)
VitD_sufficient <- numeric(N)
VitD_optimal <- numeric(N)
VitD_deficient[Gallstone_Dataset$`Vitamin D` < 12] = 1
VitD_low[Gallstone_Dataset$`Vitamin D` > 12 & Gallstone_Dataset$`Vitamin D` < 20] = 1
VitD_sufficient[Gallstone_Dataset$`Vitamin D` > 20 & Gallstone_Dataset$`Vitamin D` < 30] = 1
VitD_optimal[Gallstone_Dataset$`Vitamin D` > 30] = 1

Gallstone_Dataset$VitD_deficient <- VitD_deficient
Gallstone_Dataset$VitD_low <- VitD_low
Gallstone_Dataset$VitD_sufficient <- VitD_sufficient
Gallstone_Dataset$VitD_optimal <- VitD_optimal

Gallstone_Dataset$`Vitamin D` <- NULL



#remove some features based on the article
Gallstone_Dataset$Age <- NULL
Gallstone_Dataset$`Alanin Aminotransferaz (ALT)` <- NULL
Gallstone_Dataset$`Glomerular Filtration Rate (GFR)` <- NULL
Gallstone_Dataset$`Total Cholesterol (TC)` <- NULL
Gallstone_Dataset$`Glomerular Filtration Rate (GFR)` <- NULL
Gallstone_Dataset$Glucose <- NULL
Gallstone_Dataset$outcome <- y
usethis::use_data(Gallstone_Dataset, overwrite = TRUE)







