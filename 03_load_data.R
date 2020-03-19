# Write code here to load the data you downloaded in download_data.R
library(data.table)
adult <- data.table(readRDS("./Data/adult.rds"))
adult_test <- data.table(readRDS("./Data/adult.rds"))


# You might choose to do any resampling here to ensure it is consistent across
# models

set.seed(7482) # set seed for reproducibility

library("rsample")
adult.cv <- rsample::vfold_cv(adult, v = 3, strata = salary)
?vfold_cv
