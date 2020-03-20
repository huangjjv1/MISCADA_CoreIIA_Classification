# Write code here to download the data you are using for your report.
# DO NOT push the data to your Github repository.

# For example, to download the simple Iris data from the UCI Machine Learning
# Repository
adult <- data.table(read.csv("./adult.data"))
adult_test <- data.table(read.csv("./adult.test"))


library("dplyr")
library("forcats")
library("data.table")
library("mlr3verse")
library("rsample")
library(pROC)
library("skimr")
library(caret)
library(psych)
library("recipes")
library("keras")
library("randomForest")
library(dplyr)
library(class)

# update the class labels to be shorter
names(adult)<- c("age","workclass","fnlwgt","education","education_num","marital_status",
                 "occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week",
                 "native_country" ,"salary")
names(adult_test)<- c("age","workclass","fnlwgt","education","education_num","marital_status",
                      "occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week",
                      "native_country" ,"salary")


# Remove useless and repeated information
adult <- select(adult,-fnlwgt,-education)
adult_test <- select(adult_test,-fnlwgt,-education)


# Save into Data directory which is not pushed to Github
saveRDS(adult, "./Data/adult.rds")
saveRDS(adult_test, "./Data/adult_test.rds")

