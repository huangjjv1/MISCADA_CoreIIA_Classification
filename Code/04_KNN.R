# Load the data

source("03_load_data.R")

#KNN part
```{r}
# data splitting
# adult$salary<-ifelse(adult$salary==" <=50K",0,1)
# adult_test$salary<-ifelse(adult_test$salary==" <=50K",0,1)
adult_knn = adult
adult_test_knn = adult_test
adult_knn[,label := rep("train",nrow(adult_knn))]
adult_test_knn[,label := rep("test",nrow(adult_test_knn))]
adult_total_data <- rbind(adult_knn,adult_test_knn,fill=  TRUE)
str(adult_total_data)

library(dplyr)
library(caret)
adult_total_data_part <- select(adult_total_data,workclass,marital_status,occupation,relationship,race,sex,native_country)
dummies <- dummyVars(~., data = adult_total_data_part, levelsOnly = FALSE, fullRank = TRUE)
adult_total_data_unordered <- predict(dummies, newdata = adult_total_data_part) %>% as.data.frame()

adult_total_data_part2 <- select(adult_total_data,age,education_num,capital_gain,capital_loss,hours_per_week)
adult_total_data_numerical <- preProcess(adult_total_data_part2,method = "scale") %>% predict(.,adult_total_data_part2)

library(stringr)
adult_total_data_part3 <- select(adult_total_data,salary,label)
adult_total_data_part3[,salary := gsub("[\\. ]","",salary)]
adult_total_data_target <-adult_total_data_part3[,salary := factor(salary,levels = c("<=50K",">50K"),labels = c(0,1))]


adult_total_data_final <- cbind(adult_total_data_target,adult_total_data_unordered,adult_total_data_numerical)
adult_train_knn <- adult_total_data_final[label == "train",] %>% select(.,-label)
adult_test_knn <- adult_total_data_final[label == "test",] %>% select(.,-label)
library(class)
library(pROC)

find_k_rows= 10000
#firstly use 10 to begin with
adult_pre  <- knn(train = adult_train_knn[1:find_k_rows,-1],test = adult_test_knn[1:find_k_rows,-1],cl = adult_train_knn$salary[1:find_k_rows],k= 1)
auc <- auc(as.numeric(adult_test_knn[1:find_k_rows]$salary),as.numeric(adult_pre))
auc
# to find the best k value here
temp = 0

for (i in 1:20){
  adult_pre  <- knn(train = adult_train_knn[1:find_k_rows,-1],test = adult_test_knn[1:find_k_rows,-1],cl = adult_train_knn$salary[1:find_k_rows],k= i)
  auc <- auc(as.numeric(adult_test_knn[1:find_k_rows]$salary),as.numeric(adult_pre))
  if(auc >temp) {temp = auc;temp_k = i}
}
temp
temp_k
adult_pre  <- knn(train = adult_train_knn[1:find_k_rows,-1],test = adult_test_knn[1:find_k_rows,-1],cl = adult_train_knn$salary[1:find_k_rows],k=temp_k)
plot(roc(as.numeric(adult_test_knn$salary[1:find_k_rows]),as.numeric(adult_pre)))
auc_knn <- auc(as.numeric(adult_test_knn$salary[1:find_k_rows]),as.numeric(adult_pre))
print(paste("The best k is",temp_k,"and the auc value is:",temp))
