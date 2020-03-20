# Load the data

source("03_load_data.R")

library("recipes")

cake <- recipe(salary ~ ., data = adult) %>%
  step_meanimpute(all_numeric()) %>% # impute missings on numeric values with the mean
  step_center(all_numeric()) %>% # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) %>% # scale by dividing by the standard deviation on all numeric features
  step_unknown(all_nominal(), -all_outcomes()) %>% # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  step_dummy(all_nominal(), one_hot = TRUE) %>% # turn all factors into a one-hot coding
  prep(training = adult) # learn all the parameters of preprocessing on the training data

adult_split <- initial_split(adult)
adult_split2 <- initial_split(testing(adult_split), 0.5)
adult_validate <- training(adult_split2)

adult_train_final <- bake(cake, new_data = adult) # apply preprocessing to training data
adult_validate_final <- bake(cake, new_data = adult_validate) # apply preprocessing to validation data
adult_test_final <- bake(cake, new_data = adult_test) # apply preprocessing to testing data

adult_task <- TaskClassif$new(id = "salary",
                               backend = adult, # <- NB: no na.omit() this time
                               target = "salary",
                               positive = ' >50K')
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(adult_task)
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
res_baseline <- resample(adult_task, lrn_baseline, cv5, store_models = TRUE)
res_cart <- resample(adult_task, lrn_cart, cv5, store_models = TRUE)
res_baseline$aggregate()
res_cart$aggregate()
res <- benchmark(data.table(
  task       = list(adult_task),
  learner    = list(lrn_baseline,
                    lrn_cart),
  resampling = list(cv5)
), store_models = TRUE)
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
trees <- res$resample_result(2)
tree1 <- trees$learners[[1]]
tree1_rpart <- tree1$model
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)
plot(res$resample_result(2)$learners[[5]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[5]]$model, use.n = TRUE, cex = 0.8)
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

res_cart_cv <- resample(adult_task, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[5]]$model)
lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.016)
res <- benchmark(data.table(
  task       = list(adult_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

library("randomForest")
n<-length(names(adult))
rate=c()    #error rate list
print(paste("Times is:",n-1))
# To find the bet mtry
# for(i in 1:(n-1)){
#   rf_train<-randomForest(as.factor(adult$salary)~.,data=adult,mtry=i,ntree=1000,na.action = na.roughfix)
#   rate[i]<-mean(rf_train$err.rate)   #calculate all the error rate based on OOB model
#   # print(rf_train)
# }
# plot(rate)
# mtry best is 3
rf_train<-randomForest(as.factor(adult$salary)~.,data=adult,mtry = 3,ntree=1000 ,na.action = na.roughfix)
# ?randomForest
importance<-randomForest::importance(rf_train,type = 2)
(importance)
hist(treesize(rf_train))
barplot(rf_train$importance[,1],main="Importance of Input vaiables",cex.names = 0.65)
plot(rf_train)
pie(rf_train$importance[,1])

library("data.table")
library("mlr3verse")
library("rsample")
library(pROC)
pred<-predict(rf_train,newdata=adult_test)
pred_out_1<-predict(object=rf_train,newdata=adult_test,type="prob")
table(pred,adult_test$salary)
auc_rf <- auc(as.numeric(adult_test$salary),as.numeric(pred))
auc_rf
sum(diag(table))/sum(table)  #accuracy
