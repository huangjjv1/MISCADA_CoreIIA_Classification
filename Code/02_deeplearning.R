# Load the data

source("03_load_data.R")

# DeepLearning part
# obtaining set

# adult$salary<-ifelse(adult$salary==" <=50K",0,1)
# adult_test$salary<-ifelse(adult_test$salary==" <=50K."," <=50K"," >50K")
library("data.table")
library("mlr3verse")
library("rsample")
# First get the validate
adult_split <- initial_split(adult)
adult_split2 <- initial_split(testing(adult_split), 0.5)
adult_validate <- training(adult_split2)

library("recipes")

cake <- recipe(salary ~ ., data = adult) %>%
  step_meanimpute(all_numeric()) %>% # impute missings on numeric values with the mean
  step_center(all_numeric()) %>% # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) %>% # scale by dividing by the standard deviation on all numeric features
  step_unknown(all_nominal(), -all_outcomes()) %>% # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  step_dummy(all_nominal(), one_hot = TRUE) %>% # turn all factors into a one-hot coding
  prep(training = adult) # learn all the parameters of preprocessing on the training data

adult_train_final <- bake(cake, new_data = adult) # apply preprocessing to training data
adult_validate_final <- bake(cake, new_data = adult_validate) # apply preprocessing to validation data
adult_test_final <- bake(cake, new_data = adult_test) # apply preprocessing to testing data

library("keras")
adult_train_x <- adult_train_final %>%
  select(-starts_with("salary_")) %>%
  as.matrix()
adult_train_y <- adult_train_final %>%
  select("salary_X..50K") %>%
  as.matrix()

adult_test_x <- adult_test_final %>%
  select(-starts_with("salary_")) %>%
  as.matrix()
adult_test_y <- adult_test_final %>%
  select("salary_X..50K") %>%
  as.matrix()

adult_validate_x <- adult_validate_final %>%
  select(-starts_with("salary_")) %>%
  as.matrix()
adult_validate_y <- adult_validate_final %>%
  select("salary_X..50K") %>%
  as.matrix()

head(adult_test_y,10)

deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(adult_train_x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%

  layer_dense(units = 1, activation = "sigmoid")

deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net %>% fit(
  adult_train_x, adult_train_y,
  epochs = 50, batch_size = 64,
  validation_data = list(adult_validate_x, adult_validate_y),
)
# To get the probability predictions on the test set:
pred_test_prob <- deep.net %>% predict_proba(adult_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net %>% predict_classes(adult_test_x)

auc_dnn <- auc(as.numeric(adult_test$salary),as.numeric(pred_test_res))
auc_dnn
table(pred_test_res, adult_test_y)
yardstick::accuracy_vec(as.factor(adult_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(as.factor(adult_test_y),
                       c(pred_test_prob))
