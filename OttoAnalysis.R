library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

trainSet <- vroom("train.csv")
trainSet$target <- as.factor(trainSet$target)
testSet <- vroom("test.csv")

##EDA
ggplot(trainSet, aes(x=feat_1, y=target))+geom_jitter()

my_recipe <- recipe(target~., data=trainSet) %>%
  step_rm(id)
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model) 

## Tune smoothness and Laplace here
tuning_grid <- grid_regular(Laplace(),smoothness(),levels = 5)
## Split data for CV
folds <- vfold_cv(trainSet, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,grid=tuning_grid,metrics=metric_set(mn_log_loss)) 

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("mn_log_loss")

final_wf <-nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainSet)

## Predict
nb_preds <- predict(final_wf, new_data=testSet, type="prob") %>%
  bind_cols(., testSet) %>%
  select(id, .pred_Class_1, .pred_Class_2,.pred_Class_3,.pred_Class_4,.pred_Class_5,
         .pred_Class_6,.pred_Class_7,.pred_Class_8,.pred_Class_9) %>%
  rename(Class_1=.pred_Class_1, Class_2 =.pred_Class_2, Class_3 = .pred_Class_3,
         Class_4=.pred_Class_4, Class_5=.pred_Class_5, Class_6=.pred_Class_6,
         Class_7=.pred_Class_7, Class_8=.pred_Class_8,Class_9=.pred_Class_9) 

vroom_write(x=nb_preds, file="./NB_Preds.csv", delim=",") 


##Boosting
library(bonsai)
library(lightgbm)
my_recipe <- recipe(formula=target~., data=trainSet) %>%
  step_rm(id) %>%
  step_normalize(all_numeric_predictors())
boost_model <- boost_tree(tree_depth=1,
                          trees=2000,
                          learn_rate=.1) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_wf <- workflow()%>%
  add_recipe(my_recipe) %>%
  add_model(boost_model) %>%
  fit(data=trainSet)

boost_tuneGrid <- grid_regular(tree_depth(),trees(),learn_rate(),levels=3)
folds <- vfold_cv(trainSet, v = 5, repeats=1)
tuned_boost <- boost_wf %>%
  tune_grid(resamples=folds,grid=boost_tuneGrid,metrics=metric_set(mn_log_loss))

bestTune <- tuned_boost %>%
  select_best("mn_log_loss")

final_wf <-boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainSet)

## Predict
boost_preds <- predict(boost_wf, new_data=testSet, type="prob") %>%
  bind_cols(., testSet) %>%
  select(id, .pred_Class_1, .pred_Class_2,.pred_Class_3,.pred_Class_4,.pred_Class_5,
         .pred_Class_6,.pred_Class_7,.pred_Class_8,.pred_Class_9) %>%
  rename(Class_1=.pred_Class_1, Class_2 =.pred_Class_2, Class_3 = .pred_Class_3,
         Class_4=.pred_Class_4, Class_5=.pred_Class_5, Class_6=.pred_Class_6,
         Class_7=.pred_Class_7, Class_8=.pred_Class_8,Class_9=.pred_Class_9)

vroom_write(x=boost_preds, file="./Boost_Preds.csv", delim=",")


##Random Forest
rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")


rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod) 
rf_tuning_grid <- grid_regular(mtry(c(1, 90)), min_n(), levels = 3)

folds <- vfold_cv(trainSet, v = 5, repeats=1)

CV_results <- rf_wf %>%
  tune_grid(resamples=folds,grid=rf_tuning_grid,metrics=metric_set(mn_log_loss)) #Or leave metrics NULL


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("mn_log_loss")

final_wf <- rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainSet)

rf_preds <- predict(final_wf, new_data=testSet, type="prob") %>%
  bind_cols(., testSet) %>%
  select(id, .pred_Class_1, .pred_Class_2,.pred_Class_3,.pred_Class_4,.pred_Class_5,
         .pred_Class_6,.pred_Class_7,.pred_Class_8,.pred_Class_9) %>%
  rename(Class_1=.pred_Class_1, Class_2 =.pred_Class_2, Class_3 = .pred_Class_3,
         Class_4=.pred_Class_4, Class_5=.pred_Class_5, Class_6=.pred_Class_6,
         Class_7=.pred_Class_7, Class_8=.pred_Class_8,Class_9=.pred_Class_9)

vroom_write(x=rf_preds, file="./RFPreds.csv", delim=",") 

