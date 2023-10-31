#################################################################
## building a regression model using deep learning algorithm
## https://tensorflow.rstudio.com/tutorials/keras/regression
#################################################################

#################################################################
## loading packages and data, and cleaning the data for analysis
#################################################################
# loading the the four packages
rm(list = ls())
library(tensorflow)
library(keras)
library(tidyverse)
library(tidymodels)

# loading the data from a URL
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
col_names <- c("mpg","cylinders","displacement","horsepower","weight",
               "acceleration","model_year", "origin","car_name")
raw_dataset <- read.table(
  url,
  header = T,
  col.names = col_names,
  na.strings = "?"
)

dataset <- raw_dataset %>% select(-car_name)

# cleaning the data
lapply(dataset, function(x) sum(is.na(x))) %>% str() # Drop those rows with NA
dataset <- na.omit(dataset)

dataset <- recipe(mpg ~ ., dataset) %>%  # one-hot encode for the original column 
  step_num2factor(origin, levels = c("USA", "Europe", "Japan")) %>%
  step_dummy(origin, one_hot = TRUE) %>% # using step_dummy(one_hot = FALSE) for binary
  prep() %>% # update the recipe
  bake(new_data = NULL) # view preProcessing data 

glimpse(dataset)

#################################################################
## splitting the data into training and test sets and separating
## features from labels
#################################################################
# using train set for building model and test set for evaluation of model
split <- initial_split(dataset, 0.8)
train_dataset <- training(split)
test_dataset <- testing(split) 

# Reviewing the distribution of a few pairs of columns from the training set
train_dataset %>%
  select(mpg, cylinders, displacement, weight) %>%
  GGally::ggpairs() # help identifying the correlations among variables

# Note each feature covers a very different range
skimr::skim(train_dataset) # an alternative to summary()

# Split features from labels and normalizing feature values of train set

train_features <- train_dataset %>% select(-mpg)
test_features <- test_dataset %>% select(-mpg)

train_labels <- train_dataset %>% select(mpg)
test_labels <- test_dataset %>% select(mpg)

# see the different of each feature range and possibly skipping the step
# my_skim <- skimr::skim_with(numeric = skimr::sfl(mean, sd)) # self-defining summary
# train_dataset %>%
#   select(where(~is.numeric(.x))) %>%
#   pivot_longer(
#     cols = everything(), names_to = "variable", values_to = "values") %>%
#   group_by(variable) %>%
#   summarise(mean = mean(values), sd = sd(values))

normalizer <- layer_normalization(axis = -1L) # feature values transferred to the range [0,1]
normalizer %>% adapt(as.matrix(train_features))
print(normalizer$mean)

# first <- as.matrix(train_features[1,])
# cat('First example:', first)
# cat('Normalized:', as.matrix(normalizer(first)))

###########################################################
## the linear regression between 'mpg' and 'horsepower'
###########################################################
# Normalizing the 'horsepower' input feature 
horsepower <- matrix(train_features$horsepower) # creating a matrix of the feature
horsepower_normalizer <- layer_normalization(input_shape = shape(1), axis = NULL)
horsepower_normalizer %>% adapt(horsepower)

# Constructing a linear regression Model for 1 output
horsepower_model <- keras_model_sequential() %>%
  horsepower_normalizer() %>%
  layer_dense(units = 1)

# summary(horsepower_model)
# Using the untrained model on the first 10 ‘horsepower’ values
# predict(horsepower_model, horsepower[1:10,])

# configuring the training procedure with keras's compile()
horsepower_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.1),
  loss = 'mean_absolute_error' # to be optimized
)

# executing to train a model for 100 epochs
history <- horsepower_model %>% fit(
  as.matrix(train_features$horsepower),
  as.matrix(train_labels),
  epochs = 100,
  verbose = 0, # Suppress logging
  validation_split = 0.2 # validation on 20% of the training data
)

# Visualizing the model’s training progress 
plot(history) 

# Collecting the results on the test set
test_results <- list()
test_results[["horsepower_model"]] <- horsepower_model %>% evaluate(
  as.matrix(test_features$horsepower),
  as.matrix(test_labels),
  verbose = 0
)
test_results
###############################################################
## the linear regression with multiple inputs
###############################################################
# Constructing a linear regression Model
linear_model <- keras_model_sequential() %>%
  normalizer() %>%
  layer_dense(units = 1)

# summary(horsepower_model)
# Using the untrained model on the first 10 ‘horsepower’ values              
# predict(linear_model, as.matrix(train_features[1:10, ]))
# linear_model$layers[[2]]$kernel

# configuring the training procedure
linear_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.1),
  loss = 'mean_absolute_error'
)

# executing to train a model for 100 epochs
history <- linear_model %>% fit(
  as.matrix(train_features),
  as.matrix(train_labels),
  epochs = 100,
  verbose = 0,
  validation_split = 0.2
)
# Visualizing the model’s training progress 
plot(history) 

# Collecting the results on the test set
test_results[['linear_model']] <- linear_model %>%
  evaluate(
    as.matrix(test_features),
    as.matrix(test_labels),
    verbose = 0
  )
#############################################################
## a deep neural network (DNN) regression with a single input
#############################################################
# Normalizing the 'horsepower' input feature 
horsepower <- matrix(train_features$horsepower) # creating a matrix of the feature
horsepower_normalizer <- layer_normalization(input_shape = shape(1), axis = NULL)
horsepower_normalizer %>% adapt(horsepower)

# Constructing a linear regression Model for 1 output
dnn_horsepower_model <- keras_model_sequential() %>%
  horsepower_normalizer() %>%
  layer_dense(64, activation = 'relu') %>%
  layer_dense(64, activation = 'relu') %>%
  layer_dense(1)

# configuring the training procedure with keras's compile()
dnn_horsepower_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.1),
  loss = 'mean_absolute_error' # to be optimized
)

# executing to train a model for 100 epochs
history <- dnn_horsepower_model %>% fit(
  as.matrix(train_features$horsepower),
  as.matrix(train_labels),
  epochs = 100,
  verbose = 0, 
  validation_split = 0.2 
)

# Visualizing the model’s training progress 
plot(history) 

# Collecting the results on the test set
test_results[["dnn_horsepower_model"]] <- dnn_horsepower_model %>% evaluate(
  as.matrix(test_features$horsepower),
  as.matrix(test_labels),
  verbose = 0
)


###############################################################
## a deep neural network (DNN) regression with multiple inputs
###############################################################
# Normalizing the input features 
# normalizer <- layer_normalization(axis = -1L) # feature values transferred to the range [0,1]
# normalizer %>% adapt(as.matrix(train_features))

# Constructing a linear regression Model
dnn_model <- keras_model_sequential() %>%
  normalizer() %>%
  layer_dense(64, activation = 'relu') %>%
  layer_dense(64, activation = 'relu') %>%
  layer_dense(1)

# configuring the training procedure
dnn_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.1),
  loss = 'mean_absolute_error'
)

# executing to train a model for 100 epochs
history <- dnn_model %>% fit(
  as.matrix(train_features),
  as.matrix(train_labels),
  epochs = 100,
  verbose = 0,
  validation_split = 0.2
)
# Visualizing the model’s training progress 
plot(history) 

# Collecting the results on the test set
test_results[['dnn_model']] <- dnn_model %>%
  evaluate(
    as.matrix(test_features),
    as.matrix(test_labels),
    verbose = 0
  )

sapply(test_results, function(x) x)
##################################################
## Make predictions
#################################################
# make prediction on the test set
test_predictions <- predict(dnn_model, as.matrix(test_features))
ggplot(data.frame(pred = as.numeric(test_predictions), mpg = test_labels$mpg)) +
  geom_point(aes(x = pred, y = mpg)) +
  geom_abline(intercept = 0, slope = 1, color = "blue")

# check the error distribution
qplot(test_predictions - test_labels$mpg, geom = "density")
error <- test_predictions - test_labels

##############################################################
## saving the model and reloading it
##############################################################
save_model_tf(dnn_model, 'results/dnn_model1.h5')
reloaded <- load_model_tf('results/dnn_model1')

test_results[['reloaded']] <- reloaded %>% evaluate(
  as.matrix(test_features),
  as.matrix(test_labels),
  verbose = 0
)



