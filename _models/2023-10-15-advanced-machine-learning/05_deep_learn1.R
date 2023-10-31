#######################################
## deep learning with the dataset of iris
## https://rpubs.com/Nilafhiosagam/541333
#######################################

#################################
## Load packages and iris data
#################################

rm(list = ls())
# install.packages("reticulate")
# reticulate::py_config()
# reticulate::py_install(c('tensorflow-gpu==2.2','tensorflow-addons'), pip = TRUE)
library(keras)
#install_keras()
# library(tensorflow)
# install_tensorflow()

library(tidyverse)
library(datasets)
data(iris)

################################################################
## keras working with a matrix, which elements are the same type
## the targets are factors, the rest is numeric. Use as.numeric()
## to convert the data to numbers
################################################################

# Transfer the target into numbers, and all -1 for 0, 1, and 2
iris[,5] <- as.numeric(iris[,5]) -1

# Turn `iris` into a matrix
iris <- as.matrix(iris)

# Set iris `dimnames` to `NULL`, that is, just a matrix
dimnames(iris) <- NULL

## split data into training and test sets

set.seed(1234)
# Sample size "2" means two sets of 1 and 2, sample with replacement
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.7, 0.3))

# Split the `iris` data
xtrain <- iris[ind==1, 1:4]
xtest <- iris[ind==2, 1:4]

# Split the class attribute
ytrain <- iris[ind==1, 5]
ytest <- iris[ind==2, 5]

#######################################################
## For multi-class classification  with neural networks
## transform target from vector value (0,1,2) to matrix 
## with a boolean for each class
########################################################

# One hot encode training target values
trainLabels <- to_categorical(ytrain)

# One hot encode test target values
testLabels <- to_categorical(ytest)

# Print out the iris.testLabels to double check the result
print(testLabels)

########################################################
## Constructing Model:  initializing a sequential model
## that is fully connected, with an activation functions
########################################################

# Initialize a sequential model
model <- keras_model_sequential() 

# Add layers to the model: 4 columns
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')

# Print a summary of a model
summary(model)

#########################################################
## with two arguments of optimizer and loss, compiling and 
## fitting the model to training data
########################################################

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy', # binary classes use binary_crossentropy
  optimizer = 'adam', # can use Stochastic Gradient Descent (SGD) and RMSprop
  metrics = 'accuracy' #  for a regression use MSE.
)

# Store the fitting history in `history` 
history <- model %>% fit(
  xtrain, 
  trainLabels, 
  epochs = 200,
  batch_size = 5, 
  validation_split = 0.2
)

# Plot the history
plot(history)

###########################################################
## Visualizing the training history to make two separate
## one for the model loss and another for its model accuracy
##########################################################
# Plot the model loss of the training data
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", 
     ylab="loss", col="blue", type="l")

# Plot the model loss of the validation data
lines(history$metrics$val_loss, col="green")

# Add legend
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the accuracy of the training data 
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")

# Plot the accuracy of the validation data
lines(history$metrics$val_acc, col="green")

# Add Legend
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

###########################################################
## using the model to predict the labels for the test set
###########################################################

# Predict the classes for the test data
# for multi-class classification with softmax
# model %>% predict(x) %>% k_argmax()
# for binary classification with sigmoid
# model %>% predict(x) %>% >(0.5) %>% k_cast("int32")

classes <- model %>% 
  predict(xtest, batch_size = 28) %>% 
  k_argmax() # get the max possibility of a specific class
# Confusion matrix
table(ytest, as.vector(classes))

# Evaluate on test data and labels
score <- model %>% evaluate(xtest, testLabels, batch_size = 128)

# Print the score
print(score)

####################################################################
## Fine-tuning model: adjust the number of layers and hidden units
## as well as the number of epochs or the batch size
####################################################################

# Initialize the sequential model
model2 <- keras_model_sequential() 

# Adding layers to model
model2 %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 5, activation = 'relu') %>% 
  layer_dense(units = 3, activation = 'softmax')

model2 %>% compile(# Compile the model
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

model2 %>% fit(# Fit the model to the data
  xtrain, trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

score2 <- model2 %>% 
  evaluate(xtest, testLabels, batch_size = 128)# Evaluate the model
print(score2)# Print the score

# Adding hidden units to model

model3 <- keras_model_sequential() # Initialize a sequential model

model3 %>% # Add layers to model
  layer_dense(units = 28, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')

model3 %>% compile(# Compile the model
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

model3 %>% fit(# Fit the model to the data
  xtrain, trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

score3 <- model3 %>% 
  evaluate(xtest, testLabels, batch_size = 128)# Evaluate the model

print(score3)# Print the score

####################################################################
## Saving, Loading or Exporting the best Model
####################################################################
# save and reload the model
save_model_hdf5(model, "results/my_model.h5")
model <- load_model_hdf5("results/my_model.h5")

