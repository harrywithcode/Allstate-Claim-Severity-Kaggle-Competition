######################################################
# Final_Train_Prediction.R: Train a final model and make a prediction
# Date: Nov 27, 2016
# Developer: Gon-soo Moon
######################################################

rm(list=ls()) # remove all on workspace
# Set a working directory to the source folder
setwd("/Users/gonsoomoon/Documents/MA/MachineLearning/class/project/src/")

#  Load libraries
source("checkpoint.R")
source("lib/UtilLib.R")

# Read source data
data.train <- read.csv("data/train.csv")
data.test <- read.csv("data/test.csv")

# Exclude outliers 
outlier.indices <- GetOutlier(data.train[,"loss"],30000) # 63 returned
data.train <- data.train[-outlier.indices,]

# indices of the train data
train.index <- 1:188255

# Make indices for features, X
features.index <- 2:131
y.index <-132 # Y index, loss

# Full data set applied
data.full.X <- rbind(data.train[,features.index], data.test[,features.index])

# Retrieve column names
colnames.st <- colnames(data.full.X)
# Retrieve columns having a type of a factor
col.factor <- getType('factor',colnames.st,data.full.X)
# Retrieve columns having a type of a numeric
col.numeric <- getType('numeric',colnames.st,data.full.X)

# Apply one-hot-encoding to features having a type of a factor
# If a data file exisits on the following location, just load it, otherwise encode.
oneHotEncoding.file <- "data/oneHotEncoding.RDS"
if (file.exists(oneHotEncoding.file))
{
  oneHotEncoding.factor <- readRDS(oneHotEncoding.file)  
} else
{
  oneHotEncoding.factor <- GetOneHotEncoding(col.factor, data.full.X)
  saveRDS(oneHotEncoding.factor, file=oneHotEncoding.file)
}

# Combine the encoded features with numeric features
data.full.X <- cbind(oneHotEncoding.factor, data.full.X[,col.numeric])

# If a data file exisits on the following location, just load it, otherwise normalize(scale).
data.full.Xscaled.file <- "data/data.full.Xscaled.RDS"
if (file.exists(data.full.Xscaled.file))
{
  data.full.Xscaled <- readRDS(data.full.Xscaled.file)  
  print("Data loaded")
} else
{
  # scale only features
  data.full.Xscaled <- scale(data.full.X)
  saveRDS(data.full.Xscaled, file=data.full.Xscaled.file)
  print("Data saved")
}

# For the full scaled data, divide it into a scaled train and test data.
# The differnece between the train data originally loaded from Kaggle 
#  and the scaled train data is only whether a data is encoded and scaled.
# Likewise, a test data is the same.
data.train.Xscaled <- data.full.Xscaled[train.index,]
data.test.Xscaled <- data.full.Xscaled[-train.index,]

# extract id and loss column
id.train <- data.train[,1] # id column
loss.train <- data.train[,132] # loss column

id.test <- data.test[,1] # id column
loss.test <- 0

# combine id, features and loss into a single data set
data.preprocessed.train <- cbind(id=id.train, data.train.Xscaled, loss=loss.train)
data.preprocessed.test <- cbind(id=id.test, data.test.Xscaled, loss=loss.test)

# Initialize h2o cluster
library(h2o)
cl <- h2o.init(
  max_mem_size = "11G",
  nthreads = 6)

# Excluding the id column, 1st and the loss column, 1192th
features.index.oneHotEconding.full <- 2:1191

#==============================================================
# With optimal parameters, train Model on the train data and make a prediction on the test data
#==============================================================

# Insert a train data into the cluster
h2o.dnn.train.final <- as.h2o(
  data.preprocessed.train,
  destination_frame = "h2o.dnn.train.final")

# Insert a test data into the cluster
h2o.dnn.test.final <- as.h2o(
  data.preprocessed.test,
  destination_frame = "h2o.dnn.test.final")

# Train a final model
duration <- system.time(mt.final <- h2o.deeplearning(
  x = colnames(data.preprocessed.train)[features.index.oneHotEconding.full],
  y = "loss",
  training_frame= h2o.dnn.train.final,
  activation = "RectifierWithDropout",
  hidden = c(500,500),
  epochs = 1, # optimum # is 100
  adaptive_rate = FALSE,  
  rate = c(1e-4),
  rate_annealing = c(1e-7), 
  sparsity_beta = 0,  
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(.2,.2),
  score_training_samples = c(10000), # Optimum is 0   
  stopping_rounds = 5,
  stopping_metric = "deviance",
  stopping_tolerance = c(0),
  l1 = c(1e-4),
  l2 = c(0.00005)
  )
)


print(duration)
print(mt.final)


yhat.train.final <- as.data.frame(h2o.predict(mt.final, h2o.dnn.train.final))
confusion.train.final <- cbind(yhat.train.final, as.data.frame(h2o.dnn.train.final[,"loss"]))
print("MAE with a final model on train file")
print(mae(confusion.train.final[,"predict"],confusion.train.final[,"loss"]))


# In order to call mae, you need the package "Metrics" already loaded in the checkpoint.R
# Predict loss value with the train data set

# Predict loss value with the test data set
yhat.test <- as.data.frame(h2o.predict(mt.final, h2o.dnn.test.final))
submission <-cbind(id=id.test,loss=yhat.test)
names(submission) = c("id","loss")
write.csv(submission[,c("id","loss")], file="data/submission/submission1127_1.csv", quote=F, row.names=F)
print("submission file is created")

