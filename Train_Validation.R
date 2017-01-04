######################################################
# Train_Validaztion.R: Train a model with validation
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

# Make train and validation data with a ratio of 80% to 20%
set.seed(400) # for 95:5 ratio of train2 to validation
trainIndex <- sample(1:nrow(data.preprocessed.train), 0.95 * nrow(data.preprocessed.train)) 

# The train data is divided into a subset of it as a train data named train2 and a validation data.
data.preprocessed.train2 <- data.preprocessed.train[trainIndex, ]
data.preprocessed.validation <- data.preprocessed.train[-trainIndex, ]

# Initialize h2o cluster
library(h2o)
cl <- h2o.init(
  max_mem_size = "10G",
  nthreads = 6)

# Insert train data into the cluster
h2odnn.train <- as.h2o(
  data.preprocessed.train2,
  destination_frame = "h2omsdtrain")

# Insert validation data into the cluster
h2odnn.validation <- as.h2o(
  data.preprocessed.validation,
  destination_frame = "h2omsdvalidation")

#features.index.oneHotEconding <- 2:1154
# Excluding the id column, 1st and the loss column, 1192th
features.index.oneHotEconding.full <- 2:1191

## create parameters to try
hyperparams <- list(
  list(
    hidden = c(500,500),
    epoch = c(1), # For a test, set epoch to 1. 100 has been a proper value 
    learningRate = c(0.0001),
    rate_annealing = c(1e-7),    
    hidden_dr = c(0.2,0.2),
    score_t_sample =c(10000),
    l1_r = c(1e-4),
    l2_r = c(0.0005))
)

duration <- system.time(
    fm <- lapply(hyperparams, function(v) {
      h2o.deeplearning(
        x = colnames(data.preprocessed.train2)[features.index.oneHotEconding.full],
        y = "loss",
        training_frame= h2odnn.train,
        validation_frame = h2odnn.validation,
        activation = "RectifierWithDropout",
        hidden = v$hidden,
        epochs = v$epoch,
        adaptive_rate = FALSE,
        rate = v$learningRate,
        rate_annealing = v$rate_annealing,
        sparsity_beta = 0,
        input_dropout_ratio = 0,
        hidden_dropout_ratios = v$hidden_dr,
        score_training_samples = v$score_t_sample,                
        stopping_rounds = 5,
        stopping_metric = "deviance",
        stopping_tolerance = c(0),
        l1 = v$l1_r,
        l2 = v$l2_r
      )
    }
  )
) # End of system.time

print(duration)

print(fm[1])

