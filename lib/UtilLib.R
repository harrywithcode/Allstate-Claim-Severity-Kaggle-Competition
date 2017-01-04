######################################################
# UtilLib.R: Include utility functions
# Date: Nov 27, 2016
# Developer: Gon-soo Moon
######################################################

# Return the indices having greater than some value, threshold value
GetOutlier <- function (data.y, threshold.number)
{
  outlier.indices <- which(data.y > threshold.number) 
  return(outlier.indices)
}

# outlier.indices <- GetOutlier(loss.label,30000)
# nrow(data.frame(outlier.indices)) # 63
# 
# outlier.indices <- GetOutlier(loss.label,40000)
# nrow(data.frame(outlier.indices)) # 23
# 
# outlier.indices <- GetOutlier(loss.label,60000)
# nrow(data.frame(outlier.indices)) # 6


# Get one hot encoding matrix. In other words, for categorical values, get binary values
GetOneHotEncoding <- function(col.factor, data.X)
{
  # initialize to 0, afterward, that was excluded when returning it
  oneHotEncoding <- 0
  for (col in col.factor)
  {
    formula.factor <- paste0("~",col," -1")
    formula.object <- as.formula(formula.factor) 
    col.sparse <- data.frame(model.matrix(formula.object,data.X[,col.factor]))
    oneHotEncoding <- cbind(oneHotEncoding, col.sparse)
  }
  # exclude 1st column
  return(oneHotEncoding[,-1])
}


# Convert factor to numeric
changeFactorToNumeric <- function(columns, data.df)
{
  for (col in columns)
  {
    data.df[,col] <- as.numeric(data.df[,col])
  }
  return(data.df)
}


# return column names having factor type
getType <- function (pType, columns, data.df)
{
  result <-0
  i <- 1
  for (col in columns) 
  {
    type <- class(data.df[,col]) 
    if (type == pType)
    {
      result[i] <- col
      i <- i + 1
    }
  }
  return(result)  
}

# Compute accuracy with confusion matrix
accuracy <- function(result.df)
{
  
  result <- (result.df[1,1] + result.df[2,2]) / sum(result.df)
  return(result)
}


# Put multiple binary output to categorical output
maxidx <- function(arr) {
  return(which(arr == max(arr)))
}

# Return data with column name from the path fed as an argument
getDataFromSource <- function(fileName.s)
{
  input.df <- read.table(fileName.s, header=TRUE, sep="\t",strip.white = TRUE)
  return(input.df)
}


