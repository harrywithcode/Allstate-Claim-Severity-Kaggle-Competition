## uncomment to install the checkpoint package
#install.packages("checkpoint")

#library(checkpoint)

# Adjust the follwing R.version to your R version
#checkpoint("2016-10-22", R.version = "3.3.0")

if(!"h2o" %in% rownames(installed.packages())) 
{install.packages("h2o",dependencies=TRUE)}
library(h2o)

if(!"Metrics" %in% rownames(installed.packages()))
{install.packages("Metrics",dependencies=TRUE)}
library(Metrics)
