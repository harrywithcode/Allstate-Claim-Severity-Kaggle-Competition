# How to run R script 
  Assumption: You should have R environment. 

0. System info tested
  1) R version: 3.3.0
  2) Packages used: H20, Metrics
  3) Memory requirement: You should have at least 7GB because a cluster size is about 6.4 GB after loading all data

1. Please copy all sources(demo.R, Train_Validation.R, Final_Train_Prediction.R, checkpoint.R,
   lib, data, models and screenshot folder given to some place.

2. Run the following command for each purpose
	1) Demonstration on demo.R in the R studio
		- Scenario
			a. Before demo, do pre-processing tasks and feed data into a cluster
			b. Load the best model
			c. Predict 'loss' label values on the train data with the best model and show MAE
			d. Predict 'loss' label values on the test data and make a submission file
			e. Submit it to Kaggle.
		
	2) Run Train_Validation.R on a command line
		- Description: Load, pre-process data and train a model, 
		  then showing results of train and validation error
		- ======= Important Note================
		- At line 9, please change a path of a working directory for yours
		- It takes about 30 minutes on Intel i5 Core
		- On the 119th line, epoch is set to 1 because of a fast running. 
		  If you want, you can change it.
		- command:
					Rscript Train_Validation.R 
		- screenshot
			a. screenshot/Train_Validation/RunningCommand.png shows a running command
			b. screenshot/Train_Validation/ClusterCreation.png shows a cluster creation
			c. screenshot/Train_Validation/ModelReport.png shows a model report

	3) Run Final_Train_Prediction.R on a command line
		- Description: Load, pre-process data and train a final model, 
		  then showning a result of a train error and making a submission file to Kaggle
		- ======= Important Note================
		- At line 9, please change a path of a working directory for yours
		- It takes about 30 minutes on Intel i5 Core
		- On the 119th line, epoch is set to 1 because of a fast running. 
		  If you want, you can change it.
		- command:
					Rscript Final_Train_Prediction.R 
		- screenshot
			a. screenshot/Final_Train_Prediction/RunningCommand.png shows a running command
			b. screenshot/Final_Train_Prediction/ClusterCreation.png shows a cluster creation
			c. screenshot/Final_Train_Prediction/ModelReport.png shows a model report
		
