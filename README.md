# Udacity's Data Analyst Nanodegree:  P5 Enron email

The repository contains my submission files for Data Analyst Nanodegree(DAND)'s P5  project. The objective of this project is to train a machine learning classifier to detect person of interest (POI) in Enron case via given financial and email features. It is part of the **machine learning** section of the DAND.

This README contains files listed and outline of the analysis process.

## The files include:
- poi_id.py
	My code for the project, including creating classifiers, feature selection, tuning parameters and validation.  
- p5-enron-submission-report.pdf
	Report with my answers to the project's questions.
- tester.py
	Udacity's provided tester to check the outcome of poi_id.py
- feature_format.py
	Udacity's provided file to handle formats and transfer dataset from dictionary to numpy for scikit learn's to process.
- enron61702insiderpay.pdf
	Udacity's provided file that contains Enron employees' financial data, such as salary, bonus and stock value.
- final_project_dataset.pkl
	Udacity's provided dataset that combines Enron employees' financial and email data, such as number of email sent or number of email received from POI.
- poi_names.txt
	Udacity's provided namelist of POIs. 


## Outline of poi_id.py
### 0. Import modules and load data
### 1. About the data
- Number of datapoints
- Number of POI/non-POI
- Ratio of POI in the whole dataset

### 2. Preprocessing
- Create feature list
	- Find out each feature's number of missing values
	- Hand pick features
- Remove outliers  
	- Remove outliers with visualization
	- Remove other werid datapoints
- Create new features
	- ratio of emails received from POI 
	- ratio of emails sent to POI

### 3. Split labels-features, train-test
- Transfer dataset, features list to numpy
- Split labels and features
- Split train and test sets, using StratifiedShuffleSplit
	because this is a very unbalanced dataset

### 4. Try classifiers: Naive bayes, SVC, Decision Tree
- Get a baseline of these classifiers's default settings's performance

#### 4.1 Naive bayes
1) Preprocess
- scaling
- feature selection: Kbest
- dimension reduction: PCA
2) Tune parameters
Tune K in Kbest using grid search
3) Check their performances with test_classifier in testr.py

#### 4.2 SVC
1) Preprocess
- scaling
- feature selection: Kbest
- dimension reduction: PCA
SVC classifier's precision and recall had no improvement (both 0) after preprocessing, so I gave the classifer up.

#### 4.3 Decision tree
1) Feature selection: feature importances
2) Sort the features by importances and handpicked 
3) Check their performances with test_classifier in testr.py

#### Naive bayes performs best among the three

### 5. Dump the classifier, dataset and features list to pickle

