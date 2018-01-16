# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:33:05 2018

@author: Ningning
"""


### IMPORT

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
import pprint  

# Project
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data  # changed tester's cross_validation to model_selection

# Visualization
import matplotlib.pyplot

# Feature preprocessing
from sklearn.preprocessing import MinMaxScaler
# Split data
from sklearn.model_selection import train_test_split # Split data
 
# Evaluation metrics
from sklearn.metrics import recall_score, precision_score, classification_report
# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# Feature selection
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
# Tune parameters
### tunning svc
from sklearn.grid_search import GridSearchCV
# K-fold validation
from sklearn.model_selection import cross_val_score 
# Pipeline
from sklearn.pipeline import Pipeline
# dimensionality reduction
from sklearn.decomposition import PCA




#########################################################################
#########################################################################
########                                                      ###########
########                   Select features 1                  ###########
########                                                      ###########
#########################################################################
#########################################################################


### create features list

with open("final_project_dataset.pkl", "r") as data_file:  # with open...as...
    data_dict = pickle.load(data_file)  # load pickle


features_list = ['poi',
                 'salary', 'bonus', 'total_payments', 
                 'exercised_stock_options', 'restricted_stock', 'total_stock_value',
                 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'to_messages', 'from_messages'
                 ]  # may add shared_receipts in future



### Are there features with many missing values

countFeatures = {}
for i in features_list:
    countFeatures[i] = 0
    for item in data_dict:
        if data_dict[item][i] == "NaN":
            countFeatures[i] += 1
        else:
            pass
pprint.pprint(countFeatures)

# =============================================================================
# Missing values
# {'bonus': 64,
#  'exercised_stock_options': 44,
#  'from_messages': 60,
#  'from_poi_to_this_person': 60,
#  'from_this_person_to_poi': 60,
#  'poi': 0,
#  'restricted_stock': 36,
#  'salary': 51,
#  'to_messages': 60,
#  'total_payments': 21,
#  'total_stock_value': 20}
# =============================================================================




#########################################################################
#########################################################################
########                                                      ###########
########                   Remove outliers                    ###########
########                                                      ###########
#########################################################################
#########################################################################


# transfer data to numpy
data = featureFormat(data_dict, features_list, sort_keys = True)

# find out outliers in salary and bonus via visualization
for point in data:
    salary = point[1]
    total_payments = point[2]    
    matplotlib.pyplot.scatter( salary, total_payments )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()   

# check the max salary programatically
salaryList = []
bonousList = []
for point in data:
    salary = point[1]
    salaryList.append(salary) # add the salary in to a list 
    bonus = point[2]
    bonousList.append(bonus)  # add bonus to a list

# get the maximum salary in the dictionary
maxSalary = max(salaryList)  

# get key value of the name
for i in data_dict:  
    if data_dict[i]["salary"] == maxSalary:
        print i  # result: TOTAL  

# remove TOTAL    
data_dict.pop("TOTAL", 0)
        

# Any other outliers?
data = featureFormat(data_dict, features_list, sort_keys = True)
for point in data:
    salary = point[1]
    bonus = point[2]    
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()   

### Find out the three points that seeem to have very high salary
overpaid = []
for i in data_dict:
    # necessary to get rid of the NaN
    if data_dict[i]["salary"] != 'NaN' and data_dict[i]["bonus"] != 'NaN':
        if  data_dict[i]["salary"] > 1000000 or data_dict[i]["bonus"] > 6000000:
            overpaid.append(i)
        
print len(overpaid)  
# 4 overpaid enron people
print overpaid  
print
print "-----------------------------------------------------"
# ['LAVORATO JOHN J', 'LAY KENNETH L', 'SKILLING JEFFREY K', 'FREVERT MARK A']
# They should not be removed.

#### remove THE TRAVEL AGENCY IN THE PARK 
## according to the financial PDF (enron61702insiderpay.pdf), 
## there is also THE TRAVEL AGENCY IN THE PARK in the dictionary
## the agency account, which is co-owned by the sister of Enron's 
## former chairman, handlehandled s Enron employees business-related travels
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

# =============================================================================
# Removed TOTAL and THE TRAVEL AGENCY IN THE PARK from the dictionary
# =============================================================================





#########################################################################
#########################################################################
########                                                      ###########
########               Create new features:ratio              ###########
########                                                      ###########
#########################################################################
#########################################################################

### Create a function to get the ratio

def fraction_poi(poi_related_emails, emails):
    if poi_related_emails == "NaN" or emails == "NaN":
        ratio = "NaN"
    else:
        ratio = float(poi_related_emails) / float(emails) # float: divide operation
    return ratio

### add ratios to the data_dict
for i in data_dict:
    data_dict[i]["ratio_from_poi"] = \
    fraction_poi(data_dict[i]["from_poi_to_this_person"], \
                 data_dict[i]["to_messages"])
    data_dict[i]["ratio_to_poi"] = \
    fraction_poi(data_dict[i]["from_this_person_to_poi"], \
                 data_dict[i]["from_messages"])    

### Update features_list
features_list = ['poi',
                 'salary', 'bonus', 'total_payments', 
                 'exercised_stock_options', 'restricted_stock', 'total_stock_value',
                 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'to_messages', 'from_messages', 
                 "ratio_from_poi","ratio_to_poi"
                 ] 

### transfer the dictionary into numpy
data = featureFormat(data_dict, features_list, sort_keys = True)
### Split labels and features
labels, features = targetFeatureSplit(data)

# =============================================================================
# Two features added
#  feature1: Ratio of the person's email to poi to  total number of the person's sent messages
#  feature2: Ratio of poi's email to poi to  total number of the person's received messages
# =============================================================================



#########################################################################
#########################################################################
########                                                      ###########
########    Data, labels, features for local testing          ###########
########                                                      ###########
#########################################################################
#########################################################################


### transfer the dictionary into numpy for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
### Split labels and features
labels, features = targetFeatureSplit(data)

### Split data for training and testing
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.3, random_state=42)




#########################################################################
#########################################################################
########                                                      ###########
########                Try classifiers                       ###########
########                                                      ###########
#########################################################################
#########################################################################

print "First evaluation: get a baseline"
print

def get_baseline(clf):
    clf.fit(features_train, labels_train)
    clf_predict = clf.predict(features_test)
    clf_score = clf.score(features_test, labels_test)
    print "Accuracy: %s" % clf_score
    print "Classification report:"
    print(classification_report(labels_test, clf_predict))
     
clfNb = GaussianNB()
clfSvc = SVC()
clfDt = DecisionTreeClassifier()

print "Naive bayes classification's baseline:"
get_baseline(clfNb)

print "SVC classification's baseline:"
get_baseline(clfSvc)

print "Decision tree classification's baseline:"
get_baseline(clfDt)

print "-----------------------------------------------------"

# =============================================================================
# 
# First evaluation: get a baseline
# 
# Naive bayes classification's baseline:
# Accuracy: 0.883720930233
# Classification report:
#              precision    recall  f1-score   support
# 
#         0.0       0.92      0.95      0.94        38
#         1.0       0.50      0.40      0.44         5
# 
# avg / total       0.87      0.88      0.88        43
# 
# SVC classification's baseline:
# Accuracy: 0.883720930233
# Classification report:
#              precision    recall  f1-score   support
# 
#         0.0       0.88      1.00      0.94        38
#         1.0       0.00      0.00      0.00         5
# 
# avg / total       0.78      0.88      0.83        43
# 
# Decision tree classification's baseline:
# Accuracy: 0.837209302326
# Classification report:
#              precision    recall  f1-score   support
# 
#         0.0       0.90      0.92      0.91        38
#         1.0       0.25      0.20      0.22         5
# 
# avg / total       0.82      0.84      0.83        43
# 
# =============================================================================




#########################################################################
#########################################################################
########                                                      ###########
########      Selecting features 2                             ###########
########                                                      ###########
#########################################################################
#########################################################################


### Make a pipeline


def pipe(estimators):
    pipe = Pipeline(estimators)
    pipe.fit(features_train, labels_train)
    pipe_predict = pipe.predict(features_test)
    pipe_score = pipe.score(features_test, labels_test)
    pipe_report = classification_report(labels_test, pipe_predict)
    print "Accuracy: %s" % pipe_score
    print "Classification report:"
    print pipe_report
    print
    
scaler = MinMaxScaler()
kbest = SelectKBest(chi2)

### naive bayes

### feature scaling
estimatorsNb_scaled = [('scaler', scaler), ('clf', clfNb)]
### feature scaling + kbest
estimatorsNb_scaled_kbest = [('scaler', scaler), ('feature_selection', kbest),('clf', clfNb)]

print "Naive bayes:"

#baseline
print "Baseline:"
get_baseline(clfNb)

# feature scaling
print "After feature scaling: "
pipe(estimatorsNb_scaled)

# feature selection: kbest
print "After feature scaling and selection(kbest):"
pipe(estimatorsNb_scaled_kbest)

print "-----------------------------------------------------"

# =============================================================================
# Naive bayes: No difference from the baseline
# =============================================================================


### svc

### feature scaling
estimatorsSvc_scaled = [('scaler', scaler), ('clf', clfSvc)]

### feature scaling + kbest
estimatorsSvc_scaled_kbest = [('scaler', scaler), ('feature_selection', kbest),('clf', clfSvc)]

print "SVC:"

# baseline
print "Baseline:"
get_baseline(clfSvc)

# feature scaling
print "After feature scaling: "
pipe(estimatorsSvc_scaled)

# feature selection: kbest
print "After feature scaling and selection(kbest):"
pipe(estimatorsSvc_scaled_kbest)

print "-----------------------------------------------------"

# =============================================================================
# SVC: No difference from the baseline;
# and the performance of identifying POI is so bad, the precision and recall score are always 0.
# =============================================================================



### decision tree
### feature scaling
estimatorsDt_scaled = [('scaler', scaler), ('clf', clfDt)]

print "Decision tree:"

# baseline
print "Baseline:"
get_baseline(clfDt)

## feature 
#print "After feature scaling: "
#pipe(estimatorsDt_scaled)

# =============================================================================
# Decisiontree does not need feature scaling.
# and after I tried feature scaling, I found that the results are different in different tests:
# so gave up the feature scaling in DT
# Decision tree:
# Baseline:
# Accuracy: 0.813953488372
# Classification report:
#              precision    recall  f1-score   support
# 
#         0.0       0.88      0.92      0.90        38
#         1.0       0.00      0.00      0.00         5
# 
# avg / total       0.77      0.81      0.79        43
# 
# After feature scaling: 
# Accuracy: 0.813953488372
# Classification report:
#              precision    recall  f1-score   support
# 
#         0.0       0.88      0.92      0.90        38
#         1.0       0.00      0.00      0.00         5
# 
# avg / total       0.77      0.81      0.79        43
# -------------------------------------------------------
# Decision tree:
# Baseline:
# Accuracy: 0.813953488372
# Classification report:
#              precision    recall  f1-score   support
# 
#         0.0       0.88      0.92      0.90        38
#         1.0       0.00      0.00      0.00         5
# 
# avg / total       0.77      0.81      0.79        43
# 
# After feature scaling: 
# Accuracy: 0.837209302326
# Classification report:
#              precision    recall  f1-score   support
# 
#         0.0       0.90      0.92      0.91        38
#         1.0       0.25      0.20      0.22         5
# 
# avg / total       0.82      0.84      0.83        43
# =============================================================================


# feature importances
clfDt = DecisionTreeClassifier()
estimatorsDt = [('clf', clfDt)]
pipeDt = Pipeline(estimatorsDt)
pipeDt.fit(features_train, labels_train)
featureImportances = pipeDt.named_steps['clf'].feature_importances_
print featureImportances
pipe(estimatorsDt)



sfm = SelectFromModel(clfDt, threshold = 0.1)
print sfm
print "select from model"

# fit to training data
sfm.fit(features_train, labels_train)
# reduce featuers in etst data
features_sfm = sfm.transform(features_test)

sfm_predict = clf_sfm.predict(features_sfm)
#sfm_score = sfm.score(features_sfm, labels_test)
#print "Accuracy: %s" % sfm_score
#print "Classification report:"
#print(classification_report(features_sfm, sfm_predict))

