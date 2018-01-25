# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:44:54 2018

@author: Ningning
"""

# Import

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
import pprint  
import operator


# Project provided
from feature_format import featureFormat
from feature_format import targetFeatureSplit
from tester import dump_classifier_and_data  # changed tester's cross_validation to model_selection

# Visualization
import matplotlib.pyplot

# Feature preprocessing
from sklearn.preprocessing import MinMaxScaler
# Split data
from sklearn.model_selection import train_test_split # Split data
 
# Evaluation metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel

# Tune parameters
### tunning svc
from sklearn.model_selection import GridSearchCV
# K-fold validation
from sklearn.model_selection import cross_val_score 
# Pipeline
from sklearn.pipeline import Pipeline
# Dimensionality reduction
from sklearn.decomposition import PCA

# StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedShuffleSplit
# its very weird that i have to import from cross_validation, 
# if I chose to import from model_selection will cause error:StratifiedShuffleSplit not iterable 
# KFOLD
from sklearn.model_selection import StratifiedKFold

# to test the performance of the clf
from tester import test_classifier


###############################################################################
#############       About the dataset                           ###############
###############################################################################


# load pickle
with open("final_project_dataset.pkl", "r") as data_file:  # with open...as...
    data_dict = pickle.load(data_file)  # load pickle

# about dataset
print "Number of datapoints:", len(data_dict)

# number of POIs
poi = []
for i in data_dict:
    if data_dict[i]['poi'] == True:
        poi.append(i)
print "Number of POIs:", len(poi)
poi_ratio = float(len(poi))/float(len(data_dict))

print "POIs account for {} of the total dataset.".format(poi_ratio)
print "-----------------------------------------------------"


# =============================================================================
# Number of datapoints: 146
# Number of POIs: 18
# 0.123287671233
# POIs account for 0.123287671233 of the total dataset.
# =============================================================================



###############################################################################
#############       create features list                        ###############
###############################################################################


# all features listed
features_list = ['poi', 'bonus', 'deferral_payments', 
                      'deferred_income', 'director_fees',
                      'email_address', 'exercised_stock_options', 'expenses',
                      'from_messages', 'from_poi_to_this_person', 
                      'from_this_person_to_poi',
                      'loan_advances', 'long_term_incentive', 
                      'other','restricted_stock','restricted_stock_deferred',
                      'salary','shared_receipt_with_poi','to_messages',
                      'total_payments','total_stock_value']  

# are there features with many missing values
countFeatures = {}
for i in features_list:
    countFeatures[i] = 0
    for item in data_dict:
        if data_dict[item][i] == "NaN":
            countFeatures[i] += 1
        else:
            pass
        
# sort features by missing values, import operator
countFeatures = sorted(countFeatures.items(), key=operator.itemgetter(1))

print "Sorted features with missing values:" 
pprint.pprint(countFeatures)
print "-----------------------------------------------------"

# Update features_list
features_list = ['poi',
                 'salary', 'bonus', 'total_payments', 
                 'exercised_stock_options', 'restricted_stock', 'total_stock_value',
                 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'to_messages', 'from_messages'
                 ]  

print "-----------------------------------------------------"

# =============================================================================
# The selected features has relatively less missing values and seems very relevalnt to poi
# =============================================================================





# transfer data_dict to numpy for sklearn
data = featureFormat(data_dict, features_list, sort_keys = True)

  
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)



# create a function to get the ratio
def fraction_poi(poi_related_emails, emails):
    if poi_related_emails == "NaN" or emails == "NaN":
        # feature_format function will handle the NaN
        ratio = "NaN"  
    else:
        ratio = float(poi_related_emails) / float(emails)
        # float for divide operation
    return ratio


# add ratio featuress to data_dict
for i in data_dict:
    # ratio_from_poi
    data_dict[i]["ratio_from_poi"] = \
    fraction_poi(data_dict[i]["from_poi_to_this_person"], \
                 data_dict[i]["to_messages"])
    # ratio_to_poi
    data_dict[i]["ratio_to_poi"] = \
    fraction_poi(data_dict[i]["from_this_person_to_poi"], \
                 data_dict[i]["from_messages"])    

# update features_list
features_list = ['poi',
                 'salary', 'bonus', 'total_payments', 
                 'exercised_stock_options', 'restricted_stock', 'total_stock_value',
                 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'to_messages', 'from_messages', 
                 "ratio_from_poi","ratio_to_poi"
                 ] 



my_dataset = data_dict

# according to tester.py's test_classifier
# transfer the dictionary into numpy
data = featureFormat(my_dataset, features_list, sort_keys = True)
# split labels and features
labels, features = targetFeatureSplit(data)

# split train and test data
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )


# decision tree

# function to get a classifier's baseline: accuracy, classification report
def get_baseline(clf):
    clf.fit(features_train, labels_train)
    clf_predict = clf.predict(features_test)
    clf_score = clf.score(features_test, labels_test)
    print "Accuracy: %s" % clf_score
    print "Classification report:"
    print classification_report(labels_test, clf_predict)

     

clfDt = DecisionTreeClassifier()


print "Decision tree:"

# baseline
print "Baseline:"
get_baseline(clfDt)

# =============================================================================
# The df's baseline precision is surprisingly high....
# Classification report:
#              precision    recall  f1-score   support
# 
#         0.0       0.93      1.00      0.96        13
#         1.0       1.00      0.50      0.67         2
# 
# avg / total       0.94      0.93      0.92        15
# =============================================================================


# feature selection: feature importances
estimatorsDt = [('clf', clfDt)]
pipeDt = Pipeline(estimatorsDt)
pipeDt.fit(features_train, labels_train)
featureImportances = pipeDt.named_steps['clf'].feature_importances_

print "Feature Importances:"
indices = np.argsort(featureImportances)[::-1]
for i in indices:
    print features_list[i+1],"'s importance:",featureImportances[i]
print



# update features_list_tree after feature selection
features_list_tree = ['poi', 'exercised_stock_options', 'bonus',
'ratio_to_poi', 'from_messages','from_poi_to_this_person']

# Create new data and split labels, features
data_tree = featureFormat(my_dataset, features_list_tree, sort_keys = True)

# split labels and features
labels, features = targetFeatureSplit(data_tree)

## split train and test data
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

print "Decision tree after feature selection:"
print get_baseline(clfDt)
print test_classifier(clfDt, my_dataset, features_list_tree)
print "-----------------------------------------------------"


# Tune params: decision tree 
estimatorsDt = [('clf', clfDt)]
# make pipleline
pipe_Dt = Pipeline(estimatorsDt)
# fit pipleline
pipe_Dt.fit(features_train, labels_train)

# tune param: min_samples_split , default 2
param_grid_Dt = dict(clf__min_samples_split=[2, 3, 4, 5])

# use StratifiedKFold to make the classifier more robust!!
### this is a small dataset, with the ratio of poi and non-poi highly unbalanced
grid_search_Dt = GridSearchCV(pipe_Dt, param_grid=param_grid_Dt, cv = StratifiedKFold(10))
# fit the grid search
grid_search_Dt.fit(features_train, labels_train)
# get the best dt clf
best_Dt = grid_search_Dt.best_estimator_

best_Dt.fit(features_train, labels_train)    
best_Dt_predict = best_Dt.predict(features_test)
best_Dt_score = best_Dt.score(features_test, labels_test)
best_Dt_report = classification_report(labels_test, best_Dt_predict)



print "Best decision tree's tester classification report:"
test_classifier(best_Dt, my_dataset, features_list_tree)
