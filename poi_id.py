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




###############################################################################
#############       Remove outliers                             ###############
###############################################################################


# transfer data_dict to numpy for sklearn
data = featureFormat(data_dict, features_list, sort_keys = True)


# find out outliers in salary and bonus via visualization
print "Enron people's salary and bonus:"

for datapoint in data:
    salary = datapoint[1]  #the second column in the numpy
    total_payments = datapoint[2]    # the third column in the numpy
    matplotlib.pyplot.scatter( salary, total_payments )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()   


# get the salary and bonus list
salaryList = []
bonousList = []
for datapoint in data:
    salary = datapoint[1]
    salaryList.append(salary) # add salary into a list 
    bonus = datapoint[2]
    bonousList.append(bonus)  # add bonus into a list

# get the max salary
maxSalary = max(salaryList)  


# get the max salary reciever's name
for i in data_dict:  
    if data_dict[i]["salary"] == maxSalary:
        print "The max salary receiver is:", i  # result: TOTAL  


# remove TOTAL    
data_dict.pop("TOTAL", 0)
        

# Any other outliers?
data = featureFormat(data_dict, features_list, sort_keys = True)
print "Enron people's salary and bonus after removing 'Total':"
for point in data:
    salary = point[1]
    bonus = point[2]    
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()   

# find out the three points that seeem to have very high salary
overpaid = []
for i in data_dict:
    # necessary to get rid of the NaN
    if data_dict[i]["salary"] != 'NaN' and data_dict[i]["bonus"] != 'NaN':
        if  data_dict[i]["salary"] > 1000000 or data_dict[i]["bonus"] > 6000000:
            overpaid.append(i)
        
print "Enron's overpaid employees:", overpaid  
print len(overpaid), "people"  # 4 overpaid enron people
# =============================================================================
#  ['LAVORATO JOHN J', 'LAY KENNETH L', 'SKILLING JEFFREY K', 'FREVERT MARK A']
#  They should not be removed.
# =============================================================================


# remove THE TRAVEL AGENCY IN THE PARK, who is not an individual person
### according to the financial PDF (enron61702insiderpay.pdf), 
### the agency account is co-owned by the sister of Enron's 
### former chairman, handlehandled s Enron employees business-related travels
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

# remove LOCKHART EUGENE E, whose all values are NaN
data_dict.pop("LOCKHART EUGENE E", 0)


print "-----------------------------------------------------"

# =============================================================================
# Removed three datapoints
# TOTAL
# THE TRAVEL AGENCY IN THE PARK 
# LOCKHART EUGENE E
# =============================================================================



###############################################################################
#############        Create new features:ratio                  ###############
###############################################################################

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

# =============================================================================
# Two features added
# feature1: 
# Ratio of the person's email to poi to  total number of the person's sent messages
# feature2: 
# Ratio of poi's email to poi to  total number of the person's received messages
# Split labels and features
# =============================================================================


###############################################################################
#############     wrap up data_dict, split labels, features     ###############
###############################################################################

my_dataset = data_dict

# according to tester.py's test_classifier
# transfer the dictionary into numpy
data = featureFormat(my_dataset, features_list, sort_keys = True)
# split labels and features
labels, features = targetFeatureSplit(data)

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

# =============================================================================
# I used to use the most ordinary train and split, 
# but to get higher score in tester, it's better to use 
# the test_classifier's way for StratifiedShuffleSplit 
# * and for some weird reason, 
# I have to import StratifiedShuffleSplit from cross_validation rather than model_selection in sklearn 0.19
# =============================================================================
        
        
        
###############################################################################
#############     Try classifiers                               ###############
###############################################################################

# baseline:
# naive bayes, svc, decision tree

print "First evaluation: get a baseline"
print

# function to get a classifier's baseline: accuracy, classification report
def get_baseline(clf):
    clf.fit(features_train, labels_train)
    clf_predict = clf.predict(features_test)
    clf_score = clf.score(features_test, labels_test)
    print "Accuracy: %s" % clf_score
    print "Classification report:"
    print classification_report(labels_test, clf_predict)

     
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


###############################################################################
#############     Preprocessing for clf: scaling, kbest, pca           ###############
###############################################################################

# make a pipeline

def pipe(estimators):
    pipe = Pipeline(estimators)  # pipeline
    pipe.fit(features_train, labels_train)  # fit
    pipe_predict = pipe.predict(features_test)  # predict
    pipe_score = pipe.score(features_test, labels_test)  # accuracy
    pipe_report = classification_report(labels_test, pipe_predict)
    print "Accuracy: %s" % pipe_score
    print "Classification report:"
    print pipe_report
    print
    
# scaling
scaler = MinMaxScaler()
# select features
kbest = SelectKBest(chi2)
# reduce dimension
pca = PCA()

# =============================================================================
# This is mainly for naive bayes and svc, decision tree will use feature importances
# =============================================================================


###############################################################################
#############         naive bayes                               ###############
###############################################################################

# preprocessing
# feature scaling
estimatorsNb_scaled = [('scaler', scaler), ('clf', clfNb)]
# feature scaling + selection: kbest
estimatorsNb_scaled_kbest = [('scaler', scaler), ('feature_selection', kbest),('clf', clfNb)]
## dimension reduction: pca
#estimatorsNb_scaled_pca = [('scaler', scaler), ('reduce_dim', pca),('clf', clfNb)]
# =============================================================================
# # Previously used pca(), but after StratifiedShufleSplit, I found that Kbest performs better in NB
# # So will not remove PCA in this later
# =============================================================================

print "Naive bayes's preprocessing:"

#baseline
print "Baseline:"
get_baseline(clfNb)

# feature scaling
print "After feature scaling: "
pipe(estimatorsNb_scaled)

# feature scaling + selection: kbest
print "After feature scaling and selection(kbest):"
pipe(estimatorsNb_scaled_kbest)

## feature scaling + dimension reduction: pca
#print "After feature scaling and PCA:"
#pipe(estimatorsNb_scaled_pca)

print "-----------------------------------------------------"

# =============================================================================
# scaling + kbest for params tunning
# =============================================================================


# tune parameters

# add scaling+ pca into pipepine
pipe_Nb = Pipeline(estimatorsNb_scaled_kbest)
# fit the pipeline
pipe_Nb.fit(features_train, labels_train)

# tuning params
param_grid_Nb = dict(feature_selection__k=[3, 4, 5, 6, 7, 8, 9, 10])


# use StratifiedKFold to make the classifier more robust!!
### this is a small dataset, with the ratio of poi and non-poi highly unbalanced
grid_search_Nb = GridSearchCV(pipe_Nb, param_grid=param_grid_Nb, cv = StratifiedKFold(10))
grid_search_Nb.fit(features_train, labels_train)

# get the best Nb clf
best_Nb = grid_search_Nb.best_estimator_

# selected featuers:
print best_Nb
print "selected features:", best_Nb.named_steps['feature_selection'].get_support()
print

# fit the best NB clf
best_Nb.fit(features_train, labels_train)    
# predict the best NB clf
best_Nb_predict = best_Nb.predict(features_test)
# the clf's accuracy
best_Nb_score = best_Nb.score(features_test, labels_test)
# the clf's report
best_Nb_report = classification_report(labels_test, best_Nb_predict)

print "Best naive bayes clf's accuracy:", best_Nb_score
print "Best naive bayes clf's classification report:", best_Nb_report
print

print "Naive bayes's tester classification report"
test_classifier(best_Nb, my_dataset, features_list)

print "-----------------------------------------------------"


###############################################################################
#############         naive bayes                               ###############
###############################################################################


# preprocessing

# feature scaling
estimatorsSvc_scaled = [('scaler', scaler), ('clf', clfSvc)]

# feature scaling + kbest
estimatorsSvc_scaled_kbest = [('scaler', scaler), ('feature_selection', kbest),('clf', clfSvc)]

# dimension reduction: pca
estimatorsSvc_scaled_pca = [('scaler', scaler), ('reduce_dim', pca),('clf', clfSvc)]

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


# feature scaling + dimension reduction: pca
print "After feature scaling and PCA:"
pipe(estimatorsSvc_scaled_pca)

print "-----------------------------------------------------"
#  =============================================================================
#  SVC: No difference from the baseline;
#  and its performance on identifying POI is so bad, that the precision and recall score are always 0.
#  wont tuning params for this classifier for this classifier.
#  =============================================================================



###############################################################################
#############         decision tree                             ###############
###############################################################################


# preprocessing

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

# =============================================================================
# Feature Importances:
# exercised_stock_options 's importance: 0.341932714111
# ratio_to_poi 's importance: 0.126866365519
# -- from_messages 's importance: 0.111111111111
# from_poi_to_this_person 's importance: 0.0860707919531
# to_messages 's importance: 0.0805697278912
# total_stock_value 's importance: 0.0778388278388
# from_this_person_to_poi 's importance: 0.0590861344538
# bonus 's importance: 0.047619047619
# salary 's importance: 0.0357142857143
# restricted_stock 's importance: 0.0331909937888
# ratio_from_poi 's importance: 0.0
# total_payments 's importance: 0.0
# 
# =============================================================================

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

# =============================================================================
# # exercised_stock_options seems to be a very important factor, 
# # while total_stock_value, once added, the performance declined to 0 in precision and recall
# =============================================================================

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

#print "Best decision tree clf's report:"
#print best_Dt_score
#print best_Dt_report

print "Best decision tree's tester classification report:"
test_classifier(best_Dt, my_dataset, features_list_tree)

# =============================================================================
# Decision tree's performance is much worse than naive bayes
# =============================================================================



###############################################################################
#############         dump classifier                           ###############
###############################################################################


clf = best_Nb
my_dataset = my_dataset
features_list = features_list
dump_classifier_and_data(clf, my_dataset, features_list)