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

my_dataset = data_dict
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

### add ratios to the my_dataset
for i in my_dataset:
    my_dataset[i]["ratio_from_poi"] = \
    fraction_poi(my_dataset[i]["from_poi_to_this_person"], \
                 my_dataset[i]["to_messages"])
    my_dataset[i]["ratio_to_poi"] = \
    fraction_poi(my_dataset[i]["from_this_person_to_poi"], \
                 my_dataset[i]["from_messages"])    

### Update features_list
features_list = ['poi',
                 'salary', 'bonus', 'total_payments', 
                 'exercised_stock_options', 'restricted_stock', 'total_stock_value',
                 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'to_messages', 'from_messages', 
                 "ratio_from_poi","ratio_to_poi"
                 ] 

### transfer the dictionary into numpy
data = featureFormat(my_dataset, features_list, sort_keys = True)
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
data = featureFormat(my_dataset, features_list, sort_keys = True)
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
pca = PCA()
### naive bayes

### feature scaling
estimatorsNb_scaled = [('scaler', scaler), ('clf', clfNb)]
### feature scaling + selection: kbest
estimatorsNb_scaled_kbest = [('scaler', scaler), ('feature_selection', kbest),('clf', clfNb)]
### # dimension reduction: pca
estimatorsNb_scaled_pca = [('scaler', scaler), ('reduce_dim', pca),('clf', clfNb)]

print "Naive bayes:"

#baseline
print "Baseline:"
get_baseline(clfNb)

# feature scaling
print "After feature scaling: "
pipe(estimatorsNb_scaled)

# feature scaling + selection: kbest
print "After feature scaling and selection(kbest):"
pipe(estimatorsNb_scaled_kbest)

# feature scaling + dimension reduction: pca
print "After feature scaling and PCA:"
pipe(estimatorsNb_scaled_pca)


print "-----------------------------------------------------"

# =============================================================================
# Naive bayes: No difference from the baseline
# =============================================================================


### svc

### feature scaling
estimatorsSvc_scaled = [('scaler', scaler), ('clf', clfSvc)]

### feature scaling + kbest
estimatorsSvc_scaled_kbest = [('scaler', scaler), ('feature_selection', kbest),('clf', clfSvc)]


### # dimension reduction: pca
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

# =============================================================================
# SVC: No difference from the baseline;
# and the performance of identifying POI is so bad, the precision and recall score are always 0.
# =============================================================================

#########################################################################
#########################################################################
########                                                      ###########
########      Tunning params                                  ###########
########                                                      ###########
#########################################################################
#########################################################################


param_grid_Nb = dict(reduce_dim__n_components=[None, 2, 5, 8])


pipe_Nb = Pipeline(estimatorsNb_scaled_pca)
pipe_Nb.fit(features_train, labels_train)

grid_search_Nb = GridSearchCV(pipe_Nb, param_grid=param_grid_Nb)
grid_search_Nb.fit(features_train, labels_train)

print grid_search_Nb.best_estimator_

best_Nb = Pipeline(memory=None,
     steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', GaussianNB(priors=None))])

best_Nb.fit(features_train, labels_train)    
best_Nb_predict = best_Nb.predict(features_test)
best_Nb_score = best_Nb.score(features_test, labels_test)
best_Nb_report = classification_report(labels_test, best_Nb_predict)

print best_Nb_score
print best_Nb_report

# =============================================================================
# Pipeline(memory=None,
#      steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
#   svd_solver='auto', tol=0.0, whiten=False)), ('clf', GaussianNB(priors=None))])
# 0.906976744186
#              precision    recall  f1-score   support
# 
#         0.0       0.95      0.95      0.95        38
#         1.0       0.60      0.60      0.60         5
# 
# avg / total       0.91      0.91      0.91        43
# =============================================================================


#########################################################################
#########################################################################
########                                                      ###########
########      Dump classifier                                 ###########
########                                                      ###########
#########################################################################
#########################################################################


clf = best_Nb
my_dataset = my_dataset
features_list = features_list
dump_classifier_and_data(clf, my_dataset, features_list)