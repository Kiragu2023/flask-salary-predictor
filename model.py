# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 08:39:59 2024

@author: ADMIN
"""

import pandas as pd
import numpy as np
import seaborn as sb
import pickle
import matplotlib.pyplot as plt
#loading the data set
data=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\model\\creditcard.csv")
#explanatory data analysis
data.head()
data.info()
data.columns
data.shape
data.isnull()
#^we have no null values
data.describe()
#handling missing values
data_missing_columns=(round(((data.isnull().sum()/len(data.index))*100),2).to_frame('null')).sort_values('null',ascending=False)
data_missing_columns
#checking distribution of classes
classes=data['Class'].value_counts()
classes
#% of normal transactions
normal_share = round((classes[0]/data['Class'].count()*100),2)
print("non-fraudulent percentage:",normal_share,"%")
#% of fraudulent transactions
fraudulent_share = round((classes[1]/data['Class'].count()*100),2)
print("fraudulent percentage:",fraudulent_share,"%")
#visual bar of fraudulent vs non-fraudulent transaction
sb.countplot(x='Class',data=data)
plt.title("fraudulent vs non-fraudulent transaction")
#OUTLIERS TREATMENT
#creating fraudulent dataframe
fraud_data=data[data['Class']==1]
#creating non-fraudulent dataframe
non_fraud_data=data[data['Class']==0]
"""drop the time column since We do not see any specific pattern for the fraudulent and non-fraudulent transctions with 
respect to Time"""
data.drop('Time',axis=1,inplace=True)
#train ,test and split
from sklearn.model_selection import train_test_split
# Putting feature variables into X
X = data.drop(['Class'], axis=1)
# Putting target variable to y
y = data['Class']
# Splitting data into train and test set 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, 
train_size=0.8, test_size=0.2, random_state=100)
#feature scaling
#standardization method
from sklearn.preprocessing import StandardScaler
# Instantiate the Scaler
scaler = StandardScaler()
# Fit the data into scaler and transform
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_train.head()
#transform set test
X_test['Amount'] = scaler.transform(X_test[['Amount']])
X_test.head()
#checking skeweness
cols = X_train.columns
#Mitigate skeweness
# Importing PowerTransformer
from sklearn.preprocessing import PowerTransformer
# Instantiate the powertransformer
pt = PowerTransformer(method='yeo-johnson', standardize=True, 
copy=False)
# Fit and transform the PT on training data
X_train[cols] = pt.fit_transform(X_train)
# Transform the test set
X_test[cols] = pt.transform(X_test)
# Plotting the distribution of the variables (skewness) of all the columns
# Importing scikit logistic regression module
from sklearn.linear_model import LogisticRegression
#metrics selection
# Impoting metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
#tuning hyperparameters
# Importing libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# Creating KFold object with 5 splits
folds = KFold(n_splits=5, shuffle=True, random_state=4)
# Specify params
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}
# Specifing score as recall as we are more focused on acheiving the higher sensitivity than the accuracy
model_cv = GridSearchCV(estimator = LogisticRegression(),
 param_grid = params, 
 scoring= 'roc_auc', 
 cv = folds, 
 verbose = 1,
 return_train_score=True) 
# Fit the model
model_cv.fit(X_train, y_train)
#Fitting 5 folds for each of 6 candidates, totalling 30 fits
GridSearchCV(cv=KFold(n_splits=5, random_state=4, shuffle=True),
 estimator=LogisticRegression(),
 param_grid={'C': [0.01, 0.1, 1, 10, 100, 1000]},
 return_train_score=True, scoring='roc_auc', verbose=1)
# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results
# plot of C versus train and validation scores

# Best score with best C
best_score = model_cv.best_score_
best_C = model_cv.best_params_['C']
print(" The highest test roc_auc is {0} at C = {1}".format(best_score,
best_C))
#Logistic regression with optimal C
logistic_imb = LogisticRegression(C=0.01)
# Fit the model on the train set
logistic_imb_model = logistic_imb.fit(X_train, y_train)
# Predictions on the train set
y_train_pred = logistic_imb_model.predict(X_train)
# Confusion matrix
confusion = metrics.confusion_matrix(y_train, y_train_pred)
print(confusion)
[[227427,22],
 [ 135,261]]
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train, y_train_pred))
# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))
# Specificity
print("Specificity:-", TN / float(TN+FP))
# F1 score
print("F1-Score:-", f1_score(y_train, y_train_pred))
#classification report
print(classification_report(y_train,y_train_pred))
def draw_roc( actual, probs ):
     fpr, tpr, thresholds = metrics.roc_curve( actual, probs, drop_intermediate =False )
     auc_score = metrics.roc_auc_score( actual, probs )
     plt.figure(figsize=(5, 5))
     plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
     plt.plot([0, 1], [0, 1], 'k--')
     plt.xlim([0.0, 1.0])
     plt.ylim([0.0, 1.05])
     plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
     plt.ylabel('True Positive Rate')
     plt.title('Receiver operating characteristic example')
     plt.legend(loc="lower right")
     plt.show()
     return None
# Predicted probability
y_train_pred_proba = logistic_imb_model.predict_proba(X_train)[:,1]
# Plot the ROC curve
draw_roc(y_train, y_train_pred_proba)

# Pickle the trained Logistic Regression model
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(logistic_imb_model, file)

# Pickle the preprocessing objects
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('power_transformer.pkl', 'wb') as file:
    pickle.dump(pt, file)
import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

def load_model():
    """
    Load the trained logistic regression model.
    """
    try:
        with open('logistic_regression_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Error loading model: {e}")

def predict_fraud(model, transaction_details):
    """
    Predict whether a transaction is fraudulent.
    """
    try:
        prediction = model.predict(transaction_details)[0]
        return "Fraudulent" if prediction == 1 else "Non-Fraudulent"
    except Exception as e:
        messagebox.showerror("Error", f"Error predicting: {e}")

def classify_transaction(transaction_details):
    """
    Classify the transaction as fraudulent or non-fraudulent.
    """
    model = load_model()
    if model:
        result = predict_fraud(model, transaction_details)
        messagebox.showinfo("Prediction Result", f"The transaction is {result}")
    else:
        messagebox.showwarning("Warning", "Model could not be loaded.")
def on_submit():
    """
    Get the transaction details from the entry fields, calculate the total amount, and classify the transaction.
    """
    try:
        # Get individual transaction details and calculate total amount
        transaction_total = sum(float(entry.get().strip()) for entry in entries)
        # Add the total amount to the transaction details
        transaction_details = {'Amount': transaction_total}

        # Classify the transaction
        classify_transaction(np.array([list(transaction_details.values())]))

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numerical values.")
    except Exception as e:
        messagebox.showerror("Error", f"Error processing input: {e}")
