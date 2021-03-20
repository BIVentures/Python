# -*- coding: utf-8 -*-
"""
Created on Sep 15, 2020
#File(s): telco_customerchurn.csv
@author: BIVentures
"""

# Import the os module
import os

# Get the current working directory
cwd = os.getcwd()

# Print the current working directory
print("Current working directory: {0}".format(cwd))

# Print the type of the returned object
#print("os.getcwd() returns an object of type: {0}".format(type(cwd)))

# Change the current working directory
os.chdir('C:/DirectoryPathHere/')

# -*-Data Cleansing & Transformation for Customer Churn  -*-
#-*- Data Analysis, Modelling & Visualization -*-
#-*- Features Analysis (Numerical & Categorical), Modelling & Visualization -*-

import pickle
import os.path
import time
import warnings
warnings.filterwarnings('ignore')

##Load Libraries
import re
import pandas as pd
#import pandas.util.testing as tm
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, fbeta_score
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve

import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_formats = ['retina']
import seaborn as sns



##Loaddataset into dataframe
customer_df=pd.read_csv('telco_customerchurn.csv')
# create a dataframe 
customer_df = pd.DataFrame(customer_df) 
  
# converting each value  
# of column to a string 
customer_df = pd.DataFrame(data=customer_df)


##show top 9 (0-8 array) rows of dataframe
#print(customer_df.head(9))
#print('\n')

##show total number of rows and columns
#print(customer_df.shape)
#print('\n')

##show names of all columns
#print(customer_df.columns)
#print(customer_df.columns.values)
#print('\n')

##Check for na or missing data
print(customer_df.isna().sum())

##Data Cleaning & Data Manipulation/Transformation
##Handling NULL Data, Missing Data/Inconsistencies
#remove 11 rows with spaces in TotalCharges (0.15% missing data)
#refer to missing data in general as null, NaN, or NA values.
# df.dropna(axis='columns') #drops all columns containing a null value


#Missing numerical data use nan (Not a Number)
customer_df['TotalCharges'] = customer_df['TotalCharges'].replace('',np.nan)

##Change Total Charges string column as type float 
######*****This was painful to convert it into float because value was \n
######*****Luckily I figured it out, so yay! 
non_numeric = re.compile(r'[^\d.]+')
#number = str[ /\d+(?:.\d+)?/ ]
customer_df['TotalCharges']=customer_df.loc[customer_df['TotalCharges'].str.contains(non_numeric)]
customer_df['TotalCharges'] = customer_df['TotalCharges'].str.replace(non_numeric, '0')
customer_df['TotalCharges'] = customer_df['TotalCharges'].str.replace('', '0')
customer_df['TotalCharges'] = customer_df['TotalCharges'].str.replace(',', '')
customer_df['TotalCharges'] = customer_df['TotalCharges'].str.replace('$', '')
customer_df['TotalCharges'] = customer_df['TotalCharges'].str.replace(r'\\n',' ', regex=True)
#customer_df['TotalCharges'] = pd.to_numeric(customer_df['TotalCharges'], downcast="float")
customer_df['TotalCharges'] =customer_df['TotalCharges'].astype(float)

# Crosscheck & show the data column 
#print(customer_df['TotalCharges']) 
print(customer_df['TotalCharges'].apply(type).value_counts())

# Crosscheck Data Types now & show the datatypes 
print (customer_df.dtypes) 

##Handling Null data
#(): Generate a boolean mask indicating missing values
#notnull(): Opposite of isnull()
#dropna(): Return a filtered version of the data
#fillna(): Return a copy of the data with missing values filled or impute
#dropna The default is how='any', drops any row or column containing a null value
#dropna how='all', which will only drop rows/columns that are 'all' null values
#customer_df = customer_df.dropna(how = 'any')

##Data Preview
print ('Rows     : ', customer_df.shape[0])
print ('Columns  : ', customer_df.shape[1])
print('\nColumns or Features : \n', customer_df.columns.tolist())
print ('\nMissing values :  ', customer_df.isnull().sum().values.sum())
print ('\nUnique values :  \n', customer_df.nunique())

customer_df.info()
customer_df.isnull().sum()


##Calculate Customer Churn Count
total = float(len(customer_df['Churn']))
print('\n')
print('Customer Churn Count: \n', customer_df['Churn'].value_counts())
print('\n')
print('Customer Churn Count (%): \n',customer_df['Churn'].value_counts()/(total)*100)
print('\n')

##Visualize Customer Churn Count 
#sns.countplot(customer_df['Churn'])
#customer_df['Churn'].value_counts().plot(kind='bar');
#plt.show()

total = float(len(customer_df['Churn']))
ax=sns.countplot(x='Churn', data=customer_df, order=customer_df['Churn'].value_counts().sort_values().index)
ax.set_ylabel('Customers')
bars = ax.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]
for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.show()


##Feature(s) Related Information
#The categories of features in this dataset include the following:
#customer demographic data: Gender, SeniorCitizen, Partner, Dependents
#subscribed services: PhoneService, MultipleLine, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
#customer account information: CustomerID, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Tenure
#Target Feature/Class/Category is Churn, which has binary classes 1 (yes) and 0 (no)
##Replace values for SeniorCitizen as a categorical feature
customer_df['SeniorCitizen'] = customer_df['SeniorCitizen'].replace({1:'Yes',0:'No'})
customer_df = customer_df.dropna(how='all') # remove samples with null fields
customer_df = customer_df[~customer_df.duplicated()] # remove duplicates
customer_df[customer_df.TotalCharges == ' '] # display all 11 rows with spaces in TotalCharges column (0.15% missing data)

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
customer_df[numerical_cols].describe()

pltx=sns.pairplot(customer_df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']], 
             hue='Churn', plot_kws=dict(alpha=.3, edgecolor='none'), height=2, aspect=1.1);
print(pltx)
plt.show()


##Correlation Matrix for variables
axcor=sns.set(rc={'figure.figsize':(8,6)})
sns.heatmap(customer_df.corr(), cmap="seismic", annot=False, vmin=-1, vmax=1)
print(axcor)
plt.show()


##Numerical features analysis Histogram Charts
fig, axn = plt.subplots(1, 3, figsize=(15, 3))
customer_df[numerical_cols].hist(bins=20, figsize=(10, 7), ax=axn)
print(fig, axn)
plt.show()


##Numerical features Analysis, displaying impact on Churning
#Shows the greater TotalCharges, tenure features are, the less is the probability of churn
fig, axnf = plt.subplots(1, 3, figsize=(15, 3))
customer_df[customer_df.Churn == "No"][numerical_cols].hist(bins=35, color="blue", alpha=0.5, ax=axnf)
customer_df[customer_df.Churn == "Yes"][numerical_cols].hist(bins=35, color="orange", alpha=0.7, ax=axnf)
plt.legend(['No Churn', 'Churn'], shadow=True, loc=9)
print(fig, axnf)
plt.show()


##Categorical Features Distribution Analysis to determine Churn
# Some features are removed to reduce data
# "No Internet Service" is a repeated feature in 6 other charts

categorical_features = [
 'gender',
 'SeniorCitizen',
 'Partner',
 'Dependents',
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtection',
 'TechSupport',
 'StreamingTV',
 'StreamingMovies',
 'PaymentMethod',
 'PaperlessBilling',
 'Contract' ]

ROWS, COLS = 4, 4
fig, axcat = plt.subplots(ROWS, COLS, figsize=(18, 20) )
row, col = 0, 0
for i, categorical_feature in enumerate(categorical_features):
    if col == COLS - 1:
        row += 1
    col = i % COLS
#   customer_df[categorical_feature].value_counts().plot('bar', ax=axcat[row, col]).set_title(categorical_feature)
    customer_df[customer_df.Churn=='No'][categorical_feature].value_counts().plot(kind='bar', width=.5, ax=axcat[row, col], color='blue', alpha=0.5).set_title(categorical_feature)
    customer_df[customer_df.Churn=='Yes'][categorical_feature].value_counts().plot(kind='bar', width=.3, ax=axcat[row, col], color='orange', alpha=0.7).set_title(categorical_feature)
    plt.legend(['No Churn', 'Churn'])
    fig.subplots_adjust(hspace=0.7)
    
print(fig, axcat)
plt.show() 



#The features in this dataset include the following:
#customer demographic data: Gender, SeniorCitizen, Partner, Dependents
#subscribed services: PhoneService, MultipleLine, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
#customer account information: CustomerID, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Tenure
#Target feature is column Churn, which has binary classes 1 and 0.

# Customer Account Info Analysis, Contract & Payment Method impacts Churning
# note: users who have a month-to-month contract and Electronic check PaymentMethod are more likely to churn
fig, axcustac = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
customer_df[customer_df.Churn == 'No']['Contract'].value_counts().plot(kind='bar', ax=axcustac[0], color='blue', alpha=0.5).set_title('Contract')
customer_df[customer_df.Churn == 'Yes']['Contract'].value_counts().plot(kind='bar', width=.3, ax=axcustac[0], color='orange', alpha=0.7)
customer_df[customer_df.Churn == 'No']['PaymentMethod'].value_counts().plot(kind='bar', ax=axcustac[1], color='blue', alpha=0.5).set_title('PaymentMethod')
customer_df[customer_df.Churn == 'Yes']['PaymentMethod'].value_counts().plot(kind='bar', width=.3, ax=axcustac[1], color='orange', alpha=0.7)
plt.legend(['No Churn', 'Churn'])
print(fig, axcustac)
plt.show()



##Statistical Analysis
customer_df[numerical_cols].info()
customer_df[numerical_cols].describe()


##Calculate the Percentage of Customers 
customer_retained = customer_df[customer_df.Churn =='No'].shape[0]
print('Customers Retained Count: ', customer_retained)
customer_churned = customer_df[customer_df.Churn =='Yes'].shape[0]
print('Customers Churned Count: ', customer_churned)
Total_Customers = customer_retained + customer_churned
print('Total Customers Count: ', Total_Customers)
print('\n')


Continued_Customers = customer_retained/Total_Customers
print('Customers Continued with the Company: ', Continued_Customers)

Churned_Customers = customer_churned/Total_Customers
print('Customers left the Company: ', Churned_Customers)
print('\n')

##Calculate Percentage of Customers Contiuned with the Company
Continued_CustomersPercentage = round((customer_retained/Total_Customers*100),2)
print('Percentage of Customers Continued with the Company: ', Continued_CustomersPercentage, '%')

##Calculate Percentage of Customers Left the Company
Churned_CustomersPercentage = round((customer_churned/Total_Customers*100),2)
print('Percentage of Customers Left the Company Percentage: ', Churned_CustomersPercentage, '%')
print('\n')

#Visualize the Churn Count for Male(s) and Female(s)
#Shows Females were least likely to churn compared to males
gnx=sns.countplot(x='gender', hue='Churn', data=customer_df)
print(gnx)
plt.show()

#Visualize the Churn Count based on Internet Service
#Shows the highest number of customers that didn't churn use DSL
isx=sns.countplot(x='InternetService', hue='Churn', data=customer_df)
print(isx)
plt.show()

##Visualize Histogram using Numerical data
#Select Couple of Numerical Columns
numerical_features = ['tenure', 'MonthlyCharges']
#Visualize using Histogram
#Create Figure with 1 row, 2 columns
#To create an 28x8 pixel, 100 dots-per-inch figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(28,8), dpi=100)
customer_df[customer_df.Churn =='No'][numerical_features].hist(bins=20,color='blue',alpha=0.5, ax=axes)
customer_df[customer_df.Churn =='Yes'][numerical_features].hist(bins=20,color='orange',alpha=0.5, ax=axes)
print(fig, axes)
plt.show()


##Basic Bar Chart
customer_df['Churn'].value_counts().plot(kind='bar').set_title('Churn')
plt.show()


##Remove unneccessary columns to reduce data
##show names of all columns
#print(customer_df.columns)

#Drop CustomerID column, adds no value for analysis purposes
clean_customer_df = customer_df.drop('customerID', axis=1)

#Confirm number of rows and columns in the dataset after removing a column
print(clean_customer_df.shape)
print('\n')

##Check Data types for the clean_customer_df
print(clean_customer_df.dtypes)


##Data Transformation
##Convert non-numeric columns to numeric
##For a particular column in all columns
##If that column data type is numeric then continue with that column
##Else transform it into nummeric 
for column in clean_customer_df.columns:
    if clean_customer_df[column].dtype == np.number:
        continue
    else:
        clean_customer_df[column] =LabelEncoder().fit_transform(clean_customer_df[column])

##Show new dataset data types
print(clean_customer_df.dtypes)
#Results show object types have been transformed into numeric data types
print('\n')



