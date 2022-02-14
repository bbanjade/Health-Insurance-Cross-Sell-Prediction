# Health-Insurance-Cross-Sell-Prediction

#### Context
#### Our client is an Insurance company, wants to increase its insurance sales by providing health insurance to the current auto insurance holder. They have some past data about demographics (gender, age, region code), Vehicles (Vehicle Age, Damage), Policty (Premium, sourcing channel), and their response to hold auto insurance. Based, on this data we need to create a model, that will predict customers will be willing to participate in purchasing health insurance in future. This project will visualize the provided data, create an algorithm that best represnt the data, and test the model in the test data.

### This project includes two parts: 1) Exploratory data analysis through visualization & 2) model building: data will be cleaned and prepared for modelling based on the visualizations.

### Import all necessary models 

#### import numpy as np
#### import pandas as pd
#### import matplotlib.pyplot as plt
#### import seaborn as sns

#### from sklearn.model_selection import train_test_split

#### from sklearn.neighbors import KNeighborsClassifier
#### from sklearn.linear_model import LogisticRegression
#### from sklearn.svm import SVC
#### from sklearn.tree import DecisionTreeClassifier
#### from sklearn.ensemble import RandomForestClassifier
#### from sklearn.ensemble import GradientBoostingClassifier

#### from sklearn.model_selection import cross_val_score
#### from sklearn.metrics import confusion_matrix
#### from sklearn.metrics import accuracy_score
#### from sklearn.metrics import precision_score
#### from sklearn.metrics import recall_score
#### from sklearn.metrics import f1_score

# import datafile

#### df = pd.read_csv('train.csv')
#### df.head()
![image](https://user-images.githubusercontent.com/48388697/153886397-bec1ec76-7338-4eb5-ac83-4e6e4d9615d8.png)

1. Exploratory Data Analysis
# get a summary of the dataset 

df.info()
![image](https://user-images.githubusercontent.com/48388697/153886837-54c88166-565f-4618-bfa9-567fda1b8269.png)
# Display summary statistics in a table

df.describe()
![image](https://user-images.githubusercontent.com/48388697/153886898-51abafda-2bea-45ad-b5ed-23fc38f94af6.png)

# Group numeric and categoric variables into separate tables

#### df_cat = df[['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage']]
#### df_num = df[['Response','Age','Region_Code','Annual_Premium','Vintage','Policy_Sales_Channel']]

#### for i in df_num.columns:
    #### plt.hist(df_num[i])
    #### plt.title(i)
    #### plt.show()
 ![image](https://user-images.githubusercontent.com/48388697/153887040-06246cdc-e295-4cbc-860e-5bd6a220b2e3.png)
   
 




