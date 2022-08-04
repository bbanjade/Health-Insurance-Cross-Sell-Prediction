# Health-Insurance-Cross-Sell-Prediction

### 1. Context
#### Our client is an Insurance company, wants to increase its insurance sales by providing auto insurance to the current health insurance holder. They have some past data about demographics (gender, age, region code), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel), and their response to hold auto insurance. Based, on this data we need to create a model, that will predict customers will be willing to participate in purchasing health insurance in future. This project will visualize the provided data, create an algorithm that best represnt the data, and test the model in the test data.

#### 2. This project includes two parts: 1) Exploratory data analysis through visualization & 2) model building: data will be cleaned and prepared for modelling based on the visualizations.

#### Import all necessary models 

##### import numpy as np
##### import pandas as pd
##### import matplotlib.pyplot as plt
##### import seaborn as sns

##### from sklearn.model_selection import train_test_split

##### from sklearn.neighbors import KNeighborsClassifier
##### from sklearn.linear_model import LogisticRegression
##### from sklearn.svm import SVC
##### from sklearn.tree import DecisionTreeClassifier
##### from sklearn.ensemble import RandomForestClassifier
##### from sklearn.ensemble import GradientBoostingClassifier

##### from sklearn.model_selection import cross_val_score
##### from sklearn.metrics import confusion_matrix
##### from sklearn.metrics import accuracy_score
##### from sklearn.metrics import precision_score
##### from sklearn.metrics import recall_score
##### from sklearn.metrics import f1_score

#### import datafile

##### df = pd.read_csv('train.csv')
##### df.head()
![image](https://user-images.githubusercontent.com/48388697/153886397-bec1ec76-7338-4eb5-ac83-4e6e4d9615d8.png)

### 3. Exploratory Data Analysis
#### get a summary of the dataset 

##### df.info()
![image](https://user-images.githubusercontent.com/48388697/153886837-54c88166-565f-4618-bfa9-567fda1b8269.png)
#### Display summary statistics in a table

##### df.describe()
![image](https://user-images.githubusercontent.com/48388697/153886898-51abafda-2bea-45ad-b5ed-23fc38f94af6.png)

#### Group numeric and categoric variables into separate tables

##### df_cat = df[['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage']]
##### df_num = df[['Response','Age','Region_Code','Annual_Premium','Vintage','Policy_Sales_Channel']]

![image](https://user-images.githubusercontent.com/48388697/153985176-fbac37f9-3049-49de-b2bc-1ef4ae9966ff.png)

![image](https://user-images.githubusercontent.com/48388697/153887040-06246cdc-e295-4cbc-860e-5bd6a220b2e3.png)
![image](https://user-images.githubusercontent.com/48388697/153887508-4258b752-de34-4d7d-b48c-2101f8928428.png)
![image](https://user-images.githubusercontent.com/48388697/153887539-97fda65c-b6a5-468d-8cf3-da1d80e692da.png)
![image](https://user-images.githubusercontent.com/48388697/153887569-e62d3ba7-bf01-4b05-bc59-c4d4bf5e70bc.png)
![image](https://user-images.githubusercontent.com/48388697/153887588-d26a2315-c3c5-49d6-bb0c-990aa183473a.png)
![image](https://user-images.githubusercontent.com/48388697/153887607-d5d10db9-6d0f-4e2e-8a52-928e637393ea.png)

   
#### lets plot the correlation map
##### plt.figure(figsize = (8, 6))
#### creating mask
##### mask = np.triu(np.ones_like(df_num.corr()))
  
#### plotting a triangle correlation heatmap
##### dataplot = sns.heatmap(df_num.corr(), cmap="YlGnBu", annot=True, mask=mask)
  
#### displaying heatmap
##### plt.show()
![image](https://user-images.githubusercontent.com/48388697/153888136-d0928217-86f5-4308-9302-6fa32b0989e7.png)
#### All the correlation coefficients are low, so the correlation is not a problem in our data.
#### Find average values of variables based on response through pivot table

##### pd.pivot_table(df, index='Response', values=['Age','Region_Code','Annual_Premium','Vintage','Policy_Sales_Channel'])
![image](https://user-images.githubusercontent.com/48388697/153888281-4920ad53-b2a6-4e23-91b2-4d905c62e618.png)

![image](https://user-images.githubusercontent.com/48388697/153985099-0f08bc27-7a8c-47a9-a0e8-6d2730feeb9a.png)

    
![image](https://user-images.githubusercontent.com/48388697/153888359-e42b4685-7e17-461f-8f04-761d0e99a600.png)
![image](https://user-images.githubusercontent.com/48388697/153888383-1bced7a9-7c69-4419-a884-a704a01dcbc0.png)
![image](https://user-images.githubusercontent.com/48388697/153888417-3c514680-d418-4c7d-8137-9529b91a6da9.png)
![image](https://user-images.githubusercontent.com/48388697/153888448-87c9959f-9e46-4022-b46a-30effd6c07d8.png)
![image](https://user-images.githubusercontent.com/48388697/153888476-76b3a8cc-cc45-423d-bd99-9fee28b6717b.png)

#### Lets see if outliers is a problem in our data

##### for i in df_num.columns:
![image](https://user-images.githubusercontent.com/48388697/153985052-aa40c87d-139b-46dd-9306-3e28bfc3879d.png)

![image](https://user-images.githubusercontent.com/48388697/153888586-a64ced28-516a-482a-9a1f-5685c2994c2d.png)
![image](https://user-images.githubusercontent.com/48388697/153888639-c394c8f2-f1e5-497b-8260-c8e67a2c0b0c.png)
![image](https://user-images.githubusercontent.com/48388697/153888671-d99521e0-b872-4551-a28f-798ceceffdbe.png)
![image](https://user-images.githubusercontent.com/48388697/153888694-2f2c848b-1615-47b5-87d7-a28d282dc7af.png)
![image](https://user-images.githubusercontent.com/48388697/153888719-7c629231-39d1-40f9-9f0d-997bd3a7790f.png)
![image](https://user-images.githubusercontent.com/48388697/153888741-9ec3bb72-4f30-4f44-b901-ff5ce2a90098.png)

#### Age vs Previously Insured

##### sns.relplot(data=df,x='Age',y='Previously_Insured',kind='line')
![image](https://user-images.githubusercontent.com/48388697/153888785-50ac0332-1485-4ba1-a80e-73e37325ca29.png)

#### In our data, people around 30 and 80 years are more previously insured.

#### Relationship between Vehicle Age vs Vehicle Damage

##### sns.countplot(data=df,x='Vehicle_Age',hue='Vehicle_Damage')
![image](https://user-images.githubusercontent.com/48388697/153888878-eb60e0c4-83c5-44fb-9d8f-9c182e1ea3d0.png)

##### df1=df.groupby(['Vehicle_Age','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
##### df1

![image](https://user-images.githubusercontent.com/48388697/153888943-5b2f2161-881c-473d-bf53-c35bc048f592.png)
##### g = sns.catplot(x="Vehicle_Age", y="count",col="Response", data=df1, kind="bar", height=4, aspect=.7)
                
![image](https://user-images.githubusercontent.com/48388697/153889039-348b2134-3389-4986-b4b2-32218770350d.png)

##### df2=df.groupby(['Vehicle_Damage','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
##### df2
![image](https://user-images.githubusercontent.com/48388697/153889132-079735bb-2a52-4c29-a06a-ef1fab255e53.png)

##### g = sns.catplot(x="Vehicle_Damage", y="count",col="Response", data=df2, kind="bar",aspect=0.7)
                
![image](https://user-images.githubusercontent.com/48388697/153889223-8f4d7963-e6bb-4c68-96aa-fb29e69595a1.png)

### 4. Model Building
#### 4.1. Outliers are not big problem, but lets remove rows which have annual_premium higher than 400,000. By, doing this we are only loosing 6 rows.

##### df1 = df[df['Annual_Premium'] < 400000]
##### df1.head()
![image](https://user-images.githubusercontent.com/48388697/153889410-96797c59-35c9-450e-bf3c-5644cc242475.png)

#### 4.2. Convert character variables into dummy variables

##### df1 = pd.get_dummies(df1)
##### df1.head()
![image](https://user-images.githubusercontent.com/48388697/153889537-331d5a74-e699-4a9a-874a-f52f20d1a5d8.png)

#### 4.3. drop id, we don't have the use of this column

##### df1 = df1.drop('id', axis = 1)

#### 4.4. Lets prepare our data for model/algorithms

##### X = df1.drop(['Response'], axis=1)
##### y = df1['Response']
##### print(X.shape)
##### print(y.shape)

![image](https://user-images.githubusercontent.com/48388697/153889615-2f940458-59bd-4cd5-883d-f9c46b3a52c9.png)
#### 4.4.1. Split the data into train-test sets.

##### X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#### Lets run three algorithms (LogisticRegression, Random Forest, and GradientBoosting) based on the extent of data and simplicity and effectiveness for the models.

#### 4.5. LogisticRegression
##### logreg = LogisticRegression(C=1).fit(X_train, y_train)

##### print("logreg Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
##### print("logreg Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

#### 4.6. RandomForestClassifier
##### forest = RandomForestClassifier(n_estimators=100, random_state=0)
##### forest.fit(X_train, y_train)

##### print("forest Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
##### print("forest Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

#### 4.7. GradientBoostingClassifier
##### gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
##### gbrt.fit(X_train, y_train)

##### print("gbrt Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
##### print("gbrt Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

![image](https://user-images.githubusercontent.com/48388697/153889713-1cd010a7-a84c-4ccc-a3cb-d2275209d7db.png)

#### Based on accuracy score, the training and test scores are closer for logistic regression and gradient boosting, the randomFirest presented 1 training score and 0.867 test score representing overfitted model. So, based on accuracy score LogisticRegression and Gradient Bossting are better than Random Forest. But, our data is imbalanced, people who are willing to have health insurance are lower than who don' want. So lets try other metrics that gives us more confident to evaluate the chosen models.

#### 4.8. Grid Search
#### Lets use Grid Search to find the best parameters that improve the effective of the models.

##### from sklearn.model_selection import GridSearchCV

#### Create the parameter grid based on the results of random search 
##### param_grid = {
#####     'bootstrap': [True],
#####     'max_depth': [80, 90, 100, 110],
#####     'max_features': [2, 3],
#####     'min_samples_leaf': [3, 4, 5],
#####     'min_samples_split': [8, 10, 12],
#####     'n_estimators': [100, 200, 300, 1000]
##### }

#### Lets check if we can improve the efficacy of the RandomForest model

#### Create a based model
##### rf = RandomForestClassifier()
#### Instantiate the grid search model
##### grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

#### fit the model with gridsearch
##### grid_search.fit(X_train, y_train)
##### grid_search.best_params_

![image](https://user-images.githubusercontent.com/48388697/153889941-b39a5297-1b73-4c60-b229-3ece13be7040.png)

#### 4.9 now use the best parameters in RandomFirest Classifier.

##### forest = RandomForestClassifier(bootstrap = True, max_depth = 80, max_features = 2, min_samples_leaf = 5, min_samples_split = 10, n_estimators = 300)

##### forest.fit(X_train, y_train)

##### print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
##### print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

![image](https://user-images.githubusercontent.com/48388697/153890015-f69fa9a1-58e2-4a5e-a5ca-3d6ced79a7b3.png)
#### Grid search improved the traning-test accuracy score, so removing the overfitting model previously identified.

#### 4.10. grid search for logistic regression

##### logreg = LogisticRegression()
##### grid_values = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.009, 0.01, 0.09, 1, 5, 10, 25]}
##### grid_logreg = GridSearchCV(logreg, param_grid = grid_values,scoring = 'recall')
##### grid_logreg.fit(X_train, y_train)
![image](https://user-images.githubusercontent.com/48388697/153890124-91d0c622-4728-4b35-99c8-1b962e2025f0.png)

#### 4.11. Lets create classification report for three models

##### from sklearn.metrics import classification_report


#### create predicted values for logisticRegression
##### pred_logreg = grid_logreg.predict(X_test)

##### print("LogisticRegression ClassificationReport")
##### print(classification_report(y_test, pred_logreg, target_names=["1", "0"]))

#### create predicted values for RandomForest
##### pred_forest = forest.predict(X_test)

##### print("Random forest ClassificationReport")
##### print(classification_report(y_test, pred_forest,
                            target_names=["1", "0"]))

#### For run time, lets use the simple model of gradientBoosting Classifier earlier developed.
#### create predicted values from gradientboosting
##### pred_gbrt = gbrt.predict(X_test)

##### print("gradient boosting ClassificationReport")
##### print(classification_report(y_test, pred_gbrt, target_names=["1", "0"]))

![image](https://user-images.githubusercontent.com/48388697/153890213-999dd373-cdc8-46b0-a6a4-52469bd4cd16.png)

#### Our main interest is to find people with response 1, and capture all the possible 1 response (i.e., people buying health insurance). So Recall is the most important number. All three models LogisticRegression, RandomForest, & GradientBoosting have high in recall, precision values, and f1 scores. All three models have similar values. All of these models are good. We can choose logistic regression due to easy and fast to run the model.

















 








