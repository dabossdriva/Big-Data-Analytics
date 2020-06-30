"""
Odessa Elie
May 2020

The purpose of this project is to use machine learning algorithms to solve a research problem.
I am investigating factors affecting COVID-19 outcomes in different countries and US states.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

#DECISION TREE - factors that influence whether a country decides to impose restrictions
def decision_tree():
    
    #import the dataset 
    filename = "countries.csv"

    #prints the number of rows and columns and display the top 5 rows
    dataset = pd.read_table(filename,sep=',', header=0, encoding= 'unicode_escape')

    # Printing the dataset details  
    print ("\nDataset Details (instances, attributes): ", dataset.shape) 
        
    # Printing the first 5 instances of the dataset 
    print ("\nDataset sample: \n",dataset.head())

    #divide the data into attributes and labels
    X = dataset.drop('RestrictionsFlg', axis=1)
    y = dataset['RestrictionsFlg']

    #split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

    #training the algorithm
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    #make predictions on the test data
    y_pred = classifier.predict(X_test)
    
    # get importance
    importance = classifier.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()
    
    #classification report and confusion matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


#MULTIPLE LINEAR REGRESSION CODE - predict number of deaths in a country
def linear_regression_deaths():

    #import the dataset 
    filename = "countries.csv"

    #prints the number of rows and columns and display the top 5 rows
    dataset = pd.read_table(filename,sep=',', header=0, encoding= 'unicode_escape')

    # Printing the dataset details  
    print ("\nDataset Details (instances, attributes): ", dataset.shape) 
        
    # Printing the first 5 instances of the dataset 
    print ("\nDataset sample: \n",dataset.head())
      
    #divide the data into attributes and labels
    X = dataset.drop('TotalDeaths', axis=1)
    y = dataset['TotalDeaths']

    #split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=9)

    #create an instance of the linear regression model
    lin_reg_mod = LinearRegression()

    #fit the model on the training data
    lin_reg_mod.fit(X_train, y_train)

    #make predictions on the test set
    pred = lin_reg_mod.predict(X_test)

    #find R^2 and RMSE
    test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))

    test_set_r2 = r2_score(y_test, pred)

    #print the metrics - you want a high R^2 and a low RMSE
    print("Root Mean Squared Error: ",test_set_rmse)
    print("R-squared value: ",test_set_r2)
    print("Intercept = ", lin_reg_mod.intercept_)
    # linear regression feature importance
    # get importance
    
    importance = lin_reg_mod.coef_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

#MULTIPLE LINEAR REGRESSION - predict the number of recoveries
def linear_regression_recoveries():

    #import the dataset 
    filename = "countries.csv"

    #prints the number of rows and columns and display the top 5 rows
    dataset = pd.read_table(filename,sep=',', header=0, encoding= 'unicode_escape')

    # Printing the dataset details  
    print ("\nDataset Details (instances, attributes): ", dataset.shape) 
        
    # Printing the first 5 instances of the dataset 
    print ("\nDataset sample: \n",dataset.head())
      
    #divide the data into attributes and labels
    X = dataset.drop('TotalRecovered', axis=1)
    y = dataset['TotalRecovered']

    #split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)

    #create an instance of the linear regression model
    lin_reg_mod = LinearRegression()

    #fit the model on the training data
    lin_reg_mod.fit(X_train, y_train)

    #make predictions on the test set
    pred = lin_reg_mod.predict(X_test)

    #find R^2 and RMSE
    test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))

    test_set_r2 = r2_score(y_test, pred)

    #print the metrics - you want a high R^2 and a low RMSE
    print("Root Mean Squared Error: ",test_set_rmse)
    print("R-squared value: ",test_set_r2)
    print("Intercept = ", lin_reg_mod.intercept_)
    # linear regression feature importance
    # get importance
    importance = lin_reg_mod.coef_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

#LOGISTIC REGRESSION - classify whether restrictions were imposed or not
def logistic_regression_model():

    #import the dataset 
    filename = "countries.csv"

    #prints the number of rows and columns and display the top 5 rows
    dataset = pd.read_table(filename,sep=',', header=0, encoding= 'unicode_escape')

    # Printing the dataset details  
    print ("\nDataset Details (instances, attributes): ", dataset.shape) 
        
    # Printing the first 5 instances of the dataset 
    print ("\nDataset sample: \n",dataset.head())
      
    #divide the data into attributes and labels
    X = dataset.drop('RestrictionsFlg', axis=1)
    y = dataset['RestrictionsFlg']

    #split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)

    LR = LogisticRegression(random_state=0, max_iter=500, solver='lbfgs', multi_class='ovr').fit(X, y)
    
    #make predictions on the test set
    pred = LR.predict(X_test)

    #find R^2 and RMSE
    test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))

    test_set_r2 = r2_score(y_test, pred)

    #print the metrics - you want a high R^2 and a low RMSE
    print("Root Mean Squared Error: ",test_set_rmse)
    print("R-squared value: ",test_set_r2)
    print("Intercept = ", LR.intercept_)
    # get importance
    importance = LR.coef_[0]
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

#K-MEANS CLUSTERING - find clusters among countries based on smoking data and coronavirus deaths
def cluster_countries():

    #import the dataset 
    filename = "countries.csv"

    #prints the number of rows and columns and display the top 5 rows
    dataset = pd.read_table(filename,sep=',', header=0, encoding= 'unicode_escape')

    # Printing the dataset details  
    print ("\nDataset Details (instances, attributes): ", dataset.shape) 
        
    # Printing the first 5 instances of the dataset 
    print ("\nDataset sample: \n",dataset.head())
      
    #divide the data into attributes and labels
    #X = dataset.drop('Country', axis=1) smoking and deaths
    X=dataset.iloc[:, [23,28]].values
    y = dataset['Country']
    
    #Visualize the data
    pyplot.scatter(X[:,0],X[:,1], label='True Position')
    pyplot.show()

    #split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state=9)
    
    #create cluster with 3 clusters
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    #centroid values
    print(kmeans.cluster_centers_)

    #print the data labels
    print(kmeans.labels_)

    #visualize the cluster
    pyplot.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
    #make the centroids show in black
    pyplot.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
   
    pyplot.title('Clusters of Countries')
    pyplot.xlabel('Smoking')
    pyplot.ylabel('Death Rate')
    pyplot.show()
    
#K-MEANS CLUSTERING - find clusters based on death rate and number of ventilators
def cluster_states():

    #import the dataset 
    filename = "states.csv"

    #prints the number of rows and columns and display the top 5 rows
    dataset = pd.read_table(filename,sep=',', header=0, encoding= 'unicode_escape')

    # Printing the dataset details  
    print ("\nDataset Details (instances, attributes): ", dataset.shape) 
        
    # Printing the first 5 instances of the dataset 
    print ("\nDataset sample: \n",dataset.head())
      
    #divide the data into attributes and labels
    #X = dataset.drop('Country', axis=1) smoking and deaths
    X=dataset.iloc[:, [9,10]].values
    y = dataset['state']
    
    #Visualize the data
    pyplot.scatter(X[:,0],X[:,1], label='True Position')
    pyplot.show()

    #split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state=9)
    
    #create cluster with 3 clusters
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)

    #centroid values
    print(kmeans.cluster_centers_)

    #print the data labels
    print(kmeans.labels_)

    #visualize the cluster
    pyplot.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
    #make the centroids show in black
    pyplot.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    #pyplot.show()
   
    pyplot.title('Clusters of US States')
    pyplot.xlabel('Ventilators')
    pyplot.ylabel('Death Rate')
    pyplot.show()

#MAIN MENU
def main():
    #menu can be altered with a loop. This version does all 6 options in succession.
    print("1. Finding the most significant factors influencing whether a country decides to impose restrictions\n")
    decision_tree()
    print("2. Predicting number of deaths\n")
    linear_regression_deaths()
    print("3. Predicting number of recoveries\n")
    linear_regression_recoveries()
    print("4. Classify whether a country decides to impose restrictions\n")
    logistic_regression_model()
    print("5. Find clusters based on smoking and coronavirus deaths around the world\n")
    cluster_countries()
    print("6. Find clusters based on ventilators and coronavirus death rate in the US\n")
    cluster_states()   

if __name__ == "__main__":
    main()
