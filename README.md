# Titanic - data science project 

### Introduction
This is simple data science project designed to predict whether a given person would survive the Titanic crash.

### Code structure
This repository includes:
- 2 jupyter notebooks to run the code to train & predict the data
- data/ : folder with dataset saved in csv format
- src/ : folder with all necessary classes and functions

### Data
There are 2 csv files containing Titanic data:<br>
__train.csv__ with 802 records and 12 columns <br>
__val.csv__ with 89 records and 12 columns
Both datasets reveal whether a passanger survived or not.
<br>

Columns description:
- __passengerId__ : passenger identification number for this dataset
- __survival__ : survival outcome (0 = No, 1 = Yes)
- __pclass__ : ticket class
- __name__ : title and name
- __sex__ :	gender
- __age__ :	age in years	
- __sibsp__ : # of siblings / spouses aboard the Titanic	
- __parch__	: # of parents / children aboard the Titanic	
- __ticket__ : ticket number	
- __fare__ : passenger fare	
- __cabin__ : cabin number	
- __embarked__ : port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

### ML algorithm
RandomForestClassifier from __sklearn__ library was chosen as a machine learning algorithm to solve given problem in this project.

### Install
This project requires Python 3 and the following Python libraries installed:
- Pandas
- sklearn
- pickle

### Run
In order to run the code to train the model open a notebook __Titanic_train_data.ipynb__, set the necessary variables and click Run All.<br>

In order to run the code to get the predictions open a notebook __Titanic_predict_data.ipynb__, set the necessary variables and click Run All. (Prerequisite: run Titatnic_train_data.ipynb first to train the model used for predictions)