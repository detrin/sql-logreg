# sql-logreg
Logistic Regression with PostgreSQL

## Description
This project is a simple implementation of Logistic Regression using PostgreSQL. The goal is to predict whether the qualitz of wine is good or bad based on the wine's chemical properties. The dataset used in this project is from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

## WHY?
SQL is not meant to be used for ML, however SQL is turing complete and can be used to implement ML algorithms. This project should not encourage you to use SQL for ML, but rather to show you that it is possible to do so. I personally support to use the right tool for the right job. If you want to do ML, use Python for models outside database. In case that the model is simple and let's say you want to do a prediction in a database (because the approval proces in your company will literally take months to get a new tool), then you can use SQL to do so.

## How to run
In the project are several parts:
1. Out of the box implementation of Logistic Regression from sklearn
2. Implementation of Logistic Regression in python from scratch
3. Implementation of Logistic Regression in SQL from scratch

Create a virtual environment and install the requirements:
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Make sure that you have postgres installed and running. 

## Dependencies
Code was developed and tested with the following versions:
Python 3.11.5 (main, Aug 24 2023, 15:09:45) [Clang 14.0.3 (clang-1403.0.22.14.1)]
postgres (PostgreSQL) 14.9 (Homebrew)
```
scikit-learn==1.3.1
numpy==1.26.0
pandas==2.1.1
tqdm==4.66.1
polars==0.19.5
matplotlib==3.8.0
matplotlib-inline==0.1.6
psycopg2==2.9.8
imbalanced-learn==0.11.0
python-dotenv==1.0.0
```