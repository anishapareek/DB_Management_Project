import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time
import json
from os.path import abspath, join

dictionary = {}
train_file_path = abspath(join("setup", "Features_Variant_1.csv"))
# train_file_path = '/Users/anishapareek/Desktop/MS_CS/3rd_sem/CSCI 729/Project/Dataset/Training/Features_Variant_5.csv'
test_file_path = abspath(join("setup", "Features_TestSet.csv"))

start = time.time()
# df1 = pd.read_csv('Features_Variant_1.csv')
df = pd.read_csv(train_file_path)
end = time.time()
data_loading_time = end-start
print("Data loading time: ", data_loading_time)

start = time.time()
num_of_columns = df.shape[1]
X = df.iloc[:, :num_of_columns - 1]
y = df.iloc[:, num_of_columns - 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)
end = time.time()
training_time = end-start
print("Training time: ", training_time)

print(regr.score(X_test, y_test))

start = time.time()
y_pred = regr.predict(X_test)
end = time.time()
evaluation_time = end-start
print("Evaluation time: ", evaluation_time)

dictionary["DataLoadingTime"] = data_loading_time
dictionary["TrainingTime"] = training_time
dictionary["EvaluationTime"] = evaluation_time

with open(abspath(join("Scikit-learn", "facebook_linearregression.json")), "w") as outfile:
    json.dump(dictionary, outfile)