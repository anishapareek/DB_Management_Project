import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import time
import json
from os.path import abspath, join

dictionary = {}

np.random.seed(5)

features = pd.read_csv(abspath(join("setup", 'plants.output')))
num_of_columns = features.shape[1]
X = features.iloc[:, 1:num_of_columns-1]
Y = features.iloc[:, num_of_columns-1]

start = time.time()
iris = datasets.load_iris()
end = time.time()
data_loading_time = end - start
print("Data loading time: ", data_loading_time)

title = "6 clusters"
name = "k_means_6"

start = time.time()
est = KMeans(n_clusters=6)

fignum = 1
est.fit(X)
labels = est.labels_
end = time.time()
training_time = end-start
print("Training time: ", training_time)

start = time.time()
cluster_centers = est.cluster_centers_
MSE = 0
for index, a_point in enumerate(X):
    label = labels[index]
    center = cluster_centers[label]
    point = X.iloc[index, :]
    dist = (np.linalg.norm(point - center))
    MSE += dist
end = time.time()
evaluation_time = end-start
print("Evaluation time: ", evaluation_time)

print("SSE: ", MSE)

dictionary["DataLoadingTime"] = data_loading_time
dictionary["TrainingTime"] = training_time
dictionary["EvaluationTime"] = evaluation_time
dictionary["MeanSquaredError"] = MSE


with open(abspath(join("Scikit-learn", "plants_kmeans.json")), "w") as outfile:
    json.dump(dictionary, outfile)