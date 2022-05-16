import pandas as pd
import numpy as np
import tensorflow as tf
from joblib.numpy_pickle_utils import xrange
import time
import json

dictionary = {}
start = time.time()
features = pd.read_csv('plants_out.data')
end = time.time()
data_loading_time = end-start
print("Data Loading time: ", data_loading_time)

start = time.time()
num_of_columns = features.shape[1]
points = features.iloc[:, 1:num_of_columns-1]
num_points = features.shape[0]
dimensions = num_of_columns-1


def input_fn():
    return tf.compat.v1.train.limit_epochs(
        tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)


num_clusters = 6
kmeans = tf.compat.v1.estimator.experimental.KMeans(
    num_clusters=num_clusters, use_mini_batch=False)

# train
num_iterations = 1
previous_centers = None
for _ in xrange(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    previous_centers = cluster_centers
    # print('score:', kmeans.score(input_fn))
end = time.time()
train_time = end-start
print("Training time: ", train_time)

start = time.time()
SSE = 0
MSE = 0
# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(points):
    cluster_index = cluster_indices[i]
    center = cluster_centers[cluster_index]
    a_point = points.iloc[i, :]
    dist = (np.linalg.norm(a_point - center))
    MSE += dist

end = time.time()
evaluation_time = end-start
print("Evaluation time: ", evaluation_time)

print("MSE: ", MSE)
dictionary["DataLoadingTime"] = data_loading_time
dictionary["TrainingTime"] = train_time
dictionary["EvaluationTime"] = evaluation_time
dictionary["MeanSquaredError"] = MSE

with open("keras_plants_kmeans.json", "w") as outfile:
    json.dump(dictionary, outfile)