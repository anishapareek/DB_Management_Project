import os
import matplotlib.pyplot as plt
from collections import  namedtuple
import json
import numpy as np
import pandas as pd
from pathlib import Path

DataPoint = namedtuple('DataPoint', ['name', 'value'])

filenames = {
    "multilayerperceptron": [],
    "randomforest": [],
    "stochasticdualcoordinateascent": [],
    "linearregression": [],
    "k-means": [],
}

prettier_names = {
    "multilayerperceptron": "Multilayer Perceptron",
    "randomforest": "Random Forest",
    "stochasticdualcoordinateascent": "Stochastic Ddual Coordinate Ascent",
    "linearregression": "Linear Regression",
    "k-means": "K-Means",
}

implementations = [ "Spark", "SparkHive", "Keras", "Scikit-learn", "Pytorch", "Matlab", "MLNET" ]

values = {
    "multilayerperceptron": { "DataLoadingTime": [], "TrainingTime": [], "EvaluationTime": [], "Accuracy": [], "F1Score": [] },
    "randomforest": { "DataLoadingTime": [], "TrainingTime": [], "EvaluationTime": [], "Accuracy": [], "F1Score": [] },
    "stochasticdualcoordinateascent": { "DataLoadingTime": [], "TrainingTime": [], "EvaluationTime": [], "MicroAccuracy": [], "MacroAccuracy": [], "LogLoss": [], "LogLossReduction": [] },
    "linearregression": { "DataLoadingTime": [], "TrainingTime": [], "EvaluationTime": [], "MeanSquaredError": [], "MeanAbsoluteError": [], "RSquared": [], "RootMeanSquaredError": [] },
    "k-means": { "DataLoadingTime": [], "TrainingTime": [], "EvaluationTime": [], "AverageDistance": [] },
}

for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        for model_name in filenames:
            if model_name in file:
                filenames[model_name].append(os.path.join(root, file))

for model_name, results in filenames.items():
    for result in results:
        f = open(result)
        file_values = json.load(f)

        for implementation in implementations:
            path = os.path.normpath(result.lower())
            paths = path.split(os.sep)
            if (implementation.lower() in paths):
                for measurement in values[model_name]:
                    if (measurement in file_values):
                        values[model_name][measurement].append(DataPoint(implementation, file_values[measurement]))

def parse_points(points):
    names = []
    vals = []
    for point in points:
        names.append(point.name)
        vals.append(round(point.value, 3))

    return names, vals

for model_name, measurements in values.items():
    for measurement, points in measurements.items():
        names, vals = parse_points(points)

        title = prettier_names[model_name] + ", " + measurement
        plt.figure()
        plt.title(title)
        plt.bar(names, vals)
        plt.savefig(title)

for model_name, measurements in values.items():
    title = prettier_names[model_name] + ", Time Measurements"

    data_points = measurements["DataLoadingTime"]
    names, data_vals = parse_points(data_points)

    training_points = measurements["TrainingTime"]
    names, training_vals = parse_points(training_points)

    eval_points = measurements["EvaluationTime"]
    names, eval_vals = parse_points(eval_points)

    df = pd.DataFrame([["DataLoading"] + data_vals, ["Training"] + training_vals, ["Evaluation"] + eval_vals],
                  columns=["Frameworks"] + names)

    ax = df.plot(x='Frameworks',
        kind='bar',
        stacked=False,
        title=title, 
        rot=0,
        figsize=(13.5, 7.5))

    for container in ax.containers:
        ax.bar_label(container)

    plt.savefig(title, dpi=100)