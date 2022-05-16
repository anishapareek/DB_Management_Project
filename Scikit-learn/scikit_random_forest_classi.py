import pandas as pd
import numpy as np
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
import setup
import time
import json
from os.path import abspath, join

dictionary = {}


def random_forest_scikit(features):
    # # Display the first 5 rows of the last 12 columns
    # print(features.iloc[:, :].head(5))

    start = time.time()
    # Labels are the values we want to predict
    labels = np.array(features['Rings'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features = features.drop('Rings', axis=1)

    # Saving feature names for later use
    feature_list = list(features.columns)

    # Convert to numpy array
    features = np.array(features)

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                labels, test_size=0.25, random_state=2)

    # The baseline predictions are the historical averages
    baseline_preds = test_features[:, feature_list.index('Diameter')]

    # Baseline errors, and display average baseline error
    baseline_errors = abs(baseline_preds - test_labels)
    print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=2)

    # Train the model on training data
    rf.fit(train_features, train_labels)
    end = time.time()
    train_time = end-start
    print("Training time: ", train_time)

    start = time.time()
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    end = time.time()
    eval_time = end-start
    print("Evaluation time: ", eval_time)
    dictionary["DataLoadingTime"] = setup.data_loading_time
    dictionary["TrainingTime"] = train_time
    dictionary["EvaluationTime"] = eval_time
    dictionary["MeanAbsoluteError"] = round(np.mean(errors), 2)
    dictionary["Accuracy"] = round(accuracy, 2)
    dictionary["AverageBaselineError"] = round(np.mean(baseline_errors), 2)

    with open(abspath(join("Scikit-learn", "abalone_randomforest.json")), "w") as outfile:
        json.dump(dictionary, outfile)


if __name__ == '__main__':
    random_forest_scikit(setup.features)



