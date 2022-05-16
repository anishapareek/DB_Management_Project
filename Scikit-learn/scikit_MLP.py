from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time
import json
from os.path import abspath, join

dictionary = {}


def do_mlp():
    start = time.time()
    features = pd.read_csv(abspath(join("setup", "abalone.data")))
    end = time.time()
    data_loading_time = end-start
    print("Data loading time: ", data_loading_time)

    start = time.time()
    num_of_columns = features.shape[1]
    X = features.iloc[:, 1:num_of_columns-1]
    y = features.iloc[:, num_of_columns-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=1000, activation='relu')
    mlp.fit(X_train, y_train)
    end = time.time()
    train_time = end-start
    print("Train time: ", train_time)

    start = time.time()
    pred = mlp.predict(X_test)
    print(confusion_matrix(y_test, pred))

    print(accuracy_score(y_test, pred))
    end = time.time()
    eval_time = end-start
    print("Evaluation time: ", eval_time)
    dictionary["DataLoadingTime"] = data_loading_time
    dictionary["TrainingTime"] = train_time
    dictionary["EvaluationTime"] = eval_time

    with open(abspath(join("Scikit-learn", "abaolone_multilayerperceptron.json")), "w") as outfile:
        json.dump(dictionary, outfile)


if __name__ == '__main__':
    do_mlp()
