from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import time
import json

dictionary = {}
train_file_path = "Features_Variant_1.csv"
test_file_path = "Features_TestSet.csv"

start = time.time()
dataframe = read_csv(train_file_path)
end = time.time()
data_loading_time = end-start
print("Data loading time: ", data_loading_time)


start = time.time()
num_of_columns = dataframe.shape[1]
X = dataframe.iloc[:, :num_of_columns - 1]
Y = dataframe.iloc[:, num_of_columns - 1]


def baseline_model():
    model = Sequential()
    model.add(Dense(5, input_dim=num_of_columns - 1, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=1, batch_size=5, verbose=0)
kfold = KFold(n_splits=5)

results = cross_val_score(estimator, X, Y, cv=kfold)
mse = results.std()/10
print("Mean Squared Error: %.2f" % (results.std()))

end = time.time()
train_time = end-start
print("Train time: ", train_time)

dataframe = read_csv(test_file_path)
num_of_columns = dataframe.shape[1]
X = dataframe.iloc[:, :num_of_columns - 1]
Y = dataframe.iloc[:, num_of_columns - 1]
start = time.time()
estimator = KerasRegressor(build_fn=baseline_model, epochs=1, batch_size=5, verbose=0)
kfold = KFold(n_splits=5)

results = cross_val_score(estimator, X, Y, cv=kfold)
end = time.time()
eval_time = end-start
print("Evaluation time: ", eval_time)

dictionary["DataLoadingTime"] = data_loading_time
dictionary["TrainingTime"] = train_time
dictionary["EvaluationTime"] = eval_time
dictionary["MeanSquaredError"] = mse

with open("Keras_facebook_linearregression.json", "w") as outfile:
    json.dump(dictionary, outfile)
