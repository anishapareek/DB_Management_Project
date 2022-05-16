import pandas as pd
import numpy as np
import time
from os.path import abspath, join

start = time.time()
# Read in data and display first 5 rows
features = pd.read_csv(abspath(join("setup", "abalone.data")))
end = time.time()
data_loading_time = end-start
print("Data loading time: ", data_loading_time)

# adding header
headerList = ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight',
              'VisceraWeight', 'ShellWeight', 'Rings']

# converting data frame to csv
features.to_csv("abalone.data", header=headerList, index=False)

# Read in data and display first 5 rows
features = pd.read_csv('abalone.data')
# print(features.head(5))

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

