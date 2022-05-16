import torch
from torch.autograd import Variable
import pandas as pd
import time
import json

dictionary = {}
train_file_path = "Features_Variant_1.csv"
test_file_path = "Features_TestSet.csv"

start = time.time()
df = pd.read_csv(train_file_path)
end = time.time()
data_loading_time = end - start
print("Data loading time: ", data_loading_time)

start = time.time()
num_of_columns = df.shape[1]
x_data = df.iloc[:, :num_of_columns - 1].astype(float)
x_data = torch.tensor(x_data.values).float()
y_data = df.iloc[:, num_of_columns - 1].astype(float)
y_data = torch.tensor(y_data.values).float()


class LinearRegressionModel(torch.nn.Module):

	def __init__(self):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(num_of_columns-1, 1) # One in and one out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

# our model
our_model = LinearRegressionModel()
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)

for epoch in range(1):
	pred_y = our_model(x_data)
	loss = criterion(pred_y, y_data)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
end = time.time()
training_time = end - start
print("Training time: ", training_time)

start = time.time()
df_test = pd.read_csv(test_file_path)
num_of_columns = df_test.shape[1]
x_data = df_test.iloc[:, :num_of_columns - 1].astype(float)
x_data = torch.tensor(x_data.values).float()
y_data = df_test.iloc[:, num_of_columns - 1].astype(float)
y_data = torch.tensor(y_data.values).float()
pred_y = our_model(x_data)
loss = criterion(pred_y, y_data)
end = time.time()
evaluation_time = end-start
print("Evaluation time: ", evaluation_time)

dictionary["DataLoadingTime"] = data_loading_time
dictionary["TrainingTime"] = training_time
dictionary["EvaluationTime"] = evaluation_time

with open("PyTorch_facebook_linearregression.json", "w") as outfile:
    json.dump(dictionary, outfile)