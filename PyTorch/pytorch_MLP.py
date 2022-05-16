import setup
import numpy as np
import torch
import time
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
import json

dictionary = {}


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


def pytorch_mlp(features):
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
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=2)

    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    model = Feedforward(len(feature_list), 10)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # model.eval()
    # y_pred = model(x_test)
    # before_train = criterion(y_pred.squeeze(), y_test)
    # print('Test loss before training', before_train.item())

    model.train()
    epoch = 20
    for epoch in range(epoch):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)

        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()
    end = time.time()
    training_time = end - start
    print("Training time: ", training_time)

    start = time.time()
    model.eval()
    y_pred = model(x_test)
    after_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss after Training', after_train.item())
    end = time.time()
    eval_time = end-start
    print("Evaluation time: ", eval_time)

    dictionary["DataLoadingTime"] = setup.data_loading_time
    dictionary["TrainingTime"] = training_time
    dictionary["EvaluationTime"] = eval_time

    with open("pytorch_abalone_multilayerperceptron.json", "w") as outfile:
        json.dump(dictionary, outfile)


if __name__ == '__main__':
    pytorch_mlp(setup.features)