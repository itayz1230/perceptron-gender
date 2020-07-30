# imports
import numpy as np
from openpyxl import Workbook, load_workbook
import pandas as pd

# the class that represent the model


class Perceptron(object):

    # setting the algorithm's parameters
    def __init__(self, no_of_inputs, epoch=200, learning_rate=0.001):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    # a helper function that guesses based on the weights
    def predict(self, inputs):
        sm = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if sm > 0:
            activation = 1
        else:
            activation = 0
        return activation

    # the training model, it runs {epoch} times and each times it iterates over
    # the training inputs and computes the new weights based on the gradient descent algorithm
    # the function set the weights to the Perceptron object
    def train(self, training_inputs, labels):
        # number of times it should run
        for _ in range(self.epoch):
            # each time - iterates over the training inputs and outputs
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1] += self.learning_rate * \
                    (label - prediction) * inputs[0]
                self.weights[2] += self.learning_rate * \
                    (label - prediction) * inputs[1]
                self.weights[0] += self.learning_rate * \
                    (label - prediction)


# getting info
x = pd.read_excel('X.xlsx').values
y = pd.read_excel('Y.xlsx').values
x = x[:490, :]
y = y[:490]
y.dtype = np.uint8
y = y.reshape(490, )

# create a new network
per = Perceptron(2)
# train the network so it will have the correct weights
per.train(x[:345, :], y[:345])
# calculate the success using total and suc
total = 0.0
suc = 0.0
for x, y in zip(x[345:, :], y[345:]):
    total += 1
    pre = per.predict(x)
    if pre == y:
        suc += 1
print(f"success percentage: {suc/total*100}")


# Enter a height and weight and get the result
# Enter the height of -1 to exit
height = float(input("Enter height: "))
weight = float(input("Enter weight: "))
while height != -1:
    if per.predict([height, weight]) == 0:
        print("Man")
    else:
        print("Lady")
    print("")
    height = float(input("Enter height: "))
    weight = float(input("Enter weight: "))
