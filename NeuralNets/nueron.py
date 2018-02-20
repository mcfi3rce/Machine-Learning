import numpy as np
import random as rn

class NeuralNetworkClassifier:
    def __init__(self):
        pass

    def fit(self, data, target):
        # We want the number of columns in the data set
        num_rows, num_cols = data.shape
        # Creates a neural network with num_cols nodes
        neural_network = SinglePerceptron(data, num_cols, target)

        return NeuralNetworkModel(neural_network)


class NeuralNetworkModel:
    def __init__(self, model):
        self.model = model

    def prediNeuralNetworkModel(target)ct(self, data):
        # pass the values into the neural network
        

        return most_common_target


# A node which holds the weights between a neuron and the targets
class Node:
    def __init__(self, num_inputs, target):
        # The weights between this neuron and every target
        self.input_weights = []
        # This is the target for this node
        self.target = target
        # Accounts for a biased node
        self.bias = -1
        # Initially assigns random weights for each input and the biased node
        for _ in range(num_inputs + 1):
            self.input_weights.append(rn.uniform(-1, 1))
        # Add the output of the Node

    # Trains this vertices node to have correct weights
    def train(self, data_row, data_target):
        value = 0

        # Add the biased node
        value += self.bias * self.input_weights[0]

        # Gets the sum of the weights times the data input
        for index in range(len(data_row)):
            value += data_row[index] * self.input_weights[index + 1] #starts after the bias weight

        return value

# Holds an array of vertices between the data inputs and their targets
class SinglePerceptron:
    def __init__(self, data, num_cols, targets):
        # Will hold the vertices
        self.neural_array = []
        # All the rows of data
        self.inputs = data
        # Holds all the targets for each data row
        self.targets = targets
        # Holds only the unique targets
        self.unique_targets = np.unique(targets)
        # Creates a node for each unique target
        for unique_target in self.unique_targets:
            self.neural_array.append(Node(num_cols, unique_target))

    def calculate_output(self):
        values = []
        value, row = 0
        while row < len(data):
            for node in self.neural_array:
                value += node.train(data[row], targets[row])
                print "Value:", value
            values.append(value)
            row += 1
        




