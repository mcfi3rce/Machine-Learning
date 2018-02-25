import numpy as np
from neuron import Network
class NeuralNetClassifier(object):

    def __init__(self):
        pass

    def fit(self, data, targets):
        num_rows, num_columns = data.shape
        num_targets = np.unique(targets)
        num_hidden_layers = int(input("How many hidden layers?:"))
        num_hidden_nodes = int(input("How many nodes per layer?:"))
        return NeuralNetModel(num_columns, num_hidden_layers, num_hidden_nodes, targets)

class NeuralNetModel(object):

    def __init__(self, num_columns, num_hidden_layers, num_hidden_nodes, targets):
        self.num_columns = num_columns
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.network = Network(num_columns,num_hidden_layers,num_hidden_nodes, targets)

    def predict(self, data):
        predictions = []
        for data_row in data:
            predictions.append(self.network.predict(data_row))
