import numpy as np
import random as rn
import math

class Node(object):
    def __init__ (self, num_next_nodes):
        self.num_next_nodes = num_next_nodes
        self.weights = []
        self.error = 0

        for _ in range(0, num_next_nodes):
            self.weights.append(rn.uniform(-1,1))

    def output(self, node_input):
        outputs = []
        for weight in self.weights:
            outputs.append(weight * node_input)
        return outputs

    def calculate_error(self,previous_output, node_output, expected_output):
        self.error = (previous_output)*(expected_output - previous_output)*(previous_output - node_output)

    def change_weight(self, old_weight, learning_rate, connecting_node, error_output_previous):
        index = self.weights.index(old_weight)
        self.weights[index] = old_weight - learning_rate * connecting_node * error_output_previous

class Layer(object):

    def __init__ (self, num_nodes, num_next_nodes):
        self.nodes = []
        self.num_next_nodes = num_next_nodes

        # create the layer of Nodes
        for _ in range(0, num_nodes):
            self.nodes.append(Node(num_next_nodes))

    def calculate_activation(self, row_data):
        # add in the bias node
        row_data = np.append(-1, row_data)

        activation = []
        outputs = []
        total_output = 0

        # calculate outputs
        for node in self.nodes:
            for data in row_data:
                # this will return all the weighted values
                outputs.append(node.output(data))
        
        for index in range(0, self.num_next_nodes):
            total_output = 0
            for row_of_output in outputs:
                total_output += row_of_output[index]
                # sum all the outputs for the each node
            activation.append(self.sigmoid(total_output)) 

        return activation

    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))

class Network(object):

    def __init__ (self, num_inputs, num_hidden_layers, num_hidden_nodes, targets):
        self.layers = []
        self.targets = targets
        self.learning_rate = .1
        # add the input layer
        input_layer = Layer(num_inputs, num_hidden_nodes)
        self.layers.append(input_layer)

        hidden_layer = Layer(num_hidden_nodes, num_hidden_nodes)
        previous_layer = hidden_layer

        #loop through all the hidden layers
        for layer in range(num_hidden_layers):
            next_layer = Layer(num_hidden_nodes, num_hidden_nodes)
            self.layers.append(previous_layer)
            previous_layer = next_layer

        # add the output layer
        output_layer = Layer(num_hidden_nodes, len(targets))
        self.layers.append(output_layer)

    def predict(self, data_row):

        layer_input = data_row
        for layer in self.layers:
            layer_output = layer.calculate_activation(layer_input)
            layer_input = layer_output

        print "LAYER_OUTPUT: ", layer_output
        prediction = layer_output.index(max(layer_output))
        print "PREDICTION: ", prediction
        return self.targets[prediction]
