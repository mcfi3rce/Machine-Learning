import numpy as np
import random as rn
import math

class Node(object):
    def __init__ (self, num_next_nodes, node_type):
        self.num_next_nodes = num_next_nodes
        self.weights = []
        self.node_type = node_type
        self.final_output = 0
        self.error = 0

        if (node_type != "output"):
            for _ in range(0, num_next_nodes):
                self.weights.append(rn.uniform(-1,1))
        else:
            for _ in range(0, num_next_nodes):
                self.weights.append(1)

    def __repr__(self):
        print "\n -- "
        print "|  |"
        print " -- "
        return ""

    def output(self, node_input):
        # this function doesn't save it's output for later
        # we need to do that for when we calculate error
        outputs = []
        total_output = 0
        for weight in self.weights:
            outputs.append(weight * node_input)

        return outputs

    def calculate_error(self,previous_output, node_output, expected_output):
        self.error = (previous_output)*(expected_output - previous_output)*(previous_output - node_output)

    def change_weight(self, old_weight, learning_rate, connecting_node, error_output_previous):
        index = self.weights.index(old_weight)
        self.weights[index] = old_weight - learning_rate * connecting_node * error_output_previous

class Layer(object):

    """
    Create the nodes taking into account the number of nodes in the next row
    The only thing it needs to know is how many nodes there are in the next layer
    """
    def __init__ (self, num_nodes, num_next_nodes, layer_type):
        self.nodes = []
        self.num_next_nodes = num_next_nodes
        self.layer_type = layer_type
        # create the layer of Nodes
        for _ in range(0, num_nodes):
            self.nodes.append(Node(num_next_nodes))

    def __repr__(self):
        print self.layer_type
        print self.nodes
        return ""

    """
       This function takes in input and depending on the number of nodes in the next layer
       creates outputs for each of those nodes.
    """
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
                print "OUPUT: ", node.output(data)
                outputs.append(node.output(data))
        
        for index in range(0, self.num_next_nodes):
            total_output = 0
            for row_of_output in outputs:
                total_output += row_of_output[index] 
                # sum all the outputs for the each node
            print "INDEX-NODE: ", index
            activation.append(self.sigmoid(total_output)) 

        return activation

class Network(object):

    def __init__ (self, num_inputs, num_hidden_layers, num_hidden_nodes, targets):
        self.layers = []
        self.targets = targets
        self.learning_rate = .1
        # add the input layer
        input_layer = Layer(num_inputs, num_hidden_nodes, "input")
        self.layers.append(input_layer)

        hidden_layer = Layer(num_hidden_nodes, num_hidden_nodes, "hidden")
        previous_layer = hidden_layer

        #loop through all the hidden layers
        for layer in range(num_hidden_layers):
            next_layer = Layer(num_hidden_nodes, num_hidden_nodes, "hidden")
            self.layers.append(previous_layer)
            previous_layer = next_layer

        # add the output layer
        output_layer = Layer(num_hidden_nodes, targets, "output")#len(targets), "output")
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

def sigmoid(value):
    return 1 / (1 + math.exp(-value))

