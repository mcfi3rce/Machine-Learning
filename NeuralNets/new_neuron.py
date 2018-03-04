import numpy as np
import random
import math
import matplotlib.pyplot as plot

class Neuron():
    def __init__(self, previous_layer_nodes):
        # create weights
        self.input_weights = [random.uniform(-1.0, 1.0) for _ in range(0, previous_layer_nodes)]
        self.activation = 0
        #each node has it's own error we will save this for later
        self.error = 0
    """------------------------------------------------------------------------------------------------
    output()
    Account for bias
    ------------------------------------------------------------------------------------------------"""
    def output(self, results):
        # account for bias inputs
        biased_array = np.append(-1, inputs)

        return self.calculate_activation(biased_array)

    """------------------------------------------------------------------------------------------------
    sigmoid()
    Activation function that squashes the data 
    ------------------------------------------------------------------------------------------------"""    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))

    """------------------------------------------------------------------------------------------------
    calculate_activation()
    This function calculates the activation for a single node it uses the weights of the previous layer
    of nodes and then saves the activation for later
    ------------------------------------------------------------------------------------------------"""
    def calculate_activation(self, inputs):
        new_results = map(lambda weight, input: weight * input, self.input_weights, inputs)

        activation = 0


        # calculate the activation with the given inputs
        for result in new_results:
            activation += result

        return self.sigmoid(activation)


class NeuralNetwork():
    def __init__(self):
        pass

      def fit(self, data, targets, classes):
        self.data = data
        self.targets = targets
        self.classes = classes
        self.inputs = self.data.shape[1]
        self.num_epochs = -1
        self.num_hidden = -1
        self.features = self.data.shape[1]

        # prompt the user for hidden layers and nodes
        self.obtain_information()
        self.build_network()
        self.train()

    """-------------------------------------------------------------------------
    train()
    Where the magic happens, this is where we update the weights based on error and
    do all the back propogation.
    ------------------------------------------------------------------------------------------------"""
    def train(self):
        accuracy = []

        #keep track of the final epoch's accuracy
        final_epoch_accuracy = None

        for _ in range(0, self.num_epochs):
            single_accuracy = []
            final_epoch_accuracy =0
            count = 0

            for(data_point, target) in zip(self.data, self.targets):
                results = self.get_results(data_point)

                # the last value is the prediction
                single_accuracy.append(np.argmax(results[-1]))

                #update the network after we get the results
                self.update_network(data_point, target, results)

            # calculate accuracy for one epoch
            for (index, prediction) in enumerate(predictions):
                if (self.targets[index] == prediction):
                    count += 1

            # store final epoch accuracy
            final_epoch_accuracy = get_accuracy(count, len(self.targets))

            accuracy.append(final_epoch_accuracy)

        # graph the results
        self.plot_graph(accuracy)
        print("Accuracy: {:.2f}%").format(final_epoch_accuracy)


    """------------------------------------------------------------------------------------------------
    predict
    calls the predict_class and appends the results of the function call to the prediction array
    ------------------------------------------------------------------------------------------------"""
    def predict(self, data):
        model = []

        for item in data:
            model.append(self.predict_class(item))

        return model

    """------------------------------------------------------------------------------------------------
    predict_class
    get the results and return the argmax of the results array
    ------------------------------------------------------------------------------------------------"""
    def predict_class(self, item):
        results = self.get_results(item)

        # return the max index
        return np.argmax(results[-1])

    """------------------------------------------------------------------------------------------------
    plot_graph
    standard plotting function. takes in an array and plots each element of the array into a graph
    ------------------------------------------------------------------------------------------------"""
    def plot_graph(self, accuracy):
        print("Accuracy", accuracy)
        plot.plot(accuracy)
        plot.title("Training Accuracy - MLP")
        plot.xlabel("# of epochs")
        plot.ylabel("Accuracy")
        plot.legend([accuracy], 'Iris')
        plot.show()


    """------------------------------------------------------------------------------------------------
    build_network
    build all the layers of the neural network
    ------------------------------------------------------------------------------------------------"""
    def build_network(self):
        # build layers
        self.layers = []

        # account for bias
        total_size = self.num_hidden + 1

        """ A layer needs two things
            1) Number of nodes in the previous layer to create weights
            2) the number of nodes needed in that layer
        """
        for index in range(0, total_size):
            layer = self.build_layer(self.features(index), self.get_num_nodes(index, self.num_hidden))
            self.layers.append(layer)

    """------------------------------------------------------------------------------------------------
    build_layer
    build all the layers for a specified number of nodes
    ------------------------------------------------------------------------------------------------"""
    def build_layer(self, features_size, num):
        node_layer = []

        for _ in range(0, num):
            node_layer.append(Neuron(features_size))

        return node_layer

    """------------------------------------------------------------------------------------------------
    feature_size
    a getter for the the number of nodes for the previous layer
    ------------------------------------------------------------------------------------------------"""
    def feature_size(self, index):
        # error checking on the input layer
        if (index > 0):
            reutrn len(self.layers[index - 1])
        else:
            return self.features

    """------------------------------------------------------------------------------------------------
    get_num_nodes
    Prompt the user for the number of nodes in each hidden layer
    ------------------------------------------------------------------------------------------------"""
    def get_num_nodes(self, index, layerNum):
        numNode = -1

        if (index < layerNum):
            try:
                print("How many nodes for Layer {}?".format(layerNum + 1))
                numNode = int(input("> "))

                if (numNode < 1):
                    raise Exception("Please choose more than one node next time.")

            except (ValueError) as err:
                print("ERROR: {}".format(err))

            return numNode
        else:
            return len(self.classes)

    """------------------------------------------------------------------------------------------------
    update_network
    After an epoch is completed. Go through and update the neural network
    ------------------------------------------------------------------------------------------------"""
    def update_network(self, data, target, results):
        # CALCULATE ERROR FIRST
        self.update_errors(target, results)
        # then update the weights
        self.update_weights(data, results)

    """------------------------------------------------------------------------------------------------
    update_errors
    updates the errors for each node in a layer
    ------------------------------------------------------------------------------------------------"""
     def update_errors(self, target, results):
        for index_layer, layer in reversed(list(enumerate(self.layers))):
            for index_node, node in enumerate(layer):
                # update the error for each node
                node.error = self.get_error_node(index_node, index_layer, target, results)

    """------------------------------------------------------------------------------------------------
    get_error_node
    updates the error for a single node
    ------------------------------------------------------------------------------------------------"""
    def get_error_node(self, index_node, index_layer, target, results):
        node_result = None
        hidden_weight = None
        hidden_error = None

        # call the hidden layer error calculation first because it is different than
        if (index_layer < len(results) - 1):
            #get the specific result from the result array to calculate the errors
            node_result = results[index_layer][index_node]

            # call getters to get info
            hidden_weight = self.get_hidden_node_weights(index_node, index_layer)
            hidden_error = self.get_hidden_error(index_layer)

            return self.get_hidden_error(node_result, hidden_weight, hidden_error)
        # case for the output layers
        else:
            node_result = results[index_layer][index_node]

            # part of the algorithm is to only change nodes that don't give the correct
            # output so we can check this here
            is_match = index_node == target

    """------------------------------------------------------------------------------------------------
    get_hidden_node_error
    this is how we differentiate between the layers without using abstraction we just use a differentiate
    we just use a different function
    ------------------------------------------------------------------------------------------------"""
    def get_hidden_node_error(self, index_layer):
        array_hidden_error = []

        for node in self.layers[index_layer + 1]:
            array_hidden_error.append(node.error)

        return array_hidden_error

    """------------------------------------------------------------------------------------------------
    get_hidden_node_weights
    retrieve all the weights for a node in the hidden layer
    ------------------------------------------------------------------------------------------------"""
    def get_hidden_node_weights(self, index_node, index_layer):
        array_hidden_weights = []

        for node in self.layers[index_layer + 1]:
            array_hidden_weights.append(node.weights[index_node])

        return array_hidden_weights

    """------------------------------------------------------------------------------------------------
    get_hidden_node_weights
    retrieve all the weights for a node in the hidden layer
    ------------------------------------------------------------------------------------------------"""
    def get_hidden_error(self, result, hidden_weights, hidden_error)
        total = 0

        for (index, weight) in enumerate(hidden_weights):
            total += (weight * hidden_error[index])

        return result * (1 - result) * total

    """------------------------------------------------------------------------------------------------
    get_output_error
    normal calculation of error for the output and input layer
    ------------------------------------------------------------------------------------------------"""
    def get_output_error(self, result, target):
        return result * (1 - result) * (result - target)

    """------------------------------------------------------------------------------------------------
    update_weights
    update the weights for all layers we are going from input layer to output layer
    ------------------------------------------------------------------------------------------------"""
    def update_weights(self, data, results):
        for index, layer in enumerate(self.layers):
            for node in layer:
                #check to see if we are at the input layer
                if index > 0:
                    self.update_node_weights(node, results[index - 1])
                # this is the input layer so convert the input layer to alist for convenience
                # so we can account for bias. The results alreaddy have it appended
                else:
                    self.update_node_weights(node, data.tolist()) 
    """------------------------------------------------------------------------------------------------
    update_node_weights
    calculate a new array of weights
    ------------------------------------------------------------------------------------------------"""
    def update_node_weights(self, node, data):
        data = data + [-1]
        new_weights_array = []

        # iterate through all current weights
        for index, weight in enumerate(node.weights):
            new_weights_array.append(weight - (0.1) * data[index] * node.error)

        node.weights = new_weights_array

    """------------------------------------------------------------------------------------------------
    get_results
    calculate the activation value for each node in each layer and return the results
    ------------------------------------------------------------------------------------------------"""
    def get_results(self, input):
        results = []

        for index, layer in enumerate(self.layers):
            results.append([node.output(results[index - 1] if index > 0 else input) for node in layer])

        return results


    #obtain_information
    #
    #This function obtain the numbers of hidden layers and
    #epochs desired. The reason for number of epochs desired
    #is so I can experiment with the number of epochs and how
    #it affects the training of the neural network.
    def obtain_information(self):
        try:
            print("How many hidden layers? (1, 2)")
            self.num_hidden = int(input("> "))

            #it has been proven that if there are more than
            #2 hidden layers in a multi-layer perceptron 
            if (self.num_hidden < 1 or self.num_hidden > 2):
                raise Exception("please select either 1 or 2 hidden layers.")

            #I really don't want the neural network to overtrain
            #so I am putting a hard cap on the number of epochs
            print("How many epochs? (1 - 10,000)")
            self.num_epochs = int(input("> "))

            if (self.num_epochs < 1 or self.num_epochs > 10000):
                raise Exception("please choose between 1 and 10,000")
        except (ValueError) as err:
            print("ERROR: {}".format(err))

    
    def get_accuracy(count, length):
        return count / length * 100



