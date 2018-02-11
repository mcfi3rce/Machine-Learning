import numpy as np
import math
from bTree import Node
import entropy as enpy
import pandas as pd

class DTreeClassifier():

    def __init__(self):
        pass

    def fit(self, data, targets, headers):
        return DecisionTreeModel(data, targets, headers)

class DecisionTreeModel():
    def __init__(self, data, targets, headers):
        self.data= data
        self.targets = targets
        self.headers = headers
        frames = [data, targets]
        train_data = pd.concat(frames, axis=1)
        self.tree = enpy.build_tree(train_data, headers[:-1])

    def __repr__(self):
        enpy.print_tree(self.tree)

    def predict(self, data):
        predictions = []
        enpy.print_tree(self.tree)
        for index, row in data.iterrows():
            attributes = self.headers[:]
            prediction = self.predict_class(self.tree, row, attributes)
            print "PREDICT: ", prediction
            predictions.append(prediction)
        return predictions

    def predict_class(self, node, row_data, headers):
        print "NODE: ", node.name
        print "HEADERS: ", len(headers)
        for attribute in headers:
            if attribute == node.name:
                value = row_data[attribute]
                if value in node.children:
                    new_node = node.children[value]
                    if new_node.isLeaf():
                        print "LEAF: ", new_node.name
                        return new_node.name
                    else:
                        print "BRANCH: ", new_node.name
                        headers.remove(attribute)
                        self.predict_class(new_node, row_data, headers)
                else:
                    return 0
        return 1
