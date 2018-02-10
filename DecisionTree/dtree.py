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
        train_data.reset_index(inplace=True, drop=True)
        self.tree = enpy.build_tree(train_data, headers[:-1])

    def __repr__(self):
        enpy.print_tree(self.tree)

    def predict(self, data):
        predictions = []
        for index, row in data.iterrows():
            print(row.to_frame().T)
            predictions.append(self.predict_one(row))
        return predictions

    def predict_one(self, data):
        return 0
