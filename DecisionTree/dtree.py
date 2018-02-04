import numpy as np
import math


class Node(object):


class DTreeClassifier():

    def __init__(self):
        pass

    def fit(self, data, targets):
        return DecisionTreeModel(data, targets)

class DecisionTreeModel():
   def __init__(self, data, targets):
       self.data = data
       self.targets = targets

   def predict(self, data):
      predictions = []
      for item in data:
         predictions.append(self.predict_one(item))
      return predictions

   def predict_one(self, data):
      return "WAT"

  def build_tree(self, data, targets):

      data_set = np.unique(data)

      if len(data_set) == 1:
          return 
      ##createNode for each characteristic
      # Find the best attribute
      ##for each characteristic find entropy value
      # Calculate entropy
      ##set the lowest entropy value as the head
      # -SUM of PlogP
      ##run the next node and see which of the remaining values has the next lowest entropy
      #once a value has an entropy of zero go left
      #continue until there is no more entropy
      #return the treee
      # 
      return 0



    If all examples have the same label
    return a leaf with that label
    Else if there are no features left to test
    return a leaf with the most common label
    Else
    Consider each available feature
    Choose the one that maximizes information gain
    Create a new node for that feature

    For each possible value of the feature
        Create a branch for this value
        Create a subset of the examples for each branch
        Recursively call the function to create a new node at that branch
