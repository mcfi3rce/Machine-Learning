#from knn import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from pandas import read_csv
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
"""This code is meant to read non-numeric data from csv files and process it using the k-nearest neighbors algorithm"""

def main():
    build_cars()

def build_cars():
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "result"]
    dataset = read_csv('car.csv',header = None, names = headers)
    data_train = dataset[0:1209]
    data_test = dataset[1209:-1]


    cleanUp = {"buying" : {"low" : 0, "med" : 1, "high" : 2, "vhigh" : 3},
               "maint" : {"low" : 0, "med" : 1, "high" : 2, "vhigh" : 3},
               ""

    print type(data_train)

main()

