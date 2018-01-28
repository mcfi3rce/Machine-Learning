from knn import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from pandas import read_csv
from sklearn import preprocessing
import matplotlib as mp
from sklearn.model_selection import KFold
#from sklearn.neighbors import KNeighborsClassifier
"""This code is meant to read non-numeric data from csv files and process it using the k-nearest neighbors algorithm"""
def main():

    # Get the data from the data sets
    train_data, test_data = get_dataset()
    # Split the data into the train data
    #X_train, X_test, Y_train, Y_test = train_test_split(train_data, test_data)

    # Set Classifier
    classifier = KNeighborsClassifier()
    
    # KFold cross validation
    kf = KFold(n_splits=10)
    kf.get_n_splits(train_data, test_data)
    for train_index, test_index in kf.split(train_data, test_data):
        X_train, X_test = train_data[train_index], train_data[test_index]
        Y_train, Y_test = test_data[train_index], test_data[test_index]

    model = classifier.fit(X_train, Y_train)

    targets_predicted = model.predict(X_test)
    count = 0
    for index in range(len(X_test)):
        if targets_predicted[index] == Y_test[index]:
            count += 1

    correctness = float(count) / len(X_test) * 100

    print "Accuracy: {:.2f}".format(correctness)

def get_dataset():
    print"Which dataset do you want you want to use:\n1: Car Dataset\n2: Diabetes Dataset\n3: Car MPG Dataset \n"
    input = raw_input()
    input = int(input)

    if input == 2:
        train_data, test_data = get_diabetes()
    elif input == 3:
        train_data, test_data = get_mpg()
    else:
        train_data, test_data = build_cars()

    return train_data, test_data

def build_cars():
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "result"]
    dataset = read_csv('car.csv',header = None, names = headers)

    cleanUp = {"buying" : {"low" : 0, "med" : 1, "high" : 2, "vhigh" : 3},
               "maint" : {"low" : 0, "med" : 1, "high" : 2, "vhigh" : 3},
               "doors" : {'1': 1, '2': 2, '3': 3, '4' : 4, "5more" : 5},
               "persons" : {'1': 1, '2': 2, '3': 3, '4' : 4,"more" : 5},
               "lug_boot" : {"small" : 0, "med" : 1, "big" : 2},
               "safety" : {"low" : 0, "med" : 1, "high": 2},
               "result" : {"unacc" : 0, "acc" : 1, "good" : 2, "vgood" : 3}}
    dataset.replace(cleanUp, inplace=True)

    train_data = dataset.as_matrix(headers[0:6])
    test_data = dataset.as_matrix(headers[6:7])

    return train_data, test_data

def get_diabetes():
    headers = ["preg", "gluc", "bP", "tricep_skin", "insulin", "bmi", "dpf", "age", "class"]
    dirty = read_csv('pima.csv', header = None, names = headers)

    # clean everything except the last column
    clean = dirty[headers[0:8]]
    clean.replace(0.0, np.NaN, inplace=True)
    clean.dropna(inplace=True)
    clean["class"] = dirty["class"]

    #Convert Dataframe to np_array
    train_data = clean.as_matrix(headers[0:8])
    test_data = clean.as_matrix(headers[8:9])

    #look at the data before returning it
    print train_data[0:20]

    return train_data, test_data

def get_mpg():
    headers = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
    dataset = read_csv('mpg.csv', header = None, delim_whitespace=True, names = headers, na_values='?')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    train_data = dataset.as_matrix(headers[1:8])
    test_data = dataset.as_matrix(headers[0:1])

    print train_data
    return train_data, test_data
main()
