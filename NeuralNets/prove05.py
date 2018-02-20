from nueron import NeuralNetworkClassifier
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn import datasets
from sklearn.model_selection import KFold

def main():
# Get the data from the data sets
    train_data, test_data, headers = get_dataset()

    # Set Classifier
    classifier = NeuralNetworkClassifier()

    # KFold cross validation
    kf = KFold(n_splits=10)
    kf.get_n_splits(train_data, test_data)
    for train_index, test_index in kf.split(train_data, test_data):
        X_train, X_test = train_data[train_index], train_data[test_index]
        Y_train, Y_test = test_data[train_index], test_data[test_index]

    model = classifier.fit(X_train, X_test)
    targets_predicted = model.predict(Y_train)

    print "Model: ", model

    count = 0
    for index in range(len(Y_test)):
        if targets_predicted[index] == Y_test[index]:
            count += 1 

    correctness = float(count) / len(Y_test) * 100

    print "Accuracy: {:.2f}".format(correctness)

"""------------------------------------------------------------------------------------------------
* get_dataset
* The user interface to choose the data set you want to run. It takes in user input and gets the
* return value from each of the datasets. This is returned to main.
------------------------------------------------------------------------------------------------"""
def get_dataset():
    print "Which dataset do you want you want to use:\n1: Iris Dataset\n2: Diabetes Dataset"
    input = raw_input()
    input = int(input) 

    if input == 2:
        train_data, test_data, headers = get_diabetes()
    else:
        train_data, test_data, headers = get_iris()

    return train_data, test_data, headers

"""------------------------------------------------------------------------------------------------
* get_diabetes 
* Reads in the pima indians dataset. There are empty values that are present so those need to be
* handled. They are replaced with NaN and then these rows are dropped because it is difficult to
* compare these to others who have all their values. 
------------------------------------------------------------------------------------------------"""
def get_diabetes():
    headers = ["preg", "gluc", "bP", "tricep_skin", "insulin", "bmi", "dpf", "age", "class"]
    dirty = pd.read_csv('../DataSets/pima.csv', header = None, names = headers)

    # clean everything except the last column
    clean = dirty[headers[0:8]]
    clean.replace(0.0, np.NaN, inplace=True)
    clean.dropna(inplace=True)
    clean["class"] = dirty["class"]

    # Normalize the data with a zscore
    train_data = clean[headers[0:8]]
    train_data = train_data.apply(zscore)

    train_data = train_data.as_matrix()
    test_data = clean.as_matrix(headers[8:9])

    #look at the data before returning it
    # print train_data[0:20]

    return train_data, test_data, headers

"""------------------------------------------------------------------------------------------------
* get_iris
* Reads in the iris dataset and then converts the zscore for each of the rows and returns the
* headers, train_data, and the targets
------------------------------------------------------------------------------------------------"""
def get_iris():
    headers = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None, names = headers)


    train_data = iris[headers[0:-1]]
    train_data = train_data.apply(zscore)
    train_data = train_data.as_matrix()

    test_data = iris.as_matrix(headers[-1:])

    return train_data, test_data, headers

main()
