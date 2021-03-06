from entropy import build_tree
from entropy import calculate_entropy
from entropy import print_tree
from dtree import DTreeClassifier
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from pandas import read_csv
from sklearn import preprocessing
import matplotlib as mp
from sklearn.model_selection import KFold
"""------------------------------------------------------------------------------------------------
* Prove 03: KNN with Non-Trivial Datasets
* Thi    print "PREDICTED: ", targets_predicted
    print "VALUES: ", Y_test
s code is meant to read non-numeric data from csv files and process it using the k-nearest
* neighbors algorithm
------------------------------------------------------------------------------------------------"""
def main():

    # Get the data from the data sets
    train_data, test_data, headers = get_dataset()

    # Split the data into the train data
    X_train, Y_train, X_test, Y_test = train_test_split(train_data, test_data, test_size=.3)
 
    
    # Set Classifier
    classifier = DTreeClassifier()

    # KFold cross validation
    """
    kf = KFold(n_splits=10)
    kf.get_n_splits(train_data, test_data)
    for train_index, test_index in kf.split(train_data, test_data):
        X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
        Y_train, Y_test = test_data.iloc[train_index], test_data.iloc[test_index]
    """
    # the quirks of panda dataframes
    X_train.reset_index(inplace=True, drop=True)
    Y_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    Y_test.reset_index(inplace=True, drop=True)
    # print X_train
    model = classifier.fit(X_train, X_test, headers)
    targets_predicted = model.predict(Y_train)

    # get the column
    Y_test = Y_test[headers[-1]]
    
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
    print"Which dataset do you want you want to use:\n1: Car Dataset\n2: Diabetes Dataset\n3: Car MPG Dataset \n 4: Voting Dataset \n 5: Loan Dataset"
    input = raw_input()
    input = int(input)
    
    if input == 2:
        train_data, test_data, headers = get_diabetes()
    elif input == 3:
        train_data, test_data, headers= get_mpg()
    elif input == 4:
        train_data, test_data, headers = get_voting()
    elif input == 5:
        train_data, test_data, headers = get_loan()
    else:
        train_data, test_data, headers = build_cars()

    return train_data, test_data, headers

"""------------------------------------------------------------------------------------------------
* build_cars
* This function gets the cars dataset from the UCI repository and reads through the columns
* The data is not numerical so it replaces the values with a numerical value. Then it creates 2 
* numpy arrays and returns them.
------------------------------------------------------------------------------------------------"""
def get_voting():
    headers = ["party", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]
    dataset = read_csv('../DataSets/votes.csv', delimiter = ',', header = None, names = headers)
    replace = {"n" : 0,
               "y" : 1,
               '?' : 2, 
               "democrat": 0,
               "republican" : 1}
    dataset.replace(replace, inplace=True)
    train_data = dataset[headers[1:]]
    test_data = dataset[headers[0:1]]
    headers.remove("party")
    headers.append("party")
    return train_data, test_data, headers

"""------------------------------------------------------------------------------------------------
* build_cars
* This function gets the cars dataset from the UCI repository and reads through the columns
* The data is not numerical so it replaces the values with a numerical value. Then it creates 2 
* numpy arrays and returns them.
------------------------------------------------------------------------------------------------"""
def get_loan():
    headers = ["credit", "income", "collateral", "should_loan"]
    dataset = read_csv('../DataSets/loan.csv', delimiter = ',', header = None, names = headers)

    train_data = dataset[headers[0:3]]
    test_data = dataset[headers[3:4]]

    return train_data, test_data, headers

"""------------------------------------------------------------------------------------------------
* build_cars
* This function gets the cars dataset from the UCI repository and reads through the columns
* The data is not numerical so it replaces the values with a numerical value. Then it creates 2 
* numpy arrays and returns them.
------------------------------------------------------------------------------------------------"""
def build_cars():
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "result"]
    dataset = read_csv('../DataSets/car.csv',header = None, names = headers)

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

    return train_data, test_data, headers

"""------------------------------------------------------------------------------------------------
* get_diabetes 
* Reads in the pima indians dataset. There are empty values that are present so those need to be
* handled. They are replaced with NaN and then these rows are dropped because it is difficult to
* compare these to others who have all their values. 
------------------------------------------------------------------------------------------------"""
def get_diabetes():
    headers = ["preg", "gluc", "bP", "tricep_skin", "insulin", "bmi", "dpf", "age", "class"]
    dirty = read_csv('../DataSets/pima.csv', header = None, names = headers)

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

    return train_data, test_data, headers

"""------------------------------------------------------------------------------------------------
* get_mpg
* Read in the mpg dataset from UCI Repo. Replace the '?' with na values and split the train and
* test data
------------------------------------------------------------------------------------------------"""
def get_mpg():
    headers = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
    dataset = read_csv('../DataSets/mpg.csv', header = None, delim_whitespace=True, names = headers, na_values='?')
    dataset.dropna(inplace=True)

    train_data = dataset.as_matrix(headers[1:8])
    test_data = dataset.as_matrix(headers[0:1])
    
    print train_data
    return train_data.astype(float), test_data, headers

main()

physician-fee-freeze
0 { education-spending
    0 { 0 }
    1 { 0 }
    2 { adoption-of-the-budget-resolution
        0 { 1 }
        1 { 0 } } }
1 { synfuels-corporation-cutback
    0 { export-administration-act-south-africa
        0 { 1 }
        1 { superfund-right-to-sue
            0 { 1 }
            1 { 1 }
            2 { handicapped-infants
                0 { 0 }
                1 { 1 } } }
        2 { adoption-of-the-budget-resolution
            0 { 1 }
            1 { superfund-right-to-sue
                1 { 0 }
                2 { 1 } } } }
    1 { adoption-of-the-budget-resolution
        0 { superfund-right-to-sue
            0 { immigration
                0 { 0 }
                1 { 1 } }
            1 { water-project-cost-sharing
                0 { export-administration-act-south-africa
                    0 { 0 }
                    1 { 1 }
                    2 { 1 } }
                1 { 1 } } }
        1 { handicapped-infants
            0 { 0 }
            1 { 1 } }
        2 { 0 } }
    2 { 1 } }
2 { education-spending
    0 { 0 }
    1 { 1 }
    2 { mx-missile
        1 { 0 }
        2 { 1 } } }
