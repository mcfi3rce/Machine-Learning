#from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNeighborsClassifier
from sklearn import preprocessing

def main():
    print "Which data set would you like to load: \n"
    print "0: Iris\n 1: Digits\n "
    data = raw_input()
    print "What would you like your k to be?: "
    k = raw_input()

    build_dataset(data, k)

def build_dataset(dataset = 0, k = 3):
    #load the iris data set

    if dataset == "0":
        dataset = datasets.load_iris()
    elif dataset == "1":
        dataset = datasets.load_digits()
    print type(dataset.data)
    #returns a numpy array for each variable, this will allow us to use the variables to test our algorithm
    data_train, data_test, targets_train, test_target = train_test_split(dataset.data, dataset.target, test_size = .3)

    #Select the kNearest Neighbors
    classifier = KNeighborsClassifier(int(k))
    model = classifier.fit(data_train, targets_train)

    targets_predicted = model.predict(data_test)

    count = 0
    for index in range(len(data_test)):
        if targets_predicted[index] == test_target[index]:
            count += 1

    correctness = float(count) / len(data_test) * 100

    print "Accuracy: {:.2f}".format(correctness)

if __name__ == "__main__":
    main()
