###############################################################################
# Program:
#    Prove 11, Assignment - Ensemble Learning
#    Brother Burton, CS450
# Author:
#    Kyle West
###############################################################################

###############################################################################
# Import items from libraries that we need
###############################################################################
# useful tools
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error

# our learners
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.svm             import SVR
from sklearn.neural_network  import MLPRegressor
from sklearn.ensemble        import BaggingRegressor
from sklearn.ensemble        import AdaBoostRegressor
from sklearn.ensemble        import RandomForestRegressor

# The following are my pre-"munged" datasets
from Eur                 import EUR
from NZD                 import NZD
from GBP                 import GBP

###############################################################################
# Run a single test on of a classifier and return the accuracy
###############################################################################
def trial(dataset, testSize, classifier):
   data_train, data_test, target_train, target_test = \
      train_test_split(dataset.data, dataset.target, test_size= testSize)
   classifier.fit(data_train, target_train)
   predictions = classifier.predict(data_test)

   return mean_squared_error(target_test, predictions)


###############################################################################
# Format trial results to fit in as a row defined below
###############################################################################
def echoR(datasetName, LearnerName, error):
   print("| {: ^13} | {: ^13} | {: ^13} |"\
      .format(datasetName, LearnerName, "{:.1f}% error".format(error)))



###############################################################################
# Test and compare learners on the Car Evaluation Dataset
###############################################################################
knn = KNeighborsRegressor(n_neighbors=5)
svr  = SVR()
nn  = MLPRegressor(hidden_layer_sizes=(35,), max_iter=1000)
bag = BaggingRegressor(base_estimator=nn, n_estimators=50)
ada = AdaBoostRegressor(base_estimator=knn, n_estimators=150)
rfc = RandomForestRegressor(n_estimators=10)


print("+---------------+---------------+---------------+")
print("| Dataset       | Learner       | Accuracy      |")
print("+---------------+---------------+---------------+")
echoR("EUR",         "KNeighbors",   trial(EUR, .25, knn))
echoR("EUR",         "SVR",          trial(EUR, .25, svr))
echoR("EUR",          "MLP",          trial(EUR, .25, nn))
echoR("EUR",         "Bagging",      trial(EUR, .25, bag))
echoR("EUR",         "AdaBoost",     trial(EUR, .25, ada))
echoR("EUR",         "RandomForest", trial(EUR, .25, rfc))
print("+---------------+---------------+---------------+")


print("+---------------+---------------+---------------+")
print("| Dataset       | Learner       | Accuracy      |")
print("+---------------+---------------+---------------+")
echoR("GBP",         "KNeighbors",   trial(GBP, .33, knn))
echoR("GBP",         "SVR",          trial(GBP, .33, svr))
echoR("GBP",         "MLP",          trial(GBP, .33, nn))
echoR("GBP",         "Bagging",      trial(GBP, .33, bag))
echoR("GBP",         "AdaBoost",     trial(GBP, .33, ada))
echoR("GBP",         "RandomForest", trial(GBP, .33, rfc))
print("+---------------+---------------+---------------+")


print("+---------------+---------------+---------------+")
print("| Dataset       | Learner       | Accuracy      |")
print("+---------------+---------------+---------------+")
echoR("NZD",         "KNeighbors",   trial(NZD, .33, knn))
echoR("NZD",         "SVR",          trial(NZD, .33, svr))
echoR("NZD",          "MLP",          trial(NZD, .33, nn))
echoR("NZD",         "Bagging",      trial(NZD, .33, bag))
echoR("NZD",         "AdaBoost",     trial(NZD, .33, ada))
echoR("NZD",         "RandomForest", trial(NZD, .33, rfc))
print("+---------------+---------------+---------------+")
