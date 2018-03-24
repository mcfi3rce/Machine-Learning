import pandas
from scipy.stats import zscore

data = pandas.read_csv("./data/EUR.csv", header='infer', delimiter=',', na_values='NaN')
data.dropna(inplace=True)

data = data[[
   "EUR"]]

# get the z-score
data = (data - data.mean())/data.std()

###############################################################################
# This is for exporting our data
###############################################################################
class EmptyObject: pass
EUR = EmptyObject()
EUR.target = data.shift(-1)[:-1]
EUR.data = data[:-1]

if __name__ == "__main__":
   print(EUR.data.shape)
   print(EUR.data)
   print(EUR.target)

