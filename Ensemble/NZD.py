import pandas
from scipy.stats import zscore

data = pandas.read_csv("./data/NZD.csv", header='infer', delimiter=',')
data.dropna(inplace=True)

data = data[[
   "NZD"]]

# get the z-score
data = (data - data.mean())/data.std()

###############################################################################
# This is for exporting our data
###############################################################################
class EmptyObject: pass
NZD = EmptyObject()
NZD.target = data.shift(-1)[:-1]
NZD.data = data[:-1]

if __name__ == "__main__":
   print(NZD.data.shape)
   print(NZD.data)
   print(NZD.target)

