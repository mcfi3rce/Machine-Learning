import pandas
from scipy.stats import zscore

data = pandas.read_csv("./data/GBP.csv", header='infer', delimiter=',')
data.dropna(inplace=True)

data = data[[
   "GBP"]]

# get the z-score
data = (data - data.mean())/data.std()

###############################################################################
# This is for exporting our data
###############################################################################
class EmptyObject: pass
GBP = EmptyObject()
GBP.target = data.shift(-1)[:-1]
GBP.data = data[:-1]

if __name__ == "__main__":
   print(GBP.data.shape)
   print(GBP.data)
   print(GBP.target)

