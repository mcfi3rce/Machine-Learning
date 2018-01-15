class HardCodedClassifier():

    def __init__(self):
        pass

    def fit(self, data, target):
        return HardCodeModel()

class HardCodeModel():

    def __init__(self):
        self.model =  []

    def predict(self, data):
        for row in data:
            #it's a hard code classifier... 
            self.model.append(0)

        return self.model

