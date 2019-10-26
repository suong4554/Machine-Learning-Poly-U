import pandas as pd



#returnes the data from the Exel table:
def loadDF(dir_path, fileName):
    file = dir_path + "\\data\\" + fileName
    data = pd.read_csv(file)
    return data
    
def createDataset(directory):	
    #load the training data frame:

    trainDF = loadDF(directory, "train.csv")
        
    # create the training set: (without "target" and "id" column)
    trainX = trainDF.drop("target",  axis = 1).drop("id", axis = 1) 
    targetY = trainDF["target"]


    # take 75 examples for testing:
    testX = trainX.tail(75)
    testY = targetY.tail(75)
    # the 250-75 are used for training:
    targetY = targetY.head(250-75)
    trainX = trainX.head(250-75)
    
    retDic = {"testX":testX, "testY":testY,"targetY":targetY,"trainX":trainX}
    return retDic
    
