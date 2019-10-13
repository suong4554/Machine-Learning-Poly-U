from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import os


#returnes the data from the Exel table:
def loadDF(dir_path, fileName):
    file = dir_path + "\\data\\" + fileName
    data = pd.read_csv(file)
    return data
	
#load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
trainDF = loadDF(home_dir, "train.csv")
	
# create the training set: (without "target" and "id" column)
trainX = trainDF.drop("target",  axis = 1).drop("id", axis = 1) 
targetY = trainDF["target"]


# take 75 examples for testing:
testX = trainX.tail(75)
testY = targetY.tail(75)
# the 250-75 are used for training:
targetY = targetY.head(250-75)
trainX = trainX.head(250-75)
#print(testY)
	
#apply Logistic Regression:
lm = LogisticRegression()
lm.fit(trainX, targetY)


#predict the 75 test results: 
Y_pred = lm.predict(testX)
#print(Y_pred)
	
# counts the equal values of 2 lists:
def intersect(list1, list2):
    counter = 0
    for x in range(len(list1)):
        if list1[x] == list2[x]: 
            counter += 1
    return counter
	
# cast the results to 0 ant 1:
for x, data in enumerate(Y_pred):
    if data > 0.5: 
        Y_pred[x] = 1 
    else:
        Y_pred[x] = 0
	
#print the hits:
hits = intersect(Y_pred.tolist(), testY.tolist())
print(hits/len(testY))