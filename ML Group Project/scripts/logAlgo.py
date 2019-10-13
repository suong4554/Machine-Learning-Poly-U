from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def run(data):
    testX = data["testX"]
    testY = data["testY"]
    targetY = data["targetY"]
    trainX = data["trainX"]

        
    #apply LogisticRegression:
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
    return(hits/len(testY))