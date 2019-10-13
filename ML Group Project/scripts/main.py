import logAlgo as log, kncAlgo as knc, linearAlgo as lin
import dataIngest
import os


home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")


#Creates a dictionary:
#{"testX":testX, "testY":testY,"targetY":targetY,"trainX":trainX}
dataDict = dataIngest.createDataset(home_dir)

print("Linear Regression: ", lin.run(dataDict))
print("KNearestNeighbor Classification: ", knc.run(dataDict))
print("Logarithmic Regression: ", log.run(dataDict))