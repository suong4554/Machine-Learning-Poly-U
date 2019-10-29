from IPython import get_ipython
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import linearAlgo
import logAlgo
import kncAlgo
import ranForestClassifierAlgo as rfCA
import svmAlgo as svm
import neuralAlgo as nA
import gausClassAlgo as gca
import decTreeAlgo as dta
import discrimAnalAlgo as daa
import naiveBayAlgo as nba

import writeSubmission as ws


# returns the data from the Exel table:
def load_df(dir_path, file_name):
    file = dir_path + "\\data\\" + file_name
    data = pd.read_csv(file)
    return data


# display result accuracy of learning procedure:
def display_accuracy(y_prediction, test_y, display_message):
    print("Accuracy_Score of " + display_message + " " + str(accuracy_score(y_prediction, test_y.tolist())))


def createSubmission(arr, dir):
    dir = dir + "\\submission\\submission.csv"
    for i in range(len(arr)):
        arr[i] = [i + 250, arr[i]]
    df = pd.DataFrame(arr, columns = ["id", "target"])
    df.to_csv(dir, index=False)
    print("submission written to file")


def submit(submitB, message, dir):
    dir = dir + "\\submission\\submission.csv"
    command = 'kaggle competitions submit -c dont-overfit-ii -f ' + str(dir) + ' -m "' + str(message) + '"'
    if(submitB):
        os.system(command)
        print("\n Submitted")
    else:
        print("Not Submitted")


##########################################################################################
#####################################DATA PREPROCESSING###################################
# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
train_df = load_df(home_dir, "train.csv")

# create the training set: (without "target" and "id" column)
train_x = train_df.drop("target", axis=1).drop("id", axis=1)
train_y = train_df["target"]

"""
# take "test_amount" examples for testing:
test_amount = 75
test_x = train_x.tail(test_amount)
test_y = target_y.tail(test_amount)
# the "250-test_amount" are used for training:
train_y = target_y.head(250 - test_amount)
train_x = train_x.head(250 - test_amount)
"""


#load the testing DataFrame:
train_df = load_df(home_dir, "test.csv")

#create the testing set: (without "id" column)
test_x = train_df.drop("id", axis=1)


##########################################################################################




##########################################################################################
#####################################APPLYING ML LIBRARIES###################################

"""
################## apply Logistic Regression:##################
y_prediction = logAlgo.apply_logistic_regression(train_x, train_y, test_x)
createSubmission(y_prediction, home_dir)

submitD = True
message = "test submission for Logistic Regression no params"
submit(submitD, message, home_dir)

####################################################################################
"""


################## apply Naive Bayes GaussianNB##################

y_prediction = nba.apply_naive(train_x, train_y, test_x)
submitD = False
message = "test submission for Logistic Regression no params"
submit(submitD, message, home_dir)
####################################################################################







