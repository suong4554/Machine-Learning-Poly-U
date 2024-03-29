from sklearn.linear_model import LogisticRegression
from IPython import get_ipython
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import os




# returns the data from the Exel table:
def load_df(dir_path, file_name):
    file = dir_path + "\\data\\" + file_name
    data = pd.read_csv(file)
    return data


# display result accuracy of learning procedure:
def display_accuracy(y_prediction, test_y, display_message):
    print("Accuracy_Score of " + display_message + " " + str(accuracy_score(y_prediction, test_y.tolist())))


def apply_logistic_regression(train_x, train_y, test_x):
    # apply Logistic Regression:
    lr = LogisticRegression(
        
        C=.2, 
        fit_intercept=False,
        intercept_scaling=1, 
        
        class_weight={0:0.4, 1:0.6},
        penalty='l1',

        solver='liblinear', 
        verbose=5, 
        warm_start=False)
        
    lr.fit(train_x, train_y)

    # predict the test results:
    y_prediction = lr.predict(test_x)

    # return predictions:
    return y_prediction.tolist()

##########################################################################################
#####################################DATA PREPROCESSING###################################
# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
train_df = load_df(home_dir, "train.csv")

# create the training set: (without "target" and "id" column)
train_x = train_df.drop("target", axis=1).drop("id", axis=1)
train_x = train_x[["16","33", "45", "63", "65", "73", "91", "108", "117", "164", "189", "120", "199", "209", "217", "239"]]
print(train_x)
target_y = train_df["target"]

# take "test_amount" examples for testing:
test_amount = 75
test_x = train_x.tail(test_amount)
test_y = target_y.tail(test_amount)
# the "250-test_amount" are used for training:
train_y = target_y.head(250 - test_amount)
train_x = train_x.head(250 - test_amount)
##########################################################################################


##########################################################################################
#####################################APPLYING ML LIBRARIES###################################

################## apply Logistic Regression:##################
y_prediction = apply_logistic_regression(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "Logistic Regression")












