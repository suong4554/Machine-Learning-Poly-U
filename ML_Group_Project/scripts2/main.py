from IPython import get_ipython
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import linearAlgo as lA
import logAlgo
import neuralAlgo as nA
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
import numpy as np


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
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts2", "")
train_df = load_df(home_dir, "train.csv")

# create the training set: (without "target" and "id" column)
train_x = train_df.drop("target", axis=1).drop("id", axis=1)
train_y = train_df["target"]
#train_x = train_x[["16","33", "45", "63", "65", "73", "91", "108", "117", "164", "189", "199", "209", "217", "239"]]
#train_x = train_x[["65", '201', '297', '33', '108', '194', '0', '82', '116', '199', '256', '141', '285']]
total_y = train_df["target"]







#load the testing DataFrame:
train_df = load_df(home_dir, "test.csv")

#create the testing set: (without "id" column)
test_x = train_df.drop("id", axis=1)
#test_x = test_x[["16","33", "45", "63", "65", "73", "91", "108", "117", "164", "189", "199", "209", "217", "239"]]
#test_x = test_x[["65", '201', '297', '33', '108', '194', '0', '82', '116', '199', '256', '141', '285']]

"""
train_shape = train_x.shape[0]
total_x = RobustScaler().fit_transform(np.concatenate((train_x, test_x), axis=0))

train_x = total_x[:train_shape]
#test_x = total_x[train_shape:]
"""
"""

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=0.2, penalty="l1", dual=False).fit(train_x, train_y)
model = SelectFromModel(lsvc, prefit=True)
train_x = model.transform(train_x)
cols = model.get_support(indices=True)
"""

"""
model = LogisticRegression(
    
    C=.2, 
    fit_intercept=False,
    intercept_scaling=1, 
    
    class_weight="balanced",
    penalty='l1',

    solver='liblinear', 
    verbose=0, 
    warm_start=False)
"""
model = Lasso(alpha=0.031, tol=0.01, random_state=4, selection='random')

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
svc = SVC(kernel="linear")
train_shape = train_x.shape[0]
feature_selector = RFECV(estimator=model, min_features_to_select=16, step=5, verbose=0, cv=20, n_jobs=-1)

feature_selector.fit(train_x, total_y)


print("Optimal number of features : %d" % feature_selector.n_features_)
#features=train_x[:,feature_selector.support_]
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(feature_selector.grid_scores_) + 1), feature_selector.grid_scores_)
plt.show()


cols = feature_selector.get_support(indices=True)
print(cols)
"""
train_x = x_values[:train_shape]
test_x = x_values[train_shape:]  
"""



colString = []
for data in cols:
    colString.append(str(data))

train_x = train_x[colString]
test_x = test_x[colString]

print(test_x.shape, train_x.shape)
print(cols)

##########################################################################################




##########################################################################################
#####################################APPLYING ML LIBRARIES###################################


################## apply Logistic Regression:##################
y_prediction = logAlgo.apply_logistic_regression(train_x, train_y, test_x, total_y)
createSubmission(y_prediction, home_dir)

submitD = False
message = "submission for Logistic Regression penalty = l1"
submit(submitD, message, home_dir)

####################################################################################



################## apply MLP Classifier##################

y_prediction = nA.apply_MLPClassifier(train_x, train_y, test_x, total_y)
submitD = False
message = "submission for MLP Classifier"
submit(submitD, message, home_dir)
####################################################################################



################## apply Lasso##################

y_prediction = lA.apply_lasso(train_x, train_y, test_x, total_y)
submitD = True
message = "submission for Lasso"
submit(submitD, message, home_dir)
####################################################################################




