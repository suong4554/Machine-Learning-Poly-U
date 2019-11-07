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

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

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
train_id = train_df["id"]
# create the training set: (without "target" and "id" column)
train_x = train_df.drop("target", axis=1).drop("id", axis=1)
train_y = train_df["target"]
#train_x = train_x[["16","33", "45", "63", "65", "73", "91", "108", "117", "164", "189", "199", "209", "217", "239"]]







#load the testing DataFrame:
test_df = load_df(home_dir, "test.csv")
test_id = test_df["id"]
#create the testing set: (without "id" column)
test_x = test_df.drop("id", axis=1)
#test_x = test_x[["16","33", "45", "63", "65", "73", "91", "108", "117", "164", "189", "199", "209", "217", "239"]]



train_shape = train_x.shape[0]
total_x = RobustScaler().fit_transform(np.concatenate((train_x, test_x), axis=0))

train_x = pd.DataFrame(data=total_x[:train_shape])
test_x = pd.DataFrame(data=total_x[train_shape:])





#Logistic Regression Model Used for Feature Selection
modelLR = LogisticRegression(
    C=.2,
    fit_intercept=False,
    intercept_scaling=1,
    class_weight="balanced",
    penalty='l1',
    solver='liblinear',
    verbose=0,
    warm_start=False)

#Lasso Model Used for Feature Selection
modelLA = Lasso(alpha=0.031, tol=0.01, random_state=4, selection='random')



#Feature Selector
#https://scikit-learn.org/stable/modules/feature_selection.html
#250/25 = 10, step removes 10 per step
#16/250 .= 15% of dataset

#########################Feature Selection with Lasso Function#############################

#Use of Kfold to test various parts of data, feature selection
feature_selector = RFECV(estimator=modelLA, min_features_to_select=16, step=5, verbose=0, cv=StratifiedKFold(20), n_jobs=-1)
feature_selector.fit(train_x, train_y)


print("Optimal number of features : %d" % feature_selector.n_features_)
#Gets columns that were selected
colsLA = feature_selector.get_support(indices=True)
print("Selected Colums LA are:", colsLA)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(feature_selector.grid_scores_) + 1), feature_selector.grid_scores_)
plt.show()

#########################Feature Selection with Logarithmic Function#############################

#Use of Kfold to test various parts of data, feature selection
feature_selector = RFECV(estimator=modelLR, min_features_to_select=5, step=10, verbose=0, cv=StratifiedKFold(20), n_jobs=-1)
feature_selector.fit(train_x, train_y)


print("Optimal number of features : %d" % feature_selector.n_features_)
#Gets columns that were selected
colsLR = feature_selector.get_support(indices=True)
print("Selected Colums LR are:", colsLR)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(feature_selector.grid_scores_) + 1), feature_selector.grid_scores_)
plt.show()



#Combine Columns of feature selected Data
cols = []
for col in colsLA:
    if col not in cols:
        cols.append(col)
for col in colsLR:
    if col not in cols:
        cols.append(col)

print("Selected Colums are:", cols)
#Set data to fitted data
train_x = train_x[cols]
test_x = test_x[cols]


##########################################################################################




##########################################################################################
#####################################APPLYING ML LIBRARIES###################################


################## apply Logistic Regression:##################
y_prediction = logAlgo.apply_logistic_regression(train_x, train_y, test_x, train_y)
createSubmission(y_prediction, home_dir)
print(y_prediction)
submitD = False
message = "submission for Logistic Regression penalty = l1"
submit(submitD, message, home_dir)

####################################################################################



################## apply MLP Classifier##################

y_prediction = nA.apply_MLPClassifier(train_x, train_y, test_x, train_y)
submitD = False
message = "submission for MLP Classifier"
submit(submitD, message, home_dir)
####################################################################################



################## apply Lasso##################

y_prediction = lA.apply_lasso(train_x, train_y, test_x, train_y)
submitD = False
message = "submission for Lasso"
submit(submitD, message, home_dir)
####################################################################################
