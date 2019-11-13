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

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


# returns the data from the Exel table:
def load_df(dir_path, file_name):
    file = dir_path + "\\data\\" + file_name
    data = pd.read_csv(file)
    return data


# display result accuracy of learning procedure:
def display_accuracy(y_prediction, test_y, display_message):
    print("Accuracy_Score of " + display_message + " " + str(accuracy_score(y_prediction, test_y.tolist())))


# creates the file for submission
def create_submission(arr, dir):
    dir = dir + "\\submission\\submission.csv"
    for i in range(len(arr)):
        arr[i] = [i + 250, arr[i]]
    df = pd.DataFrame(arr, columns=["id", "target"])
    df.to_csv(dir, index=False)
    print("submission written to file")


# submits the file to kaggle.com
def submit(submitB, message, dir):
    dir = dir + "\\submission\\submission.csv"
    command = 'kaggle competitions submit -c dont-overfit-ii -f ' + str(dir) + ' -m "' + str(message) + '"'
    if (submitB):
        os.system(command)
        print("\n Submitted")
    else:
        print("Not Submitted")

def roundValues(predict):
    for x, data in enumerate(predict):
        if data > 0.5:
            predict[x] = 1
        else:
            predict[x] = 0
    return predict


# Root mean Square error => standard deviation of residuals
#Residuals => how far from the regression line data points are
def rmse_cv(model, train_x, train_y):
    rmse = np.sqrt(-cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)

##########################################################################################
#                                    DATA PREPROCESSING

# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts2", "")
train_df = load_df(home_dir, "train.csv")
train_id = train_df["id"]
# create the training set: (without "target" and "id" column)
train_x = train_df.drop("target", axis=1).drop("id", axis=1)
train_y = train_df["target"]

# load the testing DataFrame:
test_df = load_df(home_dir, "test.csv")
test_id = test_df["id"]
# create the testing set: (without "id" column)
test_x = test_df.drop("id", axis=1)

# Here we apply Standardization -> RobustScaler() which is especially good regarding outliers
# we standardize the total data and split it again afterwards
train_shape = train_x.shape[0]
total_x = RobustScaler().fit_transform(np.concatenate((train_x, test_x), axis=0))

# This are the standardized data sets:
train_x = pd.DataFrame(data=total_x[:train_shape])
test_x = pd.DataFrame(data=total_x[train_shape:])



#########################Calculate Alpha for Lasso Function#############################
#Alpha Selection for Lasso:

#Calculate Alphas to use
alphas = [0.01, 0.03, 0.04, 0.05, 0.06 , .1]
cv_lasso = [rmse_cv(Lasso(alpha = alpha), train_x, train_y).mean() for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Alpha Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")
plt.show()

#Calculate tolerances to use
tols = [0.005, 0.01, 0.015, 0.02, 0.025]
cv_lasso = [rmse_cv(Lasso(alpha = .03, tol=tol), train_x, train_y).mean() for tol in tols]
cv_lasso = pd.Series(cv_lasso, index = tols)
cv_lasso.plot(title = "Tolerance Validation")
plt.xlabel("Tolerance")
plt.ylabel("Rmse")
plt.show()
modelLA = Lasso(alpha=0.03, tol=0.01)

#########################Feature Selection with Lasso Function#############################


# -> Now we start applying feature selection and/or dimensionality reduction, as this is especially promising
# with such high dimensional data.

# Feature Selector
# https://scikit-learn.org/stable/modules/feature_selection.html
# 250/25 = 10, step removes 10 per step
# 16/250 .= 15% of dataset


# Use of Kfold to test various parts of data, feature selection (Recursive feature elimination with cross-validation):
feature_selector = RFECV(estimator=modelLA, min_features_to_select=16, step=5, verbose=0, cv=StratifiedKFold(20),
                         n_jobs=-1)
feature_selector.fit(train_x, train_y)

print("Optimal number of features : %d" % feature_selector.n_features_)
#Gets columns that were selected
cols = feature_selector.get_support(indices=True)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.title("Lasso Regression RFECV")
plt.plot(range(1, len(feature_selector.grid_scores_) + 1), feature_selector.grid_scores_)
plt.show()


print("Selected Colums are:", cols)
# Set data to fitted data
train_x = train_x[cols]
test_x = test_x[cols]


##########################################################################################
# ####################################APPLYING ML LIBRARIES###################################


# ################# apply Logistic Regression:##################
y_prediction = logAlgo.apply_logistic_regression(train_x, train_y, test_x, train_y)
y1 = y_prediction
y_prediction = roundValues(y_prediction)
create_submission(y_prediction, home_dir)
submitD = False
message = "submission for Logistic Regression penalty = l1 with Rounded values"
submit(submitD, message, home_dir)

####################################################################################


# ################# apply MLP Classifier##################

y_prediction = nA.apply_MLPClassifier(train_x, train_y, test_x, train_y)
y2 = y_prediction
y_prediction = roundValues(y_prediction)
create_submission(y_prediction, home_dir)
submitD = False
message = "submission for MLP Classifier with Rounded values"
submit(submitD, message, home_dir)
####################################################################################


# ################# apply Lasso##################

y_prediction = lA.apply_lasso(train_x, train_y, test_x, train_y)
y3 = y_prediction
y_prediction = roundValues(y_prediction)
create_submission(y_prediction, home_dir)

submitD = False
message = "submission for Lasso with Rounded values"
submit(submitD, message, home_dir)
####################################################################################


# ################# Combined Values##################

res_list = []
for i in range(0, len(y1)):
    res_list.append((y1[i][1] + y2[i][1] + y3[i][1]) / 3)
create_submission(res_list, home_dir)
submitD = False
message = "submission for Average of Logistic, Lasso, and MLP"
submit(submitD, message, home_dir)
