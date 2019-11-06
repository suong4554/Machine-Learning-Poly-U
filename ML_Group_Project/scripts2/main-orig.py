from IPython import get_ipython
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import linearAlgo as lA
import logAlgo
import numpy as np
import neuralAlgo as nA

from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import pickle

# returns the data from the Exel table:
def load_df(dir_path, file_name):
    file = dir_path + "\\data\\" + file_name
    data = pd.read_csv(file)
    return data


# display result accuracy of learning procedure:
def display_accuracy(y_prediction, test_y, display_message):
    print("Accuracy_Score of " + display_message + " " + str(accuracy_score(y_prediction, test_y.tolist())))

def return_accuracy(y_prediction, test_y):
    return str(accuracy_score(y_prediction, test_y.tolist()))
    
    
def createSubmission(arr, dir):
    dir = dir + "\\submission\\submission.csv"
    for i in range(len(arr)):
        arr[i] = [i + 250, arr[i]]
    df = pd.DataFrame(arr, columns = ["id", "target"])
    df.to_csv(dir, index=False)
    print("submission written to file")



##########################################################################################
#####################################DATA PREPROCESSING###################################
# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts2", "")
train_df = load_df(home_dir, "train.csv")

# create the training set: (without "target" and "id" column)
from itertools import combinations
train_x = train_df.drop("target", axis=1).drop("id", axis=1)

#train_x = train_x[["16","33", "45", "63", "65", "73", "91", "108", "117", "164", "189", "199", "209", "217", "239"]]
#train_x = train_x[["65", '201', '297', '33', '108', '194', '0', '82', '116', '199', '256', '141', '285']]
#train_x = train_x[["33","65","199","78","201","101","193","289","137","183"]]
target_y = train_df["target"]
total_y = train_df["target"]


from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=0.031, penalty="l1", dual=False).fit(train_x, target_y)
model = SelectFromModel(lsvc, prefit=True)
train_x = model.transform(train_x)
cols = model.get_support(indices=True)
print(cols)
print(train_x.shape)


"""
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(train_x,total_y)
#print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=train_x.columns)
print(feat_importances.nlargest(25))
feat_importances.nlargest(25).plot(kind='barh')
plt.show()
"""

#Split up the dataset for testing and training purposes
train_x, test_x, train_y, test_y = train_test_split(train_x, target_y, test_size = 0.3, random_state = 5)

#print(train_x.shape[0])
train_shape = train_x.shape[0]

#print(train_x)
#Scale the data to remove outliers
"""
data = RobustScaler().fit_transform(np.concatenate((train_x, test_x), axis=0))

train_x = data[:train_shape]
test_x = data[train_shape:]
"""

##########################################################################################




##########################################################################################
#####################################APPLYING ML LIBRARIES###################################
###########################################################################
################## apply Logistic Regression:##################
y_prediction = logAlgo.apply_logistic_regression(train_x, train_y, test_x, total_y)
accuracy = return_accuracy(y_prediction, test_y)
print(accuracy)


################## apply Linear Lasso Regression:##################
y_prediction = lA.apply_lasso(train_x, train_y, test_x, total_y)
accuracy = return_accuracy(y_prediction, test_y)
print(accuracy)


################## apply MLP:##################
#y_prediction = nA.apply_MLPClassifier(train_x, train_y, test_x, total_y)
#accuracy = return_accuracy(y_prediction, test_y)
#print(accuracy)

