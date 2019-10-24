from IPython import get_ipython
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import linearAlgo
import logAlgo
import kncAlgo
import ranForestClassifierAlgo as rfCA

# returns the data from the Exel table:
def load_df(dir_path, file_name):
    file = dir_path + "\\data\\" + file_name
    data = pd.read_csv(file)
    return data


# display result accuracy of learning procedure:
def display_accuracy(y_prediction, test_y, display_message):
    print("Accuracy_Score of " + display_message + " " + str(accuracy_score(y_prediction, test_y.tolist())))


# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
train_df = load_df(home_dir, "train.csv")

# create the training set: (without "target" and "id" column)
train_x = train_df.drop("target", axis=1).drop("id", axis=1)
target_y = train_df["target"]

# take "test_amount" examples for testing:
test_amount = 75
test_x = train_x.tail(test_amount)
test_y = target_y.tail(test_amount)
# the "250-test_amount" are used for training:
train_y = target_y.head(250 - test_amount)
train_x = train_x.head(250 - test_amount)


# apply Linear Regression:
y_prediction = linearAlgo.apply_linear_regression(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "Linear Regression")

# apply Logistic Regression:
y_prediction = logAlgo.apply_logistic_regression(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "Logistic Regression")

# apply k-Nearest-Neighbors Algorithm:
size_k = 3
y_prediction = kncAlgo.apply_logistic_regression(train_x, train_y, test_x, size_k)
display_accuracy(y_prediction, test_y, "k-Nearest-Neighbors (k=3)")
# -- Test different k values --
scores = []
k_range = list(range(1,30))
for k in k_range:
    actual_prediction = kncAlgo.apply_logistic_regression(train_x, train_y, test_x, k)
    scores.append(accuracy_score(test_y, actual_prediction))
max_accuracy = max(scores)
max_index = scores.index(max_accuracy) + 1 # function starts to count from 0
print("\t Maximal accuacy (" + str(max_accuracy) + "%) with k = " + str(max_index))

# apply Random Forest Classifier Algorithm:
max_depth = 9
estimators = 43
y_prediction = rfCA.apply_random_forest_classifier(train_x, train_y, test_x, max_depth, estimators)

display_accuracy(y_prediction, test_y, "Random Forest Classifier (max_depth = " + str(max_depth) + ", estimators = " + str(estimators) + ")")

#Test different estimator values
scores = []
estimators_range = list(range(1, 30))
max_depth_range = list(range(1, 5))

for estimators in estimators_range:
    for max_depth in max_depth_range:
        temp = []
        actual_prediction = rfCA.apply_random_forest_classifier(train_x, train_y, test_x, max_depth, estimators)
        temp.append(accuracy_score(test_y, actual_prediction))
        temp.append(estimators)
        temp.append(max_depth)
        scores.append(temp)

scores0 = [_[0] for _ in scores]

max_accuracy = max(scores0)
max_index = scores0.index(max_accuracy) + 1 # function starts to count from 0
print("\t Maximum accuacy (" + str(max_accuracy) + "%) with estimator = " + str(scores[max_index][1]) + " with max_depth = " + str(scores[max_index][2]))

# apply
