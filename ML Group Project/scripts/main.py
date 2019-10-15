import matplotlib.pyplot as plt
import pandas as pd
import os
import linearAlgo
import logAlgo
import kncAlgo


# returns the data from the Exel table:
def load_df(dir_path, file_name):
    file = dir_path + "\\data\\" + file_name
    data = pd.read_csv(file)
    return data


# counts the equal values of 2 lists:
def intersect(list1, list2):
    counter = 0
    for x in range(len(list1)):
        if list1[x] == list2[x]:
            counter += 1
    return counter


# display result accuracy of learning procedure:
def display_accuracy (y_prediction, test_y, display_message):
    hits = intersect(y_prediction, test_y.tolist())
    # Accuracy:
    print(display_message + str(hits) + " of " + str(len(test_y)) + " correctly classified! => " + str(hits / len(test_y)) + "% accuracy.")


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
display_accuracy(y_prediction, test_y, "Linear Regression accuracy: ")

# apply Logistic Regression:
y_prediction = logAlgo.apply_logistic_regression(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "Logistic Regression accuracy: ")

# apply k-Nearest-Neighbors Algorithm:
size_k = 3
y_prediction = kncAlgo.apply_logistic_regression(train_x, train_y, test_x, size_k)
display_accuracy(y_prediction, test_y, "k-Nearest-Neighbors accuracy: ")
