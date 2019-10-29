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



##########################################################################################
#####################################DATA PREPROCESSING###################################
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
##########################################################################################




##########################################################################################
#####################################APPLYING ML LIBRARIES###################################

################## apply Linear Regression: #####################
y_prediction = linearAlgo.apply_linear_regression(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "Linear Regression")
print(y_prediction)
####################################################################################

################## apply Logistic Regression:##################
y_prediction = logAlgo.apply_logistic_regression(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "Logistic Regression")

createSubmission(y_prediction, home_dir)

print()

####################################################################################

################## apply Support Vector XXXX##################
# apply svr (regression)
y_prediction = svm.apply_svr(train_x, train_y, test_x, "rbf", 1)
display_accuracy(y_prediction, test_y, "Support Vector Regression")

kernelType = ["linear", "poly", "rbf"] #linear, polynomial degree, radial kernel
degreeRange = 20
#Finds best Support Vector Regression
scores = []
for kernel in kernelType:
    for degree in range(degreeRange):
        temp = []
        if kernel == "linear" and degree > 1:
            break #Saves time 
        else:
            actual_prediction = svm.apply_svr(train_x, train_y, test_x, kernel, degree)
            temp.append(accuracy_score(test_y, actual_prediction))
            temp.append(kernel)
            temp.append(degree)
            scores.append(temp)
scores0 = [_[0] for _ in scores]

max_accuracy = max(scores0)
max_index = scores0.index(max_accuracy) + 1 # function starts to count from 0
print("\t Maximum accuacy (" + str(max_accuracy) + "%) with kernel = " + str(scores[max_index][1]) + " with degree = " + str(scores[max_index][2]))
print()
# apply svc (classification)
y_prediction = svm.apply_svc(train_x, train_y, test_x, "poly", 2)
display_accuracy(y_prediction, test_y, "Support Vector Classification")

#Finds best Support Vector Classification
scores = []
for kernel in kernelType:
    for degree in range(degreeRange):
        temp = []
        if kernel == "linear" and degree > 1:
            break #Saves time 
        else:
            actual_prediction = svm.apply_svc(train_x, train_y, test_x, kernel, degree)
            temp.append(accuracy_score(test_y, actual_prediction))
            temp.append(kernel)
            temp.append(degree)
            scores.append(temp)
scores0 = [_[0] for _ in scores]

max_accuracy = max(scores0)
max_index = scores0.index(max_accuracy) + 1 # function starts to count from 0
print("\t Maximum accuacy (" + str(max_accuracy) + "%) with kernel = " + str(scores[max_index][1]) + " with degree = " + str(scores[max_index][2]))
print()

####################################################################################

################## apply Neural Network MLP Classifier##################
y_prediction = nA.apply_MLPClassifier(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "Neural Network MLP Classification")
print()
####################################################################################

################## apply Naive Bayes GaussianNB##################
y_prediction = nba.apply_naive(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "Naive Bayes GaussianNB")

#Apply Linear and Quadratic Discrimination Analysis
solvers = ["svd", "lsqr", "eigen"]
#Eigen does not work
#apply linear
y_prediction = daa.apply_linear_disc(train_x, train_y, test_x, "svd")
display_accuracy(y_prediction, test_y, "Linear Discrimination Analysis: Solver=svd")
print()
y_prediction = daa.apply_linear_disc(train_x, train_y, test_x, "lsqr")
display_accuracy(y_prediction, test_y, "Linear Discrimination Analysis: Solver=lsqr")
print()

#apply quadratic
#No solvers for quadratic discrimination
y_prediction = daa.apply_quad_disc(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "Quadratic Discrimination Analysis")
print()
####################################################################################


##################apply Gaussian Process Classifier##################
y_prediction = gca.apply_gaus(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "Gaussian Process Classifier")
print()
####################################################################################

###################apply Decision Tree Classifier##################
y_prediction = dta.apply_tree(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "Decision Tree Classifier")
print()
####################################################################################

################## apply k-Nearest-Neighbors Algorithm:##################
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
print()
####################################################################################

################## apply Random Forest Classifier Algorithm:##################
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
print()
# apply
