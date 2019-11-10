from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def apply_logistic_regression(train_x, train_y, test_x, total_y):
    # apply Logistic Regression:
    model = LogisticRegression(

        C=.2,
        fit_intercept=False,
        class_weight="balanced",
        penalty='l1',
        solver='liblinear')


    model.fit(train_x, train_y)

    # predict the test results:
    #y_prediction = model.predict(test_x)
    y_prediction = model.predict_proba(test_x)[:,1]
    # return predictions:
    return y_prediction.tolist()
