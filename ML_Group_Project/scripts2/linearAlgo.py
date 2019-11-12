from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def apply_lasso(train_x, train_y, test_x, total_y):
    # apply Linear Regression:
    model = Lasso(alpha=0.03, tol=0.01)

    model.fit(train_x, train_y)
    # predict the results:
    #y_prediction = model.predict_proba(test_x)[:,1]
    y_prediction = model.predict(test_x)
    # return predictions:
    return y_prediction.tolist()
