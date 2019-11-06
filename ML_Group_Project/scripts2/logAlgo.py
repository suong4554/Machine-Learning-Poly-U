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
        intercept_scaling=1, 
        
        class_weight="balanced",
        penalty='l1',

        solver='liblinear', 
        verbose=0, 
        warm_start=False)
        
    """    
    train_shape = train_x.shape[0]
    feature_selector = RFECV(model, min_features_to_select=25, step=10, verbose=0, cv=25, n_jobs=-1)

    feature_selector.fit((np.concatenate((train_x, test_x), axis=0)), total_y)
    x_values = feature_selector.transform((np.concatenate((train_x, test_x), axis=0)))
    
    train_x = x_values[:train_shape]
    test_x = x_values[train_shape:]    
    """ 
    
    model.fit(train_x, train_y)
    """
    print(model.coef_)
    print(train_x.columns)
    coef = pd.Series(model.coef_, index = train_x.columns)
    imp_coef = pd.concat([coef.sort_values().head(20),
                     coef.sort_values().tail(10)])
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Coefficients in the Logistic Model")
    plt.show()
    """
    # predict the test results:
    y_prediction = model.predict(test_x)

    # return predictions:
    return y_prediction.tolist()

