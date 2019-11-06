from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def apply_lasso(train_x, train_y, test_x, total_y):
    # apply Linear Regression:
    model = Lasso(alpha=0.031, tol=0.01, random_state=4, selection='random')
    
    

    
    
    
    #Feature Selector
    #https://scikit-learn.org/stable/modules/feature_selection.html
    #250/25 = 10, step removes 10 per step
    #16/250 .= 15% of dataset

    
    #rmse_cv(model_lasso).mean()

    
    
    model.fit(train_x, train_y)
    
    """
    coef = pd.Series(model.coef_, index = train_x.columns)
    imp_coef = pd.concat([coef.sort_values().head(20),
                     coef.sort_values().tail(10)])
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show()
    """
    # predict the results:
    y_prediction = model.predict(test_x)

    # cast the results to 0 ant 1:
    for x, data in enumerate(y_prediction):
        if data > 0.5:
            y_prediction[x] = 1
        else:
            y_prediction[x] = 0

    # return predictions:
    return y_prediction.tolist()
