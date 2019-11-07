from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV
import numpy as np

def apply_MLPClassifier(train_x, train_y, test_x, total_y):
    # apply Linear Regression:
    model = MLPClassifier(hidden_layer_sizes=(2000),
    activation= 'relu',
    solver='lbfgs',
    max_iter=300)

    """
    train_shape = train_x.shape[0]
    feature_selector = RFECV(model, min_features_to_select=25, step=10, verbose=0, cv=25, n_jobs=-1)

    feature_selector.fit((np.concatenate((train_x, test_x), axis=0)), total_y)
    x_values = feature_selector.transform((np.concatenate((train_x, test_x), axis=0)))

    train_x = x_values[:train_shape]
    test_x = x_values[train_shape:]
    """

    model.fit(train_x, train_y)
    # predict the results:
    y_prediction = model.predict_proba(test_x)[:,1]

    # return predictions:
    return y_prediction.tolist()
