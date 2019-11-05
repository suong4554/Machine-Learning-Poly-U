from sklearn.neural_network import MLPClassifier


def apply_MLPClassifier(train_x, train_y, test_x):
    # apply Linear Regression:
    mlp = MLPClassifier(hidden_layer_sizes=(200,200,50,200 ), 
    activation= 'logistic', 
    solver='lbfgs',
    max_iter=300)
    mlp.fit(train_x, train_y)

    # predict the results:
    y_prediction = mlp.predict(test_x)

    # return predictions:
    return y_prediction.tolist()
