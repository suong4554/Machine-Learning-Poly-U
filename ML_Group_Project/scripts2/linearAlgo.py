from sklearn.linear_model import LinearRegression


def apply_linear_regression(train_x, train_y, test_x):
    # apply Linear Regression:
    lm = LinearRegression()
    lm.fit(train_x, train_y)

    # predict the results:
    y_prediction = lm.predict(test_x)

    # cast the results to 0 ant 1:
    for x, data in enumerate(y_prediction):
        if data > 0.5:
            y_prediction[x] = 1
        else:
            y_prediction[x] = 0

    # return predictions:
    return y_prediction.tolist()
