from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def apply_linear_disc(train_x, train_y, test_x, solverT):
    # apply Linear Regression:
    lda = LinearDiscriminantAnalysis(solver=solverT)
    lda.fit(train_x, train_y)

    # predict the results:
    y_prediction = lda.predict(test_x)

    # cast the results to 0 ant 1:
    for x, data in enumerate(y_prediction):
        if data > 0.5:
            y_prediction[x] = 1
        else:
            y_prediction[x] = 0

    # return predictions:
    return y_prediction.tolist()


#No solvers for quadratic discrimination
def apply_quad_disc(train_x, train_y, test_x):
    # apply Linear Regression:
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(train_x, train_y)

    # predict the results:
    y_prediction = qda.predict(test_x)

    # cast the results to 0 ant 1:
    for x, data in enumerate(y_prediction):
        if data > 0.5:
            y_prediction[x] = 1
        else:
            y_prediction[x] = 0

    # return predictions:
    return y_prediction.tolist()
