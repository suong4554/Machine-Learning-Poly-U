from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF

def apply_gaus(train_x, train_y, test_x):
    # apply Linear Regression:
    gpc = GaussianProcessClassifier()
    gpc.fit(train_x, train_y)

    # predict the results:
    y_prediction = gpc.predict(test_x)

    # cast the results to 0 ant 1:
    for x, data in enumerate(y_prediction):
        if data > 0.5:
            y_prediction[x] = 1
        else:
            y_prediction[x] = 0

    # return predictions:
    return y_prediction.tolist()

