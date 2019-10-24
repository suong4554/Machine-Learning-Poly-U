from sklearn import svm

#Applies Support Vector Machines
def apply_svc(train_x, train_y, test_x, kernelt, degreet):
    # apply Support Vector Regression
    svc = svm.SVC(kernel=kernelt, degree=degreet, gamma="auto")
    svc.fit(train_x, train_y) 

    # predict the results:
    y_prediction = svc.predict(test_x)

    # cast the results to 0 ant 1:
    for x, data in enumerate(y_prediction):
        if data > 0.5:
            y_prediction[x] = 1
        else:
            y_prediction[x] = 0

    # return predictions:
    return y_prediction.tolist()

def apply_svr(train_x, train_y, test_x, kernelt, degreet):
    # apply Support Vector Regression
    svr = svm.SVR(kernel=kernelt, degree=degreet, gamma="auto")
    svr.fit(train_x, train_y) 

    # predict the results:
    y_prediction = svr.predict(test_x)

    # cast the results to 0 ant 1:
    for x, data in enumerate(y_prediction):
        if data > 0.5:
            y_prediction[x] = 1
        else:
            y_prediction[x] = 0

    # return predictions:
    return y_prediction.tolist()

