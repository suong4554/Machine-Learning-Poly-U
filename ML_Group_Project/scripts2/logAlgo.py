from sklearn.linear_model import LogisticRegression



def apply_logistic_regression(train_x, train_y, test_x):
    # apply Logistic Regression:
    lr = LogisticRegression(
        
        C=.2, 
        fit_intercept=False,
        intercept_scaling=1, 
        
        class_weight="balanced",
        penalty='l1',

        solver='liblinear', 
        verbose=0, 
        warm_start=False)
    lr.fit(train_x, train_y)

    # predict the test results:
    y_prediction = lr.predict(test_x)

    # return predictions:
    return y_prediction.tolist()

