from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def apply_random_forest_classifier(train_x, train_y, test_x, max_depth, estimators):
    
    # apply Random Forest Classifier:
    clf = RandomForestClassifier(n_estimators=estimators, max_depth =max_depth)
    clf.fit(train_x, train_y)

    # predict the test results:
    y_prediction = clf.predict(test_x)

    # return predictions:
    return y_prediction.tolist()

