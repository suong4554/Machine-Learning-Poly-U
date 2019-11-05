from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from IPython import get_ipython
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import os




# returns the data from the Exel table:
def load_df(dir_path, file_name):
    file = dir_path + "\\data\\" + file_name
    data = pd.read_csv(file)
    return data


# display result accuracy of learning procedure:
def display_accuracy(y_prediction, test_y, display_message):
    print("Accuracy_Score of " + display_message + " " + str(accuracy_score(y_prediction, test_y.tolist())))


def apply_logistic_regression(train_x, train_y, test_x):
    # apply Logistic Regression:
    lr = MLPClassifier(
        activation='logistic', 
        batch_size='auto',
              
        beta_1=0.9, 
        beta_2=0.999, 
        early_stopping=False,
              
        epsilon=1e-08, 
        hidden_layer_sizes=(5, 2),
              
        learning_rate='adaptive', 
        learning_rate_init=0.001,
              
        max_iter=200, 
        momentum=0.9, 
        n_iter_no_change=10,
              
        nesterovs_momentum=True, 
        power_t=0.5, 
        random_state=1,
        shuffle=True, 
        solver='sgd', 
        tol=0.0001,
        validation_fraction=0.1, 
        verbose=False, 
        warm_start=False
    )
        
    lr.fit(train_x, train_y)

    # predict the test results:
    y_prediction = lr.predict(test_x)

    # return predictions:
    return y_prediction.tolist()

##########################################################################################
#####################################DATA PREPROCESSING###################################
# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
train_df = load_df(home_dir, "train.csv")

# create the training set: (without "target" and "id" column)
train_x = train_df.drop("target", axis=1).drop("id", axis=1)
target_y = train_df["target"]

train_x, test_x, train_y, test_y = train_test_split(train_x, target_y, test_size = 0.33, random_state = 5)

##########################################################################################


##########################################################################################
#####################################APPLYING ML LIBRARIES###################################

################## apply MLP Network:##################
y_prediction = apply_logistic_regression(train_x, train_y, test_x)
display_accuracy(y_prediction, test_y, "MLP Network")












