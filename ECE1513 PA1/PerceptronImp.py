import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def fit_perceptron(X_train, y_train):
    X0 = np.ones((X_train.shape[0], 1))  # set one column vector containing all element with 1
    X_train = np.hstack((X0, X_train))  # add one all 1 column in the front of X_train matrix
    w_current = np.zeros(X_train.shape[1])  # set the initial  all zero weight vector
    w=np.copy(w_current)  # the weight with the minimized loss function
    error=1  # the minimum of the loss function
    epoch = 5000  # set the maximum epoch up to 5000
    flag = 0 # flag=1 if the loss is 0, flag=0 otherwise

    while (flag == 0):
        epoch -= 1
        flag = 1
        for i in range(0, X_train.shape[0]):
            if ((2 * (np.dot(X_train[i], w_current) > 0) - 1) * y_train[i]) < 0:  # check whether the X_train[i] is misclassified
                flag = 0
                w_current = w_current + y_train[i] * X_train[i]  # update the current weight
                break

        #if the w_current yields a smaller loss, update w and error
        if ((errorPer(X_train, y_train, w_current))<error):
            w=w_current
            error=errorPer(X_train, y_train, w_current)
        #if 5000 epoch is reached exit the loop
        if (epoch == 0):
            break

    return w  # output the weight that best separates the two classes of training data points


def errorPer(X_train, y_train, w):
    # computes average error=num of misclassified points/total number of points
    avgError = sum(y_train!=pred(X_train[:,1:6], w))/len(y_train)
    return avgError


def confMatrix(X_train, y_train, w):
    a = 0  # position [0,0] count number
    b = 0  # position [0,1] count number
    c = 0  # position [1,0] count number
    d = 0  # position [1,1] count number
    y_pred = pred(X_train, w)  # convert predict output vector
    for i in range(0, len(y_train)):  # check each y_predict with y_train using for loop
        if y_pred[i] < y_train[i]:
            c += 1  # count the number of False negative classified point
        elif y_pred[i] > y_train[i]:
            b += 1  # count the number of False Positive classified point
        elif y_pred[i] + y_train[i] == 2:
            d += 1  # count the number of True Positive classified point
        elif y_pred[i] + y_train[i] == -2:
            a += 1  # count the number of True Negative classified point

    return np.array([[a, b], [c, d]])  # place the number of 4 types of classified and misclassified point in 2*2 confusion matrix


def pred(X_train, w):
    # set one column vector containing all element with 1
    X0 = np.ones((X_train.shape[0], 1))
    # add one all 1 column in the front of X_train matrix
    X_train = np.hstack((X0, X_train))
    # compute the predicted output with input dataset and w
    y_predict = np.dot(X_train, w)
    # classified the given input by computing the w and output the predicted output vector
    return (2 * (y_predict > 0) - 1)


def test_SciKit(X_train, X_test, Y_train, Y_test):
    # initiate a perceptron model with parameters
    #pct=Perceptron(max_iter=5000, tol=1e-5, verbose=1)
    pct = Perceptron()
    # fit the model with training data
    pct.fit(X_train, Y_train)

    # use the model to predict on the test data
    pred_pct = pct.predict(X_test)
    # compute the confusion matrix for the performance of the model
    out = confusion_matrix(Y_test, pred_pct)
    # compute the accuracy of the model for easier evaluation
    score = accuracy_score(Y_test, pred_pct)
    #print("the accuracy score of the model is:", score)
    # return the matrix
    return out


def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:], y_train[50:], test_size=0.2)

    # Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    # Pocket algorithm using Numpy
    w = fit_perceptron(X_train, y_train)
    cM = confMatrix(X_test, y_test, w)

    # Pocket algorithm using scikit-learn
    sciKit = test_SciKit(X_train, X_test, y_train, y_test)

    # Print the result
    print('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ", cM)
    print("Confusion Matrix from Part 1b is:", sciKit)


test_Part1()