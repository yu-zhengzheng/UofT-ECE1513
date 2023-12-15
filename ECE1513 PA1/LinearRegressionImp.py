import numpy as np
from numpy import linalg
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    #adds one column of 1 to X train
    shape=np.shape(X_train)
    ones = np.ones((shape[0], 1))
    X_train=np.concatenate((ones, X_train), axis=1)

    #computes best fit
    w=np.dot(np.dot(linalg.pinv(np.dot(np.transpose(X_train), X_train)), np.transpose(X_train)), y_train)
    return w

def mse(X_train,y_train,w):
    #computes prediction using w
    y_pred=pred(X_train,w)

    #computes average error
    #avgError=sum((y_train-y_pred)**2)/len(y_train)
    avgError=mean_squared_error(y_train, y_pred)
    return avgError

def pred(X_train,w):
    # adds one column of 1 to X train
    shape = np.shape(X_train)
    ones = np.ones((shape[0], 1))
    X_train=np.concatenate((ones, X_train), axis=1)

    #compute prediction and return
    return np.dot(X_train,w)

def test_SciKit(X_train, X_test, Y_train, Y_test):
    #creates a linear regression model from scikit
    LR=linear_model.LinearRegression()

    #compute linear regression using the model
    LR.fit(X_train, Y_train)
    w=np.concatenate((np.asarray([LR.intercept_]), LR.coef_))

    #computes prediction on training data using w
    train_pred = pred(X_train, w)

    #computes average error
    #avgError = sum((Y_train - train_pred) ** 2) / len(Y_train)
    avgError=mean_squared_error(Y_train,train_pred)
    return(avgError)

def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6, ], [4, 8]])
    y_train = np.asarray([1,2,3,4])

    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
     X_train, y_train = load_diabetes(return_X_y=True)
     X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)

     w=fit_LinRegr(X_train, y_train)

     #Testing Part 2a
     e=mse(X_test,y_test,w)

     #Testing Part 2b
     scikit=test_SciKit(X_train, X_test, y_train, y_test)

     print("Mean squared error from Part 2a is ", e)
     print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()
#the results from my implementation is usually slightly worse than the module from the scikit-learn library
#%%