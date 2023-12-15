import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def fit_NeuralNetwork(X_train, y_train, alpha, hidden_layer_sizes, epochs):
    # Initialize the epoch errors
    err = np.zeros((epochs, 1))

    # Initialize the architecture
    N, d = X_train.shape
    X0 = np.ones((N, 1))
    X_train = np.hstack((X0, X_train))
    d = d + 1
    L = len(hidden_layer_sizes)
    L = L + 2

    # Initializing the weights for input layer
    weight_layer = np.random.normal(0, 0.1, (d, hidden_layer_sizes[0]))  # np.ones((d,hidden_layer_sizes[0]))
    weights = []
    weights.append(weight_layer)  # append(0.1*weight_layer)

    # Initializing the weights for hidden layers
    for l in range(L - 3):
        weight_layer = np.random.normal(0, 0.1, (hidden_layer_sizes[l] + 1, hidden_layer_sizes[l + 1]))
        weights.append(weight_layer)

    # Initializing the weights for output layers
    weight_layer = np.random.normal(0, 0.1, (hidden_layer_sizes[l + 1] + 1, 1))
    weights.append(weight_layer)

    # run the algorithm for e epochs
    for e in range(epochs):
        choiceArray = np.arange(0, N)
        np.random.shuffle(choiceArray)
        errN = 0
        for n in range(N):
            index = choiceArray[n]
            x = np.transpose(X_train[index])
            # Model Update: Forward Propagation, Backpropagation
            X, S = forwardPropagation(x, weights)
            g = backPropagation(X, y_train[index], S, weights)
            # update the weight and compute the error
            weights = updateWeights(weights, g, alpha)
            errN = errN + errorPerSample(X[L - 1], y_train[index])
        err[e] = errN / N
    return err, weights


def forwardPropagation(x, weights):
    l = len(weights) + 1
    num_input = 1
    if len(x.shape) != 1:
        print("-----Warning-----")
        num_input = x.shape[0]

    retS = []
    retX = []
    currX = x
    retX.append(currX)

    # Forward Propagate for each layer
    for i in range(l - 1):
        # compute vector of sums for each x
        currS = np.dot(currX, weights[i])
        currX = currS

        # apply activation
        if i != len(weights) - 1:
            for j in range(len(currS)):
                currX[j] = activation(currS[j])

            # add bias to next layer
            if num_input == 1:
                currX = np.hstack((1, currX))
            else:
                currX = np.hstack((np.ones([num_input, 1]), currX))
        # apply output function
        else:
            currX = outputf(currS)

        retS.append(currS)
        retX.append(currX)

    return retX, retS


def errorPerSample(X, y_n):
    if y_n == 1:
        return -np.log(X)
        # implement the error function with label of y=1
    if y_n == -1:
        return -np.log(1 - X)
        # implement the error function with label of y=-1


def backPropagation(X, y_n, s, weights):
    # x:0,1,...,L
    # S:1,...,L
    # weights: 1,...,L
    l = len(X)
    delL = []

    # To be able to complete this function, you need to understand this line below
    # In this line, we are computing the derivative of the Loss function w.r.t the
    # output layer (without activation). This is dL/dS[l-2]
    # By chain rule, dL/dS[l-2] = dL/dy * dy/dS[l-2] . Now dL/dy is the derivative Error and
    # dy/dS[l-2]  is the derivative output.
    delL.insert(0, derivativeError(X[l - 1], y_n) * derivativeOutput(s[l - 2]))

    curr = 0

    # Now, let's calculate dL/dS[l-2], dL/dS[l-3],...
    for i in range(len(X) - 2, 0, -1):  # 2,1,0
        delNextLayer = delL[curr]
        WeightsNextLayer = weights[i]

        # print("WeightsNextLayer.shape=",WeightsNextLayer.shape)
        sCurrLayer = s[i - 1]

        # Init this to 0s vector
        delN = np.zeros((len(s[i - 1]), 1))
        # print("delN.shape=",delN.shape)

        # Now we calculate the gradient backward
        # Remember: dL/dS[i] = dL/dS[i+1] * W(which W???) * activation
        for j in range(len(s[i - 1])):  # number of nodes in layer i - 1
            for k in range(len(s[i])):  # number of nodes in layer i
                # calculate delta at node j
                #                                       WNL[0][] is the weights on bias
                delN[j] = delN[j] + delNextLayer[k] * WeightsNextLayer[j + 1][k] * derivativeActivation(sCurrLayer[j])  # Fill in the rest

        # print("delN.shape=",delN.shape)
        delL.insert(0, delN)

        # print("delL",delL)

    # We have all the deltas we need. Now, we need to find dL/dW.
    # It's very simple now, dL/dW = dL/dS * dS/dW = dL/dS * X
    g = []
    for i in range(len(delL)):
        rows, cols = weights[i].shape
        gL = np.zeros((rows, cols))
        currX = X[i]
        currdelL = delL[i]
        # print("currX.shape=",currX.shape)
        # print("currdelL.shape=",currdelL.shape)
        for j in range(rows):
            for k in range(cols):
                # Calculate the gradient using currX and currdelL
                gL[j, k] = currdelL[k] * currX[j]  # Fill in here
        # append curret layer's gradient to the list
        g.append(gL)
    return g


def updateWeights(weights, g, alpha):
    nW = []
    for i in range(len(weights)):
        rows, cols = weights[i].shape
        currWeight = weights[i]
        currG = g[i]
        for j in range(rows):
            for k in range(cols):
                # Gradient Descent Update
                currWeight[j, k] = currWeight[j, k] - alpha * currG[j, k]
        nW.append(currWeight)
    return nW


# implement relu function as the activation fuction with respect of s
def activation(s):
    return (s + abs(s)) / 2


# implement derivative relu function as the activation fuction with respect of s
def derivativeActivation(s):
    if s > 0:
        return 1
    else:
        return 0


# implement the output function as the logistic function
def outputf(s):
    return 1 / (1 + np.exp(-s))


# implement the derivative output function
def derivativeOutput(s):
    return outputf(s) * (1 - outputf(s))


# implement the derivative error function
def derivativeError(x_L, y):
    if y == 1:
        return -1 / x_L  # implement the derivative error with label of y=1
    if y == -1:
        return 1 / (1 - x_L)  # implement the derivative error with label of y=-1


def pred(x_n, weights):
    # Enter implementation here
    #     initialize the class vector
    c = []
    #    use forward propagation function to compute the sum of dot product of input and weight as S, updated input as X
    N, d = x_n.shape
    X0 = np.ones((N, 1))
    X_train = np.hstack((X0, x_n))
    X, S = forwardPropagation(X_train, weights)

    return (X[3] >= 0.5) * 2 - 1


def confMatrix(X_train, y_train, w):
    # Enter implementation here
    # use the prediction function to predict on the input
    predc = pred(X_train, w)
    # use scikit-learn to compute the confusion matrix on the true class and prediction class
    # print("y_train: ",y_train)
    # print("predc: ", predc)
    cm = confusion_matrix(y_train, predc)
    return cm


def plotErr(e, epochs):
    # Enter implementation here
    #     set the x axis by the epoch number
    x_ticks = range(epochs)
    #     plot the error at each epoch with labels and title
    plt.plot(x_ticks, e)
    plt.legend(['avg error'])
    plt.xlabel('epoch')
    plt.ylabel('avg error')
    plt.title('Error over the Training Epochs')
    plt.show()


def test_SciKit(X_train, X_test, Y_train, Y_test):
    # Enter implementation here
    # build the model and train by training data
    pct = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(30, 10), random_state=1, max_iter=300, verbose=True)
    pct.fit(X_train, Y_train)
    # Pass in the test features into the trained model
    pred_pct_test = pct.predict(X_test)
    pred_pct_train = pct.predict(X_train)
    # obtain the testing accuracy
    accuracyTest = accuracy_score(Y_test, pred_pct_test)
    print("testing accuracy is ", accuracyTest)
    # obtain the training accuracy
    accuracyTrain = accuracy_score(Y_train, pred_pct_train)
    print("training accuracy is ", accuracyTrain)
    # obtain the confusion matrix on test data
    cm = confusion_matrix(Y_test, pred_pct_test)
    return cm


def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:], y_train[50:], test_size=0.2, random_state=1)

    for i in range(80):
        if y_train[i] == 1:
            y_train[i] = -1
        else:
            y_train[i] = 1
    for j in range(20):
        if y_test[j] == 1:
            y_test[j] = -1
        else:
            y_test[j] = 1

    err, w = fit_NeuralNetwork(X_train, y_train, 1e-2, [30, 10], 100)

    plotErr(err, 100)

    # print("Xt= ", X_test.shape)
    # print("yt= ", y_test.shape)
    cM = confMatrix(X_test, y_test, w)

    sciKit = test_SciKit(X_train, X_test, y_train, y_test)

    print("Confusion Matrix is from Part 1a is: ", cM)
    print("Confusion Matrix from Part 1b is:", sciKit)


test_Part1()
