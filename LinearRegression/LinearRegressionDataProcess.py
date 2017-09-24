import numpy as np
import pandas as pd

def LinearRegressionGD(X, y, eta=0.0001, n_iter=10000):
    """
    Linear Regression by Gradient Descent (GD)
    X : Training dataset matrix
    y : Target values matrix
    eta : Learning rate
    n_iter : Number of iterations to train for

    Fitting to a polnomial of order 2
    """

    weights = np.zeros(3)

    for n in range(n_iter):
        for i in range(X.size):
            phi = np.array([1, X[i], X[i] ** 2])
            error = y[i] - np.dot(phi, weights)
            weights += eta * error * phi
            """
            if (n % 2000 == 0 or n == n_iter - 1) and i % 100 == 0:
                print("Iteration: ", n)
                print("weights:", weights)
                print("error:", error)
            """
    #print("weights_GD", weights)
    return weights

def LinearRegressionCF(X, y):
    """
    Linear regression closed form

    Phi is the design matrix (eq. 3.16 in book)
    The function fits to a polynomial of order 2
    """
    # Design matrix (eq. 3.16)
    Phi = np.array([np.ones(X.size), X, X ** 2])
    # Moore-Penrose pseudoinverse (3.17)
    pinvPhi = np.transpose(np.linalg.pinv(Phi))
    # Normal equation (3.15)
    weights_ML = np.matmul(pinvPhi, y)
    #print("weights_ML", weights_ML)
    return weights_ML

def collectiveError(X, y, weights):
    error = 0
    for i in range(X.size):
        phi = np.array([1, X[i], X[i] ** 2])
        # eq 1.2, 3.12
        error += (y[i] - np.dot(phi, weights)) ** 2
    error = 0.5 * error
    #errorRMS = (2 * error / X.size) ** 0.5
    return error

def errorRMS(X, error):
    return (2 * error / X.size) ** 0.5

def getDataMatrix(X):
    return X.iloc[:, 2].values
def getTargetMatrixRegistered(y):
    return y.iloc[:, 3].values
def getTargetMatrixCasual(y):
    return y.iloc[:, 4].values

def processData(trainingSet, validationSet):
    # Create arrays
    X_train = getDataMatrix(trainingSet)
    y_train_registered = getTargetMatrixRegistered(trainingSet)
    y_train_casual = getTargetMatrixCasual(trainingSet)

    X_validation = getDataMatrix(validationSet)
    y_validation_registered = getTargetMatrixRegistered(validationSet)
    y_validation_casual = getTargetMatrixCasual(validationSet)

    # Calculate registered weights using Gradient Descent (GD) and Maximum Likelihood (ML)
    weights_GD_registered = LinearRegressionGD(X_train, y_train_registered)
    weights_ML_registered = LinearRegressionCF(X_train, y_train_registered)
    myData.write("\t\t\t\t  w0\t      w1\t  w2\n")
    print("weights_GD_registered", weights_GD_registered)
    myData.write("weights_GD_registered\t")
    myData.write(np.array_str(weights_GD_registered))
    myData.write("\n")
    print("weights_ML_registered", weights_ML_registered)
    myData.write("weights_ML_registered\t")
    myData.write(np.array_str(weights_ML_registered))
    myData.write("\n")

    # Calculate registered error
    error_GD = collectiveError(X_validation, y_validation_registered, weights_GD_registered)
    error_GD_RMS = errorRMS(X_validation, error_GD)
    myData.write("error_GD\t")
    myData.write(str(error_GD))
    myData.write("\n")
    myData.write("error_GD_RMS\t")
    myData.write(str(error_GD_RMS))
    myData.write("\n")

    error_ML = collectiveError(X_validation, y_validation_registered, weights_ML_registered)
    error_ML_RMS = errorRMS(X_validation, error_ML)
    myData.write("error_ML\t")
    myData.write(str(error_ML))
    myData.write("\n")
    myData.write("error_ML_RMS\t")
    myData.write(str(error_ML_RMS))
    myData.write("\n")

# Get dataset and target matrix from csv file
# 0 Year, 1 Week of Year, 2 Day of Week, 3 Registered norm, 4 Casual norm

# Outputs stored in data.txt
myData = open("data.txt", "w")
myData.write("Registered riders fit\neta = 0.0001\nn_iter = 10000\n")
myData.write("polynomial being fit: y(x,w) = w0, w1 * x, w2 * x^2\n")

print("\nDataset00")
myData.write("Dataset00\n")
train00 = pd.read_csv('.\Data\\train.00.csv', header=None)
validation00 = pd.read_csv('.\Data\\validation.00.csv', header=None)
processData(train00, validation00)

print("\nDataset01")
myData.write("\nDataset01\n")
train01 = pd.read_csv('.\Data\\train.01.csv', header=None)
validation01 = pd.read_csv('.\Data\\validation.01.csv', header=None)
processData(train01, validation01)

print("\nDataset02")
myData.write("\nDataset02\n")
train02 = pd.read_csv('.\Data\\train.02.csv', header=None)
validation02 = pd.read_csv('.\Data\\validation.02.csv', header=None)
processData(train02, validation02)

print("\nDataset03")
myData.write("\nDataset03\n")
train03 = pd.read_csv('.\Data\\train.03.csv', header=None)
validation03 = pd.read_csv('.\Data\\validation.03.csv', header=None)
processData(train03, validation03)

print("\nDataset04")
myData.write("\nDataset04\n")
train04 = pd.read_csv('.\Data\\train.04.csv', header=None)
validation04 = pd.read_csv('.\Data\\validation.04.csv', header=None)
processData(train04, validation04)
