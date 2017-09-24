import numpy as np
import pandas as pd

# Get dataset and target matrix from csv file
# 0 Year, 1 Week of Year, 2 Day of Week, 3 Registered norm, 4 Casual norm
train00 = pd.read_csv('.\Data\\train00.csv', header=None)
# Data matrix
X_train00 = train00.iloc[:, 2].values
# Target matrix
y_train00 = train00.iloc[:, 3].values

print("\nRaw data matrix:")
print(train00.head())
print("Input matrix:")
print(X_train00)
print("Target matrix:")
print(y_train00)
#print("Weights matrix")
#print(weights)


def LinearRegressionGD(X, y, eta=0.00001, n_iter=20000):
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
            if (n % 2000 == 0 or n == n_iter - 1) and i % 100 == 0:
                print("Iteration: ", n)
                print("weights:", weights)
                print("error:", error)
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
    print("weights_ML", weights_ML)
    return weights_ML

LinearRegressionGD(X_train00, y_train00)
LinearRegressionCF(X_train00, y_train00)
