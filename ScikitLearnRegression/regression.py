import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import sys
import csv
import random
rng = np.random
trainX, testX, trainY, testY = [],[],[],[]

print("Importing data...")
sys.stdout.flush()

# Read training data
with open("train.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        trainX.append(float(row[0])+random.uniform(0,0.02)-0.01)
        trainY.append(float(row[1]))

# Read testing data
with open("test.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        testX.append(float(row[0])+random.uniform(0,0.02)-0.01)
        testY.append(float(row[1]))

print("Initializing...")
sys.stdout.flush()

degrees = [1, 2, 3, 5]

X = np.asarray(trainX)
y = np.asarray(trainY)
X_test = np.asarray(testX)
y_test = np.asarray(testY)

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    print('Fitting degree {}'.format(degrees[i]))
    sys.stdout.flush()
    
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=5)

    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, y_test, label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 6))
    plt.ylim((0, 1))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))

plt.savefig('fits'.format(degrees[i]))
