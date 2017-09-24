import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import csv
import sys
rng = numpy.random

trainX, testX, trainY, testY = [],[],[],[]

print("Importing data...")
sys.stdout.flush()

# Read training data
with open("train.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        trainX.append(float(row[0]))
        trainY.append(row[1])

# Read testing data
with open("test.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        testX.append(float(row[0]))
        testY.append(row[1])

print("Initializing...")
sys.stdout.flush()

# Parameters
train_X = numpy.asarray(trainX)
train_Y = numpy.asarray(trainY)
test_X = numpy.asarray(testX)
test_Y = numpy.asarray(testY)
learning_rate = 0.01
training_epochs = 1000
display_step = 50000
step = 0
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    print("Training....")
    sys.stdout.flush()
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

            # Display logs per epoch step
            if (step+1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                    "W=", sess.run(W), "b=", sess.run(b))
                sys.stdout.flush();
            step+=1

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line vs Training')
    plt.legend()
    plt.savefig("fittedLineAgainstTraining.png")

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line vs Testing')
    plt.legend()
    plt.savefig("fittedLineAgainstTest.png")
