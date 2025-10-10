import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate a data set
x, y = make_blobs(n_samples=300, n_features=2,
                  centers=[[0., 0.], [0.5, 0.1]],
                  cluster_std=0.2, center_box=(-1., 1.))
y = y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Visually see the data.
plt.figure(figsize=(6, 4))
color = [['red', 'blue'][a] for a in y.reshape(-1, )]
plt.scatter(x[:, 0], x[:, 1], s=100, c=color, alpha=0.3)
plt.show()

# create an ANN with a hidden layer
n_input = x.shape[1]  # number of input neurons
n_output = 1  # number of output neurons
n_hidden = 8  # number of hidden neurons
lr = 0.05  # learning rate

# initialize the parameters
wh = tf.Variable(np.random.normal(size=(n_input, n_hidden)))
bh = tf.Variable(np.zeros(shape=(1, n_hidden)))
wo = tf.Variable(np.random.normal(size=(n_hidden, n_output)))
bo = tf.Variable(np.zeros(shape=(1, n_output)))
parameters = [wh, bh, wo, bo]


# loss function
def binary_cross_entropy(y, y_hat):
    return -tf.reduce_mean(y * tf.math.log(y_hat) + (1. - y) * tf.math.log(1. - y_hat))

def predict(x, proba = True):
    o_hidden = tf.nn.relu(tf.matmul(x, parameters[0]) + parameters[1])
    o_output = tf.nn.sigmoid(tf.matmul(o_hidden, parameters[2]) + parameters[3])
    if proba: return o_output           # return sigmoid output as is
    else: return (o_output > 0.5) * 1   # return class

def fit(x_train, y_train, x_val, y_val, epochs, batch_size):
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        # training with mini-batch
        for batch in range(int(x_train.shape[0] / batch_size)):
            idx = np.random.choice(x_train.shape[0], batch_size)
            x_bat = x_train[idx]
            y_bat = y_train[idx]

            # automatic differentiation
            with tf.GradientTape() as tape:
                loss = binary_cross_entropy(y_bat, predict(x_bat))

            # find the gradients of loss
            grads = tape.gradient(loss, parameters)

            # update parameters by the gradient descent
            for i in range(len(parameters)):
                parameters[i].assign_sub(lr * grads[i])

        # loss history
        loss = binary_cross_entropy(y_train, predict(x_train))
        train_loss.append(loss.numpy())

        loss = binary_cross_entropy(y_val, predict(x_val))
        val_loss.append(loss.numpy())

        if epoch % 10 == 0:
            print(f"epoch: {epoch} --- train loss: {train_loss[-1]:.4f} --- val loss: {val_loss[-1]:.4f}")

    return train_loss, val_loss

train_loss, val_loss = fit(x_train, y_train, x_test, y_test, epochs= 200, batch_size= 50)

# Visually see the loss history
plt.plot(train_loss, c='blue', label='train loss')
plt.plot(val_loss, c='red', label='validation loss')
plt.legend()
plt.show()
# Check the accuracy of the test data
y_pred = predict(x_test, proba=False).numpy()
acc = np.mean(y_pred == y_test)
print("\nAccuracy of the test data = {:4f}".format(acc))








