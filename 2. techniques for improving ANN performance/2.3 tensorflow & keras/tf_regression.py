import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate a data set
x = np.random.random((1000, 1))
y = 2.0 * np.sin(2.0 * np.pi * x) + np.random.normal(0.0, 0.8, (1000, 1))

# Generate training, test data set
x_train, x_test, y_train, y_test = train_test_split(x, y)
x_pred = np.linspace(0, 1, 200).reshape(-1, 1)

# Visually see the data.
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.3, label='train')
plt.scatter(x_test, y_test, s=20, c='red', alpha=0.3, label='valid')
plt.legend()
plt.show()

# create an ANN with a hidden layer
n_input = x.shape[1]
n_output = 1
n_hidden = 8
lr = 0.05

# initialize the parameters
wh = tf.Variable(np.random.normal(size= (n_input, n_hidden)))
bh = tf.Variable(np.zeros(shape= (1, n_hidden)))
wo = tf.Variable(np.random.normal(size= (n_hidden, n_output)))
bo = tf.Variable(np.zeros(shape= (1, n_output)))
parameters = [wh, bh, wo, bo]

opt = optimizers.Adam(learning_rate = 0.01)

# loss function
def mse(y, y_hat):
    return tf.reduce_mean(tf.math.square(y - y_hat))

def predict(x):
    o_hidden = tf.nn.tanh(tf.matmul(x, parameters[0]) + parameters[1])
    o_output = tf.matmul(o_hidden, parameters[2]) + parameters[3]
    return o_output

def fit(x_train, y_train, x_val, y_val, epochs, batch_size):
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        # training with mini-batch
        for batch in range(int(x_train.shape[0] / batch_size)):
            idx = np.random.choice(x_train.shape[0], batch_size)
            x_bat = x_train[idx]
            y_bat = y_train[idx]

            # automatic differentiation and update parameters
            with tf.GradientTape() as tape:
                loss = mse(y_train, predict(x_train))
            grads = tape.gradient(loss, parameters)
            opt.apply_gradients(zip(grads, parameters))


        # loss history
        loss = mse(y_train, predict(x_train))
        train_loss.append(loss.numpy())

        loss = mse(y_val, predict(x_val))
        val_loss.append(loss.numpy())

        if epoch % 10 == 0:
            print(f"epoch: {epoch} --- train loss: {train_loss[-1]:.4f} --- val loss: {val_loss[-1]:.4f}")

    return train_loss, val_loss

# training
trn_loss, val_loss = fit(x_train, y_train, x_test, y_test, epochs=200, batch_size=50)

# Visually see the loss history.
plt.plot(trn_loss, c='blue', label='train loss')
plt.plot(val_loss, c='red', label='validation loss')
plt.legend()
plt.show()

# Visually check the prediction result.
y_pred = predict(x_pred)
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.3,
 label='train')
plt.scatter(x_test, y_test, s=20, c='red', alpha=0.3,
 label='validation')
plt.scatter(x_pred, y_pred, s=5, c='red', label='test')
plt.legend()
plt.show()



