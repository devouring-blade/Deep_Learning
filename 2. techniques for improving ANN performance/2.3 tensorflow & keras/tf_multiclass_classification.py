import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate a dataset
x, y = make_blobs(n_samples=400, n_features=2,
                  centers=[[0., 0.], [0.5, 0.1], [1., 0.]],
                  cluster_std=0.15, center_box=(-1., 1.))
n_class = len(np.unique(y)) # the number of classes

# one-hot encode class y, y = [0,1,2]
y_ohe = np.eye(n_class)[y]
x_train, x_test, y_train, y_test = train_test_split(x, y_ohe)

# Visually see the data.
plt.figure(figsize=(5,4))
color = [['red', 'blue', 'green'][a] for a in y.reshape(-1,)]
plt.scatter(x[:,0],x[:,1],s=100,c=color,alpha=0.3)
plt.show()

# create an ANN with a hidden layer
n_input = x_train.shape[1]
n_output = n_class
n_hidden = 8
lr = 0.05

# initialize the parameters
wh = tf.Variable(np.random.normal(size= (n_input, n_hidden)))
bh = tf.Variable(np.zeros(shape= (1, n_hidden)))
wo = tf.Variable(np.random.normal(size= (n_hidden, n_output)))
bo = tf.Variable(np.zeros(shape= (1, n_output)))
parameters = [wh, bh, wo, bo]

opt = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)

# loss function
def cross_entropy(y, y_hat):
    return -tf.reduce_mean(tf.reduce_sum(y * tf.math.log(y_hat), axis= 1))

# predict
def predict(x, proba = True):
    o_hidden = tf.nn.relu(tf.matmul(x, parameters[0]) + parameters[1])
    o_output = tf.nn.softmax(tf.matmul(o_hidden, parameters[2]) + parameters[3])
    if proba: return o_output
    else: return tf.math.argmax(o_output, axis= 1)

def fit(x_train, y_train, x_val, y_val, epochs, batch_size):
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        # training with mini_batch
        for batch in range(int(x_train.shape[0] / batch_size)):
            idx = np.random.choice(x_train.shape[0], batch_size)
            x_bat = x_train[idx]
            y_bat = y_train[idx]
            
            # automatic differentiation
            with tf.GradientTape() as tape:
                loss = cross_entropy(y_bat, predict(x_bat))

            # find the gradient and update the parameters
            grads = tape.gradient(loss, parameters)
            opt.apply_gradients(zip(grads, parameters))

        # loss history
        loss = cross_entropy(y_train, predict(x_train))
        train_loss.append(loss)

        loss = cross_entropy(y_val, predict(x_val))
        val_loss.append(loss)

        if epoch % 10 == 0:
            print(f"epoch: {epoch} --- train loss: {train_loss[-1]:.4f} --- val loss: {val_loss[-1]:.4f}")

    return train_loss, val_loss

# training
train_loss, val_loss = fit(x_train, y_train, x_test, y_test, epochs= 200, batch_size= 50)

# Visually see the loss history
plt.plot(train_loss, c='blue', label='train loss')
plt.plot(val_loss, c='red', label='test loss')
plt.legend()
plt.show()

# Check the accuracy of the test data
y_pred = predict(x_test, proba=False).numpy()
acc = np.mean(y_pred == np.argmax(y_test, axis=1))
print("Accuracy of test data = {:4f}".format(acc))









