from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from gradient_descent import gradient_descent
from matplotlib.colors import ListedColormap

np.random.seed(42)

# Generate a dataset
x, y = make_blobs(n_samples=400, n_features=2,
                  centers=[[0., 0.], [0.5, 0.1], [1., 0.]],
                  cluster_std=0.15, center_box=(-1., 1.))
y[y == 2] = 0 # y = [0, 1, 2] → [0, 1]
y = y.reshape(-1, 1)

# See the data visually.
plt.figure(figsize=(7,5))
color = [['red', 'blue'][a] for a in y.reshape(-1,)]
plt.scatter(x[:, 0], x[:, 1], s=50, c=color, alpha=0.5)
plt.show()

# generate the training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# create two-layered ANN model
n_input = x.shape[1]    # the number neuron of input layer
n_output = 1    # the number neuron of output layer
n_hidden = 16   # the number neuron of hidden layer
alpha = 0.1     # learning rate

# initialize the parameters randomly
wh = np.random.normal(size= (n_input, n_hidden))    # weights of hidden layer
bh = np.zeros(shape= (1, n_hidden))     # biases of hidden layer
wo = np.random.normal(size= (n_hidden, n_output))   # weights of output layer
bo = np.zeros(shape= (1, n_output))     # bias of output layer
parameters = [wh, bh, wo, bo]   # parameters list

# activation function
def sigmoid(x): return 1. / (1. + np.exp(-x))
def relu(x): return np.maximum(0, x)

# loss function: binary cross entropy
def loss(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1. - y) * np.log(1. - y_hat))

# output from the ANN model: prediction process
def predict(x, proba= True):
    h_out = relu(np.dot(x, parameters[0]) + parameters[1])  # output from hidden layer
    o_out = sigmoid(np.dot(h_out, parameters[2]) + parameters[3])   # output from output layer
    if proba: return o_out  # return probability
    else: return (o_out > 0.5) * 1  # return class

# perform training and track the loss history
def train(x_train, y_train, x_val, y_val, epochs, batch_size):
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        # measure the losses during training
        train_loss.append(loss(y, predict(x)))      # train loss
        val_loss.append(loss(y_val, predict(x_val)))    # val loss

        # perform training using mini-batch gradient descent
        for batch in range(int(x_train.shape[0] / batch_size)):
            idx = np.random.choice(x_train.shape[0], batch_size)
            gradient_descent(x_train[idx], y_train[idx], alpha,
                             loss, predict, parameters)

        if epoch % 10 == 0:
            print(f"epoch: {epoch} --- train loss: {train_loss[-1]} --- val loss: {val_loss[-1]}")

    return train_loss, val_loss

# perform training
train_loss, val_loss = train(x_train, y_train, x_test, y_test,
                             epochs= 200, batch_size= 50)

# Visually check the loss history.
plt.plot(train_loss, c='blue', label='train loss')
plt.plot(val_loss, c='red', label='test loss')
plt.legend()
plt.show()

# check the accuracy of training data
y_pred = predict(x_train, proba= False)
acc = (y_pred == y_train).mean()
print(f"accuracy of training data: {acc:4f}")

# check the accuracy of val data
y_pred = predict(x_test, proba= False)
acc = (y_pred == y_test).mean()
print(f"accuracy of training data: {acc:4f}")

# Visualize the non-linear decision boundary
# reference : https://psrivasin.medium.com/
# plotting-decision-boundaries-using-numpy-and-matplotlib-
# f5613d8acd19
x_min, x_max = x_test[:, 0].min() - 0.1, x_test[:,0].max() + 0.1
y_min, y_max = x_test[:, 1].min() - 0.1, x_test[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
np.linspace(y_min, y_max, 50))
x_in = np.c_[xx.ravel(), yy.ravel()]

# Predict the classes of the data points in the x_in variable.
y_pred = predict(x_in, proba=False).astype('int8')
y_pred = y_pred.reshape(xx.shape)

plt.figure(figsize=(5, 5))
m = ['o', '^']
color = ['red', 'blue']
for i in [0, 1]:
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1], c = color[i], marker = m[i], s = 40,
                edgecolor = 'black', alpha = 0.5, label='class-' + str(i))
plt.contour(xx, yy, y_pred, cmap=ListedColormap(color), alpha=0.5)
plt.axis('tight')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
















