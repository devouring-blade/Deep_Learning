import numpy as np
from gradient_descent import gradient_descent
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

np.random.seed(42)

# generate a dataset for binary classification
x, y = make_blobs(n_samples = 400, n_features = 2,
                  centers = [[0.0, 0.0], [0.5, 0.1]],
                  center_box = (-1.0, 1.0), cluster_std = 0.15)
y = y.reshape(-1, 1)

# see the data visually
plt.figure(figsize = (7, 5))
color = [["red", "blue"][a] for a in y.reshape(-1,)]
plt.scatter(x[: , 0], x[: , 1], s = 50, c = color, alpha = 0.5)
plt.show()

# split data to train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y)

# create an single-layered ANN model
n_input = x.shape[1]     # the number of input neurons
n_output = 1     # the number of output neurons
alpha = 0.1     # learning rate

# initialize the parameters randomly
wo = np.random.normal(size= (n_input, n_output))    # weights of output layer
bo = np.zeros(shape= (1, n_output))     # bias of output layer
parameters = [wo, bo]   # parameter list


# activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# loss function: binary cross entropy
def loss(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))


# output from the ANN model: prediction process
def predict(x, proba = True):
    output = sigmoid(np.dot(x, parameters[0]) + parameters[1])
    if proba: return output     # return probability
    else: return (output > 0.5) * 1.0   # return class


# perform training and track the loss history
def train(x, y, x_val, y_val, epochs, batch_size):
    train_loss = []  # loss history of training data
    val_loss = []   # loss history of testing data
    for epoch in range(epochs):
        # measure the losses during training
        train_loss.append(loss(y, predict(x)))      # train loss
        val_loss.append(loss(y_val, predict(x_val)))    # val loss

        # perform training using mini-batch gradient descent
        for batch in range(int(x.shape[0] / batch_size)):
            idx = np.random.choice(x.shape[0], batch_size)
            gradient_descent(x[idx], y[idx], alpha, loss, predict, parameters)

        if epoch % 10 == 0:
                print(f"epoch: {epoch} --- train loss: {train_loss[-1]:.4f} --- val loss: {val_loss[-1]:.4f}")
    return train_loss, val_loss


# perform training
train_loss, val_loss = train(x_train, y_train, x_test, y_test, epochs= 200, batch_size= 50)

# visually check the loss history
plt.plot(train_loss, c = "blue", label = " train loss")
plt.plot(val_loss, c = "red", label = "val loss")
plt.legend()
plt.show()

# check the accuracy
y_pred = predict(x_train, proba = False)
acc = (y_pred == y_train).mean()
print(f"accuracy of training data: {acc:.4f}")

y_pred = predict(x_test, proba = False)
acc = (y_pred == y_test).mean()
print(f"accuracy of validation data: {acc:.4f}")


x_min, x_max = x_test[:, 0].min() - 0.1, x_test[:,0].max() + 0.1
y_min, y_max = x_test[:, 1].min() - 0.1, x_test[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                     np.linspace(y_min, y_max, 50))
x_in = np.c_[xx.ravel(), yy.ravel()]

# Predict the classes of the data points in the x_in variable.
y_pred = predict(x_in, proba=False).astype('int8')
y_pred = y_pred.reshape(xx.shape)

plt.figure(figsize=(6, 5))
m = ['o', '^']
color = ['red', 'blue']
for i in [0, 1]:
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1],
                c = color[i],
                marker = m[i],
                s = 40,
                edgecolor = 'black',
                alpha = 0.5,
                label='class-' + str(i))
plt.contour(xx, yy, y_pred, cmap=ListedColormap(color), alpha=0.5)
plt.axis('tight')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()





