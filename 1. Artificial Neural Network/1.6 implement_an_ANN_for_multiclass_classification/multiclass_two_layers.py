from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from gradient_descent import gradient_descent
from matplotlib.colors import ListedColormap

np.random.seed(42)

# Generate a dataset for multiclass classification
x, y = make_blobs(n_samples=400, n_features=2,
                  centers=[[0., 0.], [0.5, 0.1], [1., 0.]],
                  cluster_std=0.15, center_box=(-1., 1.))
n_class = np.unique(y).size     # the number of classes
y_ohe = np.eye(n_class)[y]      # one-hot encode class y. y = [0, 1, 2]

# See the data visually.
plt.figure(figsize=(7,5))
color = [['red', 'blue', 'green'][a] for a in y.reshape(-1,)]
plt.scatter(x[:, 0], x[:, 1], s=50, c=color, alpha=0.5)
plt.show()

#generate the training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y_ohe)

# create an two-layered ANN model
n_input = x.shape[1]        # number of neuron in input layer
n_output = n_class      # number of neuron in output layer
n_hidden = 16       # number of neuron in hidden layer
alpha = 0.05        # learning rate

# initialize the parameters randomly
wh = np.random.normal(size= (n_input, n_hidden))
bh = np.zeros(shape= (1, n_hidden))
wo = np.random.normal(size= (n_hidden, n_output))
bo = np.zeros(shape= (1, n_output))
parameters = [wh, bh, wo, bo]

# activation function
def softmax(x):
    a = np.max(x, axis= 1, keepdims= True)
    e = np.exp(x - a)
    return e / np.sum(e,  axis= 1, keepdims= True)

def relu(x):
    return np.maximum(0, x)

# loss function: cross entropy
def loss(y, y_hat):
    ce = -np.sum(y * np.log(y_hat),  axis= 1)
    return np.mean(ce)

#output from ANN model: prediction process
def predict(x, proba= True):
    h_out = relu(np.dot(x, parameters[0]) + parameters[1])
    o_out = softmax(np.dot(h_out, parameters[2]) + parameters[3])
    if proba: return o_out      # return probability distribution
    else: return np.argmax(o_out, axis= 1)      # return class

# perform training and track the loss history
def train(x, y, x_val, y_val, epochs, batch_size):
    train_loss = []     # loss history of training data
    val_loss = []       # loss history of val data
    for epoch in range(epochs):
        # measure the losses during training
        train_loss.append(loss(y, predict(x)))      # train loss
        val_loss.append(loss(y_val, predict(x_val)))    # val loss

        # perform training using mini-batch gradient descent
        for batch in range(int(x.shape[0] / batch_size)):
            idx = np.random.choice(x.shape[0], batch_size)
            gradient_descent(x[idx], y[idx], alpha, loss, predict, parameters)

        if epoch % 10 == 0:
            print(f"epoch: {epoch} --- train loss: {train_loss[-1]} --- val loss: {val_loss[-1]}")

    return train_loss, val_loss

# Perform training
train_loss, val_loss = train(x_train, y_train, x_test, y_test,
                             epochs=250, batch_size=50)

# Visually check the loss history.
plt.plot(train_loss, c='blue', label='train loss')
plt.plot(val_loss, c='red', label='validation loss')
plt.legend()
plt.show()

# Check the accuracy.
y_pred = predict(x_train, proba=False)
acc = (y_pred == np.argmax(y_train, axis=1)).mean()
print("Accuracy of training data = {:4f}".format(acc))

y_pred = predict(x_test, proba=False)
acc = (y_pred == np.argmax(y_test, axis=1)).mean()
print("Accuracy of test data = {:4f}".format(acc))

# Visualize the decision boundaries.
# reference :
# https://psrivasin.medium.com/
# plotting-decision-boundaries-using-numpy-and-matplotlib-
# f5613d8acd19
x_min, x_max = x_test[:, 0].min() - 0.1, x_test[:,0].max() + 0.1
y_min, y_max = x_test[:, 1].min() - 0.1, x_test[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
x_in = np.c_[xx.ravel(), yy.ravel()]

# Predict the classes of the data points in the x_in variable.
y_pred = predict(x_in, proba=False).astype('int8')
y_pred = y_pred.reshape(xx.shape)
plt.figure(figsize=(7, 5))
m = ['o', '^', 's']
color = ['red', 'blue', 'green']
for i in [0, 1, 2]:
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1], c = color[i], marker = m[i],
                s = 80, edgecolor = 'black', alpha = 0.5, label='class-' + str(i))
plt.contour(xx, yy, y_pred, cmap=ListedColormap(color), alpha=0.5)
plt.axis('tight')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()


















