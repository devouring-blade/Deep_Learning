from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
from gradient_descent import gradient_descent
from sklearn.model_selection  import train_test_split
from matplotlib.colors import ListedColormap


# Generate a dataset
x, y = make_blobs(n_samples=400, n_features=2,
                  centers=[[0., 0.], [0.5, 0.1], [1., 0.]],
                  cluster_std=0.15, center_box=(-1., 1.))
y[y == 2] = 0
y = y.reshape(-1, 1)

color = [["red", "blue"][i] for i in y.ravel()]
plt.scatter(x[: , 0], x[: , 1], c= color)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, shuffle= True)

n_input = x.shape[-1]
n_hidden = 16
n_output = 1

wh = np.random.normal(size= (n_input, n_hidden))
bh = np.zeros(shape= (1, n_hidden))
wo = np.random.normal(size= (n_hidden, n_output))
bo = np.zeros(shape= (1, n_output))
parameters = [wh, bh, wo, bo]

def sigmoid(x): return 1. / (1. + np.exp(-x))
def relu(x): return np.maximum(0, x)
def bce(y, y_hat): return np.mean(y * np.log(1 / y_hat) + (1. - y) * np.log(1. / (1. - y_hat)))

def predict(x, proba= True):
    y_hat = sigmoid(relu(x @ wh + bh) @ wo + bo)
    if proba: return y_hat
    else: return (y_hat > 0.5) * 1.

def train(x, y, x_val, y_val, epochs, batch_size):
    train_loss = [bce(y, predict(x))]
    valid_loss = [bce(y_val, predict(x_val))]
    for epoch in range(epochs):
        for batch in range(int(x.shape[0] / batch_size)):
            idx = np.random.choice(x.shape[0], batch_size)
            gradient_descent(x[idx], y[idx], predict, bce, parameters, 0.1)

        train_loss.append(bce(y, predict(x)))
        valid_loss.append(bce(y_val, predict(x_val)))

        if epoch % 10 == 0: print(f"epoch: {epoch} ----- train loss: {train_loss[-1]:.2f} ----- val loss: {valid_loss[-1]:.2f}")
    print(f"epoch: {epochs} ----- train loss: {train_loss[-1]:.2f} ----- val loss: {valid_loss[-1]:.2f}")
    return train_loss, valid_loss

train_loss, val_loss = train(x, y, x_test, y_test, 200, 50)

plt.plot(train_loss, color= "red", label= "train loss")
plt.plot(val_loss, color= "blue", label= "val loss")
plt.legend()
plt.show()


# Check the accuracy.
y_pred = predict(x_train, proba=False)
acc = (y_pred == y_train).mean()
print("Accuracy of training data = {:4f}".format(acc))
y_pred = predict(x_test, proba=False)
acc = (y_pred == y_test).mean()
print("Accuracy of test data = {:4f}".format(acc))
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







