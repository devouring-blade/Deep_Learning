from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from gradient_descent import gradient_descent
from matplotlib.colors import ListedColormap


# Generate a dataset for multiclass classification
x, y = make_blobs(n_samples=400, n_features=2,
                  centers=[[0., 0.], [0.5, 0.1], [1., 0.]],
                  cluster_std=0.15, center_box=(-1., 1.))

color = [["red", "blue", "green"][i] for i in y.reshape(-1)]
plt.scatter(x[:, 0], x[: , 1], c= color, alpha= 0.4)
plt.show()

n_input = x.shape[-1]
n_hidden = 16
n_output = len(np.unique(y))

wh = np.random.normal(size= (n_input, n_hidden))
bh = np.zeros(shape= (1, n_hidden))
wo = np.random.normal(size= (n_hidden, n_output))
bo = np.zeros(shape= (1, n_output))
parameters = [wh, bh, wo, bo]

y_ohe = np.eye(n_output)[y]

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, test_size= 0.2, shuffle= True)

def softmax(x):
    max_value = np.max(x, axis= 1).reshape(-1, 1)
    return np.exp(x - max_value) / np.sum(np.exp(x - max_value), axis= 1).reshape(-1, 1)

def relu(x): return np.maximum(0, x)

def ce(y, y_hat): return np.mean(np.sum(y * np.log(1 / y_hat), axis= 1))

def predict(x, proba= True):
    y_hat = softmax(relu(x @ wh + bh) @ wo + bo)
    if proba: return y_hat
    else: return np.argmax(y_hat, axis= 1)

def train(x, y, x_val, y_val, epochs, batch_size):
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        for batch in range(int(x.shape[0] / batch_size)):
            idx = np.random.choice(x.shape[0], batch_size)
            gradient_descent(x[idx], y[idx], predict, ce, parameters, 0.05)

        train_loss.append(ce(y, predict(x)))
        valid_loss.append(ce(y_val, predict(x_val)))

        if epoch % 10 == 0: print(f"epoch: {epoch} --- train loss: {train_loss[-1]:.2f} --- val loss: {valid_loss[-1]:.2f}")
    print(f"epoch: final --- train loss: {train_loss[-1]:.2f} --- val loss: {valid_loss[-1]:.2f}")

    return train_loss, valid_loss

train_loss, val_loss = train(x_train, y_train, x_test, y_test, 200, 50)

plt.plot(train_loss, color= "red", label= "train loss")
plt.plot(val_loss, color= "blue", label= "val loss")
plt.legend()
plt.show()


# Check the accuracy.
y_pred = predict(x_train, proba=False)
acc = (y_pred == np.argmax(y_train, axis=1)).mean()
print("Accuracy of training data = {:4f}".format(acc))
y_pred = predict(x_test, proba=False)
acc = (y_pred == np.argmax(y_test, axis=1)).mean()
print("Accuracy of test data = {:4f}".format(acc))



x_min, x_max = x_test[:, 0].min() - 0.1, x_test[:,0].max() + 0.1
y_min, y_max = x_test[:, 1].min() - 0.1, x_test[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
np.linspace(y_min, y_max, 50))
x_in = np.c_[xx.ravel(), yy.ravel()]
# Predict the classes of the data points in the x_in variable.
y_pred = predict(x_in, proba=False).astype('int8')
y_pred = y_pred.reshape(xx.shape)
plt.figure(figsize=(7, 5))
m = ['o', '^', 's']
color = ['red', 'blue', 'green']
for i in [0, 1, 2]:
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1],
c = color[i],
marker = m[i],
s = 80,
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
