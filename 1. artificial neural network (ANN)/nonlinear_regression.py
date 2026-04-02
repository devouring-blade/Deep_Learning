import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from gradient_descent import gradient_descent


# Generate training data set
x = np.random.random((1000, 1))
y = 2.0 * np.sin(2.0 * np.pi * x) + np.random.normal(0.0, 0.8, (1000, 1))

# Generate training, validation, and test data set
x_train, x_valid, y_train, y_valid = train_test_split(x, y)
x_test = np.linspace(0, 1, 200).reshape(-1, 1)

# See the data visually.
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.3, label='train')
plt.scatter(x_valid, y_valid, s=20, c='red', alpha=0.3, label='valid')
plt.legend()
plt.show()

n_input = x.shape[-1]
n_hidden = 16
n_output = 1
alpha = 0.01

wh = np.random.normal(size= (n_input, n_hidden))
bh = np.zeros(shape= (1, n_hidden))
wo = np.random.normal(size= (n_hidden, n_output))
bo = np.zeros(shape= (1, n_output))
parameters = [wh, bh, wo, bo]

def relu(x): return np.maximum(0, x)

def tanh(x): return (1 - np.exp(-x)) / (1 + np.exp(-x))

def mse(y, y_hat): return np.mean(np.square(y - y_hat))

def predict(x): return tanh(x @ wh + bh) @ wo + bo

def train(x, y, x_val, y_val, epochs, batch_size):
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        for batch in range(int(x.shape[0] / batch_size)):
            idx = np.random.choice(x.shape[0], batch_size)
            gradient_descent(x[idx], y[idx], predict, mse, parameters, alpha)
        train_loss.append(mse(y, predict(x)))
        valid_loss.append(mse(y_val, predict(x_val)))

        if epoch % 10 == 0: print(f"epoch: {epoch} --- train loss: {train_loss[-1]:.2f} --- val loss: {valid_loss[-1]:.2f}")
    print(f"epoch: {epochs} --- train loss: {train_loss[-1]:.2f} --- val loss: {valid_loss[-1]:.2f}")
    return train_loss, valid_loss

train_loss, val_loss = train(x_train, y_train, x_valid, y_valid, 1000, 50)


# Visually check the loss history.
plt.plot(train_loss, c='blue', label='train loss')
plt.plot(val_loss, c='red', label='validation loss')
plt.legend()
plt.show()
# Visually check the prediction result.
y_pred = predict(x_test)
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.3, label='train')
plt.scatter(x_valid, y_valid, s=20, c='red', alpha=0.3,
label='validation')
plt.scatter(x_test, y_pred, s=5, c='red', label='test')
plt.legend()
plt.show()


