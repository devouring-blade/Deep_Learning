import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from gradient_descent import gradient_descent

# Generate training data set
x = np.random.random((1000, 1))
y = 2.0 * np.sin(2.0 * np.pi * x) + np.random.normal(0.0, 0.8, (1000, 1))

# Generate training, validation, and test data set
x_train, x_valid, y_train, y_valid = train_test_split(x, y)
x_test = np.linspace(0, 1, 200).reshape(-1, 1)

# See the data visually.
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.5,
label='train')
plt.scatter(x_valid, y_valid, s=20, c='red', alpha=0.5,
label='valid')
plt.legend()
plt.show()

# create a two-layered ANN model
n_input = x.shape[1]
n_output = 1
n_hidden = 16
alpha = 0.01

# initialize the parameters randomly
wh = np.random.normal(size= (n_input, n_hidden))
bh = np.zeros(shape= (1, n_hidden))
wo = np.random.normal(size= (n_hidden, n_output))
bo = np.zeros(shape= (1, n_output))
parameters = [wh, bh, wo, bo]

# loss function: mean squared error
def loss(y, y_hat):
    return np.mean((y - y_hat)**2)

# output from ANN model
def predict(x):
    h_output = np.tanh(np.dot(x, parameters[0]) + parameters[1])
    o_output = np.dot(h_output, parameters[2]) + parameters[3]
    return o_output

# perform training and track the loss history
def train(x_train, y_train, x_valid, y_valid, epochs, batch_size):
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        # measure the losses during training
        train_loss.append(loss(y_train, predict(x_train)))
        valid_loss.append(loss(y_valid, predict(x_valid)))

        # perform training using mini-batch gradient descent
        for batch in range(int(x_train.shape[0] / batch_size)):
            idx = np.random.choice(x_train.shape[0], batch_size)
            gradient_descent(x_train[idx], y_train[idx], alpha, loss, predict, parameters)

        if epoch % 10 == 0:
            print(f"epoch: {epoch} --- train loss: {train_loss[-1]:.4f} --- valid loss: {valid_loss[-1]:.4f}")

    return train_loss, valid_loss

# perform training
train_loss, valid_loss = train(x_train, y_train, x_valid, y_valid,
                               epochs= 1000, batch_size= 50)

# Visually check the loss history.
plt.plot(train_loss, c='blue', label='train loss')
plt.plot(valid_loss, c='red', label='validation loss')
plt.legend()
plt.show()

# Visually check the prediction result.
y_pred = predict(x_test)
plt.figure(figsize=(7,5))

plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.5, label='train')
plt.scatter(x_valid, y_valid, s=20, c='red', alpha=0.5, label='validation')
plt.scatter(x_test, y_pred, s=5, c='black', label='test')
plt.legend()
plt.show()


