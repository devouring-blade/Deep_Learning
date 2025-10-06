import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from gradient_descent import gradient_descent

# Generate training data set
# y = 0.5x + 0.3 + noise
x = np.random.normal(0.0, 0.5, (1000, 1))
y = 0.5 * x + 0.3 + np.random.normal(0.0, 0.2, (1000, 1))

# Generate training, validation, and test data set
x_train, x_valid, y_train, y_valid = train_test_split(x, y)
x_test = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)

# See the data visually.
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.5,
label='train')
plt.scatter(x_valid, y_valid, s=20, c='red', alpha=0.5,
label='test')
plt.legend()
plt.show()

# create a single-layered ANN model
n_input = x.shape[1]
n_output = 1
alpha = 0.01

# initialize the parameters randomly
wo = np.random.normal(size= (n_input, n_output))
bo = np.zeros(shape= (1, n_output))
parameters = [wo, bo]

# loss function: mean squared error
def loss(y, y_hat):
    return np.mean((y - y_hat)**2)

# output from the ANN model: prediction process
def predict(x):
    o_output = np.dot(x, parameters[0]) + parameters[1]
    return o_output

# perform training and track the loss history
def train(x_train, y_train, x_val, y_val, epochs, batch_size):
    train_loss = []     # loss history of training data
    valid_loss = []       # loss history of valid data
    for epoch in range(epochs):
        # measure the losses during training
        train_loss.append(loss(y_train, predict(x_train)))
        valid_loss.append(loss(y_val, predict(x_val)))

        # perform training using mini-batch gradient descent
        for batch in range(int(x_train.shape[0] / batch_size)):
            idx = np.random.choice(x_train.shape[0], batch_size)
            gradient_descent(x_train[idx], y_train[idx], alpha, loss,
                             predict, parameters)

        if epoch % 10 == 0:
            print(f"epoch: {epoch} --- train loss: {train_loss[-1]:.4f} --- valid loss: {valid_loss[-1]:.4f}")

    return train_loss, valid_loss

# perform training
train_loss, valid_loss = train(x_train, y_train, x_valid, y_valid, epochs= 100, batch_size= 50)

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
plt.scatter(x_test, y_pred, s=5, c='red', label='test')
plt.legend()
plt.show()
print(parameters)








