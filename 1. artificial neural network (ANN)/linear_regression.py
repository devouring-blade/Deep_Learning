import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from gradient_descent import gradient_descent


# Generate training data set
# y = 0.5x + 0.3 + noise
x = np.random.normal(0.0, 0.5, (1000, 1))
y = 0.5 * x + 0.3 + np.random.normal(0.0, 0.2, (1000, 1))

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size= 0.15, shuffle= True)
x_test = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)

plt.figure(figsize= (7, 5))
plt.scatter(x_train, y_train, color= "red", label= "train data", s= 20, alpha= 0.4)
plt.scatter(x_val, y_val, color= "blue", label= "val data", s= 20, alpha= 0.4)
plt.legend()
plt.show()

n_input = x.shape[-1]
n_output = 1
alpha= 0.01

wo = np.random.normal(size= (n_input, n_output))
bo = np.zeros(shape= (1, n_output))

parameters = [wo, bo]

def relu(x): return np.maximum(0, x)

def mse(y, y_hat): return np.mean(np.square(y - y_hat))

def predict(x): return x @ wo + bo

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

train_loss, val_loss = train(x_train, y_train, x_val, y_val, 100, 50)

# Visually check the loss history.
plt.plot(train_loss, c='blue', label='train loss')
plt.plot(val_loss, c='red', label='validation loss')
plt.legend()
plt.show()

# Visually check the prediction result.
y_pred = predict(x_test)
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.3,
label='train')
plt.scatter(x_val, y_val, s=20, c='red', alpha=0.3,
label='validation')
plt.scatter(x_test, y_pred, s=5, c='red', label='test')
plt.legend()
plt.show()
print(parameters)





