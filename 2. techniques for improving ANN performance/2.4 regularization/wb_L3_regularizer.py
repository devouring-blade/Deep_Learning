import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers


# Generate a dataset
x, y = make_blobs(n_samples=300, n_features=2,
centers=[[0., 0.], [0.5, 0.1]],
cluster_std=0.25, center_box=(-1., 1.))
y = y.reshape(-1, 1).astype('float32')
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Visually see the distribution of the data points
plt.figure(figsize=(5, 5))
color = [['red', 'blue'][int(a)] for a in y_train.reshape(-1,)]
plt.scatter(x_train[:, 0], x_train[:, 1], s=100, c=color, alpha=0.3)
plt.show()

# Create an ANN model.
n_input = x_train.shape[1] # number of input neurons
n_output = 1 # number of output neurons
n_hidden = 32 # number of hidden neurons
R = 0.01 # regularization constant
adam = optimizers.Adam(learning_rate=0.005)

# Custom regularizer for L3 regularization
# L3 regularization is rarely used, but if you want to use
# it for some reason, you can implement it using a custom
# regularizer like this.
class reg_L3(regularizers.Regularizer):
    def __init__(self, reg_lambda):
        self.R = reg_lambda

    def __call__(self, x):
    # The w or b of a layer is passed to x.
        return self.R*tf.reduce_sum(tf.math.pow(tf.math.abs(x),3))

# The data is simple, but we intentionally added many hidden
# layers to the model to demonstrate the effect of regularization.
x_input = Input(batch_shape=(None, n_input))
h = Dense(n_hidden, activation = 'relu', kernel_regularizer=reg_L3(R), bias_regularizer=reg_L3(R))(x_input)

# 4 more hidden layers
for i in range(4):
    h = Dense(n_hidden, activation = 'relu', kernel_regularizer=reg_L3(R), bias_regularizer=reg_L3(R))(h)
y_output = Dense(n_output, activation='sigmoid', kernel_regularizer=reg_L3(R), bias_regularizer=reg_L3(R))(h)
model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', optimizer = adam)
h = model.fit(x_train, y_train, epochs=100, batch_size=50, validation_data=[x_test, y_test])

# Visually see the loss history
plt.plot(h.history['loss'], c='blue', label='train loss')
plt.plot(h.history['val_loss'], c='red', label='validation loss')
plt.legend()
plt.show()

# Check the accuracy of the test data
y_pred = (model.predict(x_test) > 0.5) * 1
acc = (y_pred == y_test).mean()
print("\nAccuracy of the test data = {:4f}"\
 .format(acc))

# --- Vẽ Decision Boundary ---
# Tạo lưới điểm phủ toàn bộ không gian dữ liệu
xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 300),
    np.linspace(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5, 300)
)

# Ghép thành (N,2) để đưa qua model
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Dự đoán xác suất cho mỗi điểm trong lưới
Z = model.predict(grid_points)
Z = Z.reshape(xx.shape)

# Vẽ contour (ranh giới phân chia)
plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=["red", "blue"])
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

# Vẽ lại dữ liệu thật
color = [['red', 'blue'][int(a)] for a in y_train.reshape(-1,)]
plt.scatter(x_train[:, 0], x_train[:, 1], s=100, c=color, edgecolor='k', alpha=0.6)

plt.title("Decision Boundary learned by ANN")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.show()









