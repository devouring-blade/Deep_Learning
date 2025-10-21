import numpy as np
from keras import Model
from keras.layers import Layer, Dense, Input
from keras.initializers import GlorotUniform, Zeros
import tensorflow as tf
from keras.optimizers import Adam
from matplotlib import pyplot as plt

# Generate training data: 2 noisy sine curves
n = 1000 # the number of data points
n_step = 20 # the number of time steps
s1 = np.sin(np.pi * 0.06 * np.arange(n)) + np.random.random(n)
s2 = 0.5*np.sin(np.pi * 0.05 * np.arange(n)) + np.random.random(n)
data = np.vstack([s1, s2]).T # shape = (1000, 2)

m = np.arange(0, n - n_step)
x_train = np.array([data[i: (i + n_step), : ] for i in m])
y_train = np.array([data[i + n_step, : ] for i in m])

class MyRNN(Layer):
    def __init__(self, nf, nh):
        super().__init__()
        self.nh = nh
        self.wx = self.add_weight(shape= (nf, nh), initializer= GlorotUniform, name= "wx", trainable= True)
        self.wh = self.add_weight(shape= (nh, nh), initializer= GlorotUniform, name= "wh", trainable= True)
        self.bi = self.add_weight(shape= (1, nh), initializer= Zeros, name= "bias", trainable= True)

    def call(self, x):
        h = tf.zeros(shape= (tf.shape(x)[0], self.nh))
        for t in range(x.shape[1]):
            z = tf.matmul(x[: , t, : ], self.wx) + tf.matmul(h, self.wh) + self.bi
            h = tf.math.tanh(z)
        return h

# create a simple RNN model
n_feat, n_output, n_hidden = x_train.shape[-1], y_train.shape[-1], 50
x_input = Input(batch_shape= (None, n_step, n_feat))
h = MyRNN(n_feat, n_hidden)(x_input)
y_output = Dense(n_output)(h)
model = Model(x_input, y_output)
model.compile(loss= "mse", optimizer= Adam(learning_rate= 0.001))
model.summary()

# Training
hist = model.fit(x_train, y_train, epochs=100, batch_size=50)

# Loss history
plt.figure(figsize=(5, 3))
plt.plot(hist.history['loss'], color='red')
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


# n = 100
# ax = np.arange(1, n + 1)
# x = data[-(n + n_step): -n]
# for i in range(n):
#     px = x[-n_step: ].reshape(1, n_step, n_feat)
#     y_hat = model.predict(px, verbose= 0)
#     x = np.vstack([x, y_hat])
# plt.plot(ax, data[-n: , 0], '-o', c='blue', markersize=3, label='Actual time series 1', linewidth=1)
# plt.plot(ax, data[-n: , 1], '-o', c='red', markersize=3, label='Actual time series 2', linewidth=1)
# plt.plot(ax, x[-n:, 0], '-o', c='green', markersize=3, label='Estimated time series 1')
# plt.plot(ax, x[-n:, 1], '-o', c='orange', markersize=3, label='Estimated time series 2')
# plt.legend()
# plt.show()

# Predict future values for the next 50 periods.
# After predicting the next value, re-enter the predicted value
# to predict the next value. Repeat this process 50 times.
n_future = 50
n_last = 100
last_data = data[-n_last: ]
for i in range(n_future):
    # predict the next value with the last n_step data points.
    px = last_data[-n_step:].reshape(1, n_step, n_feat)

    # predict the next value
    y_hat = model.predict(px, verbose= 0)

    # Append the predicted value to the last_data array.
    # In the next iteration, the predicted value is input
    # along with the existing data points.
    last_data = np.vstack([last_data, y_hat])

p = last_data[:-n_future]
f = last_data[-(n_future + 1):]

# Plot past and future time series.
plt.figure(figsize=(12, 6))
ax1 = np.arange(1, len(p) + 1)
ax2 = np.arange(len(p), len(p) + len(f))
plt.plot(ax1, p[:, 0], '-o', c='blue', markersize=3, label='Actual time series 1', linewidth=1)
plt.plot(ax1, p[:, 1], '-o', c='red', markersize=3, label='Actual time series 2', linewidth=1)
plt.plot(ax2, f[:, 0], '-o', c='green', markersize=3, label='Estimated time series 1')
plt.plot(ax2, f[:, 1], '-o', c='orange', markersize=3, label='Estimated time series 2')
plt.axvline(x=ax1[-1], linestyle='dashed', linewidth=1)
plt.legend()
plt.show()












