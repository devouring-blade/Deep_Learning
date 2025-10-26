import numpy as np
from keras.layers import Layer, Dense, Input
import tensorflow as tf
from keras.models import Model
from matplotlib import pyplot as plt

class MyPeepholeLSTM(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        # weights for x
        self.wf = self.add_weight(shape= (input_shape[-1], self.units))
        self.wi = self.add_weight(shape= (input_shape[-1], self.units))
        self.wc = self.add_weight(shape= (input_shape[-1], self.units))
        self.wo = self.add_weight(shape= (input_shape[-1], self.units))

        # weights for h
        self.rf = self.add_weight(shape= (self.units, self.units))
        self.ri = self.add_weight(shape= (self.units, self.units))
        self.rc = self.add_weight(shape= (self.units, self.units))
        self.ro = self.add_weight(shape= (self.units, self.units))

        # peephole connections
        self.pf = self.add_weight(shape= (self.units, self.units))
        self.pi = self.add_weight(shape= (self.units, self.units))
        self.po = self.add_weight(shape= (self.units, self.units))

        # bias
        self.bf = self.add_weight(shape= (self.units, ))
        self.bi = self.add_weight(shape= (self.units, ))
        self.bc = self.add_weight(shape= (self.units, ))
        self.bo = self.add_weight(shape= (self.units, ))

    def call(self, x):
        def compute(x, c, h):
            f_gate = tf.math.sigmoid((x @ self.wf) + (h @ self.rf) + (c @ self.pf) + self.bf)
            i_gate = tf.math.sigmoid((x @ self.wi) + (h @ self.ri) + (c @ self.pi) + self.bi)
            c_tild = tf.math.tanh((x @ self.wc) + (h @ self.rc) + self.bc)
            c_state = (f_gate * c) + (i_gate * c_tild)
            o_gate = tf.math.sigmoid((x @ self.wo) + (h @ self.ro) + (c_state @ self.po) + self.bo)
            h_state = tf.math.tanh(c_state) * o_gate
            return c_state, h_state

        c = tf.zeros(shape= (tf.shape(x)[0], self.units))
        h = tf.zeros(shape= (tf.shape(x)[0], self.units))
        for t in range(x.shape[1]):
            c, h = compute(x[: , t, : ], c, h)
        return h

# Generate training data: 2 noisy sine curves
n = 1000 # the number of data points
n_step = 20 # the number of time steps
s1 = np.sin(np.pi * 0.06 * np.arange(n)) + np.random.random(n)
s2 = 0.5*np.sin(np.pi * 0.05 * np.arange(n)) + np.random.random(n)
data = np.vstack([s1, s2]).T # shape = (1000, 2)

x_train = np.array([data[i: (i + n_step)] for i in range(n - n_step)])
y_train = np.array([data[i + n_step] for i in range(n - n_step)])

n_feat = x_train.shape[-1]
n_output = y_train.shape[-1]
n_hidden = 50 # the number of hidden units
# Build a peephole LSTM model
x_input = Input(batch_shape= (None, n_step, n_feat))
hidden = MyPeepholeLSTM(units= n_hidden)(x_input)
y_output = Dense(units= n_output)(hidden)

model = Model(x_input, y_output)
model.compile(loss= "mse", optimizer= "adam")
model.summary()

# training
hist = model.fit(x_train, y_train, epochs= 50, batch_size= 50)

# visually see the loss history
plt.figure(figsize= (5, 3))
plt.plot(hist.history["loss"], color= "red")
plt.title("loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Predict future values for the next 50 periods.
# After predicting the next value, re-enter the predicted value
# to predict the next value. Repeat this process 50 times.
n_last = 100
n_future = 50
last_data = data[-n_last: ]
for i in range(n_future):
    # Predict the next value with the last n_step data points.
    px = last_data[-n_step: ].reshape(1, n_step, -1)

    # Predict the next value
    y_hat = model.predict(px)

    # Append the predicted value to the last_data array.
    # In the next iteration, the predicted value is input
    # along with the existing data points.
    last_data = np.vstack([last_data, y_hat])

p = last_data[: n_last]
f = last_data[n_last - 1: ]

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









