import numpy as np
from keras.layers import Input, Dense, SimpleRNN
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

# Generate training data: 2 noisy sine curves
n = 1000 # the number of data points
n_step = 20 # the number of time steps
s1 = np.sin(np.pi * 0.06 * np.arange(n)) + np.random.random(n)
s2 = 0.5*np.sin(np.pi * 0.05 * np.arange(n)) + np.random.random(n)
data = np.vstack([s1, s2]).T

m = np.arange(0, n - n_step)
x_train = np.array([data[i: i + n_step] for i in m])
y_train = np.array([data[i + n_step] for i in m])

n_feat = x_train.shape[-1]
n_output = y_train.shape[-1]
n_hidden = 50

# create a many-to-one RNN model
x_input = Input(batch_shape= (None, n_step, n_feat))
h = SimpleRNN(units= n_hidden)(x_input)
y_output = Dense(units= n_output)(h)
model = Model(x_input, y_output)
model.compile(loss= "mse", optimizer= Adam(learning_rate= 0.001))
model.summary()

# training
hist = model.fit(x_train, y_train, epochs= 50, batch_size= 50)

# loss history
plt.figure(figsize= (5, 3))
plt.plot(hist.history["loss"], color= "red")
plt.title("loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Predict future values for the next 50 periods.
# After predicting the next value, re-enter the predicted value
# to predict the next value. Repeat this process 50 times.
n_future = 50
n_last = 100
last_data = data[-n_last: ]
for i in range(n_future):
    px = last_data[-20: ].reshape(1, n_step, n_feat)
    y_hat = model.predict(px)
    last_data = np.vstack([last_data, y_hat])

p = last_data[: n_last]
f = last_data[-(n_future + 1): ]

# Plot past and future time series.
plt.figure(figsize= (12, 6))
ax1, ax2 = np.arange(1, len(p) + 1), np.arange(len(p), len(p) + len(f))
plt.plot(ax1, p[: , 0], "-o", color= "blue", markersize= 3, label='Actual time series 1', linewidth=1)
plt.plot(ax1, p[: , 1], "-o", color= "red", markersize= 3, label='Actual time series 2', linewidth=1)
plt.plot(ax2, f[: , 0], "-o", color= "green", markersize= 3, label='Predicted time series 1', linewidth=1)
plt.plot(ax2, f[: , 1], "-o", color= "orange", markersize= 3, label='Predicted time series 1', linewidth=1)
plt.axvline(x= ax1[-1], linestyle='dashed', linewidth=1)
plt.legend()
plt.show()




