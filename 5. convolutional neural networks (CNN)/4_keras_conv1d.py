import numpy as np
from keras.layers import Input, Dense, Activation, AveragePooling1D, Flatten, Conv1D
from keras.models import Model
from MyConv1D import conv_1d
from matplotlib import pyplot as plt


# Generate training data: 2 noisy sine curves
n = 3000 # the number of data points
s1 = np.sin(np.pi * 0.06 * np.arange(n)) + np.random.random(n)
s2 = 0.5*np.sin(np.pi * 0.05 * np.arange(n)) + np.random.random(n)
data = np.vstack((s1, s2)).T

n_step = 30 # the number of time steps
m = np.arange(n - n_step)
x_train = np.array([data[i: i + n_step, : ] for i in m])
y_train = np.array([data[i, : ] for i in (m + n_step)])
print(x_train.shape)

n_emb = 20 # time series embedding size
k_size = 5 # kernel size
n_kernel = 10 # number of filters
p_size = 10 # pooling filter size
n_feat = x_train.shape[-1] # the number of features

x_input = Input(batch_shape= (None, n_step, x_train.shape[-1]))
emb = Dense(units= n_emb, activation= "tanh", use_bias= False)(x_input)
conv = Conv1D(filters= n_kernel, kernel_size= k_size, padding= "same", strides= 1)(emb)
conv = Activation("relu")(conv)
conv = Conv1D(filters= n_kernel, kernel_size= k_size, padding= "same", strides= 1)(emb)
conv = Activation("relu")(conv)
pool = AveragePooling1D(pool_size= p_size, strides= 1)(conv)
flat = Flatten()(pool)
y_output = Dense(units= y_train.shape[-1])(flat)

model = Model(x_input, y_output)
model.load_weights(r"E:\pycharm\PycharmProjects\deep_learning\weights\result_4.weights.h5")
model.summary()


# predict
n_future = 50
n_last = 100
last_data = data[-n_last: , :]
for i in range(n_future):
    px = last_data[-n_step: , : ].reshape(1, n_step, -1)
    y_hat = model.predict(px, verbose= 0)
    last_data = np.vstack((last_data, y_hat))

p = last_data[: n_last, : ]
f = last_data[n_last: , : ]

# Plot past and future time series.
plt.figure(figsize=(12, 6))
ax1 = np.arange(1, len(p) + 1)
ax2 = np.arange(len(p), len(p) + len(f))
plt.plot(ax1, p[:, 0], '-o', c='blue', markersize=3,
label='Actual time series 1', linewidth=1)
plt.plot(ax1, p[:, 1], '-o', c='red', markersize=3,
label='Actual time series 2', linewidth=1)
plt.plot(ax2, f[:, 0], '-o', c='green', markersize=3,
label='Estimated time series 1')
plt.plot(ax2, f[:, 1], '-o', c='orange', markersize=3,
label='Estimated time series 2')
plt.axvline(x=ax1[-1], linestyle='dashed', linewidth=1)
plt.legend()
plt.show()
