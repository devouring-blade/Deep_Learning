import numpy as np
from keras.layers import Input, Dense, Activation, AveragePooling1D, Flatten
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

# build a CNN model
x_input = Input(batch_shape= (None, n_step, n_feat))
print(x_input.shape)
emb = Dense(units= n_emb, use_bias= False, activation= "tanh")(x_input)
print(emb.shape)
conv = conv_1d(n_kernels= n_kernel,  kernel_size= k_size, padding= "same")(emb)
conv = Activation("relu")(conv)
print(conv.shape)
pool = AveragePooling1D(pool_size= p_size, strides= 1)(conv)
print(pool.shape)
flat = Flatten()(pool)
print(flat.shape)
y_output = Dense(units= y_train.shape[-1])(flat)
print(y_output.shape)

model = Model((x_input), (y_output))
model.compile(loss= "mse", optimizer= "adam")
model.summary()

# training
hist = model.fit(x= x_train,y= y_train, epochs= 200, batch_size= 100)

plt.figure(figsize= (5, 3))
plt.plot(hist.history["loss"], color= "red")
plt.title("loss history")
plt.xlabel("epcochs")
plt.ylabel("loss")
plt.show()


n_future = 50
n_last = 100
last_data = data[-n_last: ]
for i in range(n_future):
    px = last_data[-n_step: , : ].reshape(1, n_step, 2)
    y_hat = model.predict(px, verbose= 0)
    last_data = np.vstack((last_data, y_hat))

p = last_data[: n_last]
f = last_data[n_last: ]

plt.figure(figsize= (12, 6))
ax1 = np.arange(1, len(p) + 1)
plt.plot(ax1, p[:, 0], '-o', c='blue', markersize=3,
         label='Actual time series 1', linewidth=1)
plt.plot(ax1, p[:, 1], '-o', c='red', markersize=3,
         label='Actual time series 2', linewidth=1)
ax2 = np.arange(len(p), len(p) + len(f))
plt.plot(ax2, f[:, 0], '-o', c='green', markersize=3,
         label='estimated time series 1', linewidth=1)
plt.plot(ax2, f[:, 1], '-o', c='yellow', markersize=3,
         label='estimated time series 2', linewidth=1)
plt.axvline(x=ax1[-1], linestyle='dashed', linewidth=1)
plt.legend()
plt.show()

















