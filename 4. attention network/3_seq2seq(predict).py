import pickle
from keras.layers import Input, Dense, GRU, TimeDistributed
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt


with open("dataset.pkl", "rb") as f:
    data, _, _, _ = pickle.load(f)

n_hidden = 100
n_step = 50
n_feat = 2

# encoder
i_enc = Input(batch_shape= (None, n_step, n_feat))
h_enc = GRU(units= n_hidden, return_sequences= False)(i_enc)

# decoder
single_step_GRU = GRU(units= n_hidden, return_sequences= True, return_state= True)
many_2_many_output = TimeDistributed(Dense(units= n_feat))

i_dec = Input(batch_shape= (None, 1, n_feat))
o_dec, _ = single_step_GRU(i_dec, initial_state= h_enc)
y_output = many_2_many_output(o_dec)
model = Model([i_enc, i_dec], y_output)
model.load_weights("seq2seq.weights.h5")

# encoder model
encoder = Model(i_enc, h_enc)

# decoder model
i_pre = Input(batch_shape= (None, n_hidden))
o_pre, h_pre = single_step_GRU(i_dec, initial_state= i_pre)
y_pre = many_2_many_output(o_pre)
decoder = Model([i_dec, i_pre], [y_pre, h_pre])


# prediction
e_seed = data[-50: ].reshape(-1, 50, 2)
he = encoder.predict(e_seed, verbose= 0)

d_seed = data[-1].reshape(-1, 1, 2)

n_future = 50
y_pred = []
for i in range(n_future):
    yd, hd = decoder.predict([d_seed, he], verbose= 0)
    y_pred.append(yd.reshape(2, ))

    he = hd
    d_seed = yd
y_pred = np.array(y_pred)

# Plot the past time series and the predicted future time series.
y_past = data[-100:]
plt.figure(figsize=(12, 6))
ax1 = np.arange(1, len(y_past) + 1)
ax2 = np.arange(len(y_past), len(y_past) + len(y_pred))
plt.plot(ax1, y_past[:, 0], '-o', c='blue', markersize=3, label='Original time series 1', linewidth=1)
plt.plot(ax1, y_past[:, 1], '-o', c='red', markersize=3, label='Original time series 2', linewidth=1)
plt.plot(ax2, y_pred[:, 0], '-o', c='green', markersize=3, label='Predicted time series 1')
plt.plot(ax2, y_pred[:, 1], '-o', c='orange', markersize=3, label='Predicted time series 2')
plt.axvline(x=ax1[-1], linestyle='dashed', linewidth=1)
plt.legend()
plt.show()








