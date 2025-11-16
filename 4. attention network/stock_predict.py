import pickle
from keras.layers import Input, Dense
from transformer import Encoder, Decoder
from keras.models import  Model
import numpy as np
from matplotlib import pyplot as plt


with open("stock_data.pkl", "rb") as f:
    x_train, x_test, _, _, _ = pickle.load(f)

seq_len = 60
n_feat = x_train.shape[1]
d_model = 120
n_future = 20

emb_dense = Dense(units= d_model, use_bias= False)

i_enc = Input(batch_shape= (None, seq_len, n_feat))
h_enc = emb_dense(i_enc)
encoder = Encoder(num_layer= 2, num_feat= d_model, num_head= 4, num_ff= 128, dropout_rate=0.5)
o_enc = encoder(h_enc)

i_dec = Input(batch_shape= (None, n_future, n_feat))
h_dec = emb_dense(i_dec)
decoder = Decoder(num_layer= 2, num_feat= d_model, num_head= 4, num_ff= 128, dropout_rate=0.5)
o_dec = decoder(h_dec, o_enc)
y_dec = Dense(units= n_feat)(o_dec)

model = Model(inputs= (i_enc, i_dec), outputs= y_dec)
model.load_weights("stock.weights.h5")

# prediction
n_past = 50
e_data = x_train[-seq_len: ].reshape(1, seq_len, n_feat)
d_data = np.zeros(shape= (1, n_future, n_feat))
d_data[0, 0, : ] = x_train[-1]

for i in range(n_future):
    y_hat = model.predict((e_data, d_data), verbose= 0)

    if i < n_future - 1:
        d_data[0, i + 1, : ] = y_hat[0, i, : ]

# Plot the past time series and the predicted future time series.
y_hat = np.vstack([x_train[-1], d_data[0,:20,:]])
y_past = np.vstack([x_train[-n_past:], x_test])
plt.figure(figsize=(12, 6))
ax1 = np.arange(1, len(y_past) + 1)
ax2 = np.arange(n_past-1, n_past + n_future)
plt.plot(y_past[:, 0], '-o', c='blue', markersize=3,
alpha=0.5, label='S&P500', linewidth=1)
plt.plot(y_past[:, 1], '-o', c='black', markersize=3,
alpha=0.5, label='DOW', linewidth=1)
plt.plot(y_past[:, 2], '-o', c='red', markersize=3,
alpha=0.5, label='NASDAQ', linewidth=1)
plt.plot(ax2, y_hat[:, 0], c='blue', label='Predicted S&P500')
plt.plot(ax2, y_hat[:, 1], c='black',label='Predicted DOW')
plt.plot(ax2, y_hat[:, 2], c='red', label='Predicted NASDAQ')
plt.axvline(x=n_past-1, linestyle='dashed', linewidth=2)
plt.legend()
plt.show()









