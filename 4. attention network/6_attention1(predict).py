import pickle
from keras.layers import Dot, Activation, Concatenate
from keras.layers import Input, Dense, GRU, TimeDistributed
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt


def attention_layer(e, d):
    dot_product = Dot(axes= (2, 2))([d, e])
    score = Activation("softmax")(dot_product)
    value = Dot(axes= (2, 1))([score, e])
    return Concatenate()([value, d])

with open("dataset.pkl", "rb") as f:
    data, _, _, _ = pickle.load(f)

n_hidden = 100
n_step = 50
n_feat = 2

# layers
trained_enc_GRU = GRU(units= n_hidden, return_sequences= True, return_state= True)
trained_dec_GRU = GRU(units= n_hidden, return_sequences= True, return_state= True)
trained_FFN = TimeDistributed(Dense(units= n_feat))

# trained encoder
i_enc = Input(batch_shape= (None, n_step, n_feat))
o_enc, h_enc = trained_enc_GRU(i_enc)

# trained decoder
i_dec = Input(batch_shape= (None, n_feat, n_feat))
o_dec, _ = trained_dec_GRU(i_dec, initial_state= h_enc)
attention = attention_layer(o_enc, o_dec)
output = trained_FFN(attention)

model = Model([i_enc, i_dec], output)
model.load_weights("attention1.weights.h5")

# encoder model for prediction
i_enc_pre = Input(batch_shape= (None, n_step, n_feat))
o_enc_pre, h_enc_pre = trained_enc_GRU(i_enc_pre)
encoder = Model(i_enc_pre, [o_enc_pre, h_enc_pre]) # encoder model

# decoder model for prediction
i_dec_pre = Input(batch_shape= (None, 1, n_feat))
i_stat = Input(batch_shape= (None, n_hidden))
i_h_enc = Input(batch_shape= (None, n_step, n_hidden))
o_dec_pre, h_dec_pre = trained_dec_GRU(i_dec_pre, initial_state= i_stat)
attention_pre = attention_layer(i_h_enc, o_dec_pre)
predict = trained_FFN(attention_pre)
decoder = Model([i_dec_pre, i_stat, i_h_enc], [predict, h_dec_pre])

# prediction
e_seed = data[-50: ].reshape(-1, 50, 2)
d_seed = data[-1].reshape(-1, 1, 2)

oe, he = encoder.predict(e_seed, verbose= 0)

n_future = 50
y_predict = []

for i in range(n_future):
    yd, hd = decoder.predict([d_seed, he, oe], verbose= 0)
    y_predict.append(yd.reshape(2, ))
    he = hd
    d_seed = yd
y_predict = np.array(y_predict)


# Plot the past time series and the predicted future time series.
y_past = data[-100:]
plt.figure(figsize=(12, 6))
ax1 = np.arange(1, len(y_past) + 1)
ax2 = np.arange(len(y_past), len(y_past) + len(y_predict))
plt.plot(ax1, y_past[:, 0], '-o', c='blue', markersize=3, label='Original time series 1', linewidth=1)
plt.plot(ax1, y_past[:, 1], '-o', c='red', markersize=3, label='Original time series 2', linewidth=1)
plt.plot(ax2, y_predict[:, 0], '-o', c='green', markersize=3, label='Predicted time series 1')
plt.plot(ax2, y_predict[:, 1], '-o', c='orange', markersize=3, label='Predicted time series 2')
plt.axvline(x=ax1[-1], linestyle='dashed', linewidth=1)
plt.legend()
plt.show()







