from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import pickle
from transformer import Encoder, Decoder
import numpy as np

with open("dataset.pkl", "rb") as f:
    data, _, _, _ = pickle.load(f)

n_step = 50
n_feat = data.shape[-1]
d_model = 100
n_future = 60

emb_dense = Dense(d_model, use_bias=False)

#  encoder
i_enc = Input(batch_shape= (None, n_step, n_feat))
h_enc = emb_dense(i_enc)
encoder = Encoder(num_layer= 1, seq_len= n_step, num_feat= d_model, num_head= 5, num_ff= 64, dropout_rate= 0.5)
o_enc = encoder(h_enc)

#  decoder
i_dec = Input(batch_shape= (None, None, n_feat))
h_dec = emb_dense(i_dec)
decoder = Decoder(num_layer= 1, seq_len= n_future, num_feat= d_model, num_head= 5, num_ff= 64, dropout_rate= 0.5)
o_dec = decoder(h_dec, o_enc)
y_dec = Dense(n_feat)(o_dec)

model = Model(inputs=[i_enc, i_dec], outputs=y_dec)
model.load_weights("transformer.weights.h5")

# prediction
e_data = data[-n_step:].reshape(-1, n_step, n_feat)
d_data = np.zeros(shape=(1, n_future, n_feat))
d_data[0, 0, :] = data[-1]

for i in range(n_future):
    y_hat = model.predict([e_data, d_data], verbose=0)
    if i < n_future - 1:
        d_data[0, i + 1, :] = y_hat[0, i, :]
    print(i + 1, ':', y_hat[0, i, :])

# Plot the past time series and the predicted future time series.
y_past = data[-100:]
y_hat = np.vstack([y_past[-1], d_data[0,:,:]])
plt.figure(figsize=(12, 6))
ax1 = np.arange(1, len(y_past) + 1)
ax2 = np.arange(len(y_past), len(y_past) + len(y_hat))
plt.plot(ax1, y_past[:, 0], '-o', c='blue', markersize=3, label='Original time series 1', linewidth=1)
plt.plot(ax1, y_past[:, 1], '-o', c='red', markersize=3, label='Original time series 2', linewidth=1)
plt.plot(ax2, y_hat[:, 0], '-o', c='green', markersize=3, label='Predicted time series 1')
plt.plot(ax2, y_hat[:, 1], '-o', c='orange', markersize=3, label='Predicted time series 2')
plt.axvline(x=ax1[-1], linestyle='dashed', linewidth=1)
plt.legend()
plt.show()



