import pickle
from keras.layers import Dense, Input, TimeDistributed
from attention import Encoder, Decoder
from keras.models import Model
from matplotlib import pyplot as plt
from keras.layers import Dot, Activation, Concatenate, Reshape
from keras.layers import Dense, GRU, Layer
import tensorflow as tf





with open("dataset.pkl", "rb") as f:
    _, xi_enc, xi_dec, xp_dec = pickle.load(f)

n_hidden = 100
n_emb = 30
n_feed = 30
n_step = xi_enc.shape[1]
n_feat = xi_enc.shape[2]

# time series embedding layer
embed_input = Dense(units= n_emb, activation= "tanh")

# encoder
i_enc = Input(batch_shape= (None, n_step, n_feat))
e_enc = embed_input(i_enc)
o_enc, h_enc = Encoder(n_hidden)(e_enc)

# decoder
i_dec = Input(batch_shape= (None, n_step, n_feat))
e_dec = embed_input(i_dec)
o_dec = Decoder(n_hidden, n_feed)(e_dec, o_enc, h_enc)
output = TimeDistributed(Dense(units= n_feat))(o_dec)

model = Model([i_enc, i_dec], output)
model.compile(loss= "mse", optimizer= "adam")
model.summary()

# training: teacher forcing
hist = model.fit([xi_enc, xi_dec], xp_dec, batch_size= 500, epochs= 200)

# save the model trained
model.save_weights("attention2.weights.h5")

# Visually see the loss history
plt.plot(hist.history['loss'], color='red')
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()





