import pickle
from keras.layers import Input, GRU, TimeDistributed, Dense
from keras.models import Model
from matplotlib import pyplot as plt


with open("dataset.pkl", "rb") as f:
    data, xi_enc, xi_dec, xo_dec = pickle.load(f)

n_hidden = 100
n_step = xi_enc.shape[1]
n_feat = xi_enc.shape[2]

# encoder
i_enc = Input(batch_shape= (None, n_step, n_feat))
h_enc = GRU(units= n_hidden, return_sequences= False)(i_enc)

# decoder
i_dec = Input(batch_shape= (None, n_step, n_feat))
o_dec = GRU(units= n_hidden, return_sequences= True)(i_dec, initial_state= h_enc)
y_dec = TimeDistributed(Dense(units= n_feat))(o_dec)

model = Model([i_enc, i_dec], y_dec)
model.compile(loss= "mse", optimizer= "adam")
model.summary()

# training
hist = model.fit(x=[xi_enc, xi_dec],y= xo_dec, batch_size= 200, epochs= 100)

# Save the trained model
model.save_weights("seq2seq.weights.h5")

# Visually see the loss history
plt.plot(hist.history['loss'], color='red')
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()





