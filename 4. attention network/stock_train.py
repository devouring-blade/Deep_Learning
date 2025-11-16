import pickle
from keras.layers import Input, Dense
from transformer import  Encoder, Decoder
from keras.models import Model
from matplotlib import pyplot as plt


with open("stock_data.pkl", "rb") as f:
    x_train, x_test, xi_enc, xi_dec, xo_dec = pickle.load(f)

seq_len = xi_enc.shape[1]
n_feat = xi_enc.shape[2]
d_model = 120

emb_dense = Dense(units= d_model, use_bias= False)

i_enc = Input(batch_shape= (None, seq_len, n_feat))
h_enc = emb_dense(i_enc)
encoder = Encoder(num_layer= 2, num_feat= d_model, num_head= 4, num_ff= 128, dropout_rate= 0.5)
o_enc = encoder(h_enc)

i_dec = Input(batch_shape= (None, seq_len, n_feat))
h_dec = emb_dense(i_dec)
decoder = Decoder(num_layer= 2, num_feat= d_model, num_head= 4, num_ff= 128, dropout_rate= 0.5)
o_dec = decoder(h_dec, o_enc)
y_dec = Dense(units= n_feat)(o_dec)

model = Model(inputs= (i_enc, i_dec), outputs= y_dec)
model.compile(loss= "mse", optimizer= "adam")
model.summary()

hist = model.fit((xi_enc, xi_dec), xo_dec, epochs= 100, batch_size= 200)
model.save_weights("stock.weights.h5")


# Visually see the loss history
plt.plot(hist.history['loss'], label='Train loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()



