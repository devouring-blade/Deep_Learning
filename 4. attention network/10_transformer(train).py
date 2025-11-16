import pickle
from keras.layers import Dense, Input
from keras.models import Model
from matplotlib import pyplot as plt
from transformer import Encoder, Decoder



# read data
with open("dataset.pkl", "rb") as f:
    _, xi_enc, xi_dec, xp_dec = pickle.load(f)

n_step = xi_enc.shape[1] # 50
n_feat = xi_enc.shape[2] # 2
d_model = 100

emb_dense = Dense(d_model, use_bias=False)

#  encoder
i_enc = Input(batch_shape= (None, n_step, n_feat))
h_enc = emb_dense(i_enc)
encoder = Encoder(num_layer= 1, num_feat= d_model, num_head= 5, num_ff= 64, dropout_rate= 0.5)
o_enc = encoder(h_enc)

#  decoder
i_dec = Input(batch_shape= (None, None, n_feat))
h_dec = emb_dense(i_dec)
decoder = Decoder(num_layer= 1, num_feat= d_model, num_head= 5, num_ff= 64, dropout_rate= 0.5)
o_dec = decoder(h_dec, o_enc)
y_dec = Dense(n_feat)(o_dec)

model = Model(inputs= (i_enc, i_dec), outputs= y_dec)
model.compile(loss= "mse", optimizer= "adam")
model.summary()

hist = model.fit((xi_enc, xi_dec), xp_dec, epochs= 100, batch_size= 200)

model.save_weights("transformer.weights.h5")

# Visually see the loss history
plt.plot(hist.history['loss'], label='Train loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()














