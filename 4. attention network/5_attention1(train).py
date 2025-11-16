import pickle
from keras.layers import Dot, Activation, Concatenate
from keras.layers import Input, GRU, TimeDistributed, Dense
from keras.models import Model
from matplotlib import pyplot as plt

def attention_layer(e, d):
    dot_product = Dot(axes= (2, 2))([d, e])
    score = Activation("softmax")(dot_product)
    value = Dot(axes= (2, 1))([score, e])
    return Concatenate()([value, d])

with open("dataset.pkl", "rb") as f:
    _, xi_enc, xi_dec, xp_dec = pickle.load(f)

n_hidden = 100
n_step = 50
n_feat = 2

# encoder
i_enc = Input(batch_shape= (None, n_step, n_feat))
o_enc, h_enc = GRU(units= n_hidden, return_sequences= True, return_state= True)(i_enc)

# decoder
i_dec = Input(batch_shape= (None, n_step, n_feat))
o_dec = GRU(units= n_hidden, return_sequences= True)(i_dec, initial_state= h_enc)
attention = attention_layer(o_enc, o_dec)
output = TimeDistributed(Dense(units= n_feat))(attention)

model = Model([i_enc, i_dec], output)
model.compile(loss= "mse", optimizer= "adam")

# training
hist = model.fit([xi_enc, xi_dec], xp_dec, batch_size= 500, epochs= 200)

# save the trained model
model.save_weights("attention1.weights.h5")

# Visually see the loss history
plt.plot(hist.history['loss'], color='red')
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()








