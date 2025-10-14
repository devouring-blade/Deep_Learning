# Observing the parameters inside Batch Normalization layer

import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Model

x = np.random.normal(size=(100, 3))
y = np.random.choice([0,1], 100).reshape(-1,1)
e = 0.001; rho = 0.99

x_input = Input(batch_shape= (None, 3))
h = Dense(units= 4, use_bias= False, name= "hn")(x_input)
r = BatchNormalization(momentum= rho, epsilon= e, name= "bn")(h)
h_act = Activation("relu")(r)
y_output = Dense(units= 1, activation= "sigmoid")(h_act)
model = Model(x_input, y_output)
model.compile(loss= "mse", optimizer= "adam")
model.summary()

model_h = Model(x_input, h)
model_r = Model(x_input, r)

# Initial values of the parameters in Batch Normalization layer
gamma, beta, me, var = model.get_layer("bn").get_weights()
print(f"gamma: {gamma.round(3)}")
print(f"beta: {beta.round(3)}")
print(f"E(h): {me.round(3)}")
print(f"var(h): {var.round(3)}")

# Training. Gamma and beta are also learned, and moving mu and var are calculated and stored.
model.fit(x, y, epochs=10, batch_size=10, verbose=0)

# outputs of the hidden layer
print('After training: prediction stage')
ho = model_h.predict(x, verbose=0)[:3]
print('h = '); print(ho.round(3))

# outputs of the Batch Normalization layer
ro = model_r.predict(x, verbose=0)[:3]
print('\nr = '); print(ro.round(3))

# Parameters stored in Batch Normalization layer
gamma, beta, me, var = model.get_layer("bn").get_weights()
print(f"gamma: {gamma.round(3)}")
print(f"beta: {beta.round(3)}")
print(f"E(h): {me.round(3)}")
print(f"var(h): {var.round(3)}")

# Let's manually calculate the outputs of the BatchNormalization layer.
rm = gamma * (ho - me) / np.sqrt(var + e) + beta
print('\nManual calculation (r)')
print(rm.round(3)) # This matches r above.










