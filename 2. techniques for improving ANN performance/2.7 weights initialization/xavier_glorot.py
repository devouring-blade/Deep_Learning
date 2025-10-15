import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

method = 'Uniform'

if method == 'Normal':
    init_w1 = initializers.GlorotNormal()
    init_w2 = initializers.GlorotNormal()
else:
    init_w1 = initializers.GlorotUniform()
    init_w2 = initializers.GlorotUniform()

n_in = 50
n_h1 = 80
n_h2 = 100

x = np.random.normal(size=(1000, n_in))

x_input = Input(batch_shape= (None, n_in))
h1 = Dense(units= n_h1, kernel_initializer= init_w1, activation= "tanh", name= "h1")(x_input)
h2 = Dense(units= n_h2, kernel_initializer= init_w2, activation= "tanh", name= "h2")(h1)
y_output = Dense(units= 1, activation= "sigmoid")(h2)

model = Model(x_input, y_output)
w1 = model.get_layer("h1").get_weights()[0].flatten()
w2 = model.get_layer("h2").get_weights()[0].flatten()

h1_model = Model(x_input, h1)
h2_model = Model(x_input, h2)

h1_output = h1_model.predict(x, verbose= 0)
h2_output = h2_model.predict(x, verbose= 0)


plt.hist(x.flatten(), bins=50); plt.show()
plt.hist(h1_output.flatten(), bins=50); plt.show()
plt.hist(h2_output.flatten(), bins=50); plt.show()

if method == 'Normal':
    print('[Keras ] σ of w1 = {:.3f}'.format(w1.std()))
    print('[Formula] σ of w1 = {:.3f}'.format(np.sqrt(2 / (n_in + n_h1))))
    print('[Formula] mean of w1 = {:.3f}'.format(w1.mean()))

    print('\n[Keras ] σ of w2 = {:.3f}'.format(w2.std()))
    print('[Formula] σ of w2 = {:.3f}'.format(np.sqrt(2 / (n_h1 + n_h2))))
    print('[Formula] mean of w2 = {:.3f}'.format(w2.mean()))

    print(f"var of h1 output: {h1_output.var()}")
    print(f"var of h2 output: {h2_output.var()}")
else:
    print('[Keras ] a of w1 = {:.3f} ~ {:.3f}'.format(w1.min(), w1.max()))
    print('[Formula] a of w1 = ±{:.3f}'.format(np.sqrt(6 / (n_in + n_h1))))

    print('\n[Keras] a of w2 = {:.3f} ~ {:.3f}'.format(w2.min(), w2.max()))
    print('[Formula] a of w2 = ±{:.3f}'.format(np.sqrt(6 / (n_h1 + n_h2))))

    print(f"var of h1 output: {h1_output.var()}")
    print(f"var of h2 output: {h2_output.var()}")




