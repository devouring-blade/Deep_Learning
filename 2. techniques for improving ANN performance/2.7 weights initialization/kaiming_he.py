import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import initializers
from matplotlib import pyplot as plt


method = "Uniform"
if method == "Normal":
    init_w1 = initializers.HeNormal()
    init_w2 = initializers.HeNormal()
else:
    init_w1 = initializers.HeUniform()
    init_w2 = initializers.HeUniform()

n_input = 50
n_h1 = 80
n_h2 = 100
n_output = 1

x = np.random.normal(size= (1000, n_input))

# create an ANN model
x_input = Input(batch_shape= (None, n_input))
h1 = Dense(units= n_h1, kernel_initializer= init_w1, activation= "relu", name= "w1")(x_input)
h2 = Dense(units= n_h2, kernel_initializer= init_w2, activation= "relu", name= "w2")(h1)
y_output = Dense(units= n_output, activation= "sigmoid")(h2)

# model
model = Model(x_input, y_output)
h1_model = Model(x_input, h1)
h2_model = Model(x_input, h2)

h1_out = h1_model.predict(x, verbose= 0).flatten()
h2_out = h2_model.predict(x, verbose= 0).flatten()

# visualize
plt.hist(x.flatten(), bins= 50); plt.show()
plt.hist(h1_out, bins= 50); plt.show()
plt.hist(h2_out, bins= 50); plt.show()


w1 = model.get_layer("w1").get_weights()[0].flatten()
w2 = model.get_layer("w2").get_weights()[0].flatten()
if method == "Normal":
    # w1
    print(f"keras   σ of w1: {w1.std():.3f}")
    print(f"formula σ of w1: {np.sqrt(2 / n_input):.3f}")

    # w2
    print(f"keras   σ of w2: {w2.std():.3f}")
    print(f"formula σ of w2: {np.sqrt(2 / n_h1):.3f}")
else:
    # w1
    print(f"keras   a of w1: {w1.min():.3f} ~ {w1.max():.3f}")
    print(f"formula a of w1: ±{np.sqrt(6 / n_input)}")

    # w1
    print(f"keras   a of w2: {w2.min():.3f} ~ {w2.max():.3f}")
    print(f"formula a of w2: ±{np.sqrt(6 / n_h1)}")







