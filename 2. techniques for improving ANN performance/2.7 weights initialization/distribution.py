# Observe the distribution of hidden layer outputs according to initial weights

import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# generate a simple dataset with 1000 data points
x = np.random.normal(size= (1000, 100))

# create an ANN model with 2 hidden layers
n_input = x.shape[1]
n_hidden = 100

# Let's check the distribution of hidden layer. outputs by changing std in N(0, std).
std = [1.0, 0.3, 0.15, 0.1, 0.05, 0.01]
for sigma in std:
    # create an ANN model
    w0 = initializers.RandomNormal(mean= 0.0, stddev= sigma)
    x_input = Input(batch_shape= (None, n_input))
    s1 = Dense(n_hidden, kernel_initializer= w0)(x_input)
    s2 = Dense(n_hidden, kernel_initializer= w0)(s1)

    s1_model = Model(x_input, s1)
    s2_model = Model(x_input, s2)

    i = 0
    s1_out = s1_model.predict(x, verbose=0)[:, i]
    s2_out = s2_model.predict(x, verbose=0)[:, i]

    # Check the distribution of the hidden layer outputs.
    plt.figure(figsize=(8, 2))
    plt.subplot(121)
    plt.hist(s1_out, bins=30, color='blue', alpha=0.5)
    plt.title('w_std=' + str(sigma) + ', \
    s1_std=' + str(s1_out.std().round(3)))
    plt.subplot(122)
    plt.hist(s2_out, bins=30, color='red', alpha=0.5)
    plt.title('s2_std=' + str(s2_out.std().round(3)))
    plt.show()










