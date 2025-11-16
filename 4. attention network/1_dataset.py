import numpy as np
import pickle


# Generate a dataset consisting of two noisy sine curves
n = 5000 # the number of data points
s1= np.sin(np.pi * 0.06 * np.arange(n))+np.random.random(n)
s2= 0.5*np.sin(np.pi * 0.05 * np.arange(n))+np.random.random(n)
data = np.vstack([s1, s2]).T


n_step = 50
m = range(0, n - 2*n_step + 1)
xi_enc = np.array([data[i: i + n_step] for i in m])
xi_dec = np.array([data[i + n_step - 1: i + 2*n_step - 1] for i in m])
xo_dec = np.array([data[i + n_step: i + 2*n_step] for i in m])

# Save the training data for later use
with open("dataset.pkl", "wb") as f:
    pickle.dump([data, xi_enc, xi_dec, xo_dec], f)

print("The shape of the dataset:", data.shape)
print("The shape of the encoder input:", xi_enc.shape)
print("The shape of the decoder input:", xi_dec.shape)
print("The shape of the decoder output:", xo_dec.shape)

