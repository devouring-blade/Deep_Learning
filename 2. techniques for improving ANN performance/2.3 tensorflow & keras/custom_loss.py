import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

# Generate a data set
x = np.random.random((1000, 1))
y = 2.0 * np.sin(2.0 * np.pi * x) + np.random.normal(0.0, 0.8, (1000, 1))

# Generate training, test data set
x_train, x_test, y_train, y_test = train_test_split(x, y)
x_pred = np.linspace(0, 1, 200).reshape(-1, 1)

# Visually see the data.
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=50, c='blue', alpha=0.5, label='train')
plt.scatter(x_test, y_test, s=20, c='red', alpha=0.5, label='valid')
plt.legend()
plt.show()

# custom loss: applying L2 regularization to the loss function
class regularized_loss(tf.keras.losses.Loss):
    def __init__(self, C, h_layer, o_layer):
        super(regularized_loss, self).__init__()
        self.C = C
        self.h_layer = h_layer
        self.o_layer = o_layer

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.math.square(y_true - y_pred))
        wh = self.h_layer.weights[0]
        wo = self.o_layer.weights[0]
        mse += self.C * tf.reduce_sum(tf.math.square(wh))
        mse += self.C * tf.reduce_sum(tf.math.square(wo))
        return mse

# create an ANN model
n_input = x.shape[1]
n_output = 1
n_hidden = 8
adam = optimizers.Adam(learning_rate= 0.01)

h_layer = Dense(n_hidden, activation= "tanh")
o_layer = Dense(n_output, activation= "linear")

x_input = Input(batch_shape= (None, n_input))
h = h_layer(x_input)
y_output = o_layer(h)
model = Model(x_input, y_output)

myloss = regularized_loss(0.001, h_layer, o_layer)
model.compile(loss= myloss, optimizer= adam)

# Training
f = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=50)

# Visually see the loss history.
plt.plot(f.history['loss'], c='blue', label='train loss')
plt.plot(f.history['val_loss'], c='red', label='validation loss')
plt.legend()
plt.show()

# Visually see the prediction result.
y_pred = model.predict(x_pred)
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.3,
label='train')
plt.scatter(x_test, y_test, s=20, c='red', alpha=0.3,
label='validation')
plt.scatter(x_pred, y_pred, s=5, c='red', label='test')
plt.legend()
plt.show()









