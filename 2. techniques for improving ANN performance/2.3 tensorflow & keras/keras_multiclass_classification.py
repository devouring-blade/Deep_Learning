import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

# Generate a dataset for multiclass classification
x, y = make_blobs(n_samples=400, n_features=2,
                  centers=[[0., 0.], [0.5, 0.1], [1., 0.]],
                  cluster_std=0.15, center_box=(-1., 1.))
x_train, x_test, y_train, y_test = train_test_split(x, y)
n_class = np.unique(y).shape[0] # the number of classes

# Visually see the data.
plt.figure(figsize=(5,4))
color = [['red', 'blue', 'green'][a] for a in y.reshape(-1,)]
plt.scatter(x[:, 0], x[:, 1], s=70, c=color, alpha=0.3)
plt.show()

# create an ANN with a hidden layer
n_input = x.shape[1]
n_output = n_class
n_hidden = 8
adam = optimizers.Adam(learning_rate= 0.01)

# create an ANN model
x_input = Input(batch_shape= (None, n_input))
h = Dense(n_hidden, activation= "relu")(x_input)
y_output = Dense(n_output, activation= "softmax")(h)
model = Model(x_input, y_output)
model.compile(loss= "sparse_categorical_crossentropy", optimizer= adam)

# training
f = model.fit(x_train, y_train, validation_data= (x_test, y_test), epochs= 200, batch_size= 50)

# Visually see the loss history
plt.plot(f.history['loss'], c='blue', label='train loss')
plt.plot(f.history['val_loss'], c='red', label='validation loss')
plt.legend()
plt.show()

# Check the accuracy of the test data
y_pred = model.predict(x_test)
acc = (np.argmax(y_pred, axis=1) == y_test).mean()
print("Accuracy of test data = {:.2f}".format(acc))

