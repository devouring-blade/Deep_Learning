import numpy as np
from keras import initializers
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt

# read MNIST dataset
x, y = fetch_openml(name= "mnist_784", return_X_y= True)

# visualize data
labels = np.unique(y)
samples = []
for label in labels:
    indices = np.where(y == label)[0]
    random_index = np.random.choice(indices, 1)
    samples.append([x.iloc[random_index].values, label])
plt.figure(figsize= (10, 5))
for i, [image, label] in enumerate(samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image.reshape(28, 28), cmap= "gray")
    plt.title(f"label: {label}")
    plt.axis("off")
plt.show()

x, y = np.array(x) / 255, np.array(y.to_numpy().astype("int8")).reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y)

n_input = x.shape[1]
n_hidden = 20
n_output = len(labels)

# highway network
def high_way(x, n_layers):
    for i in range(n_layers):
        h = Dense(units= n_input, kernel_initializer= initializers.HeNormal, bias_initializer= "zeros", activation= "relu")(x)
        t = Dense(units= n_input, kernel_initializer= initializers.GlorotNormal, bias_initializer= "zeros", activation= "sigmoid")(x)
        x = h*t + x*(1 - t)
    return x

# create an ANN model with highway networks
x_input = Input(batch_shape= (None, n_input))
hidden = high_way(x_input, 20)
y_output = Dense(units= n_output, activation= "softmax")(hidden)

model = Model(x_input, y_output)
model.compile(loss= "sparse_categorical_crossentropy", optimizer= "adam")
model.summary()

# training
f = model.fit(x_train, y_train, batch_size= 1000, epochs= 50, validation_data= (x_test, y_test))

# visually see the loss history
plt.plot(f.history['loss'], label='Train loss')
plt.plot(f.history['val_loss'], label='Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

y_prob = model.predict(x_train)
y_pred = np.argmax(y_prob, axis= 1).reshape(-1, 1)
acc = np.mean(y_pred == y_test)
print(f"accuracy of the test data: {acc:.4f}")

# Let's check out some misclassified images.
n_samples = len(labels)
miss_idxs = np.where(y_test != y_pred)[0]
miss_samples = np.random.choice(miss_idxs, n_samples)
fig, ax = plt.subplots(2, n_samples, figsize=(14,4))
for i, miss in enumerate(miss_samples):
    x = x_test[miss] * 255
    ax[i].imshow(x.reshape(28, 28))
    ax[i].axis('off')
    ax[i].set_title(str(y_test[miss]) + ' / ' + str(y_pred[miss]))



