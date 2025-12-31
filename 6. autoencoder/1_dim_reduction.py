from keras.layers import Dense, Dropout, Input
from keras.datasets import mnist
from keras.regularizers import L2
from keras.optimizers import Adam
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def Encoder(x):
    x1 = Dense(units= 300, activation= "relu")(x)
    x1 = Dropout(0.2)(x1)
    x2 = Dense(units= 100, activation= "relu")(x1)
    x2 = Dropout(0.2)(x2)
    return x2

def Decoder(x):
    x1 = Dense(units= 100, activation= "relu")(x)
    x1 = Dropout(0.2)(x1)
    x2 = Dense(units= 300, activation= "relu")(x1)
    x2 = Dropout(0.2)(x2)
    return  x2

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)

x_input = Input(batch_shape= (None, *x_train.shape[1: ]))
encode = Encoder(x_input)
z = Dense(units= 2, activation= "linear", activity_regularizer= L2(1e-4))(encode)
decode = Decoder(z)
x_output = Dense(units= x_train.shape[1], activation= "sigmoid")(decode)

model = Model(x_input, x_output)
# model.compile(loss= "binary_crossentropy", optimizer= Adam(0.001))
model.summary()

# hist = model.fit(x= x_train, y= x_train, epochs= 300, batch_size= 500,
#                  validation_data= (x_test, x_test))
# plt.plot(hist.history["loss"], color= "red")
# plt.plot(hist.history["val_loss"], color= "blue")
# plt.legend()
# plt.show()

model.load_weights(r"E:\pycharm\PycharmProjects\deep_learning\weights\result1.weights.h5")

encoder = Model(x_input, z)
x_pred = encoder.predict(x_test)


plt.figure(figsize= (8, 8))
for i in np.unique(y_test):
    idxs = np.where(y_test == i)[0]
    z = x_pred[idxs]
    plt.scatter(z[: , 0], z[: , 1], alpha= 0.5, label= str(i))
plt.legend()
plt.show()

pca = PCA(n_components= 2)
pca.fit(x_train)
x_pred_pca = pca.transform(x_test)

plt.figure(figsize= (8, 8))
for i in np.unique(y_test):
    idxs = np.where(y_test == i)[0]
    z = x_pred_pca[idxs]
    plt.scatter(z[:, 0], z[: ,1], alpha= 0.5, label= str(i))
plt.legend()
plt.show()



# x_train1 = x_train[: 10000]
# x_test1 = x_test[: 5000]
# y_test1 = y_test[: 5000]
#
# pca = KernelPCA(n_components= 3)
# pca.fit(x_train1)



