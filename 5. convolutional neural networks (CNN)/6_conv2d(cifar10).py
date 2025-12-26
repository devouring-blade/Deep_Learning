from keras.datasets import cifar10
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Activation
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers import RandomFlip, RandomRotation
import numpy as np
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train, y_test = np.reshape(y_train, (y_train.shape[0], )), np.reshape(y_test, (y_test.shape[0], ))
categories = set(y_test)

def conv2d_pool(x, n_filters, k_size, p_size):
    x = Conv2D(filters= n_filters, kernel_size= k_size)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D(pool_size= p_size)(x)
    return x

x_input = Input(batch_shape= (None, *x_train.shape[1: ]))
x = RandomFlip(mode= "horizontal")(x_input)
x = RandomRotation(0.1)(x)
h = conv2d_pool(x, 16, k_size= (3, 3), p_size= (2, 2))
h = conv2d_pool(h, 32, k_size= (3, 3), p_size= (2, 2))
h = conv2d_pool(h, 64, k_size= (3, 3), p_size= (2, 2))
h = Flatten()(h)
h = Dense(units= 128, activation= "relu")(h)
h = Dropout(0.5)(h)
y_output = Dense(units= len(categories), activation= "softmax")(h)

model = Model(x_input, y_output)
model.load_weights(r"E:\pycharm\PycharmProjects\deep_learning\weights\resutl_6.weights.h5")
# model.compile(loss= "sparse_categorical_crossentropy", optimizer= "adam")
model.summary()

# hist = model.fit(x= x_train, y= y_train, epochs= 300, batch_size= 1000,
#                  validation_data= (x_test, y_test), shuffle= True)

# plt.figure(figsize= (7, 5))
# plt.plot(hist.history["loss"], color= "red")
# plt.plot(hist.history["val_loss"], color= "blue")
# plt.title("history training")
# plt.legend()
# plt.show()

y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis= 1)
print(classification_report(y_test, y_pred))

accuracy = np.mean(y_test == y_pred)
print(f"accuracy of test data: {accuracy:.4f}")

cate = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

n_sample = 20
miss_idxs = np.random.choice(np.where(y_pred != y_test)[0], n_sample, replace= False)
fig, ax = plt.subplots(2, 10, figsize= (14, 7))
for i, miss_idx in enumerate(miss_idxs):
    x = x_test[miss_idx]
    if i < 10:
        ax[0][i % 10].imshow(x)
        ax[0][i % 10].set_title(str(cate[y_test[miss_idx]]) + "/\n" + str(cate[y_pred[miss_idx]]))
    else:
        ax[1][i % 10].imshow(x)
        ax[1][i % 10].set_title(str(cate[y_test[miss_idx]]) + "/" + str(cate[y_pred[miss_idx]]))
plt.show()


