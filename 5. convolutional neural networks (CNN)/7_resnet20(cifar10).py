from keras.layers import (ZeroPadding2D, Conv2D, BatchNormalization, Activation, Add,
                          GlobalAveragePooling2D, Input, RandomFlip, RandomRotation, Dense)
from keras.datasets import cifar10
from keras.models import Model
import numpy as np
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt


def res_block(x, filters, strides):
    x1 = ZeroPadding2D(padding= (1, 1))(x)
    x1 = Conv2D(filters= filters, kernel_size= (3, 3), strides= strides[0])(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)

    x2 = ZeroPadding2D(padding= (1, 1))(x1)
    x2 = Conv2D(filters= filters, kernel_size= (3, 3), strides= strides[1])(x2)
    x2 = BatchNormalization()(x2)

    if strides[0] == 2:
        residual = Conv2D(filters= filters, kernel_size= (1, 1), strides= 2)(x)
        residual = BatchNormalization()(residual)
    else:
        residual = x

    output = Add()((x2, residual))
    output = Activation("relu")(output)
    return output

def resnet20(x):
    x0 = ZeroPadding2D(padding= (1, 1))(x)
    x0 = Conv2D(filters= 16, kernel_size= (3, 3), strides= 1)(x0)
    x0 = BatchNormalization()(x0)
    x0 = Activation("relu")(x0)

    x1 = res_block(x0, 16, (1, 1))
    x1 = res_block(x1, 16, (1, 1))
    x1 = res_block(x1, 16, (1, 1))

    x2 = res_block(x1, 32, (2, 1))
    x2 = res_block(x2, 32, (1, 1))
    x2 = res_block(x2, 32, (1, 1))

    x3 = res_block(x2, 64, (2, 1))
    x3 = res_block(x3, 64, (1, 1))
    x3 = res_block(x3, 64, (1, 1))

    output = GlobalAveragePooling2D()(x3)
    return output


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train, y_test = y_train.reshape((-1, )), y_test.reshape((-1, ))

x_input = Input(batch_shape= (None, *x_train.shape[1: ]))
pre = RandomFlip(mode= "horizontal_and_vertical")(x_input)
pre = RandomRotation(0.2)(pre)
h = resnet20(pre)
y_output = Dense(units= 10, activation= "softmax")(h)

model = Model(x_input, y_output)
# model.compile(loss= "parse_categorical_crossentropy", optimizer= "adam")
model.load_weights(r"E:\pycharm\PycharmProjects\deep_learning\weights\result_7.weights.h5")
model.summary()


y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis= 1)

print(classification_report(y_test, y_pred))

accuracy = np.mean(y_pred == y_test)
print(f"accuracy of the test data: {accuracy:.4f}")

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

n_samples = 20
wrong_idxs = np.random.choice(np.where(y_pred != y_test)[0], n_samples, replace= False)

fig, ax = plt.subplots(2, 10, figsize= (10, 7))
for i, wrong_idx in enumerate(wrong_idxs):
    sample = x_test[wrong_idx]
    if i < 10:
        ax[0][i % 10].imshow(sample)
        ax[0][i % 10].set_title(str(class_names[y_test[wrong_idx]]) + "/\n" +
                                str(class_names[y_pred[wrong_idx]]))
    else:
        ax[1][i % 10].imshow(sample)
        ax[1][i % 10].set_title(str(class_names[y_test[wrong_idx]]) + "/\n" +
                                str(class_names[y_pred[wrong_idx]]))
plt.show()





