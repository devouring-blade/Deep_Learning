from keras.datasets import mnist
from keras.layers import Input, Conv1D, Activation, MaxPool1D, Flatten, Dense
from keras.models import Model
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = mnist.load_data()

n_class = len(set(y_test))
k_size = 5 # kernel size
n_kernel = 20 # number of kernels
p_size = 10 # pooling filter size
n_rows = x_train.shape[1] # number of rows of an image
n_cols = x_train.shape[2] # number of columns of an image

x_input = Input(batch_shape= (None, n_rows, n_cols))
conv = Conv1D(filters= n_kernel, kernel_size= k_size, strides= 1)(x_input)
act = Activation("relu")(conv)
pool = MaxPool1D(pool_size= p_size, strides= 1)(act)
flat = Flatten()(pool)
y_output = Dense(units= n_class, activation= "softmax")(flat)

model = Model(x_input, y_output)
model.load_weights(r"E:\pycharm\PycharmProjects\deep_learning\weights\resutl_5.weights.h5")
model.summary()

# model.compile(loss= "sparse_categorical_crossentropy", optimizer= "adam")
# hist = model.fit(x= x_train, y= y_train, epochs= 300, batch_size= 1000,
#                  validation_data= (x_test, y_test))
# plt.figure(figsize= (5, 3))
# plt.plot(hist.history["loss"], color= "red")
# plt.plot(hist.history["val_loss"], color= "blue")
# plt.title("history train")
# plt.xlabel("epochs")
# plt.ylabel("loss value")
# plt.legend()
# plt.show()

y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis= 1)
acc = np.mean(y_pred == y_test)
print(f"accuracy of the test data: {acc:.4f}")

print(classification_report(y_pred, y_test))

n_samples = 10
wrong_idxs = np.random.choice(np.where(y_pred != y_test)[0], n_samples, replace= False)
fig, ax = plt.subplots(1, n_samples, figsize= (14, 5))
for i, wrong_idx in enumerate(wrong_idxs):
    x = x_test[wrong_idx]
    ax[i].imshow(x)
    ax[i].set_title(str(y_test[wrong_idx]) + "/" + str(y_pred[wrong_idx]))
plt.show()








