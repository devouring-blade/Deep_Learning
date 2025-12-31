from keras.datasets import mnist
from keras.layers import ZeroPadding2D, Conv2D, Conv2DTranspose, Input
from matplotlib import pyplot as plt
from keras.models import Model
import numpy as np

def Encoder(x):
    x1 = ZeroPadding2D((1, 1))(x)
    x2 = Conv2D(filters= 10, kernel_size= (3, 3), strides= 2, activation= "relu")(x1)
    return x2

def Decoder(x):
    x1 = Conv2DTranspose(filters= 10, kernel_size= (3, 3), activation= "relu",
                         strides= 2, padding= "same")(x)
    x2 = Conv2D(filters= 1, kernel_size= (3, 3), strides= 1, padding= "same",
                activation= "sigmoid")(x1)
    return  x2

def show_image(x, idxs):
    plt.figure(figsize= (14, 4))
    for i, idx in enumerate(idxs):
        ax = plt.subplot(1, len(idxs), i+1)
        ax.imshow(x[idx], cmap= "gray")
        ax.axis("off")
    plt.show()


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
x_train, x_test = x_train.reshape((*x_train.shape, 1)), x_test.reshape((*x_test.shape, 1))


x_input = Input(batch_shape= (None, *x_train.shape[1: ]))
x_enc = Encoder(x_input)
x_dec = Decoder(x_enc)

model = Model(x_input, x_dec)
model.compile(loss= "binary_crossentropy", optimizer= "adam")
model.summary()

# hist = model.fit(x= x_train, y = x_train, epochs= 20, batch_size= 300,
#                  validation_data= (x_test, x_test))
# plt.plot(hist.history["loss"], color= "red")
# plt.plot(hist.history["val_loss"], color= "blue")
# plt.legend()
# plt.show()

# Chọn 10 ảnh ngẫu nhiên từ 10 lớp
indices = [np.random.choice(np.where(y_test == i)[0]) for i in range(10)]

# 1. Hiển thị ảnh gốc
print("Original Images:")
show_image(x_test.reshape(-1, 28, 28), indices)

# 2. Hiển thị ảnh sau khi nén (Feature Maps)
# Lấy 10 mẫu đã chọn
encoder = Model(x_input, x_enc)
x_samples = x_test[indices]
r_samples = encoder.predict(x_samples, verbose=0)

print("Latent Features (Mean of 10 filters):")
# r_samples lúc này có shape (10, 14, 14, 10)
show_image(np.mean(r_samples, axis=-1), range(10))

# 3. Hiển thị ảnh sau khi giải nén (Reconstructed)
print("Reconstructed Images:")
decoded_samples = model.predict(x_samples, verbose=0)
show_image(decoded_samples.reshape(-1, 28, 28), range(10))


