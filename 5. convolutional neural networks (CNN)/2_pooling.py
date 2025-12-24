import numpy as np
import pickle
from matplotlib import pyplot as plt


def convolution(x, w):
    return np.sum(x * w)

def feat_map(x, w, s):
    rx, cx = x.shape
    rw, cw = w.shape
    rf, cf = int((rx - rw)/s + 1), int((cx - cw)/s + 1)
    feat = np.zeros(shape= (rf, cf))
    for i in range(rf):
        for j in range(cf):
            px = x[(i*s): (i*s + rw), (j*s): (j*s + cw)]
            feat[i, j] = convolution(px, w)
    return np.maximum(feat, 0)

def pooling(x, w, s, method):
    rx, cx = x.shape
    rw, cw = w.shape
    rf, cf = int((rx - rw) / s + 1), int((cx - cw) / s + 1)
    feat = np.zeros(shape=(rf, cf))
    for i in range(rf):
        for j in range(cf):
            px = x[(i * s): (i * s + rw), (j * s): (j * s + cw)]
            if method == "max": feat[i, j] = np.max(px)
            else: feat[i, j] = np.mean(px)
    return feat

def show_image(x, size, title):
    plt.figure(figsize= size)
    plt.imshow(x)
    plt.title(title)
    plt.show()

w1 = np.random.normal(size = (7, 7))
w2 = np.empty(shape= (3, 3))

with open("dataset/mnist.pkl", "rb") as f:
    x, y = pickle.load(f)

xi = x[13]
show_image(xi, (7, 7), "input image")

feat = feat_map(xi, w1, 1)
h, w = feat.shape
feat_size = (h/xi.shape[0] * 7, w/xi.shape[1] * 7)
show_image(feat, feat_size, "feature map")

max_pooling = pooling(feat, w2, 1, "max")
h, w = max_pooling.shape
max_size = (h/xi.shape[0] * 7, w/xi.shape[1] * 7)
show_image(max_pooling, max_size, "max pooling")

avg_pooling = pooling(feat, w2, 1, "avg")
h, w = avg_pooling.shape
avg_size = (h/xi.shape[0] * 7, w/xi.shape[1] * 7)
show_image(avg_pooling, avg_size, "average pooling")





