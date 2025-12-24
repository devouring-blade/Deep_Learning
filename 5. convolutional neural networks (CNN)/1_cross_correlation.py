import numpy as np
import pickle
from matplotlib import pyplot as plt

def cross_correlation(x, w):
    return np.sum(x * w)

def convolution(x, w):
    fw = np.flipud(np.fliplr(w))
    return np.sum(x * fw)

def feat_map(x, w, s, method):
    rx, cx = x.shape
    rw, cw = w.shape
    rf, cf = int((rx - rw) / s + 1), int((cx - cw) / s + 1)
    feat = np.zeros(shape= (rf, cf))
    for i in range(rf):
        for j in range(cf):
            px = x[(i*s): (i*s + rw), (j*s): (j*s + cw)]
            if method == "CROSS": feat[i, j] = cross_correlation(px, w)
            else: feat[i, j] = convolution(px, w)
    return np.maximum(feat, 0)

with open("dataset/mnist.pkl", "rb") as f:
    x, y = pickle.load(f)

xi = x[12]
print(xi.shape)
plt.imshow(xi)
plt.show()

w1 = np.random.normal(size= (7, 7))
w2 = np.random.normal(size= (7, 7))

feat1 = feat_map(xi, w1, 1, "CROSS")
feat2 = feat_map(xi, w2, 1, "CROSS")
fig, ax = plt.subplots(1, 2, figsize= (10, 10))
fig.suptitle("feat map by cross correlation ")
ax[0].imshow(feat1)
ax[1].imshow(feat2)
plt.show()

feat1 = feat_map(xi, w1, 1, "CONV")
feat2 = feat_map(xi, w2, 1, "CONV")
print(feat1)
fig, ax = plt.subplots(1, 2, figsize= (10, 10))
fig.suptitle("feat map by convolution ")
ax[0].imshow(feat1)
ax[1].imshow(feat2)
plt.show()


