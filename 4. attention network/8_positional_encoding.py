import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt


def positional_encoding(n_position, d_model):
    position_dims = np.arange(n_position)[: , np.newaxis]
    embed_dims = np.arange(d_model)[np.newaxis, : ]
    angle_rates = 1 / np.power(10000.0, (2 * (embed_dims // 2)) / d_model)
    angle_rads = position_dims * angle_rates

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    return np.concatenate([sines, cosines], axis=-1)

pe = positional_encoding(6, 8)
print(np.round(pe, 3))

for i in range(pe.shape[0] - 1):
    d = euclidean_distances(pe[i].reshape(1, -1), pe[i + 1].reshape(1, -1))
    norm = np.linalg.norm(pe[i])
    dot = np.dot(pe[i], pe[i + 1])
    print("%d - %d : distance = %.4f, norm = %.4f, dot = %.4f" % (i, i + 1, d[0, 0], norm, dot))

PE = positional_encoding(20, 2)
plt.figure(figsize=(6,6))
plt.plot(PE[:,0], PE[:,1], marker='o', linewidth=1.0, color='red')
plt.show()

PE = positional_encoding(20, 3)
fig = plt.figure(figsize=(6,6), dpi=100)
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot(PE[:,0], PE[:,1], PE[:, 2], marker='o', linewidth=1.0, color='red')
plt.show()

