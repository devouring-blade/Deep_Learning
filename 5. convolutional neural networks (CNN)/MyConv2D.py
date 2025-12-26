from keras.layers import Layer
import tensorflow as tf
from numpy.ma.core import shape
import numpy as np


class conv_2d(Layer):
    def __init__(self, n_filters, kernel_size, stride= 1, padding= "VALID"):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def build(self, input_shape):
        self.w = self.add_weight(shape= (self.n_filters, self.kernel_size[0], self.kernel_size[1], input_shape[-1]),
                                 initializer= "glorot_uniform",
                                 trainable= True)
        self.b = self.add_weight(shape= (self.n_filters, ), initializer= "zeros", trainable= True)

    def call(self, x):
        if self.padding == "same":
            pad_rows = int((x.shape[1]*(self.stride - 1) + self.kernel_size[0] - self.stride) / 2)
            pad_cols = int((x.shape[2]*(self.stride - 1) + self.kernel_size[1] - self.stride) / 2)
            px = tf.pad(x, ((0, 0), (pad_rows, pad_rows), (pad_cols, pad_cols), (0, 0)))
            n_rows, n_cols = x.shape[1], x.shape[2]
        else:
            px = x
            n_rows = int((x.shape[1] - self.kernel_size[0])/self.stride) + 1
            n_cols = int((x.shape[2] - self.kernel_size[1])/self.stride) + 1

        feat_map = []
        for i in range(n_rows):
            for j in range(n_cols):
                p = px[: , (i*self.stride): (i*self.stride + self.kernel_size[0]), (j*self.stride): (j*self.stride + self.kernel_size[1]), : ]
                mul = self.w[tf.newaxis, : , : , : , : ] * p[: , tf.newaxis, : , : , :]
                feat_map.append(tf.reduce_sum(mul, axis= (-3, -2, -1)))

        feat_map = tf.stack(feat_map)
        feat_map = tf.transpose(feat_map, (1, 0, 2))
        feat_map = tf.reshape(feat_map, (x.shape[0], n_rows, n_cols, -1))
        feat_map += self.b[tf.newaxis, tf.newaxis, tf.newaxis, : ]
        return feat_map

class pool_2d(Layer):
    def __init__(self, pooling_size, stride, method= "avg"):
        super().__init__()
        self.pooling_row = pooling_size[0]
        self.pooling_col = pooling_size[1]
        self.stride = stride
        self.method = method

    def call(self, x):
        n_rows = (x.shape[1] - self.pooling_row) // self.stride + 1
        n_cols = (x.shape[2] - self.pooling_col) // self.stride + 1

        feat_map = []
        for i in range(n_rows):
            for j in range(n_cols):
                px = x[: , (i*self.stride): (i*self.stride + self.pooling_row), (j*self.stride) : (j*self.stride + self.pooling_col), : ]
                if self.method == "avg": pooled = tf.reduce_mean(px, axis= (1, 2))
                else: pooled = tf.reduce_max(px, axis= (1, 2))
                feat_map.append(pooled)

        feat_map = tf.stack(feat_map)
        feat_map = tf.transpose(feat_map, (1, 0, 2))
        feat_map = tf.reshape(feat_map, (x.shape[0], n_rows, n_cols, -1))
        return feat_map











