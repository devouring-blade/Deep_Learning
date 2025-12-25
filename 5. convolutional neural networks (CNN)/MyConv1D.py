import tensorflow as tf
from keras.layers import Layer


class conv_1d(Layer):
    def __init__(self, n_kernels, kernel_size, padding = "VALID"):
        super().__init__()
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):
        self.w = self.add_weight(shape= (self.n_kernels, self.kernel_size, input_shape[-1]),
                                 initializer= "glorot_uniform",
                                 trainable= True)
        self.b = self.add_weight(shape= (self.n_kernels, ),
                                 initializer= "zeros",
                                 trainable= True)

    def call(self, x):
        if self.padding == "same":
            n_pads = self.kernel_size // 2
            px = tf.pad(x, ((0, 0), (n_pads, n_pads), (0, 0)))
            n_rows = x.shape[1]
        else:
            px = x
            n_rows = x.shape[1] - self.kernel_size + 1

        feat_maps = []
        for i in range(n_rows):
            p = px[: , i: (i + self.kernel_size), : ]
            feat_maps.append(self.w[tf.newaxis, : , : , : ] * p[: , tf.newaxis, : , : ])

        feat_maps = tf.stack(feat_maps)
        feat_maps = tf.reduce_sum(feat_maps, (-2, -1))
        feat_maps = tf.transpose(feat_maps, (1, 0, 2))
        feat_maps += self.b[tf.newaxis, tf.newaxis, : ]
        return feat_maps







