import numpy as np
import tensorflow as tf


# set the parameters as variables in TensorFlow
x1 = tf.Variable(2.)
x2 = tf.Variable(5.)

# perform automatic differentiation
# forward pass
with tf.GradientTape() as tape:
    y = tf.math.log(x1) + x1 * x2 - tf.math.sin(x2)

# backward pass
dx1, dx2 = tape.gradient(y, [x1, x2])

print(f"dx1: {dx1.numpy()}")
print(f"dx2: {dx2.numpy()}")
