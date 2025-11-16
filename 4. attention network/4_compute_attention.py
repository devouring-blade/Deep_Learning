import numpy as np
from keras.layers import Dot, Activation, Concatenate


E = np.array([[[0.707, 0.616, 0.852],
               [0.19 , 0.113, 0.123],
               [0.757, 0.022, 0.236],
               [0.54 , 0.923, 0.412]]])

D = np.array([[[0.786, 0.634, 0.873],
               [0.796, 0.949, 0.872],
               [0.704, 0.314, 0.912],
               [0.293, 0.075, 0.73 ]]])


def attention_layer(e, d):
    dot_product = Dot(axes= (2, 2))([d, e])
    score = Activation("softmax")(dot_product)
    value = Dot(axes= (2, 1))([score, e])
    output = Concatenate()([value, d])
    return dot_product, score, value, output

d, s, v, o = attention_layer(E, D)
print("\nDot-product:")
print(np.round(d, 3))
print("\nScore:")
print(np.round(s, 3))

print("\nAttention values:")
print(np.round(v, 3))
print("\nAttentional hidden states:")
print(np.round(o, 3))






