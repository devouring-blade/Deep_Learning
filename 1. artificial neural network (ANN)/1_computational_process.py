import numpy as np

def sigmoid(x): return 1. / (1. + np.exp(-x))

# example data
x = np.array([[-0.05, 0.14],
              [0.04, 0.17],
              [0.67, -0.09],
              [0.51, 0.08],
              [0.00, 0.09],
              [0.11, 0.09],
              [0.62, -0.06],
              [0.32, -0.03],
              [0.06, -0.10],
              [0.58, 0.12],
              [0.46, -0.02],
              [0.14, 0.03]])
y = [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0]

wh = np.array([[ 1.17, -1.20, -1.07, 0.58],
               [-1.31, -0.12, 1.11, -0.68]])
bh = np.array([0.18, 0.66, 0.70, 0.12])
wo = np.array([[1.17], [-0.81], [-0.67], [1.48]])
bo = np.array([-0.38])

y_linear = (x @ wh + bh) @ wo + bo
y_sigmoid = sigmoid(y_linear)
y_pred = (y_sigmoid > 0.5) * 1.0

print("y_linear y_sigmoid y_pred y")
for a, b, c, d in zip(y_linear, y_sigmoid, y_pred, y):
    print(f"{a[0]:.2f}        {b[0]:.2f}         {c[0]}      {d}")










