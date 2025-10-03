import numpy as np
from sklearn.metrics import accuracy_score

class ActivationFunction:
    def ReLU(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

if __name__ == '__main__':
    # Input data
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
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0])

    # Hidden layer weights
    Wh = np.array([[1.17, -1.20, -1.07, 0.58],
                   [-1.31, -0.12, 1.11, -0.68]])

    # Hidden layer bias
    bh = np.array([0.18, 0.66, 0.70, 0.12])

    # Output layer weights
    Wo = np.array([[1.17], [-0.81], [-0.67], [1.48]])

    # Output layer bias
    bo = np.array([-0.38])

    activator = ActivationFunction()
    outputs_hidden_layer = activator.ReLU(np.dot(x, Wh) + bh)
    y_pred = np.int64(activator.sigmoid(np.dot(outputs_hidden_layer, Wo) + bo) > 0.5)
    for i, j in zip(y_pred, y):
        print("prediction:", i, "ground truth:", j)
    print(accuracy_score(y_true=y, y_pred= y_pred))





