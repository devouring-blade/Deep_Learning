import numpy as np


def numerical_differentiation(x, y, f_predict, f_loss, parameters, h= 1e-8):
    all_grads = []
    for parameter in parameters:
        grads = np.zeros_like(parameter)
        n_rows, n_cols = parameter.shape
        for row in range(n_rows):
            for col in range(n_cols):
                p_ori = parameter[row, col]

                parameter[row, col] = p_ori + h
                y_hat = f_predict(x)
                f1 = f_loss(y, y_hat)

                parameter[row, col] = p_ori - h
                y_hat = f_predict(x)
                f2 = f_loss(y, y_hat)

                # center difference approximation
                grads[row, col] = (f1 - f2) / (2 * h)

                parameter[row, col] = p_ori

        all_grads.append(grads)
    return all_grads

def gradient_descent(x, y, f_predict, f_loss, parameters, lr):
    all_grads = numerical_differentiation(x, y, f_predict, f_loss, parameters)
    for parameter, grad in zip(parameters, all_grads):
        parameter -= lr * grad




