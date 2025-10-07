import numpy as np

# we compute the gradient of the loss function with respect to each parameter via numerical differentiation.
# and then we use gradient descent to update all parameters

h = 1e-4    # small value for numerical differentiation
def numerical_differentiation(x, y, f_loss, f_predict, parameters):
    gradients = []
    for i in range(len( parameters)):
        # ex: p[0] -> wh, p[1] -> bh, p[2] -> wo, p[3] -> bo
        rows, cols = parameters[i].shape
        grads = np.zeros_like(parameters[i])

        # apply numerical differentiation to all elements in p[i]
        for row in range(rows):
            for col in range(cols):
                # measures the change in loss when the p[i][row, col] element changes by h.
                # the remaining elements are fixed. this is an approximate gradient.
                p_org = parameters[i][row, col] # original value of p

                parameters[i][row, col] = p_org + h # the element at position (row, col) increases by h.
                y_hat = f_predict(x) # calculate y_hat
                f1 = f_loss(y, y_hat) # the amount of change in loss

                parameters[i][row, col] = p_org - h  # the element at position (row, col) decreases by h.
                y_hat = f_predict(x)  # calculate y_hat
                f2 = f_loss(y, y_hat)  # the amount of change in loss

                parameters[i][row, col] = p_org # restore p back to it's original value.

                # gradient at position (row, col)
                grads[row, col] = (f1 - f2) / (2.0 * h)

        gradients.append(grads)
    return gradients

# perform gradient descent
def gradient_descent(x, y, alpha, f_loss, f_predict, parameters):
    gradients = numerical_differentiation(x, y, f_loss, f_predict, parameters)
    for i in range(len(parameters)):
        parameters[i] -= alpha * gradients[i]   # update in-place








