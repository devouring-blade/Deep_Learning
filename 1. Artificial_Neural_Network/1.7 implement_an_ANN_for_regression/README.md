# implement an ANN using numerical differentiation and gradient descent for regression
linear regression can be implemented using a single-layer Perceptron (SLP), and non-linear regression can be implemented using a 
multi-layer Perceptron (MLP)
The output layer uses a linear activation function (f(z) = z), and the hidden layer uses nonlinear activation function, such as sigmoid, tanh, ReLU.
Mean squared error (MSE) is used as the loss function.

<img width="1311" height="549" alt="{E597754D-FE13-4256-A899-7CF6A5CEE51B}" src="https://github.com/user-attachments/assets/df5b835b-7504-4ac8-b471-05dfc14b0fec" />

## single-layered Perceptron
Create a single-layered ANN model and perform the linear regression using numerical differentiation and gradient descent.
To calculate the gradient accurately, you must use automatic differentiation. However, here we use numerical differentiation to approximate the gradient.
Gradient descent with automatic differentiation will be discussed in detail in the backpropagation part later.

<img width="1342" height="563" alt="{F0438AAD-129F-4283-88D1-B05F835D2541}" src="https://github.com/user-attachments/assets/d0303fbd-3505-4180-bedb-5719406b4ebf" />

<img width="1349" height="644" alt="{AB494E8C-E824-46B8-84CE-FAD73BEDAEE1}" src="https://github.com/user-attachments/assets/787ce498-95c3-4f7d-b8d0-57ae5ee3be51" />

## two-layered Perceptron
<img width="1348" height="641" alt="{5802E153-78E4-4DF3-A266-FB9C4CE909A3}" src="https://github.com/user-attachments/assets/4b9f7bb7-4153-471f-ac56-4b94f218f32b" />

<img width="1344" height="636" alt="{C27E9D28-06A4-4404-A5F9-505322C56FDC}" src="https://github.com/user-attachments/assets/24a3e8be-2e52-4aba-8c9e-1a47e2b5939b" />



