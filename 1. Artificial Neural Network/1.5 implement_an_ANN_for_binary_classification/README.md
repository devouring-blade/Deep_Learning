# implement an ANN using numerical differentiation and gradient descent for binary classification

## numerical differentiation and gradient descent
we compute the gradient of the loss function with respect to each parameter via numerical differentiation. and then we use gradient descent to update all parameters

<img width="1348" height="582" alt="{AB38FC1E-92C2-4AE1-A5D7-AC0983ABD5BF}" src="https://github.com/user-attachments/assets/4c4118ab-de81-40d0-96d7-76f824bd6dbd" />

## single-layered Perceptron
create an single - layered ANN model and perform binary classification using numerical differentiation and gradient descent.
To calculate the gradient accurately, you must use automatic differentiation. However, here we use numerical differentiation to approximate the gradient.
Gradient descent with automatic differentiation will be discussed in detail in the Backpropagation part.

<img width="1215" height="500" alt="{6D6B3199-AC4A-4EC2-890F-38C946888A3E}" src="https://github.com/user-attachments/assets/d193f791-24e7-4d14-bd2e-3274374c25d7" />

<img width="1213" height="583" alt="{B28DDCEB-0270-4A5F-97A2-6C201D2C656E}" src="https://github.com/user-attachments/assets/156caec5-7e39-41e8-a287-40dce2185601" />

<img width="1214" height="581" alt="{5E0AD020-3141-4988-AB0C-790E0134C821}" src="https://github.com/user-attachments/assets/05e1e245-c9cc-40a8-b922-88cb6b6a546c" />

## two-layered Perceptron
Create an two-layered ANN and perform binary classification using numerical differentiation and gradient descent.
To calculate the gradient accurately, you must use automatic differentiation. However, here we use numerical differentiation to approximate the gradient.
Gradient descent with automatic differentiation will be discussed in detail in the Backpropagation part.

<img width="1373" height="565" alt="{8D68F59C-AC94-462B-8FD4-F4AD022507AB}" src="https://github.com/user-attachments/assets/8b2e0b61-8014-4f0f-97bf-13bfc092022e" />

<img width="1374" height="652" alt="{9F139853-E331-41FD-B5A5-071740B7FF39}" src="https://github.com/user-attachments/assets/f883a0a0-3297-4c43-81f6-afa5ba3d7586" />

<img width="1372" height="658" alt="{34DA4203-E68B-4E66-91E8-2AB0AA2077CB}" src="https://github.com/user-attachments/assets/554eda7e-d2bc-46aa-a23a-296f424a146e" />

## two-layred Perceptron with linear activation function 
Even with a multi-layer perceptron, nonlinear problems cannot be solved if a linear activation function is used in the hidden layers.
Let's check this with an experiment.
just remove ReLU in predict function.

<img width="1376" height="595" alt="{0A7EA4EA-3FC6-4654-B499-CD7C715F3659}" src="https://github.com/user-attachments/assets/95bc9ace-6f0f-4d5b-9f85-5dedbab4c91a" />





