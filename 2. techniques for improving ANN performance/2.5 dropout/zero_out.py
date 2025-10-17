from tensorflow.keras.layers import Input, Dense, Multiply
from tensorflow.keras.models import Model
import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# A mask that sets the output of the third neuron in the hidden layer to 0.
mask = np.array([[1., 1., 0., 1.]])

# create a simple network
x_input =  Input(batch_shape= (None, 2))
h1 = Dense(4, name= "h1", activation= "sigmoid")(x_input)
h2 = Multiply(name= "h2")([h1, mask]) # set the output of the third neuron to 0.
y_output = Dense(1, name='y', activation='sigmoid')(h2)

model = Model(inputs= x_input, outputs= y_output)
model.compile(loss= 'mse', optimizer='adam')

print('\n# weights of h1 before training:')
print(model.get_layer('h1').get_weights()[0])
print('\n# weights of y before training:')
print(model.get_layer('y').get_weights()[0])

# Training
print('\n----- train -----')
model.fit(x, y, epochs=100, verbose=0)

# After training, check the weights of the h1 layer.
# Compare with before training.
# Check if the 3rd column is the same as before training.
# Check if w's in the 3rd column are not updated.
print('\n# weights of h1 after training:')
print(model.get_layer("h1").get_weights()[0])

# After training, check the weights of the y layer.
# Compare with before training.
# Check if w's in the 3rd column are not updated.
print('\n# weights of y after training:')
print(model.get_layer('y').get_weights()[0])

# Check the output h1.
model_h1 = Model(inputs= x_input, outputs= h1) # model for checking h1 outputs
print('\n# h1 output:\n')
print(model_h1.predict(x, verbose= 0))

# Check the output h2. Check that the outputs of the third neuron are all 0.
model_h2 = Model(inputs= x_input, outputs= h2) # model for checking h2 outputs
print('\n# h2 output:\n')
print(model_h2.predict(x, verbose= 0))







