import numpy as np
from creating_training_data import create_data
from dense_layer_functions import LayerDense
from activation_function import Activation_ReLU


# Create dataset
X, y = create_data(100, 3)

# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense(2, 3)  # first dense layer, 2 inputs (each sample has 2 features), 3 outputs

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Fwd pass thru activation func. Takes in output from prev layer
activation1.forward(dense1.output)

# Let's see output of few first samples:
print(activation1.output[:5])
# This array has been rectified removing negative values.

