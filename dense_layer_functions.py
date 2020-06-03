import numpy as np

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.05
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class LayerDense:
    def __init__(self, inputs, neurons):
        # Initialize weights and biases
        self.weights = np.random.randn(inputs, neurons) * 0.01  # The parameter placement is intentional to ensure no transposition required.
        # Drawing to aid understanding.
        self.biases = np.zeros((1, neurons))
        # self.output = None
        # Forward pass

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        # self.weights1 = np.random.randn(self.output, 4)
        # Draw a diagram to understand this better...

# randomly_gen_array = np.random.randn(2, 4) * 0.01
# print(randomly_gen_array)


# Create dataset
X, y = create_data(100, 3)
# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense(2, 3)  # first dense layer, 2 inputs (each sample has 2 features), 3 outputs
# Make a forward pass of the training data through this layer
dense1.forward(X)
# Let's see output for first few samples:
print(dense1.output[:5])


# I will improvise and continue the neural network forward through a set of more neurons

# print(dense1.weights1)
