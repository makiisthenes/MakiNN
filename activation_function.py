# Rectified Linear Activation Function
# Sigmoid Activation Function Graph
# Rectified Linear Units Activation Function
# The Softmax Activation Function
import numpy as np
import math
# Basic Example of Simple Rectified Linear Activation Function...
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
for input_index in inputs:
    # print(input_index)
    if input_index > 0:
        output.append(input_index)
    else:
        output.append(0)
print(output)
# Another way to do it is...
output = []
for i in inputs:
    output.append(max(0, i))
print(output)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities






# The Softmax Activation Function - To determine probabilities and whether or not a certain classifier is right.
layer_outputs = [4.8, 1.21, 2.385]
print(math.e)
expo_outputs = []
final_output = []
# First all values are multiplied by ^e, then each orginal value is divided by the sum of all e values that gives us a probability out of 1.
for output in layer_outputs:
    expo_outputs.append(math.e**output)
sum_expo = sum(expo_outputs)
for output in expo_outputs:
    final_output.append(output/sum_expo)
print(final_output)
print(sum(final_output))

# Or there is a more easier way to do this with numpy
exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values)
print(norm_values)




# get unnormalized probabilities
exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
# normalize them for each sample
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)



