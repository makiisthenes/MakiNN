
# Lesson 1: Mapping of 4 inputs to 1 node, example neural network.

# structure of an neuron, with 3 previous neurons connected to it.
inputs = [1, 2, 3, 2.5]  # the output from the previous neurons will be the input of the current neuron
weights = [0.2, 0.8, -0.5, 1.0]  # given that every neuron has a unique weight attached to each input, we add them here
bias = 2  # every unique neuron has a unique bias
# Steps for neuron to take are:
# --> Add up (all the inputs * all the weights) plus bias
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias  #  + inputs[3]*weights[3]

print(output)  # this would be the static output of the neuron.
# >>> 2.3
