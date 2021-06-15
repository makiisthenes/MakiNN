from numba import cuda  # Run code using GPU instead
import numpy as np      # Used to deal with array.
import os

input = [1.0, 2.0, 3.0, 2.5]
weight = [0.2, 0.8, -0.5, 1.0]
bias = 2.0
outputs = np.dot(weight, input) + bias
print(outputs)
if tuple.__itemsize__ != 4:
    print(f"GPUs on this system:: {cuda.gpus}")
else:
    print("Python interpreter is 32 bit, use 64 bit for GPU support")
# Learn full functionality of np.array() [ ]

# What defines a neuron in NN is that they need to have all inputted weights and inputs they same homogenous array and also the same number of bais.

# From 4 outputs to 3 nuerons input.
weights = [[2,4,3,2],
           [4,5,1,2],
           [1,4,2,3]]
input = [2,3,5,1]
biases = [10, 10, 10]

print("----- Full Output Example Tests -----")
print("Numpy dot product using transpose feature:: ")
print((np.dot(input, np.array(weights).T) + biases))
# The parameter of the dot product on the left needs to be one with the array so it can be treated as a vector of an vector.

print("Numpy using the weights and then inputs do you the array shapes. Input:: 4 and Weights:: 3,4")
print(np.dot(weights, input))
# Due to this being a matrix and a vector, numpy will automatically treat the matrix as a list of vectors
# and the answer will be a vector.

input = [1.0, 2.0, 3.0, 2.5]
weight = [0.2, 0.8, -0.5, 1.0]

weight = np.array(weight).T
output = np.dot(input, weight) + bias
print(output)


# Outputting Neuron Calculation using Matrixes of Inputs and Weights for each node.
# Level 1, using 1 input and 3,4 weights matrix dot product. correct [x]

input   =  [1.0  , 2.0  , 3.0 , 2.5]
weights =  [[0.2 , 0.8  , -0.5, 1.0],
           [0.5  , -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
bais1    =  [2.0   , 3.0    , 0.5]
# This shows that there are 4 initial inputs that are used to go to the nodes.
# There are 3 sets of weights so we are looking at 3 nodes.
print(f"Output Level 1: {np.dot(weights, input) + bais1}")

# Level 2, using a batch of inputs now instead...             correct [x]
# Now we will be using 2 matrixes so we will need to pay attention to tranposing.
input   =  [[1.0,2.0,3.0,2.5],
            [2.0,5.0,-1.0,2.0],
            [-1.5,2.7,3.3,-0.8],
            [2.0,1.0,1.0,3.0],
            [4.0,2.0,1.0,3.0],
            [4.0,1.0,2.0,2.0]]

print(f"Output Array should have a shape {np.size(input,0)}, {np.size(weights,0)}")
# Weights axis is 0 because it hasnt been tranposed yet.
print("The first layer should be equal to level 1.")
print(f"Output Level 2: \n {np.dot(input, np.array(weights).T) + bais1}")



# Page 61 - Hidden Layers
# We now have a 3 node input, that needs to go to a x number of hidden layer nodes.
output_layer1 = np.dot(input, np.array(weights).T) + bais1
# x x x Shape (6,3)
# x x x
# 6 node hidden layer is the output, so we make a 6,x shape matrix.
# Given there is 3 output nodes to the hidden layer, the matrix shape is 6,3.
weights2 = [[1.0, 2.5, 3.0],
            [1.2, 3.2, 2.1],
            [0.7, 2.1, 1.5],
            [1.3, 2.4, 3.4],
            [0.4, 1.3, 1.7],
            [2.1, 2.5, 3.1]]
bais2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# outputs (6,3)
output_hidden_layer2 = np.dot(output_layer1, np.array(weights2).T) + bais2
print(output_hidden_layer2)



# Training Data

# Creating Data [NON-LINEAR DATA]
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()
import matplotlib.pyplot as plt

# Creates 100 pieces of data with shape x,2 as these are coords.
print("Generating data points X")
X, y = spiral_data(samples=100, classes = 3)
plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
# plt.show()
print(f"Class Y Actual Data Points:: {y}")
print(f"Target Y Class Shape {np.array(y).shape}")



# Dense Layer Class Pg66
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
    # Calculate output values from inputs
        self.output = np.maximum(0, inputs)


# This Activation Method is used for the output layer for classification network.
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# Calculating Network Error with Loss Pg 111
layer1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
print("----- Output Layer -----")
print(activation2.output[:5])
# Prints out the first 5 outputted layer

# Depending on the target of the neural network you set up, there will be different goals in mind,
# The shape of the nueral network will also differ and also the functions involved as well.
# For the hidden layers there are many activation functions we can use, these includes:
# ReLU    Activation Function
# Sigmoid Activation Function
# Linear  Activation Function
# Tanh    Activation Function

# The output layer would also have a activation function but this would depend on what type of network we are using.
# Due to our objective for this network being classification we will use the softmax activation function to get probalities.
# We also need a corresponding loss function this will be related to what activation function we have used for the output activation functions.
# In which we will be using categorical cross-entropy loss.

# Categoric Cross-Entropy Loss Function
import math

# An example output from the output layer of the neural network
softmax_output = [0.7, 0.1, 0.2]
# Ground truth
target_output = [1, 0, 0]
loss = -(math.log(softmax_output[0])*target_output[0] +
math.log(softmax_output[1])*target_output[1] +
math.log(softmax_output[2])*target_output[2])
print(f"Loss Function Output:: {loss}")

# log(x) ~ x < 1, Gives us a value that is negative always. Base natural e.

softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]
class_targets = [0, 1, 1] # dog, cat, cat

# We want to have the loss value of this batch of outputs.
for target_index, distribution_output in zip(class_targets, softmax_outputs):
    print(f"Value selected:: {distribution_output[target_index]}")
test = np.array(softmax_outputs)[[0,1,2], [1,1,1]]
confidence_lvls = np.array(softmax_outputs)[range(len(softmax_outputs)), class_targets]
loss_outputs = -np.log(confidence_lvls)
print(f"Testing np array index of list iteration:: {test}.")
print(f"Actual Correct Index Confidence Level:: {loss_outputs}")
print("An average loss value that is closer to 0, is much more useful.")
print(f"Average Loss for this batch[using python]:: {np.mean(loss_outputs)}")
# The target data needs to be the same as number of rows of output batches.


# Making the loss function more accesible and dynamic to different clasification training answer data.
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

# This is if the class_target answers is just a vector/list containing the correct indexes.
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)),class_targets]

elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_outputs*class_targets, axis=1)

loss = -np.log(correct_confidences)
average_loss = np.mean(loss)
print(f"Average Loss for this batch [using numpy]:: {np.mean(loss_outputs)}")

class Loss:
    # X means data, y means target data.
    def calculate(self, output, y):
    # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, output, y_true):
        samples = len(output)
        y_pred_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods



# Now we use the loss function on the existing softmax outputs.
loss_function = Loss_CategoricalCrossentropy()
likelihood = loss_function.calculate(softmax_outputs, class_targets)
print(likelihood)


# Continuing Actual Nueral Network Code/
print("Our Nueral Network Loss Function.")
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print(f"Loss:: {loss}")



# Accuracy Calculation
# Although Loss is a good metric for determining how close or far
# the value of target was fully confidently selected, its not the only
# indicator used.

# Accuracy determines how many times the network correctly selected the right (classifier in this case)
# Using np.argmax()  -- https://www.geeksforgeeks.org/numpy-argmax-python/
softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])
predictions = np.argmax(softmax_outputs, axis=1)     # argmax of predictions
if len(class_targets.shape) == 2:
    # argmax of class_targets, if matrix, will look for max value index which is 1.
    class_targets = np.argmax(class_targets, axis=1)
print(predictions == class_targets)
accuracy = np.mean(predictions == class_targets)
print('Accuracy:', accuracy)

# Accuracy Function -- Given the book did it in a non-function way, I'm going to do another way.

def get_accuracy(output, y):
    predictions = np.argmax(output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    # Print accuracy
    print(f"Accuracy:: {accuracy}%")
    return accuracy
    # >>> Accuracy:: 34% | This makes sense because its a random set, with 3 classes, so it makes sense to be 33% accurate.

# >>>Continuation of Neural Network
accuracy = get_accuracy(activation2.output, y)



# Optimization - Here we are tweaking the network to make it work...

# This is where we need to think, we could just randomise the weight and biases and record the highest
# accuracy or lowest loss, so we will try this, but this is not an efficent way of doing this.

print("Optimisation | Chapter 6 | Pg131")
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
# This is a dataset with 3 classification but not spiral but vertical.
nnfs.init()
X1, y1 = vertical_data(samples=100, classes=3)
# plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=40, cmap='brg')  # I dont know what s parameter is udsed for.
# plt.show()

def example_random_exhuastive():
    dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
    activation2 = Activation_Softmax()
    # Create loss function
    loss_function = Loss_CategoricalCrossentropy()
    lowest_loss = 9999999  # some initial value
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()
    for iteration in range(10000):
        # Generate a new set of weights for iteration
        dense1.weights = 0.05 * np.random.randn(2, 3)
        dense1.biases = 0.05 * np.random.randn(1, 3)
        dense2.weights = 0.05 * np.random.randn(3, 3)
        dense2.biases = 0.05 * np.random.randn(1, 3)
        # Perform a forward pass of the training data through this layer
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        # Perform a forward pass through activation function
        # it takes the output of second dense layer here and returns loss
        loss = loss_function.calculate(activation2.output, y)
        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)
        # If loss is smaller - print and save weights and biases aside
        if loss < lowest_loss:
            print('New set of weights found, iteration:', iteration,
                  'loss:', loss, 'acc:', accuracy)
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss

print("--> Using random exhuastive method:: ")
example_random_exhuastive()
# With this random exhaustive approach we can see that its not working as effiecently
# and doesnt yield results even after 1 billion iterations.

# Tweaking wieghts and bais depedning on loss value.
def tweak_example_random_exhaustive(iterations=100):
    X, y = vertical_data(samples=100, classes=3)
    # Create model
    dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
    activation2 = Activation_Softmax()
    # Create loss function
    loss_function = Loss_CategoricalCrossentropy()
    # Helper variables
    lowest_loss = 9999999  # some initial value
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()
    for iteration in range(iterations):
        # Update weights with some small random values
        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)
        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)
        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        # Perform a forward pass through activation function
        # it takes the output of second dense layer here and returns loss
        loss = loss_function.calculate(activation2.output, y)
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)
        # If loss is smaller - print and save weights and biases aside
        if loss < lowest_loss:
            print('New set of weights found, iteration:', iteration,
                  'loss:', loss, 'acc:', accuracy)
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
            # Revert weights and biases
        else:
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()

print("--> Using tweaking method based on best loss value:: ")
tweak_example_random_exhaustive()
# This method was quite successful tbh, WOW, I was shocked.
# This is not always the case, and its to do with "local minimum of loss"...

# Gradients, Partial Derivatives, and the Chain Rule.
# Derivatives                 [x]
# Partial Derivatives Sum     [x]
# Partial Derivatives Product [x]
# Partial Derivatives Max     [x]
# The Gradient                [x]
# The Chain Rule              [x]

# Backpropagation | Pg180 - This is to determine the gradients and partial derivatives required.

# Here we will use a one nueron example to understand backpropagation.

def basic_back_propagation():
    print("Example Back Propagation")
    x = [1.0, -2.0, 3.0]   # Inputs for this example
    w = [-3.0, -1.0, 2.0]  # Weights for this example
    b = 1.0                # Bais for this example
    xw0 = x[0]*w[0]
    xw1 = x[1]*w[1]
    xw2 = x[2]*w[2]
    print(f"--> Combined Input 0: {xw0} \n--> Combined Input 1: {xw1} \n--> Combined Input 2: {xw2} \n--> Bais: {b}")
    total = xw0 + xw1 + xw2 + b
    # ReLU activation function
    total = max(total, 0)
    print(f"--> Total Output: {total}")

    # y = ReLU(sum(mul(x0, w0), mul(x1, w1), mul(x2, w2), b))   -- This is the chained function used for this neuron.

    # Now we are going to calculate the impact of weight w0 on the output.

    # Backward pass
    # First going to determine the derivative of ReLU with respect to its input(nuerons output).
    # The derivative from the next layer [random value].
    dvalue = 1.0
    # Derivative of ReLU and the chain rule
    z = xw0 + xw1 + xw2 + b
    drelu_dz = dvalue * (1. if z > 0 else 0.)
    print(f"--> Derivative ReLU with respect to input functions:: {drelu_dz}\n")
    # We have calculated the derivative of ReLU w.r.t the input which is the prior function.

    # Now we need to look at the previous function which is the sum function and find each partial derivative and use chain rule.
    # Pg 187

    # Due to the sum of partial derivatives always being 1 w.r.t all weighted inputs.
    # Sum Partial Derivatives Theory Background [x] -- Done in paint.
    dsum_dxw0 = 1
    dsum_dxw1 = 1
    dsum_dxw2 = 1
    dsum_dxb  = 1
    drelu_dxw0 = drelu_dz * dsum_dxw0
    drelu_dxw1 = drelu_dz * dsum_dxw1
    drelu_dxw2 = drelu_dz * dsum_dxw2
    drelu_db   = drelu_dz * dsum_dxb
    # Print out all values.
    print(f"--> Derivative ReLU with respect of weighted input 0:: {drelu_dxw0}\n"
          f"--> Derivative ReLU with respect of weighted input 1:: {drelu_dxw1}\n"
          f"--> Derivative ReLU with respect of weighted input 2:: {drelu_dxw1}\n"
          f"--> Derivative ReLU with respect of ****bais input 0:: {drelu_db}\n")

    # Now we want to go further back, back to the multiplication from the sums.
    # Multplicative Partial Derivatives Theory Background [x] -- Done in paint.
    dmul_dx0 = w[0]
    dmul_dx1 = w[1]
    dmul_dx2 = w[2]
    dmul_dw0 = x[0]
    dmul_dw1 = x[1]
    dmul_dw2 = x[2]
    drelu_dx0 = drelu_dxw0 * dmul_dx0
    drelu_dx1 = drelu_dxw1 * dmul_dx1
    drelu_dx2 = drelu_dxw2 * dmul_dx2
    drelu_dw0 = drelu_dxw0 * dmul_dw0
    drelu_dw1 = drelu_dxw1 * dmul_dw1
    drelu_dw2 = drelu_dxw2 * dmul_dw2
    # Print out calculated derivations.
    print("------- Don't Affect ReLU Result [not tweakable] ------- ")
    print(f"--> Derivative ReLU with respect of input  0:: {drelu_dx0}\n"
          f"--> Derivative ReLU with respect of input  1:: {drelu_dx1}\n"
          f"--> Derivative ReLU with respect of input  2:: {drelu_dx2}\n"
          f"------- These Affect ReLU Result [tweakable] ------- \n"
          f"--> Derivative ReLU with respect of weight 0:: {drelu_dw0}\n"
          f"--> Derivative ReLU with respect of weight 1:: {drelu_dw1}\n"
          f"--> Derivative ReLU with respect of weight 2:: {drelu_dw2}\n")

    # Pg 200 - Derivation of ReLU w.r.t x or w or b equation
    drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]

    # only need these when working backwards on nuerons and need thier output.
    dx = [drelu_dx0, drelu_dx1, drelu_dx2]
    dw = [drelu_dw0, drelu_dw1, drelu_dw2]  # gradients on weights
    db = drelu_db  # gradient on bias...just 1 bias here

    # Given the actual gradients of these wieghts on the ReLU function we can subtract them

    '''
    Weights Gradient Outputs from one neuron example.
    >>> dw = [1.0, -2.0, 3.0]

    '''
    print(f"Weights: {w}\nBias: {b}\n")
    print("Tweaking Weights given thier bias to try and lower the ReLU output [example of optimiser].")
    w[0] += -0.001 * dw[0]
    w[1] += -0.001 * dw[1]
    w[2] += -0.001 * dw[2]
    b += -0.001 * db
    print(f"--> Weights: {w}\n--> Bias: {b}\n")
    # Now testing the new weights that have been tweaked.
    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]
    print(f"-->New Output [tweaked]:: {max(xw0 + xw1 + xw2 + b, 0)}")
    # Pg 204 - Mapping a example full network using partial derivations and back propagation.

    # Given in back propagation we deal with partial differentiation, however we dont really differentiate here,
    # we only use rules and known outputs for derivatives and use shortcut equations to get values required.
    # We are going to simulate a 4 input, 3 node hidden layer.
    dvalues = np.array([[1., 1., 1.]])
    weights = np.array([[0.2, 0.8, -0.5, 1],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]]).T
    # We want to determine the values of delta input value from output.





    # To understand the this code below, look at image back_propagation2.png in dir.
    dx0 = sum(weights[0]* dvalues[0])
    dx1 = sum(weights[1]* dvalues[0])
    dx2 = sum(weights[2]* dvalues[0])
    dx3 = sum(weights[3]* dvalues[0])
    dinputs = np.array([dx0, dx1, dx2, dx3])
    print(f"Dinputs {dinputs}")
    d_inputs = np.dot(dvalues[0], weights.T)
    print(f"Dinputs {dinputs}")
    print(f"Gradients of Inputs w.r.t network function:: {dinputs}")

    # Now lets use a list of dvalues instead of a row vector
    # [I dont understand where this 2d matrix is derived from]
    # I think its to do with batch of input vectors at the start inputted.
    dvalues = np.array([[1., 1., 1.],
                        [2., 2., 2.],
                        [3., 3., 3.]])
    dinputs = np.dot(dvalues, weights.T)
    print(f"Delta function w.r.t inputs:: {dinputs}")

    # Now we need to determine the gradients of the wieghts of a given layer that can have batch of samples.
    # We need 2 arrays.
    # The dvalue array of the next layer, and also inputs of the current layer.
    dvalues = np.array([[1., 1., 1.],
                        [2., 2., 2.],
                        [3., 3., 3.]])
    inputs = np.array([[1, 2, 3, 2.5],
                       [2., 5., -1., 2],
                       [-1.5, 2.7, 3.3, -0.8]])
    # To get the gradient of the weights we perform a dot product and sum all of the inputs row wise, after each calculation.
    # This is already achieved by the dot product.
    d_wieghts = np.dot(inputs.T, dvalues)

    # Now we need to determine gradient of bias
    # this is really simple, we dont even need the biases, we need the dvalues from the next layer.
    biases = np.array([[2, 3, 0.5]])
    dbiases = np.sum(dvalues, axis=0, keepdims=True)
    print(f"Delta biases:: {dbiases}")


    # Determining the gradient of the reLU function on the array.
    z = np.array([[1, 2, -3, -4],
                  [2, -7, -1, 3],
                  [-1, 2, 5, -1]])
    dvalues = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12]])
    # ReLU activation's derivative
    drelu = np.zeros_like(z)
    drelu[z > 0] = 1
    print(drelu)
    # The chain rule
    drelu *= dvalues
    print(drelu)

    # Another way of doing the same thing is::
    drelu = dvalues.copy()
    drelu[z <= 0] = 0
    print(drelu)

basic_back_propagation()


def basic_forward_backward_pass():
    # Here will make example forward and backward pass in our neural network which will allow us to optimise our network.
    # We first need to define our network.
    dvalues = np.array(
        [[1., 1., 1.],
        [2., 2., 2.],
        [3., 3., 3.]])

    inputs = np.array(
        [[1, 2, 3, 2.5],
        [2., 5., -1., 2],
        [-1.5, 2.7, 3.3, -0.8]])

    weights = np.array(
        [[0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]).T

    biases = np.array([[2, 3, 0.5]])

    # Forward Pass
    layer_outputs = np.dot(inputs, weights) + biases
    relu_outputs = np.maximum(0, layer_outputs)

    # Backward Pass
    drelu = relu_outputs.copy()
    drelu[layer_outputs <= 0] = 0
    print(f"Shape of drelu is {drelu.shape}")
    # dReLU Array Shape 3,3

    # Recap what needs to be multiplied by what when determining the dinputs and dweights

    # Delta Inputs = drelu * wieghts [needs configuration of shape]
    d_inputs = np.dot(drelu, weights.T)

    # Delta Weights = drelu * inputs [needs configuration of shape]
    d_weights = np.dot(drelu, inputs)
    # print(f"My dot product{d_weights}")
    dweights = np.dot(inputs.T, drelu)
    # print(f"textbook dot product{d_weights}")

    # Delta Biases = sum(drelu) for each neuron in layer.
    dbiases = np.sum(drelu, axis=0, keepdims=True)



    # Then given all this info we can now optimise the nueron weights and bais to better fit output.
    # I currently dont know why we give this factor when updating weights and biases.
    weights += -0.001 * dweights
    biases += -0.001 * dbiases


basic_forward_backward_pass()

# Now we append the backward_pass function to the dense layer and the activation relu function.

# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0



# Now we need to know the derivative of Softmax and the Cat Loss Function (-log function with small threshold).

# Derivation of Loss Function

# Inherited Loss Class from class above...
class Loss_CategoricalCrossentropy(Loss):
    # Forward Pass
    def forward(self, output, y_true):
        samples = len(output)
        y_pred_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
            negative_log_likelihoods = -np.log(correct_confidences)
            return negative_log_likelihoods
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
            negative_log_likelihoods = -np.log(correct_confidences)
            return negative_log_likelihoods

    # Backward pass (Pg 218)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Derivation of Softmax Activation. (Pg 220)
softmax_output = [0.7, 0.1, 0.2]
# Makes output a column array.
softmax_output = np.array(softmax_output).reshape(-1, 1)
print(softmax_output)
# I understand the theory of the derivation but not much the code after the dot product of itself.

class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
        keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
        keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)


class Loss:
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss



class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Derivation of Softmax and Loss Combined in one function
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Due to the cba in me to learn derivation of functions of softmax and loss functions, I will just copy and paste
# I understand each seperate derivation but not the combined derivation


print("A full network without the optmisation")
X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)
print(loss_activation.output[:5])
print('loss:', loss)
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print('acc:', accuracy)
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)


# Optimiser | Pg 249 - This will be used to actually tweak the weights and biases to decrease loss.
# Constant Learning Rate.
# SGD optimizer
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

def train_model(samples=10000, epoch_num=100, learning_rate=1.0):
    X, y = spiral_data(samples, classes=3)
    # Create Dense layer with 2 input features and 64 output values
    dense1 = Layer_Dense(2, 64)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    # Create second Dense layer with 64 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = Layer_Dense(64, 3)
    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    # Create optimizer
    optimizer = Optimizer_SGD(learning_rate)
    # Train in loop
    for epoch in range(epoch_num+1):
        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)
        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forward(activation1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        loss = loss_activation.forward(dense2.output, y)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')
        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        # Update weights and biases
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

# print("Training with learning rate:: 1.00")
# train_model(samples=10000, epoch_num=10000, learning_rate=1.0)
# print("Training with learning rate:: 0.85")
# train_model(samples=10000, epoch_num=10000, learning_rate=0.85)
# Learning Rate can be adjusted to allow us to get base minimums rather than local minimums of the gradient.

# We will now explore learning rate decay. Pg 274
starting_learning_rate = 1.
learning_rate_decay = 0.1
for step in range(20):
    learning_rate = starting_learning_rate * (1. / (1 + learning_rate_decay * step))
    print(learning_rate)

class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    # This learning rate will decay over time
    def __init__(self, learning_rate=1., decay=0.):
    	# Defualt value lr=1, decay=0
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        # Update parameters
    def update_params(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases
        # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# Now we will create the train model function with addition of variable decay learning rate.
# Create dataset
# @cuda.jit
def training_data_with_decay_lr():
    print("Running on GPU")
    X, y = spiral_data(samples=100, classes=3)
    # Create Dense layer with 2 input features and 64 output values
    dense1 = Layer_Dense(2, 64)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    # Create second Dense layer with 64 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = Layer_Dense(64, 3)
    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    # Create optimizer
    optimizer = Optimizer_SGD(decay=1e-3)
    # Train in loop
    for epoch in range(100001):
        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)
        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forward(activation1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        loss = loss_activation.forward(dense2.output, y)
        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}, ' +
                  f'lr: {optimizer.current_learning_rate}')
        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

# training_data_with_decay_lr()
# Best Settings, decay 1e-4, 1000001
# Best Weights found acc: 0.930, loss:0.140, epoch:980000


# One way to improve optimisation, is the introduction of momentum with SGD Optimiser [default]
# Pg 282
class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        # Update parameters

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            
            # This line below stores the value of momentum in order to use in the next iteration for weights.
            layer.weight_momentums = weight_updates
            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            
            # This line below stores the value of momentum in order to use in the next iteration for bias.
            layer.bias_momentums = bias_updates
            # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
            # Update weights and biases using either
            # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


def training_with_momentum_decay(samples, epoch_num,learning_rate=1.0, decay=1e-3, momentum=0.5):
    print(f"Training model using Momentum:: {momentum}")
    # Create dataset
    X, y = spiral_data(samples, classes=3)
    # Create Dense layer with 2 input features and 64 output values
    dense1 = Layer_Dense(2, 64)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    # Create second Dense layer with 64 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = Layer_Dense(64, 3)
    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    # Create optimizer
    optimizer = Optimizer_SGD(learning_rate, decay, momentum)
    # Train in loop
    for epoch in range(epoch_num+1):
        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)
        # takes outputs of activation function of first layer as inputs
        dense2.forward(activation1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        loss = loss_activation.forward(dense2.output, y)
        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}, ' +
                  f'lr: {optimizer.current_learning_rate}')
                  # f'Gradient {dense1.weights}')

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

# Sometimes increasing momentum yeilds better results, if stuck on local minimum.
# An decreased decay will allow a shallow step of learning rate, stopping local minimum at start of training.
training_with_momentum_decay(100, 5000000,learning_rate=1.0, decay=1e-3, momentum=0.8)


# Review the different types of Optimisers in Paint window.
# SGD Default Optimiser Pg 250[x]
# SGD Optimiser with Momentum Pg 283[x]
# AdaGrad Optimiser Pg 293[x]
# RMSProp Optimiser Pg 298[]
# Adam Optimiser Pg 304[]

# When we look at the optimiser, this is what will tweak the weights and bias according to the derivative.
# We first tweak it depending on the learning rate.


# For these 2 optimiser use youtube, as book doesn't really make it understandable, just implements it.

# Another Optimiser is AdaGrad SGD.
# Adaptive Gradient allows the change in gradient, affecting the learning rate after a number of iterations. This doesnt use learning decay, and instead uses its own method and a constant initial learning rate.
# https://www.youtube.com/watch?v=GSmW59dM0-o [this video is really helpful and useful for me to understand the optimiser].


# One problem with AdaGrad SGD is the fact alpha t can become really high, and so is solved by AdaDelta and RMSProp SGD.
# Another Optimiser is RMSProp SGD/ & AdaDelta.
# This is basically the same thing like AdaGrad however has some restrictions and used averged wieght, not really sure about this one.
# https://www.youtube.com/watch?v=9wFBbAQixBM [this video was not really helpful but is the next in the playlist with this lecturer].

# Coding both Adagrad and RMSProp Optimisers. [These 2 optimisers are less accurate than the SGD Momentum Optimiser].

# Adagrad optimizer
class Optimizer_Adagrad:
    def __init__(self, learning_rate, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

    # Look at theory again to understand this code again when revisiting this class.

# We don't really need to care about these optimisers because they are useless compared to SGD Momentum.
# The only better optimiser is Adam which is the main optimiser used today.



# Regularization
# Dropout
