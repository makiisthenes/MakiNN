import matplotlib
import numpy as np


# I bought the eBook for $29 from sendex's website.
# Start of eBook tutorial...
# In lesson 2 we introduced an concept where 4 inputs went into 3 nodes and we printed the output, of course this is tedious to do if we have a large neural network.
# So we use for loops instead.
# I never seen for loops that have 2 arguments passed for an zip() object..
# For my own understanding I will try to make a looped function for an end output array as an result.
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
layer_outputs = []
for neuron_weight, neuron_bias in zip(weights, biases):
    print(
        f'Neuron Weight: {neuron_weight}')  # for this increment, the wieghts will be associated with that specific node.
    print(f'Neuron Biases: {neuron_bias}')
    neuron_output = 0  # this will reset for each for increment.
    for specific_weight, specific_input in zip(neuron_weight, inputs):
        print(f'Neuron Tuple {specific_input}, {specific_weight}')
        neuron_output += specific_input * specific_weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(f'The output for these {len(weights)} is {layer_outputs}')  # which gives the same answer as book :)

'''
>>>
Neuron Weight: [0.2, 0.8, -0.5, 1]
Neuron Biases: 2
Neuron Tuple 1, 0.2
Neuron Tuple 2, 0.8
Neuron Tuple 3, -0.5
Neuron Tuple 2.5, 1
Neuron Weight: [0.5, -0.91, 0.26, -0.5]
Neuron Biases: 3
Neuron Tuple 1, 0.5
Neuron Tuple 2, -0.91
Neuron Tuple 3, 0.26
Neuron Tuple 2.5, -0.5
Neuron Weight: [-0.26, -0.27, 0.17, 0.87]
Neuron Biases: 0.5
Neuron Tuple 1, -0.26
Neuron Tuple 2, -0.27
Neuron Tuple 3, 0.17
Neuron Tuple 2.5, 0.87
The output for these 3 is [4.8, 1.21, 2.385]
'''

# This is the proposed code they use...

''' 
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
layer_outputs = []  # Output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0  # Output of given neuron
    for input, weight in zip(inputs, neuron_weights):
        neuron_output += input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
'''

# Now we are on data structures and tensors. The concept of Numpy and Tensorflow.

lolol = [[[1, 5], [6, 2]],
         [[3, 2], [1, 3]],
         [[5, 2], [1, 2]]]
# An Tensor is an array...
# Implementation of numpy in our code instead of pure python code...

'''
    A matrix is literally 2D Array, with columns and rows, 
'''

list_matrix_array = [[4, 2],
                     [5, 1],
                     [8, 2]]

# Numpy basic matrix multiplication
matrix_multi_array = np.array(weights) @ np.array(inputs)
print(matrix_multi_array)
print('------------------')
print(matrix_multi_array.T)
print('------------------')
print(np.array(weights))  # The arrays are flipped.
print('------------------')
print(np.array(weights).T)
print('------------------')
print(np.array(inputs).T)
# We took each list of weights * by its individual inputs --> £ calculations were done

'''
    weight1     weight2     weight3
       x           y           z      inputs
Calculation gave us the 3 answers as an numpy array.
'''

layer_outputs = np.array(weights) @ np.array(inputs).T + np.array(biases).T
print(f'Layer Outputs: {layer_outputs}')
# an array with 4 rows and 3 columns (i.e., 4x3).
array_a = [[1, 2],
           [2, 3]]
# practice_a = np.matmul(array_a, array_a)  <-- Same thing...
practice_a = np.array(array_a) @ np.array([1, 2]).T * np.array([2, 3]).T  # Understood
print(practice_a)  # I got it correct and understand the matrix multiplication process. Row Array 1 * Column Array 2


# The dot product -- Not really sure of this method, will revert back to np.matmul() function.
x = [1, 2]
y = [3, 4]
dot_product = (x[0]*y[0]) + (x[1]*y[1])
print(dot_product)
# Dot function is basically the same as np.matmul, but of course its not a matrix. Not 2d anymore...
print(np.dot(weights, inputs))
layer_outputs = np.dot(weights, np.array(inputs).T) + np.array(biases)
# layer_outputs = np.dot(weights, inputs) + np.array(biases)

print(layer_outputs)

# Usually models are trained using batches of inputs instead of one, so the model is tweaked rather than

inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

layer_outputs = np.dot(inputs, np.array(weights).T) + np.array(biases)
print(layer_outputs)


# Adding more layers, neural networks Input, Output// 2 hidden layers.

'''

inputs = [[1, 2, 3, 2.5], [2., 5., -1., 2], [-1.5, 2.7, 3.3, -0.8]] 
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5] 
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5] 


'''

lll = [[[1, 5, 6, 2], [3, 2, 1, 3]],
         [[5, 2, 1, 2], [6, 4, 8, 4]],
         [[2, 8, 5, 3], [1, 1, 9, 4]]]  # Look at the outer most bracket and you can determine the dimension of the array.


# Dot Product Rule
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0
# It will times each input and weight from each index and add the bias, in which is what we want...
output = np.dot(weights, inputs) + bias
print(output)

# -------------------->
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)
# To learn, the function of np.dot()



