# If ever in doubt about the matrix, vector and determining shape of array look at image _classification.jpg
# Look at drawing for determining the calculation of dot function, matrix multiplication is exhausted...
import numpy as np
# Adding layers
inputs = [[1, 2, 3, 2.5], [2., 5., -1., 2], [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

first_output = np.dot(inputs, np.array(weights).T) + biases
second_output = np.dot(first_output, np.array(weights2).T) + biases2
print(second_output)
# Printed through second set of neurons, done successfully :) [UNDERSTOOD]
