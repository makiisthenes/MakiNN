# Lesson 3 Python Code.
import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]
# Numpy Vectors, Matrices, Dot Products, Shape.


# inputs = [1, 2, 3, 2.5]
# weights=[0.2,0.8,-0.5,1.0]
# bias = 2
# output = np.dot(inputs, np.array(weights).T) + biases


# 2 parameters, but positions need to be determined...
# 3,4
# 3,4
layer_outputs = np.dot(inputs, np.array(weights).T)
# biases = [[2,2,2], [3,3,3], [0.5,0.5,0.5]]
layer_outputs1 = np.dot(weights, np.array(inputs).T)
print(layer_outputs)
print(layer_outputs1)