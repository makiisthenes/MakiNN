# Here we need to iterate through the NN until we have a very low loss value and high accuracy value, this can be done a number of ways...
# Here we first need to write all the code again from the beginning to make sure we understand:
'''
    --> Make Dense Layer of 2 Inputs and 3 neurons.
    --> Pass the ReLu Activation Function through the outputs of neuron.
    --> Make another Dense Layer of Input 3 and Output 3.
    --> Apply the SoftMax Activation Function on the outputs and determine confidence levels.
    --> Determine Loss Function from using Actual Answer Indexes
    --> Find out accuracy of model by calculating mean of correct argmax values in sample.
    --> Optimization includes finding the most suitable configuration of weights and biases.
    --> In which there are many ways of determining this, using gradients and partial differentiation.
    --> We will look into these topics eventually when we reach BackPropagation etc...
'''
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
def create_data(points, classes):
    X = np.zeros((points*classes, 2)) # list of given number of points per each class, containing pairs of values
    y = np.zeros(points*classes, dtype='uint8')  # same as above, but containing simple values - classes
    # print(X)
    # print(y)
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))  # index in class
        # print(ix)
        X[ix] = np.c_[np.random.randn(points) * .1 + class_number / 3, np.random.randn(points) * .1 + 0.5]
        # print(X[ix])
        y[ix] = class_number

    return X, y

X, y = create_data(100, 3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
class Layer_Dense:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred = y_pred[range(samples), y_true]
        negative_log_likelihoods = -np.log(y_pred)
        data_loss = np.mean(negative_log_likelihoods)
        return data_loss
X, y = create_data(100, 3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()
lowest_loss = 9999999
best_dense1_weights = dense1.weights
best_dense1_biases = dense1.biases
best_dense2_weights = dense2.weights
best_dense2_biases = dense2.biases

for iteration in range(100000):
    # This is where I randomly change the weights and biases in order to find a better value of biases...

    # Generate a new set of weights for iteration
    '''dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.randn(1, 3)
    '''
    dense1.weights += 0.01 * np.random.randn(2, 3)
    dense1.biases += 0.01 * np.random.randn(1, 3)
    dense2.weights += 0.01 * np.random.randn(3, 3)
    dense2.biases += 0.01 * np.random.randn(1, 3)

    # Make a forward pass of the training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate loss (from activation output, softmax activation here) and accuracy
    loss = loss_function.forward(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis
    accuracy = np.mean(predictions == y)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration, '|loss:', loss, '|acc:', accuracy)
        best_dense1_weights = dense1.weights
        best_dense1_biases = dense1.biases
        best_dense2_weights = dense2.weights
        best_dense2_biases = dense2.biases
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights
        dense1.biases = best_dense1_biases
        dense2.weights = best_dense2_weights
        dense2.biases = best_dense2_biases
print(best_dense1_biases)
print(best_dense1_weights)
print(best_dense2_weights)
print(best_dense2_biases)


'''
Best Set of biases and weights given random adjustments.
>>>
    [[-1.95133681 -0.58110745 -1.61583587]]
    
    [[-0.34589432  3.34096442 -1.11056796]
    [-0.27898945  1.09910392  1.89399122]]
    
    [[-3.76026482  5.60585809 -2.18420502]
    [ 4.33006958  1.7944321   0.80116426]
    [-4.0377977   1.38991568  6.0409604 ]]
    
    [[ 1.2981156  -0.16160226  4.78433761]]
(/iteration: 733/ /loss: 1.085384982677742/ /acc: 0.5366666666666666/)
'''