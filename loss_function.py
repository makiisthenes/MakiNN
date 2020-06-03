import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]  # example output from the output layer of the neural network.
target_output = [1, 0, 0]  # ground truth

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

# Simplified Version of above is:
loss_simple = -(math.log(softmax_output[0]))  # 0th index, is the desired output solution.
# print(loss)
# print(loss_simple)
print(np.log(1) == math.log(1))
# Or you could use numpy versions, also anything timed by 0 is 0, and 1 is itself. Respectively.

# print(np.log(math.e))

# When training a NN we need to know what classifier the specific output was meant to fit.
# We can give a list of expected indexes we can use an iteration to help the NN. For example,

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
# Class targets tells us which index is suppose to be the strongest confidence and we base our loss function of this.
class_targets = [0, 1, 1]

print('|------ Target Index Probabilities ------|')
complex_way = []
for target_index, probability_distribution in zip(class_targets, softmax_outputs):
    print(probability_distribution[target_index])
    # print(f'Loss for this sample: {-np.log(probability_distrubution[target_index])}')  # This is a python 3.7 version.
    complex_way.append(-np.log(probability_distribution[target_index]))
print(f'Complex way result >>>{np.array(complex_way)}')


# It can be written in a more simple way, but I don't understand this way completely...
simpler_way = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])  # For the range 0 to 2 index, but actual batches will be dependant on the len of these batches and so we can use len() instead.
print(f'Simpler way result >>> {simpler_way}')


average_loss = np.mean(complex_way)
print(f'Average Loss: {average_loss}')

# Making a Class Method for this:
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Probabilities for target values
        y_pred = y_pred[range(samples), y_true]  # links correct index to an list.
        print(f' The indexes {y_pred}')

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        # Overall loss
        data_loss = np.mean(negative_log_likelihoods)
        return data_loss

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.forward(softmax_outputs, class_targets)
print(f'    Loss: {loss}')
# Loss being the actual value that is required to be changed in order to make it the same as actual answer.
# But accuracy is used instead, which is a better indicator for us.


softmax_outputs = np.array([[0.7, 0.2, 0.1],
                   [0.5, 0.1, 0.4],
                   [0.02, 0.9, 0.08]])
targets = [0, 1, 1]
predictions = np.argmax(softmax_outputs, axis=1)  # It will find the index of highest value for each row.
accuracy = np.mean(predictions == targets)  # Finds the mean amount of values that the same as the prediction, returning accuracy.
print(f'Accuracy: {accuracy}')
# Obviously a higher accuracy and lower loss is preferred model set...
'''
predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis
accuracy = np.mean(predictions==y)

'''
m = Loss_CategoricalCrossentropy()
m.forward(softmax_outputs, targets)