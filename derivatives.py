import matplotlib.pyplot as plt
import numpy as np

# Basic function of f(x)=2x
def f(x):
    return 2*x
x = np.array(range(5))  # [0, 1, 2, 3, 4]
# print(x)
y = f(x)
# print(y)
plt.plot(x, y)  # 2 arrays incremented automatically.
plt.show()
# The impact of an weight or bias on an output is to do with gradients.
linear_gradient = (y[1]-y[0]) / (x[1]-x[0])
print(f'Linear Gradient: {linear_gradient}')

def g(x):
    return 2 * x ** 2
# x = np.array(range(5))
x = np.arange(0, 5, 0.001)
y = g(x)
nonlinear_gradient1 = (y[1]-y[0]) / (x[1]-x[0])
nonlinear_gradient2 = (y[3]-y[2]) / (x[3]-x[2])
print(f'Non-Linear Gradient1: {nonlinear_gradient1}')
print(f'Non-Linear Gradient2: {nonlinear_gradient2}')
plt.plot(x, y)

p2_delta = 0.0001
for x1 in range(5):
    x2 = x1 + p2_delta
    y1 = g(x1)
    y2 = g(x2)
    apx_derivative = (y2 - y1) / (x2 - x1)  # apx short for approximate
    print(f'Non-Linear Derivative: {apx_derivative}')
    b = y2-(apx_derivative*x2)

    def tang_line(x):
        return (apx_derivative*x) + b

    to_plot = [x1-0.9, x1, x1+0.9]
    # print([i for i in to_plot], [tang_line(i) for i in to_plot])  # Linked for loops.
    plt.plot([i for i in to_plot], [tang_line(i) for i in to_plot])
    print(f"Approximate derivative for g({x1}) is {apx_derivative}")
plt.show()

# The gradient will be used to determine the impact of a specific weight or bias on the overall loss function.
# Partial derivatives and chain rule.
