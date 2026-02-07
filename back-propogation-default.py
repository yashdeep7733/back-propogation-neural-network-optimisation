import numpy as np

# Example inputs, weights, and bias for a multiple neuron layer
weights = np.array([[0.1 , 2.0 , 0.3 , 0.4],
                   [0.5 , 0.6 , 0.7 , 0.8],
                   [0.9 , 1.0 , 1.1 , 1.2]])

bias = np.array([0.1, 0.2, 0.3])

inputs = np.array([1, 2, 3, 4])

learning_rate = 0.001

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0) # The derivative of ReLU is 1 for x > 0 and 0 for x <= 0

for iterations in range(200):
    # forward pass
    z = np.dot(weights, inputs) + bias
    a = relu(z)
    y = np.sum(a)

    # Calculate the loss
    loss = y ** 2

    # Backpropagation
    # Gradient of the loss with respect to output y
    dL_dy = 2 * y

    # Gradient of y with respect to a
    dy_da = np.ones_like(a) # Since y is the sum of a, the gradient is 1 for each element of a

    # Gradient of the loss with respect to a
    dL_da = dL_dy * dy_da

    # Gradient of a with respect to z
    da_dz = relu_derivative(z)

    # Gradient of loss with respect to z
    dL_dz = dL_da * da_dz

    # Gradient of z with respect to weights and bias
    dL_dW = np.outer(dL_dz , inputs)
    dL_db = dL_dz 

    # Update weights and bias
    weights -= learning_rate * dL_dW
    bias -= learning_rate * dL_db

    print(f"Iteration {iterations + 1}, Loss: {loss}")

print("Final Weights:", weights)
print("Final Bias:", bias)