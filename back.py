import numpy as np

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize variables
x = np.array([0.5, 0.8])  # Input values
y_true = 1  # True output

# Weights
weights_input_hidden = np.array([
    [0.1, 0.2],  # Weights for h1
    [0.3, 0.4],  # Weights for h2
    [0.5, 0.6]   # Weights for h3
])

weights_hidden_output = np.array([0.7, 0.8, 0.9])  # Weights from hidden to output

# Learning rate
learning_rate = 0.01

# Training for 2 iterations
for iteration in range(2):
    print(f"Iteration {iteration + 1}")

    # Forward propagation
    # Calculate hidden layer activations
    z_hidden = np.dot(weights_input_hidden, x)  
    h_hidden = sigmoid(z_hidden)  

 
    z_output = np.dot(weights_hidden_output, h_hidden)  # Weighted sum for output
    y_pred = sigmoid(z_output)  # Apply activation function
    print(f"  Predicted output: {y_pred}")

 
    delta_output = (y_pred - y_true) * sigmoid_derivative(z_output)

    # Gradients for weights_hidden_output
    grad_hidden_output = delta_output * h_hidden

    # Error at hidden layer
    delta_hidden = delta_output * weights_hidden_output * sigmoid_derivative(z_hidden)

    # Gradients for weights_input_hidden
    grad_input_hidden = np.outer(delta_hidden, x)

    # Update weights
    weights_hidden_output -= learning_rate * grad_hidden_output
    weights_input_hidden -= learning_rate * grad_input_hidden

    # Display updates
    print(f"  Updated weights (input to hidden):\n{weights_input_hidden}")
    print(f"  Updated weights (hidden to output): {weights_hidden_output}\n")
