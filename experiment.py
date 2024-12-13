import numpy as np

# Define the KL divergence function
def kl_divergence(theta, q):
    # theta and q should be numpy arrays with 4 elements
    return np.sum(theta * np.log(theta / q))

# Define the gradient of KL divergence with respect to theta
def kl_gradient(theta, q):
    # Gradient of KL divergence w.r.t each theta_i
    return np.log(theta) + 1 - np.log(q)

# Initialize the parameters (theta values)
theta = np.array([0.25, 0.25, 0.25, 0.25])  # Example initial values for theta
q = np.array([0.1, 0.2, 0.3, 0.4])  # Example values for q (these should sum to 1)

# Set the learning rate
learning_rate = 0.01              

# Number of iterations for gradient descent
iterations = 30

# Gradient descent loop
for _ in range(iterations):
    # Compute the gradient of the KL divergence
    gradient = kl_gradient(theta, q)
    
    # Update theta using the gradient and learning rate
    theta -= learning_rate * gradient
    
    # Optionally, ensure that the theta values stay valid probabilities (i.e., non-negative)
    theta = np.clip(theta, 1e-10, None)  # Avoid any zero or negative values
    
    # Optionally, normalize theta to ensure it sums to 1 (since it's a probability distribution)
    theta /= np.sum(theta)
    
    # Print the current values of theta and the KL divergence for monitoring
    print(f"Updated theta: {theta}")
    print(f"KL Divergence: {kl_divergence(theta, q)}")

# Final output
print("Final theta:", theta)
