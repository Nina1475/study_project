import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# merge all functions 

def update_vector(theta,q,learning_rate):
    epsilon = 1e-10
     # Update theta[0] using the gradient of KL divergence
    theta = theta + learning_rate * (1 + np.log(np.clip(theta[0], epsilon, None)) - np.log(q[0]))* np.array([1, 0, 0, 0])
    theta = theta / theta.sum()  # Normalize the theta vector
    
    # Update theta[1] using the gradient of KL divergence
    theta = theta + learning_rate * (1 + np.log(np.clip(theta[1], epsilon, None)) - np.log(q[1]))* np.array([0, 1, 0, 0])
    theta = theta / theta.sum()  # Normalize the theta vector
    
    # Update theta[2] using the gradient of KL divergence
    theta = theta + learning_rate * (1 + np.log(np.clip(theta[2], epsilon, None)) - np.log(q[2]))*np.array([0, 0, 1, 0]) 
    theta = theta / theta.sum()  # Normalize the theta vector
    
    # Update theta[3] using the gradient of KL divergence
    theta = theta + learning_rate * (1 + np.log(np.clip(theta, epsilon, None)) - np.log(q[3]))*np.array([0, 0, 0, 1])
    theta = theta / theta.sum()  # Normalize the theta vector
    """"
    theta = theta + learning_rate * (1+np.log(theta[0])-np.log(q[0])) * np.array([1, 0, 0, 0])
    theta = theta / theta.sum() 
    theta = theta + learning_rate * (1+np.log(theta[1])-np.log(q[1])) * np.array([0, 1, 0, 0])
    theta = theta / theta.sum() 
    theta = theta + learning_rate * (1+np.log(theta[2])-np.log(q[2])) * np.array([0, 0, 1, 0])
    theta = theta / theta.sum() 
    theta = theta + learning_rate * (1+np.log(theta[3])-np.log(q[3])) * np.array([0, 0, 0, 1])
    theta = theta / theta.sum() 
   
    #theta = np.maximum(theta, epsilon)  # Ensure all theta elements are positive
    #q = np.maximum(q, epsilon)  # Ensure all q elements are positive
    epsilon = 1e-10
    theta = np.maximum(theta, epsilon)  # Ensure all theta elements are positive
    q = np.maximum(q, epsilon)
    for i in range(len(theta)):
        update_direction = np.zeros_like(theta)
        update_direction[i] = 1
        theta = theta + learning_rate * (1 + np.log(theta[i]) - np.log(q[i])) * update_direction
        theta = np.maximum(theta, epsilon)  # Avoid negative or zero values
        theta /= theta.sum()  # Normalize to maintain a valid probability distribution
"""
    return theta
  






def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler Divergence D_KL(P || Q)
    :param p: Probability distribution P (true distribution)
    :param q: Probability distribution Q (approximation)
    :return: KL divergence
    """
    p = np.array(p)
    q = np.array(q)
    
    # Prevent log(0) and division by 0
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)  # Avoid log(0)
    q = np.clip(q, epsilon, 1)  # Avoid division by 0
    
    # Compute KL divergence
    return np.sum(p * np.log(p / q))




def generate_samples(n, p_star):
    # Define the possible outcomes as rows of X
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=int)
    
    # Define the probabilities of each outcome (example: p_star = [0.1, 0.3, 0.2, 0.4])
    outcomes = [0, 1, 2, 3]
    
    # Generate random sample indices from p_star
    samples = np.random.choice(outcomes, size=n, p=p_star)
    samples= np.array(samples)
    print(samples.sort())
    # Map indices to corresponding rows in X
    mapped_samples = X[samples]
    print(samples)
    # Print for demonstration delete we need the true distribution for the plotting 
    list=[]
    for i in range(4):
        count = 0
        for j in (samples):
            if j==i:
                count+=1
        list.append(count/n)

    print(list)
    list = np.array(list)
    
    return list, mapped_samples






# rename it in a better convention 
theta = [0.25,0.25,0.25,0.25]
p_true = [0.1, 230.3, 0.2, 0.4]  # this is the true probability 
n = 100  # Number of samples to generate
# generate the samples and the probability distribution
#p_star, mapped_samples = generate_samples(n,p_staro)
print(list)
learning_rate = 0.01
kl_values = []
# think about not going to zero 
for i in range(30):
    theta = update_vector(np.array(theta),p_true,learning_rate)
    kl_values.append(kl_divergence(theta,p_true))
     
   
print(theta)

print (kl_divergence(theta,p_true))
 
# Plot the KL divergence
plt.figure(figsize=(10, 5))
plt.plot(range(30), kl_values, label="KL Divergence ")
plt.xlabel("Iterations")
plt.ylabel("KL Divergence")
plt.title("KL Divergence Over Iterations")
plt.legend()
plt.grid(True)
plt.show()
#writing in latex 
# github repository 