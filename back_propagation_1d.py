import numpy as np 


def Feedforward(x,weight):
    return weight * x 

def cost_function(y_expected,output):
    return np.square(output-y_expected)

def cost_function_derivative(weight):
    return 4.5* weight - 1.5





y_expected = 0.5
x = 1.5
weight = 0.8
learninng_rate= 0.1
output = Feedforward(x, weight)

for i in range(100):
    w1 = weight - learninng_rate * cost_function_derivative(weight)
    print(w1)
    weight = w1



    

print(weight)


