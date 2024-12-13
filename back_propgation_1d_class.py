import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.x = 1.5
        self.y_expected = 0.5
        self.weight = 0.8
        self.learninng_rate= 0.1

    def Feedforwardx(weight,x):
        return weight * x 

    def cost_function_derivative(weight):
        return 4.5* weight - 1.5 
    
    def backPropagation(self,weight):
        w1 = weight - self.learninng_rate * self.cost_function_derivative(weight)
        weight = w1
        return weight

    def train(self):
       output = self.Feedforward(self.x)
       self.backPropagation(self.weight)
    



NN =NeuralNetwork()
 
for i in range(1000):
    NN.train()


print("perdicted output",NN.feedForward())    