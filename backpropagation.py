import numpy as np 

#X hours of sleeping y test score of the student

X = np.array(([2,9], [1,5], [3,6]),dtype=float)
print(np.size(X[0]))
y = np.array(([92],[86],[89]),dtype=float)
#scale unit  np.max goes column wise
X=X/np.amax(X, axis=0)
y = y/100



class NeuralNetwork(object):
    def __init__(self):
        #parameters 
        self.inputSize = 2 
        self.outputSize = 1
        self.hiddenSize = 3
      # 3x2 
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)




    def feedForward( self, X):

        #forward propagation 
        self.z = np.dot (X,self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        output = self.sigmoid(self.z3)
        return output


    def sigmoid(self, s,  deriv= False):
        if(deriv== True ):
            return s * (1-s)
        return 1/(1+np.exp(-s))
    
    def backward(self,X,y,output):
        #backward propagate throught the network first layer 
        self.output_error = y - output
        print("this is output error of the secound layer: \n", self.output_error )
        self.output_delta = self.output_error * self.sigmoid(output,deriv= True)
        print("this is the derivative of the activation function evaluated at the output\n",self.sigmoid(output,deriv= True))
        print ("this is the delta outpu:\n",self.output_delta)

        #backward propagate through the secound layer 
        self.z2_error = self.output_delta.dot(self.W2.T) #z2 how much our hidden layer weights contribute to output layer 
        print("this is the hidden layer:\n",self.z2_error)
        self.z2_delta  = self.z2_error * self.sigmoid(self.z2,deriv=True)
        
        
        
        #applying the derivative of sigmoid to z2 error
        print("derivative of the first layer:\n",self.z2_delta)
         
        self.W1 += X.T.dot(self.z2_delta) #adjusting first set (input->hidden) weights
        self.W2 += self.z2.T.dot(self.output_delta) #adjustinng hidden->output layer
    def train (self,X,y):
           output = self.feedForward(X)
           self.backward(X,y,output)


NN = NeuralNetwork()
for i in range(2):
    NN.train(X,y)

print("Input", X)
print("actual Output",y)
print("perdicted output",NN.feedForward(X))