import numpy as np 

#X hours of sleeping y test score of the student

X = np.array(([2,9], [1,5], [3,6]),dtype=float)
print(np.size(X[0]))
y = np.array(([92],[86],[89]),dtype=float)
#scale unit  np.max goes column wise
X=X/np.amax(X, axis=0)
y = y/100

def feedfForward(x,W1):
    z1 = W1.dot(x)
    sigmoid(z1)

def sigmoid( s,  deriv= False):
    if(deriv== True ):
        return s * (1-s)
    return 1/(1+np.exp(-s))

def Cost_function(y,y_expected):
    return 0.5 * np.square(y-y_expected)

def delta_output(y,y_expected):
    output_error= y_expected-y
    return sigmoid(y,deriv=True) * output_error

def gradient_output_layer():
    delta_output(y_expected)*feedfForward()



