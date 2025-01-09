""" why is the activation function used?
basically the nueral netwrok or the model of the the RELu
can mimic the function better , hence provide better results in such case.
the activation function decides whether a neuron should be activated as information
moves through the network.

import numpy as np
np.random.seed(0)
X = [[1,2,3,2.5],[2.0 , 5.0 , -1.0 , 2.0] ,[-1.5 , 2.7 , 3.3 , -0.8]]

class Layer_dense:
    def __init__(self , n_inputs , n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs , n_neurons)
        # the inputs are taken first because during the foward calculation we
        # dont reqiure the transpose
        self.biases = np.zeros((1,n_neurons))
    def foward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases


layer1 = Layer_dense(4,5)
layer2 = Layer_dense(5,2)

layer1.foward(X)
print(layer1.output)
layer2.foward(layer1.output)
print(layer2.output)
"""
import numpy as np
np.random.seed(0)
X = [[1,2,3,2.5],[2.0 , 5.0 , -1.0 , 2.0] ,[-1.5 , 2.7 , 3.3 , -0.8]]

inputs = [0 , 2, -1 , 3.3 , -2.7 , -1.1 , 2.2 , -100]
output = []

for i in inputs:
    if i > 0:
        output.append(i)
    elif i == 0:
        output.append(0)
print(output)

# or we could take

for i in inputs:
    output.append(max(0,i))

print(output)
