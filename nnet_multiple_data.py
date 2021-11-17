"""
------------------------------------------------------------------------------------------------------------------------
Using feedforward neural net to work with multiple data points.
Achieved approx. 87% accuracy after 1000 epochs (with batch size of 32, lr=5 and 15 hidden neurons).
I'm not sure why this is learning so slowly in comparison to Nielsen's code...
------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt

from nnet_single_datum import * 
from data_handling import *

"""
------------------------------------------------------------------------------------------------------------------------
Functions to execute SGD and test model accuracy.
------------------------------------------------------------------------------------------------------------------------
"""



"""
------------------------------------------------------------------------------------------------------------------------
Initialise and train model.
------------------------------------------------------------------------------------------------------------------------
"""

my_net = Network([784,30,10])
my_net.set_params()
my_net.lr = 3
epochs, accuracies = SGD(30, 30, my_net)

plt.plot(epochs, accuracies)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy/%')
plt.show()



"""
def nielsen_backprop(self, x, y):

		
		Return a tuple "(nabla_b, nabla_w)" representing the
		gradient for the cost function C_x.  "nabla_b" and
		"nabla_w" are layer-by-layer lists of numpy arrays, similar
		to "self.biases" and "self.weights".
		I think x is an input, and y is the label vector.
		
		
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] #list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
			sigmoid_deriv(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-1].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
		for l in range(2, self.depth):
			z = zs[-l]
			sp = sigmoid_deriv(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l].transpose())

		return (nabla_b, nabla_w)
		
	def cost_derivative(self, output_activations, y):
		Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
		return (output_activations-y)
"""