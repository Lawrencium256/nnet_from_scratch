"""
------------------------------------------------------------------------------------------------------------------------
Design code for feedforward neural network.
------------------------------------------------------------------------------------------------------------------------
"""
import numpy as np
import random
import sys

from data_handling import *

"""
------------------------------------------------------------------------------------------------------------------------
Activation functions.
------------------------------------------------------------------------------------------------------------------------
"""

def sigmoid(x):
	"""
	Sigmoid activation.
	"""
	y = 1/(1+np.exp(-x))
	return y

def sigmoid_deriv(x):
	"""
	Sigmoid derivative.
	"""
	return np.exp(-x)*((1+np.exp(-x))**(-2))

def tanh(x):
	"""
	Tanh activation.
	"""
	return np.tanh(x)

def tanh_deriv(x):
	"""
	Tanh derivative.
	"""
	return (np.cosh(x))**(-2)


def relu(x):
	"""
	ReLU activation.
	"""
	return max(0.0, x)

def relu_deriv(x):
	"""
	ReLU derivative.
	"""
	if x<=0:
		return 0
	else:
		return x

"""
------------------------------------------------------------------------------------------------------------------------
Loss functions.
------------------------------------------------------------------------------------------------------------------------
"""

def mse_loss(predicted, truth):
	"""
	Mean-squared error loss.
	"""
	difference = predicted - truth
	return 0.5*np.dot(difference, difference)		#Factor of 0.5 is convention.

def mse_loss_deriv(predicted, truth):
	"""
	Derivative of MSE loss w.r.t predicted.
	"""
	return predicted-truth

"""
------------------------------------------------------------------------------------------------------------------------
Neural network class.
------------------------------------------------------------------------------------------------------------------------
"""

class Network():
	"""
	Neural network class.
	"""
	def __init__(self, widths, lr=0.1):
		self.widths = widths 				#List object containing the width of each layer.
		self.depth = len(self.widths) 		
		self.lr = lr 						
		self.weights = []		
		self.biases = []	
		self.zs = []						#Weighted neuron input values.	
		self.activs = []					#Neuron activations.
		self.bias_grads = []				#Bias gradients.
		self.weight_grads = []				#Weight gradients.

	def set_params(self):
		"""
		Initialise network parameters and activations.
		Parameters chosen from random normalised Gaussian distribution.
		Returns: list objects containing arrays of weight and bias information.
		"""
		#Weight matrix for layer i has shape (mxn), where widths[i] = m, widths[i-1] = n.
		#Bias for layer i is row vector with shape (m,).																		
		self.weights = [np.random.randn(j,i) for i, j in zip(self.widths, self.widths[1:])]	
		self.biases = [np.random.randn(i) for i in self.widths[1:]]							
		
		#Initilise activations to take placeholder value 0.5.
		self.zs = [np.zeros(i) for i in self.widths]
		self.activs = [sigmoid(z) for z in self.zs]


	def count_params(self):
		"""
		Count network parameters.
		"""
		ws = sum([np.size(i) for i in self.weights])
		bs = sum([np.size(j) for j in self.biases])
		return(ws+bs)


	def forward_prop(self, input_vec):
		"""
		Perform a forwards run of the network for a single input.
		Stores the activations of each layer as a class attribute, for use during backprop.
		Input: input_vec (the input data).
		"""
		if self.activs:
			self.activs[0] = input_vec
			for i in range(self.depth-1):
				z = np.matmul(self.weights[i],self.activs[i])+self.biases[i]
				a = sigmoid(z)
				self.zs[i+1] = z
				self.activs[i+1] = a

		else:
			print('Network parameters have not been initialised prior to forward prop.')
			sys.exit()


	def backprop(self, truth):
		"""
		Obtain the gradient of the loss surface in parameter space using backprop.
		Stores this as class attributes, self.weight_grads and self.bias_grads.
		Does this for a single input.
		"""
		
		error_vecs = []

		#Error vector for the final layer.
		output = self.activs[-1]
		error = mse_loss_deriv(output,truth)*sigmoid_deriv(self.zs[-1])
		error = np.reshape(error, (np.size(error),1))							#Turn into column vector (necessary for iterative step).
		error_vecs.append(error)

		#Iteratively compute error vectors for previous layers.
		for i in reversed(range(self.depth-2)):
			error = np.matmul(np.transpose(self.weights[i+1]),error_vecs[-1])*np.reshape(sigmoid_deriv(self.zs[i+1]), (len(self.zs[i+1]),1))
			error_vecs.append(error)											#Column vectors.

		error_vecs.reverse()													#Reverse errors list, since errors are calculated from back-to-front.

		#Calculate the gradients of the loss w.r.t weights and biases.
		bias_grads = [i.reshape(np.size(i)) for i in error_vecs]				#Reshape into row vector.
		weight_grads = []
		for i in range(self.depth-1):
			weight_grads.append(np.outer(error_vecs[i], self.activs[i]))

		self.bias_grads = bias_grads
		self.weight_grads = weight_grads
		
		
	def update_params(self):
		"""
		Update network parameters.
		"""
		if hasattr(self, 'weight_grads'):
			for i in range(self.depth-1):
				self.weights[i] -= self.lr*self.weight_grads[i]
				self.biases[i]  -= self.lr*self.bias_grads[i]

			self.bias_grads.clear()
			self.weight_grads.clear()

		else:
			print("Network doesn't possess any gradient attributes for parameters update.")



	def train_accuracy(self, train_data, train_labels):
		"""
		Obtain classification accuracy of network on training data.
		"""
		correct = 0
		for i in range(0,len(train_labels)):
			train_input = train_data[i]
			truth = train_labels[i]
			self.forward_prop(train_input)
			output = self.activs[-1]
			pred = np.argmax(output)
			if pred == truth:
				correct += 1

		return (correct/len(train_labels))*100


	def test_accuracy(self, test_data, test_labels):
		"""
		Obtain classification accuracy of network on test data.
		"""
		correct = 0
		for i in range(0,len(test_labels)):
			test_input = test_data[i]
			truth = test_labels[i]
			self.forward_prop(test_input)
			output = self.activs[-1]
			pred = np.argmax(output)
			if pred == truth:
				correct += 1

		return (correct/len(test_labels))*100


	def SGD(self, 	train_data, train_labels, 
					test_data, test_labels, 
					batch_size=30, epoch=50):
		"""
		Performs stochastic gradient descent using training data.
		"""
		i=0
		epochs = []
		train_accuracies = []
		test_accuracies = []
		while i<epoch:
			running_weight_grads = []
			running_bias_grads = []
			randomlist = random.sample(range(0,len(train_labels)), batch_size)

			#train_acc = self.train_accuracy(train_data, train_labels)
			test_acc = self.test_accuracy(test_data, test_labels)
			print('Epoch ' + str(i) + ': ' + "{:.1f}".format(test_acc) + str('%'))

			epochs.append(i)
			#train_accuracies.append(train_acc)
			test_accuracies.append(test_acc)

			for j in randomlist:
				input = train_data[j]
				truth = np.zeros(10)
				truth[train_labels[j]] = 1
				self.forward_prop(input)
				self.backprop(truth)

				if not running_bias_grads:
					running_weight_grads = self.weight_grads
					running_bias_grads = self.bias_grads

				else:
					running_weight_grads = [a + b for a, b in zip(running_weight_grads, net.weight_grads)]
					running_bias_grads = [a + b for a, b in zip(running_bias_grads, net.bias_grads)]
					
			average_weight_grads = [w/batch_size for w in running_weight_grads]
			average_bias_grads = [b/batch_size for b in running_bias_grads]

			self.weight_grads = average_weight_grads
			self.bias_grads = average_bias_grads

			self.update_params()
			running_weight_grads.clear(), running_bias_grads.clear()

			i+=1

		return epochs, train_accuracies, test_accuracies
		

"""
------------------------------------------------------------------------------------------------------------------------
Network is operated as below.
------------------------------------------------------------------------------------------------------------------------
"""
net = Network([784,30,10])
net.set_params()
net.lr = 3
epochs, train_accuracies, test_accuracies = net.SGD(train_data, train_labels, test_data, test_labels, epoch=1000)

plt.plot(epochs, test_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy/%')
plt.ylim(0,100)
plt.show()




