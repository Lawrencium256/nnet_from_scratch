"""
------------------------------------------------------------------------------------------------------------------------
Code to confirm that my neural net for a single datum is functioning correctly, through verification with PyTorch.
When calling net.update_params() before PyTorch pred(model) function, they disagree. Am yet to explain this behaviour.
------------------------------------------------------------------------------------------------------------------------
"""
import numpy as np
import sys
import torch
from torch import nn

from nnet_single_datum import * 

"""
------------------------------------------------------------------------------------------------------------------------
PyTorch set up.
------------------------------------------------------------------------------------------------------------------------
"""
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, final_dim):
        super(NeuralNetwork, self).__init__()
       	self.input_dim = input_dim
       	self.hidden_dim = hidden_dim
       	self.final_dim = final_dim
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.final_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

"""
------------------------------------------------------------------------------------------------------------------------
Model comparison.
------------------------------------------------------------------------------------------------------------------------
"""
#Initialise my hand-made model.
input_dim = 10
hidden_dim = 5
final_dim = 2

net = Network([input_dim,hidden_dim,final_dim])
net.set_params()
my_inputs = np.random.rand(input_dim)
my_truth = np.random.rand(final_dim)

#Initialise PyTorch model.
model = NeuralNetwork(input_dim, hidden_dim, final_dim).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

inputs = torch.from_numpy(my_inputs).float()
inputs = torch.reshape(inputs, (1,input_dim))
y = torch.from_numpy(my_truth).float()
y = torch.reshape(y, (1,final_dim))

#Set hand-made model params to be the same as PyTorch model.
params = []
for i in model.parameters():
	params.append(i)

net.weights[0] = params[0].detach().numpy()
net.weights[1] = params[2].detach().numpy()
net.biases[0] = params[1].detach().numpy()
net.biases[1] = params[3].detach().numpy()

#Backprop hand-made model.
net.forward_prop(my_inputs)			#Make sure I call this after I have defined the network parameters.
net.backprop(my_truth)
net.update_params()

#Obtain PyTorch model prediction and do backprop.
pred = model(inputs)
loss = loss_fn(pred, y)
optimizer.zero_grad()
loss.backward()


#Compare output and gradients.
print('PyTorch model output:')
print(pred)
print('\n')
print('Hand-made model output:')
print(net.activs[-1])
print('\n')

print('PyTorch model gradients:')
for param in model.parameters():
	print(param.grad)
print('\n')

print('Hand-made model gradients:')
print(net.bias_grads[1])
print('\n')