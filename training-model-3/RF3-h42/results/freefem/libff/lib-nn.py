###! /usr/bin/env python
### -*- coding:utf-8 -*-

import os
import torch
import torch.nn.functional as F
import numpy

####################### SETUP DEFINITION

size_hidden     = 80    # number of neurons per hidden layer (1st if flexible)
n_hidden        = 4     # number of hidden layers -> n-1 hidden-to-hidden-mappings
nvar_in         = 8     # number of inputs
nvar_out        = 1     # number of outputs

model_directory = "./../best/"

######################## MLP DEFINITION

class NN(torch.nn.Module):
    
    #----- Initialize layer modules
    def __init__(self, D_in, D_out, layer_size, num_hidden):
        super(NN, self).__init__()

        self.input_linear = torch.nn.Linear(D_in, layer_size) # First layer
        self.hidden_linear = torch.nn.ModuleList([torch.nn.Linear(layer_size, layer_size)
                                                  for i in range(num_hidden)]) # Hidden layers
        self.output_linear = torch.nn.Linear(layer_size, D_out) # Output layer
        self.activ = torch.nn.ReLU()
        
    #----- Forward feeding function
    def forward(self, x):
        
        # First layer
        z = self.input_linear(x)
        y = self.activ(z)
        
        # Hidden layers
        for i,layer in enumerate(self.hidden_linear):
            z = layer(y)
            y = self.activ(z)
            
        # Output layer
        y = self.output_linear(y)
        # y = self.activ(z)
        
        return y

if  torch.cuda.is_available():
    print('...Using CUDA')
    device      = torch.device("cuda")
else:
    print('...Computing on CPU')
    device      = torch.device("cpu")
    
######### FILE NAMES
modelname = ("model_in"+str(nvar_in)+"_out"+str(nvar_out)+"_"+str(n_hidden)+"hid"+str(size_hidden))

########## MODEL
model = NN(nvar_in, nvar_out, size_hidden, n_hidden)
model.load_state_dict(torch.load(model_directory+modelname+'_best.pth',map_location=lambda storage, loc: storage))
print("Model loaded")
model.eval()

#===============================================================
# COMPUTE OUTPUT
#===============================================================
input_collection = numpy.loadtxt("./features.txt")
input_collection = input_collection.astype(numpy.float32)
inputs = torch.from_numpy(input_collection).to(device); print(inputs[0])
pred_outputs = model(inputs).detach().numpy(); print(pred_outputs[0])

numpy.savetxt("field_param.txt",pred_outputs)
