#===============================================================
# NEURAL NETWORK
#===============================================================

import os
import torch
import torch.nn.functional as F
import numpy as np

####################### LOADING DATA

####################### SETUP DEFINITION

size_hidden     = 80    # number of neurons per hidden layer (1st if flexible)
n_hidden        = 4     # number of hidden layers -> n-1 hidden-to-hidden-mappings
nvar_in         = 8    # number of inputs
nvar_out        = 1     # number of outputs

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

######################## WRITE MODEL IN FREEFEM FORMAT

######### FILE NAMES
modelname      = ("model_in"+str(nvar_in)    #or train_pyfeatures/
                  +"_out"+str(nvar_out)+"_"+str(n_hidden)+"hid"+str(size_hidden))
filename       = ("./results/best/parameters_in"+str(nvar_in)
                  +"_out"+str(nvar_out)+"_"+str(n_hidden)+"hid"+str(size_hidden))

########## MODEL

model          = NN(nvar_in, nvar_out, size_hidden, n_hidden)
model.load_state_dict(torch.load('./results/best/'+modelname+'_best.pth',map_location=lambda storage, loc: storage))

for p in model.parameters():
    p.requires_grad = False

########## WRITE TO FILE

fmtstr = '%.18e'
with open(filename+"_FreeFem.dat","w") as fparam:
    # structure
    fparam.write(str(nvar_in)+'\n'+str(size_hidden)+'\n'+str(nvar_out)+'\n'+str(n_hidden)+'\n')
    
    # first layer weights
    mat = model.input_linear.weight.numpy()
    mat = np.asarray(mat).reshape(-1)
    np.savetxt(fparam,mat,fmt=fmtstr)

    # hidden layer weights
    for l in range(n_hidden):
        mat = model.hidden_linear[l].weight.numpy()
        mat = np.asarray(mat).reshape(-1)
        np.savetxt(fparam,mat,fmt=fmtstr)

    # output layer weights
    mat = model.output_linear.weight.numpy()
    mat = np.asarray(mat).reshape(-1)
    np.savetxt(fparam,mat,fmt=fmtstr)
    
    # biases 1st and hidden, then output bias
    mat = model.input_linear.bias.numpy()
    mat = np.asarray(mat).reshape(-1)
    np.savetxt(fparam,mat,fmt=fmtstr)

    for l in range(n_hidden):
        mat = model.hidden_linear[l].bias.numpy()
        mat = np.asarray(mat).reshape(-1)
        np.savetxt(fparam,mat,fmt=fmtstr)

    mat = model.output_linear.bias.numpy()
    mat = np.asarray(mat).reshape(-1)
    np.savetxt(fparam,mat,fmt=fmtstr)
