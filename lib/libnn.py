# import os, sys
import numpy
import operator
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from math import *

def rot_matrix2d(angle):
    mat   = numpy.zeros((2,2),dtype=numpy.float32)
    sin   = numpy.sin(angle)
    cos   = numpy.cos(angle)
    mat[0,0] = cos
    mat[0,1] = sin
    mat[1,0] = -sin
    mat[1,1] = cos
    return mat
    
def rotate_f(f,angle):
    Q_rot = rot_matrix2d(angle)
    for i in range(len(f)):
        f[i] = numpy.matmul( Q_rot , f[i] )
    return  f
    
#===============================================================
# DATASET CLASS
#===============================================================
class Data:
    
    def read(self, cases, header_size, nvar, connectivity = False):
        print('Loading data...\n')
        
        nnodes          = []            # Initialize nodes list
        nelements       = []            # Initialize elements list
        data            = []            # Initialize data array
        connect         = []
        
        for case in cases:
            print('File: '+ case)
            
            eof = False
            i = 0
            nodes = 1
            elements = 1
            
            with open(case) as f:
                while eof == False:
                    line = f.readline()
                    if (i == header_size - 1):
                        nodes = int(line[line.find('N=')+2:line.find('E=')-1])
                        elements = int(line[line.find('E=')+2:line.find(',',line.find('E='))])
                        nnodes.append(nodes)
                        nelements.append(elements)
                        
                    elif (i > header_size - 1 and i < header_size + nodes):
                        myarray = numpy.fromstring(line, dtype=float, sep=' ')
                        data = numpy.concatenate([data , myarray])
                    
                    elif (i >= header_size + nodes and i < header_size + nodes + elements) :
                        if connectivity:
                            myarray = numpy.fromstring(line, dtype=int, sep=' ')
                            connect = numpy.concatenate([connect,myarray])
                        else:
                            break
                    elif (i == header_size + nodes + elements):
                        break
                    
                    i+=1
                    
        # Reshape data so that we have blocks of lines corresponding to each case 
        self.size       = sum(nnodes)    # Total size of data with all the cases
        self.nnodes     = nnodes
        self.nelements  = nelements
        self.nvar       = nvar
        self.base       = numpy.reshape(data, (self.size,nvar))
        if connectivity:
            self.connectivity    = numpy.reshape(connect,(sum(nelements),3))
        
    def collections(self, input_loc,output_loc,useRotation=False):
        inputs          = self.base[:,input_loc]   # Features
        outputs         = self.base[:,output_loc]  # Flow variables outputs
        
        if(useRotation):
          print("Using data augmentation: rotating training data...")
          angles      = list(range(-180,180,10))

          # Temp arrays
          input_tmp   = inputs
          output_tmp  = outputs

          # Rotation variable: coord=[cos\alpha, sin\alpha]
          coord       = numpy.zeros_like(outputs) ; coord[:,0] = 1.

          # Data without rotation
          self.inputs_norot   = numpy.r_['-1',input_tmp,coord]
          self.outputs_norot  = numpy.r_['-1',output_tmp]
          self.inputs_norot   = self.inputs_norot.astype(numpy.float32)
          self.outputs_norot  = self.outputs_norot.astype(numpy.float32)#numpy.array(self.outputs_norot,dtype=numpy.float32)
          if ( len(output_loc) == 1 ) : self.outputs_norot = self.outputs_norot.reshape((self.size,1))
          
          self.size  = self.size * len(angles)
          for i,a in enumerate(angles):
              f_rot     = rotate_f(output_tmp,a)
              coord_rot = rotate_f(coord,a)
              if i == 0:
                  inputs  = numpy.r_['-1',input_tmp,coord_rot]
                  outputs = numpy.r_['-1',f_rot]
              else:
                  inputs  = numpy.append(inputs,numpy.r_['-1',input_tmp,coord_rot],axis=0)
                  outputs = numpy.append(outputs,numpy.r_['-1',f_rot],axis=0)
                  
          if self.size != len(inputs) :
              print("Input does not have dataset size: ",self.size,len(inputs))
              quit()
              
        self.inputs     = inputs.astype(numpy.float32)
        self.outputs    = numpy.array(outputs,dtype=numpy.float32)
        if ( len(output_loc) == 1 ) : self.outputs = self.outputs.reshape((self.size,1))
    
    def mask(self,limits):
        
        indices = []
        nodes0 = 0
        newnodes = []
        
        for k, nodes in enumerate(self.nnodes):
            count = 0
            for i in range(nodes0,nodes+nodes0):
                if self.outputs[i] > limits[k]:
                    indices.append(i)
                    count += 1
            newnodes.append(nodes-count)
            nodes0 += nodes
        
        self.inputs = numpy.delete(self.inputs,indices, axis = 0)
        self.outputs = numpy.delete(self.outputs,indices, axis = 0)
        
        self.nnodes = newnodes
        self.size = sum(newnodes)
        
    def split_data(self, model, training_split, shuffle = True, load = False):
    
        directory = model.directory
        modelname = model.name
        device = model.device
        
        if load:
            tr_indices = numpy.load(directory+'train_indices_'+modelname+'.npy')
            val_indices = numpy.load(directory+'val_indices_'+modelname+'.npy')
            
        else:
            indices = list(range(self.size))
            if shuffle :
                numpy.random.shuffle(indices)
            split = int(numpy.floor(training_split * self.size))
            tr_indices, val_indices = indices[:split], indices[split:]
        
            numpy.save(directory+'train_indices_'+modelname+'.npy',tr_indices)
            numpy.save(directory+'val_indices_'+modelname+'.npy',val_indices)
    
        # Extract training and validation inputs/outputs from collections
        self.split = int(numpy.floor(training_split * self.size))
        self.tr_inputs   = torch.from_numpy(self.inputs[tr_indices]).to(device)
        self.tr_outputs  = torch.from_numpy(self.outputs[tr_indices]).to(device)
        self.val_inputs  = torch.from_numpy(self.inputs[val_indices]).to(device)
        self.val_outputs = torch.from_numpy(self.outputs[val_indices]).to(device)

#===============================================================
# PYTORCH NEURAL NETWORK CLASS
#===============================================================
# REFERENCE: 
# https://stackoverflow.com/questions/54916135/what-is-the-class-definition-of-nn-linear-in-pytorch
# https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf pg14

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
        return y
    
#===============================================================
# WEIGHT AND BIAS INITIALIZATION
#===============================================================
# REFERENCE:
# AUX FCT: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

def init_lin_mod(mod):
    if type(mod) == torch.nn.Linear:
        # Weight initialization 
        torch.nn.init.xavier_uniform_(mod.weight) # he et al rather kaiming_normal_ xavier_uniform_
        
        # Bias initialization with b = 0.1
        mod.bias.data.fill_(0.1)
        
#===============================================================
# MODEL CLASS DEFINITION
#===============================================================

class Model:
               
    def __init__(self, nvar_in, nvar_out, n_hid, size_hid):
        self.nvar_in = nvar_in
        self.nvar_out = nvar_out
        self.n_hid = n_hid
        self.size_hid = size_hid
        
        self.lays     = [n_hid]
        self.sizes    = [size_hid]
            
        # Create file to save training results
        self.directory = './results/training/'
        self.filename = "training.dat"        

    def gridsearch(self, gs_n_hid, gs_size_hid):
        self.lays     = gs_n_hid     # List of number of hidden layers
        self.sizes    = gs_size_hid  # List of hidden layers sizes
        
        # Create file to save gridsearch results
        self.directory = './results/gridsearch/'
        self.filename = "gridsearch.dat"
            
    def setup_device(self, device = 'cuda'):
        if device == 'cuda':
            if  torch.cuda.is_available():
                print('GPU available...')
                print('...Using CUDA')
                self.device      = torch.device("cuda")
            else:
                print('Problem! No GPU available...')
                print('...Computing on CPU')
                self.device      = torch.device("cpu")
                
        elif device == 'cpu':
            print('...Computing on CPU')
            self.device      = torch.device("cpu")

    def set_name(self):
        self.name = ("model_in"+str(self.nvar_in) +"_out"+str(self.nvar_out)+"_"\
                              +str(self.n_hid)+"hid"+str(self.size_hid))
            
        self.logfile        = "logloss_"+self.name
        self.vallogfile     = "val_logloss_"+self.name
        self.fitfile        = 'fitlog_'+self.name


    def create_net(self, learn_rate = 0.2, load = False): 
        
        print('#---New model: '+str(self.n_hid)+' hidden layers of size '+str(self.size_hid))
        
        # Define NN model object
        self.net = NN(self.nvar_in, self.nvar_out, self.size_hid, self.n_hid).to(self.device)
        
        # Initialize weights and biases
        if load:
            self.net.load_state_dict(torch.load('./results/best/'+self.name+'_best.pth', map_location = self.device))
        else:
            self.net.apply(init_lin_mod)
        
        # Define loss function form
        self.criterion    = torch.nn.MSELoss(reduction='sum')
        
        # Initialize LBFGS optimizer for the model parameters
        self.optimizer    = torch.optim.LBFGS(self.net.parameters(), lr=learn_rate)
        
        # Print total number of weights and biases in model
        #total_params = sum(p.numel() for p in model.parameters())
        #print(total_params,'Total number of parameters in Model')
        #print(len(tr_inputs),'Size of training data set \n')
        
    def check_net(self, inputs, outputs):
        inputs      = torch.from_numpy(inputs).to(self.device)
        outputs     = torch.from_numpy(outputs).to(self.device)
        
        pred_outputs = self.net(inputs) # Predicted output from input data
        fit = (1 - torch.sum((pred_outputs-outputs)**2) /
           torch.sum((torch.mean(outputs)-outputs)**2)).item()
        
        pred_outputs = pred_outputs.cpu().detach().numpy()
        if ( len(outputs) == 1 ) : pred_outputs = numpy.reshape(pred_outputs,len(outputs))
        
        return pred_outputs, fit
    

#===============================================================
# LBFGS LOOPS FOR TRAINING AND VALIDATION
#===============================================================
def lbfgs_loop(model, database, num_epochs, istop, early_stop):
    
    net = model.net
    criterion = model.criterion
    optimizer = model.optimizer
    tr_inputs = database.tr_inputs
    tr_outputs = database.tr_outputs
    val_inputs = database.val_inputs
    val_outputs = database.val_outputs
    tr_size = database.split
    val_size = database.size-database.split
    directory = model.directory
    modelname = model.name
   
    global TLOSS, VLOSS, FIT
    
    TLOSS  = numpy.empty((0,2))
    VLOSS = numpy.empty((0,2))
    FIT = numpy.empty((0,2))
    
    print('Beginning Model Training...\n')
    
    # Loop for given number of epochs
    for i in range(num_epochs):
        
        #----- TRAINING STEP
        
        # Set model to training mode
        net.train()

        def closure():
            global tloss
            
            # Set gradients to zero
            optimizer.zero_grad()
                      
            # Obtain loss function from output currently predicted by the model 
            tr_outputs_pred = net(tr_inputs) # Predicted output from input data
            tr_loss = criterion(tr_outputs_pred, tr_outputs) # Calculate loss function
            tloss = tr_loss.item()
            # Backpropagation
            tr_loss.backward()
            
            return tr_loss

        optimizer.step(closure)
        
        # Save loss in global array
        TLOSS = numpy.append(TLOSS, numpy.array([[i,tloss/tr_size]]), axis = 0)
        
        #----- VALIDATION STEP
        
        # Set model to evaluation mode
        net.eval()

        with torch.no_grad(): # Disable gradient calculation
        
            # Obtain loss function from output currently predicted by the model 
            val_outputs_pred = net(val_inputs) # Predicted output from input data
            val_loss = criterion(val_outputs_pred, val_outputs) # Calculate loss function
            
            # Save loss in global array
            VLOSS = numpy.append(VLOSS,numpy.array([[i,val_loss.item()/val_size]]), axis = 0)
            
            # Print validation status
            fit = (1 - torch.sum((val_outputs_pred-val_outputs)**2) /
                   torch.sum((torch.mean(val_outputs)-val_outputs)**2)).item()
            
            FIT = numpy.append(FIT,numpy.array([[i,fit]]), axis = 0)
            
            print('Epoch ', i)
            print('Fit(%)=',fit * 100)
            
            print('Validation loss: ', VLOSS[i,1])
            print('Training loss: ', TLOSS[i,1], '\n')

            if (i > istop):
                for p in (VLOSS[i-3:i,1]-VLOSS[i-2:i+1,1]):
                    print(p)
                
            # Activate early stopping after [istop] for threshold [early_stop]
            if( (i > istop) and ( (VLOSS[i,1] - TLOSS[i,1] > 0.1*VLOSS[i,1]) or \
                              (all(p < early_stop for p in VLOSS[i-3:i,1]-VLOSS[i-2:i+1,1])) ) ):
                print('Early stop... \n')
                break

            # Stop loop if validation loss is blown up
            if numpy.isnan(VLOSS).any():
                print('Blown up...\n')
                break
            
    # Save model
    torch.save(net.state_dict(), directory+modelname+'.pth')
            
    print('Finished Model Training. \n')
    
    return FIT, TLOSS, VLOSS
    
