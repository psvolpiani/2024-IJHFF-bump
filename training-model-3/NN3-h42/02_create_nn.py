###! /usr/bin/env python
### -*- coding:utf-8 -*-

import os, sys
sys.path.append("../../lib/")
from libnn import *

import numpy
import operator
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from math import *

#===============================================================
# NEURAL NETWORK SETUP DEFINITION
#===============================================================
print('Setting up network parameters...')

#----- Model parameters
nvar_in         = 8               # number of input variables in file
nvar_out        = 1               # number of output variables in file
size_hid        = 80              # number of neurons per hidden layer (1st if flexible)
n_hid           = 4               # number of hidden layers -> n-1 hidden-to-hidden-mappings
useRotation     = False           # use data augmentation?

#----- Dataset parameters
bumps           = [20,26,31,38]
header_size     = 3               # Lines in file header
nvar            = 17              # Total number of variables in file
input_loc       = range(6, 6+nvar_in)
output_loc      = [16]
if useRotation : nvar_in = nvar_in+2

#----- Training parameters
training_split  = .75             # fraction of training vs validation samples
shuffle_dataset = True            # randomize?
load_model      = False
load_indices    = False
num_epochs      = 250             # epochs for full dataset training
early_stop      = 1e-2            # early stopping threshold on error
istop           = 30              # epochs to activate early stopping
lr              = 0.1             # optimizer learning rate
random_seed     = 747             # user def. seed -> no ensemble of computations yet

#----- Grid search parameters
gridsearch      = False           # use grid search to find hyperparameters
blown_up        = True            # grid search repetition if a run blows up
gs_bestof       = 6               # redo run this many times and take best of them
gs_n_hid        = list(range(2,5))# list of number of hidden layers to search
gs_size_hid     = [100,110,120]   # list of hidden layers sizes to search

plotfrom        = 5
plotfig         = True
#mask_lims       = [1.25, 1.5, 3.0]

#===============================================================
# LOAD DATA FOR TRAINING
#===============================================================

#----- Define training cases
cases = []
for bump in bumps:
    cases = numpy.concatenate([cases, ["./features/training-les-features-h"+str(bump)+".dat"]])

#----- Create data object
data = Data()
data.read(cases, header_size, nvar)
data.collections(input_loc,output_loc,useRotation)
#data.mask(mask_lims)

try: os.mkdir('./results')
except OSError as error: print(error)
try: os.mkdir('./results/gridsearch')
except OSError as error: print(error)
try: os.mkdir('./results/training')
except OSError as error: print(error)
try: os.mkdir('./results/best')
except OSError as error: print(error)
try: os.mkdir('./results/verification')
except OSError as error: print(error)
try: os.mkdir('./results/freefem')
except OSError as error: print(error)

#===============================================================
# GRID SEARCH/TRAINING
#===============================================================

model = Model(nvar_in, nvar_out, n_hid, size_hid)
if gridsearch:  model.gridsearch(gs_n_hid, gs_size_hid)
model.setup_device()

file     = open(model.directory+model.filename,"w")  
file.write("n_hidden size_hidden best_fit \n")

bestfit  = numpy.zeros((len(model.lays), len(model.sizes)),dtype=numpy.float32)  

# Loop for grid search (layers, sizes)
for II, n_hid in enumerate(model.lays):
    
    model.n_hid = n_hid
    
    for JJ, size_hid in enumerate(model.sizes):
        
        model.size_hid = size_hid
        model.set_name()
        
        gs = 0
        while blown_up or gs<gs_bestof:
            
            if not blown_up: gs += 1
            
            blown_up = False  
                   
            #----- DEFINE TRAINING AND VALIDATION DATASETS
            data.split_data(model, training_split, shuffle = shuffle_dataset, load = load_indices)
            
            #----- CREATE NN MODEL
            model.create_net(learn_rate = lr, load = load_model)

            #----- TRAINING
            FIT, TLOSS, VLOSS = lbfgs_loop(model, data, num_epochs, istop, early_stop)
            
            #----- SAVE LOG FILES
            numpy.savetxt(model.directory+model.logfile+"_iter"+str(gs)+".dat", TLOSS, fmt='%i %f', header = 'epoch tr_loss')
            numpy.savetxt(model.directory+model.vallogfile+"_iter"+str(gs)+".dat", VLOSS, fmt='%i %f', header = 'epoch val_loss')
            numpy.savetxt(model.directory+model.fitfile+"_iter"+str(gs)+".dat", FIT, fmt='%i %f', header = 'epoch fit')
            
            #----- PLOT LOSSES
            plt.plot(VLOSS[plotfrom:,0], VLOSS[plotfrom:,1], 'k', label = 'Validation loss')
            plt.plot(TLOSS[plotfrom:,0], TLOSS[plotfrom:,1], 'r', label = 'Training loss')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.grid()
            plt.legend()
            plt.savefig(model.directory+'plot_loss_'+model.name+'_iter'+str(gs)+'.png', bbox_inches='tight')
            if plotfig : plt.show()
            
            #----- PLOT FIT
            plt.plot(FIT[:,0], FIT[:,1], 'k')
            plt.grid()
            plt.ylabel('Fit')
            plt.xlabel('Epochs')
            plt.ylim(bottom = 0., top = 1.)
            plt.savefig(model.directory+'plot_fit_'+model.name+'_iter'+str(gs)+'.png', bbox_inches='tight')
            if plotfig : plt.show()
            
            #----- CHOOSE BEST FIT
            if (not blown_up) and (FIT[-1,1] > bestfit[II,JJ]):
                print("Better model. Saving...")
                torch.save(model.net.state_dict(), './results/best/'+model.name+'_best.pth')
                bestfit[II,JJ] = FIT[-1,1]
            
            #----- WRITE GRIDSEARCH RESULTS
            file.write(str(n_hid)+'  '+str(size_hid)+'  '+str(FIT[-1,1])+"\n")
            file.flush()

            print('Testing on entire test case without rotation: ')
            if(useRotation) :
              _, all_fit = model.check_net(data.inputs_norot, data.outputs_norot)
            else:
              _, all_fit = model.check_net(data.inputs, data.outputs)
            print("FIT=",all_fit)

file.close()

# Print best fit
all_best = numpy.max(bestfit) # Find best result
at_grid  = numpy.argwhere(bestfit==all_best)[0] # Find best result location
print('Best performance: '
      +str(all_best)+' with: '
      +str(model.lays[at_grid[0]])+' layers, '
      +str(model.sizes[at_grid[1]])+' n/layer')
