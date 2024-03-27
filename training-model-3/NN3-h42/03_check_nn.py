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
plt.rcParams['agg.path.chunksize'] = 10000
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
bumps           = [20, 26, 31, 38, 42]
header_size     = 3               # Lines in file header
nvar            = 17              # Total number of variables in file
input_loc       = range(6, 6+nvar_in)
output_loc      = [16]
if useRotation : nvar_in = nvar_in+2

C    = 305.0 /1000.0
Uref = 16.8
deltatheta = 0.0036
Retheta = 2500
nu = Uref * deltatheta / Retheta

#===============================================================
# LOAD DATA FOR TESTING
#===============================================================

#----- Define testing cases
cases = []
for bump in bumps:
    cases = numpy.concatenate([cases, ["./features/training-les-features-h"+str(bump)+".dat"]])

#----- Create data object
data = Data()
data.read(cases, header_size, nvar, connectivity = True)
data.collections(input_loc,output_loc,useRotation)

#===============================================================
# LOAD MODEL
#===============================================================
model = Model(nvar_in, nvar_out, n_hid, size_hid)
model.setup_device()
model.set_name()
model.directory = './results/verification/'
model.filename = 'verification.dat'
model.create_net(load = True)

#===============================================================
# PLOT RESULTS
#===============================================================
#----- Set plot parameters -----#
plt.rc('font', family='serif')
plt.rc('font', size=16)
plt.rc('legend',**{'fontsize':14})
plt.rc({'axes.labelsize': 14})

#----- Plot -----#
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(data.outputs, data.outputs, linewidth= 1.0, c= 'k', label = 'Reference data')
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(data.outputs, data.outputs, linewidth= 1.0, c= 'k', label = 'Reference data')

#===============================================================
# COMPUTE RESULTS
#===============================================================
nodes0 = 0
elements0 = 0

bump_fit = numpy.empty(len(bumps))
solution = numpy.empty((0,nvar+nvar_out))
    
for i, bump in enumerate(bumps):
    nodes = data.nnodes[i]
    elements = data.nelements[i]
    
    var = data.base[nodes0:nodes0+nodes,:6]
    
    if(useRotation) :
      inputs = data.inputs_norot[nodes0:nodes0+nodes,:]
      outputs = data.outputs_norot[nodes0:nodes0+nodes,:]
    else:
      inputs = data.inputs[nodes0:nodes0+nodes,:]
      outputs = data.outputs[nodes0:nodes0+nodes,:]
    connectivity = data.connectivity[elements0:elements0+elements,:]
    
    pred_outputs, fit = model.check_net(inputs, outputs)
    pred_outputs = numpy.resize(pred_outputs, (nodes,nvar_out))
    
    if bump == 20:
        ax.scatter(outputs, pred_outputs,s = 1.0 , c = 'C1', label = 'Training data h20')
    elif bump == 31:
        ax.scatter(outputs, pred_outputs,s = 1.0 , c = 'C0', label = 'Training data h31')
    elif bump == 38:
        ax.scatter(outputs, pred_outputs,s = 1.0 , c = 'C4', label = 'Training data h38')
    elif bump == 26:
        ax.scatter(outputs, pred_outputs,s = 1.0 , c = 'C3', label = 'Training data h26')
    elif bump == 42:
        ax2.scatter(outputs, pred_outputs,s = 1.0 , c = 'C2', label = 'Testing data h42')
        
    bump_fit[i] = fit
    
    solution = data.base[nodes0:nodes0+nodes,:]
    solution = numpy.append(solution, pred_outputs, axis = 1)

    tecfile = model.directory+'training-les-solution-h'+str(bumps[i])+'.dat'
    with open(tecfile, 'w') as t:
        header = 'TITLE = BUMP \n'+\
             'VARIABLES = "X" ,  "Y", "ub" , "vb" , "pb" , "nutb", '+\
             '"Q1" , "Q2", "Q3" ,"Q4" ,"Q5" , "Q6", "Q7" , "Q8" ,'+\
             '"dnuT", "nuT", "nuTe", "outputNN" \n'+\
             'ZONE   N='+str(data.nnodes[i])+',E='+str(data.nelements[i])+',F=FEPOINT,ET=TRIANGLE\n'
        t.write(header)
        
    with open(tecfile, 'a') as t:
        numpy.savetxt(t, solution, fmt = '%.9e')
        numpy.savetxt(t, connectivity, fmt = '%i')
    
    nodes0 += nodes
    elements0 += elements
    
model_directory = "results/"
ax.legend(borderpad = 0.65)
ax.set_ylabel(r'$N({Q}_{ref})$')
ax.set_xlabel(r'$N_{ref}$')
ax.tick_params(axis='both', labelsize=14)
ax.grid()
plt.show()
fig.savefig(model_directory+'verification/verification-scatter-1-nn3.png', bbox_inches='tight')

ax2.legend(borderpad = 0.65)
ax2.set_ylabel(r'$N({Q}_{ref})$')
ax2.set_xlabel(r'$N_{ref}$')
ax2.tick_params(axis='both', labelsize=14)
ax2.grid()
plt.show()
fig2.savefig(model_directory+'verification/verification-scatter-2-nn3.png', bbox_inches='tight')


if(useRotation) :
  _, all_fit = model.check_net(data.inputs_norot, data.outputs_norot)
else:
  _, all_fit = model.check_net(data.inputs, data.outputs)
bump_fit = numpy.append(bump_fit, all_fit)

#===============================================================
# WRITE RESULTS
#===============================================================
file     = open(model.directory+model.filename,"w")  
file.write("n_hidden size_hidden \n")
file.write(str(model.n_hid)+' '+str(model.size_hid)+"\n")
file.write("h20 h26 h31 h38 h42 all_fit \n")
numpy.savetxt(file, bump_fit, fmt = '%f')
file.close()

