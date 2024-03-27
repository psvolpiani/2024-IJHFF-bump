###! /usr/bin/env python
### -*- coding:utf-8 -*-

import os, sys

sys.path.append("../../lib/")

import time
import numpy
import operator
import pandas as pd
#import torch
#import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import *
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from libnn import *

#===============================================================
# NEURAL NETWORK SETUP DEFINITION
#===============================================================
print('Setting up network parameters...')

#----- Dataset parameters
bumps           = [20,26,31,38,42]      # Bumps
header_size     = 3                     # Lines in file header
nvar            = 17                    # Total number of variables in file
input_loc       = [6,7,8,9,10,11,12,13] # Input location in file
output_loc      = [16]                  # Output location in file
nvar_in         = len(input_loc)        # Number of input variables in file
nvar_out        = len(output_loc)       # Number of output variables in file
useRotation     = False                 # use data augmentation?
if useRotation : nvar_in = nvar_in+2
model_directory = "results/"

#===============================================================
# LOAD DATA FOR TRAINING
#===============================================================

#----- Define training cases
cases = []
for bump in bumps:
    cases = numpy.concatenate([cases, ["./features/training-les-features-h"+str(bump)+".dat"]])

#----- Create data object
data = Data()
data.read(cases, header_size, nvar, connectivity = True)
data.collections(input_loc,output_loc,useRotation)

#===============================================================
# LOAD THE MODEL USING JOBLIB
#===============================================================
import joblib
filename = model_directory+'best/rf_model.joblib'

model = joblib.load(filename)
print("Model loaded")

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

    pred_outputs = model.predict(inputs)
    pred_outputs = numpy.resize(pred_outputs, (nodes,nvar_out))

    if bump == 20:
        ax.scatter(outputs, pred_outputs,s = 1.0 , c = 'C1', label = 'Training data h20')
    elif bump == 31:
        ax.scatter(outputs, pred_outputs,s = 1.0 , c = 'C0', label = 'Training data h31')
    elif bump == 38:
        ax.scatter(outputs, pred_outputs,s = 1.0 , c = 'C4', label = 'Training data h38')
    elif bump == 26:
        ax2.scatter(outputs, pred_outputs,s = 1.0 , c = 'C3', label = 'Training data h26')
    elif bump == 42:
        ax2.scatter(outputs, pred_outputs,s = 1.0 , c = 'C2', label = 'Testing data h42')

    print("bump =",bump)
    print("MSE =",mean_squared_error(outputs,pred_outputs))
    
    solution = data.base[nodes0:nodes0+nodes,:]
    solution = numpy.append(solution, pred_outputs, axis = 1)

    tecfile = model_directory+'verification/training-les-solution-h'+str(bumps[i])+'.dat'
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

ax.legend(borderpad = 0.65)
ax.set_ylabel(r'$N({Q}_{ref})$')
ax.set_xlabel(r'$N_{ref}$')
ax.tick_params(axis='both', labelsize=14)
ax.grid()
plt.show()
fig.savefig(model_directory+'verification/verification-scatter-1-rf3.png', bbox_inches='tight')

ax2.legend(borderpad = 0.65)
ax2.set_ylabel(r'$N({Q}_{ref})$')
ax2.set_xlabel(r'$N_{ref}$')
ax2.tick_params(axis='both', labelsize=14)
ax2.grid()
plt.show()
fig2.savefig(model_directory+'verification/verification-scatter-2-rf3.png', bbox_inches='tight')
