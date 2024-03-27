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
from libnn import *

plt.rc('font', family='serif')
plt.rc('lines', linewidth=1.)
plt.rc('font', size=12)
plt.rc('legend',**{'fontsize':12})
plt.rc({'axes.labelsize': 13})

#===============================================================
# NEURAL NETWORK SETUP DEFINITION
#===============================================================
print('Setting up network parameters...')

#----- Dataset parameters
bumps           = [20,26,31,38]         # Bumps
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
data.read(cases, header_size, nvar)
data.collections(input_loc,output_loc,useRotation)

#===============================================================
# MODEL FITTING USING RANDOM FOREST
#===============================================================

x = data.inputs
y = data.outputs
if ( len(output_loc) == 1 ) : y = y.reshape((y.size,1))
feature_names = [f"feature {i}" for i in range(x.shape[1])]
output_names = [f"output {i}" for i in range(y.shape[1])]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Instantiate the Model
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

# Print Metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#===============================================================
# FEATURE IMPORTANCE BASED ON MEAN DECREASE IN IMPURITY
#===============================================================
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py

start_time = time.time()
importances = rf.feature_importances_
std = numpy.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

rf_importances_1 = pd.Series(importances, index=feature_names)
std_1 = std

fig, ax = plt.subplots(figsize=(6.5,4))
rf_importances_1.plot.bar(yerr=std, ax=ax)
bars = ('q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8')
y_pos = numpy.arange(len(bars))
plt.xticks(y_pos, bars, rotation=0)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
fig.savefig(model_directory+'best/verification-scatter-mdi-rf1.png', bbox_inches='tight')

#===============================================================
# FEATURE IMPORTANCE BASED ON FEATURE PERMUTATION
#===============================================================

#start_time = time.time()
#result = permutation_importance(rf, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
#elapsed_time = time.time() - start_time
#print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
#
#rf_importances_2 = pd.Series(result.importances_mean, index=feature_names)
#std_2 = result.importances_std
#
#fig, ax = plt.subplots(figsize=(6.5,4))
#rf_importances_2.plot.bar(yerr=result.importances_std, ax=ax)
#bars = ('q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8')
#y_pos = numpy.arange(len(bars))
#plt.xticks(y_pos, bars, rotation=0)
#ax.set_title("Feature importances using permutation on full model")
#ax.set_ylabel("Mean accuracy decrease")
#fig.tight_layout()
#plt.show()
#fig.savefig(model_directory+'best/verification-scatter-permutation-rf1.png', bbox_inches='tight')

#===============================================================
# BOTH GRAPHICS
#===============================================================
#fig, ax = plt.subplots(figsize=(6,4))
#bars = ('q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8')
#legends = ['Mean decrease in impurity (MDI method)','Mean accuracy decrease (permutation)']
#y_pos = numpy.arange(len(bars))*2
#plt.xticks(y_pos, bars, rotation=0)
#width=0.5
#ax.bar(y_pos-width/2,rf_importances_1, yerr=std_1, align='center', alpha=0.9, color='C0')
#ax.bar(y_pos+width,rf_importances_2, yerr=std_2, align='center', alpha=0.9, color='C2')
#ax.set_title("Feature importances using MDI/permutation")
#plt.legend(legends)
#fig.tight_layout()
#plt.show()
#fig.savefig(model_directory+'best/verification-scatter-permutation-rf1-all.png', bbox_inches='tight')

#===============================================================
# SAVE AND LOAD THE MODEL USING JOBLIB
#===============================================================
import joblib
filename = model_directory+'best/rf_model.joblib'

joblib.dump(rf, filename)
print("Model saved")
model = joblib.load(filename)
print("Model loaded")
