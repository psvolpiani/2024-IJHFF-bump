#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys
import matplotlib.mathtext as mathtext
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import operator
import pandas as pd

from scipy.interpolate import griddata
from scipy import interpolate
from math import *
from numpy import *
from matplotlib import *
from matplotlib import rc


rc('text', usetex=True)
rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=16)
plt.rc('legend',**{'fontsize':14})
plt.rc({'axes.labelsize': 20})


# FIT
file   = "fitlog_model_in8_out1_4hid80_iter4.dat"
header = 1
x = numpy.loadtxt(file,skiprows=header, usecols=(0,))
y = numpy.loadtxt(file,skiprows=header, usecols=(1,))

file   = "../../../train-29-loc8-best/results/training/fitlog_model_in8_out1_4hid80_iter2.dat"
header = 1
x2 = numpy.loadtxt(file,skiprows=header, usecols=(0,))
y2 = numpy.loadtxt(file,skiprows=header, usecols=(1,))

fig, ax = plt.subplots(figsize=(5,3.8))
ax.plot(x2,y2,color='C2',linestyle='-', label = 'NN1')
ax.plot(x,y,color='C1',linestyle='-.', label = 'NN2')
##ax.plot(x_da,c_da,color='C3')
#ax.set_xlim([-0.25, 1.5])
#ax.set_ylim([-0.05, 3.035])
#ax.xaxis.set_ticks(np.linspace(0, 9, 10))
#ax.yaxis.set_ticks(np.linspace(0, 3, 4))
plt.legend()
ax.set_xlabel( "Epoch",fontsize=18)
ax.set_ylabel( "Fit",fontsize=18)
fig.subplots_adjust(left=0.2)
fig.subplots_adjust(bottom=0.20)
fig.savefig("fig_fit_nn1_nn2.pdf", dpi=300, bbox_inches='tight')

# VAL/TRA LOSS
file   = "logloss_model_in8_out1_4hid80_iter4.dat"
header = 1
xt = numpy.loadtxt(file,skiprows=header, usecols=(0,))
yt = numpy.loadtxt(file,skiprows=header, usecols=(1,))
file   = "val_logloss_model_in8_out1_4hid80_iter4.dat"
header = 1
xv = numpy.loadtxt(file,skiprows=header, usecols=(0,))
yv = numpy.loadtxt(file,skiprows=header, usecols=(1,))

fig, ax = plt.subplots(figsize=(5,3.8))
ax.plot(xt,yt,color='C0',linestyle='--', label = 'Training loss')
ax.plot(xv,yv,color='C3',linestyle='-', label = 'Validation loss')
#ax.set_xlim([-0.25, 1.5])
#ax.set_ylim([-0.05, 3.035])
#ax.xaxis.set_ticks(np.linspace(0, 9, 10))
#ax.yaxis.set_ticks(np.linspace(0, 3, 4))
plt.legend()
ax.set_xlabel( "Epoch",fontsize=18)
ax.set_ylabel( "Loss",fontsize=18)
fig.subplots_adjust(left=0.2)
fig.subplots_adjust(bottom=0.20)
fig.savefig("fig_loss_nn2.pdf", dpi=300, bbox_inches='tight')



plt.show()



