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
from scipy import interpolate
from sklearn.metrics import mean_squared_error
from scipy import integrate


rc('text', usetex=True)
rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=16)
plt.rc('legend',**{'fontsize':13})
plt.rc({'axes.labelsize': 20})

# -----------------------------
# Parameters
# -----------------------------
nummesh = input('Insert bump number:')
h = int(nummesh)
c    = 305.0 /1000.0
Uref = 16.8
q = 0.5*Uref**2 # Reference dynamic pressure

# LES data
file   = '../../../../les/Cfh'+nummesh+'.dat'
header = 1
x  = numpy.loadtxt(file,skiprows=header, usecols=(0,))
cf = numpy.loadtxt(file,skiprows=header, usecols=(1,))
mk  = np.arange(1, len(x), 10)
cf_les = [x[mk],cf[mk]]

file   = '../../../../les/Cph'+nummesh+'.dat'
header = 1
x  = numpy.loadtxt(file,skiprows=header, usecols=(0,))
cp = numpy.loadtxt(file,skiprows=header, usecols=(1,))
mk  = np.arange(1, len(x), 10)
cp_les = [x[mk],cp[mk]]

file = '../../../../baseline-shorter/results/h'+nummesh+'/baseline-X-Y-Cp-Cf-h'+nummesh+'-Re-2500.dat'
header  = 0
x_sa   = numpy.loadtxt(file,skiprows=header, usecols=(0,))
y_sa   = numpy.loadtxt(file,skiprows=header, usecols=(1,))
p_sa   = numpy.loadtxt(file,skiprows=header, usecols=(2,))
c_sa   = numpy.loadtxt(file,skiprows=header, usecols=(3,))

file = './results/h'+nummesh+'/X-Y-Cp-Cf-h'+nummesh+'.dat'
header  = 0
x_nn1   = numpy.loadtxt(file,skiprows=header, usecols=(0,))
y_nn1   = numpy.loadtxt(file,skiprows=header, usecols=(1,))
p_nn1   = numpy.loadtxt(file,skiprows=header, usecols=(2,))
c_nn1   = numpy.loadtxt(file,skiprows=header, usecols=(3,))

f1 = interpolate.interp1d(x, cp)
cp_ref = f1(x_sa/c)
f2 = interpolate.interp1d(x, cf)
cf_ref = f2(x_sa/c)

E_Cp_SA = integrate.cumtrapz( (cp_ref-p_sa/q)**2, x_sa/c)
E_Cp_ML = integrate.cumtrapz( (cp_ref-p_nn1/q)**2, x_sa/c)
print("Ratio Cp SA/ML =",E_Cp_ML[-1]/E_Cp_SA[-1])
E_Cf_SA = integrate.cumtrapz( (cf_ref-c_sa/q)**2, x_sa/c)
E_Cf_ML = integrate.cumtrapz( (cf_ref-c_nn1/q)**2, x_sa/c)
print("Ratio Cf SA/ML =",E_Cf_ML[-1]/E_Cf_SA[-1])

fig, ax = plt.subplots(figsize=(6,4))
plt.axhline(y=0., color='k', linestyle='-', linewidth=0.5)
ax.scatter(cp_les[0],cp_les[1],marker='o',color='k',facecolors='none',label='LES')
ax.plot(x_sa/c,p_sa/q,color='C0',linestyle='-',label='RANS-SA')
ax.plot(x_nn1/c,p_nn1/q,color='C3',linestyle='--',label='RANS-NN3')
ax.legend()
ax.set_xlim([-0.25, 1.5])
ax.set_xlabel( r"$x/c$",fontsize=18)
ax.set_ylabel( r"$C_p$",fontsize=18)
fig.subplots_adjust(left=0.2)
fig.subplots_adjust(bottom=0.20)
fig.savefig('./results/h'+nummesh+'/fig_cp_loc_h'+nummesh+'_nn3.pdf', dpi=300)

fig, ax2 = plt.subplots(figsize=(6,4))
plt.axhline(y=0., color='k', linestyle='-', linewidth=0.5)
ax2.scatter(cf_les[0],cf_les[1],marker='o',color='k',facecolors='none',label='LES')
ax2.plot(x_sa/c,c_sa/q,color='C0',linestyle='-',label='RANS-SA')
ax2.plot(x_nn1/c,c_nn1/q,color='C3',linestyle='--',label='RANS-NN3')
ax2.legend()
ax2.set_xlim([-0.25, 1.5])
ax2.set_xlabel( r"$x/c$",fontsize=18)
ax2.set_ylabel( r"$C_f$",fontsize=18)
fig.subplots_adjust(left=0.2)
fig.subplots_adjust(bottom=0.20)
fig.savefig('./results/h'+nummesh+'/fig_cf_loc_h'+nummesh+'_nn3.pdf', dpi=300)
plt.show()



