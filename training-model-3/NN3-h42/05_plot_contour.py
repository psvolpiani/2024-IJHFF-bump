#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys
sys.path.append("../../lib/")
from libplot import *
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
nummesh = input('Insert bump number:')
h = int(nummesh)
C    = 305.0 /1000.0
Uref = 16.8
deltatheta = 0.0036
Retheta = 2500
nu = Uref * deltatheta / Retheta

nutmaxvals = {'20': 5*14, 
              '26': 6*14,
              '31': 7*14,
              '38': 12*14,
              '42': 15*14}
nutmax = nutmaxvals[nummesh]

directory1 = './results/verification/'
file1 = directory1+'training-les-solution-h'+str(h)+'.dat'
header1 = 3
nvar1 = 18
#label1 = 'RANS-SA'

# -----------------------------------------------------------------------------------------
# Read files
# -----------------------------------------------------------------------------------------
data1 = Data()
data1.read_features_8(file1, header1, nvar1)
data1.grid(h,1600,1000)

# -----------------------------------------------------------------------------------------
# Plot turbulent viscosity contour
# -----------------------------------------------------------------------------------------
print('Plotting turbulent viscosity contour... \n')

data1.label = r"$\nu_t/\nu$"
nuti = data1.interp(data1.nuT_nu)
filename = './figures/h'+nummesh+'/baseline-contour-nuT_nu-h'+str(h)+'.png'
plot_contour_dnuT(data1, nuti, ref_length = C, filename = filename)

data1.label = r"$\nu_t^{LES}/\nu$"
nuti = data1.interp(data1.nuTe_nu)
filename = './figures/h'+nummesh+'/baseline-contour-nuTe_nu-h'+str(h)+'.png'
plot_contour_nuT(data1, nuti, ref_length = C, filename = filename, vmin = 0.0, vmax = nutmax)

data1.label = r"$\dnu_t/\nu$"
nuti = data1.interp(data1.dnuT_nu)
filename = './figures/h'+nummesh+'/baseline-contour-dnuT_nu-h'+str(h)+'.png'
plot_contour_dnuT(data1, nuti, ref_length = C, filename = filename)

data1.label = r"$\nu_t^{NN}/\nu$"
nuti = data1.interp(data1.nuTNN_nu)
filename = './figures/h'+nummesh+'/baseline-contour-nuTNN3_nu-h'+str(h)+'.png'
plot_contour_nuT(data1, nuti, ref_length = C, filename = filename, vmin = 0.0, vmax = nutmax)

