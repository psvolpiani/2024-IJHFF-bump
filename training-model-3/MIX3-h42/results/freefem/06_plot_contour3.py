#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys
sys.path.append("../../../../lib/")
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

directory1 = '../../../../reference/results/h'+nummesh+'/'
file1 = directory1+'reference-Ue-Ve-Pe-nute-h'+str(h)+'-tecplot.dat'
header1 = 3
nvar1 = 7
label1 = 'LES'

directory2 = '../../../../baseline-shorter/results/h'+nummesh+'/'
file2 = directory2+'baseline-solution-h'+str(h)+'-Re-2500-tecplot.dat'
header2 = 3
nvar2 = 7
label2 = 'RANS - SA'

directory3 = './results/h'+nummesh+'/'
file3 = directory3+'nn-solution-h'+str(h)+'-Re-2500-tecplot.dat'
header3 = 3
nvar3 = 6
label3 = 'RANS - NN2'

# -----------------------------------------------------------------------------------------
# Read files
# -----------------------------------------------------------------------------------------
data1 = Data()
data1.read(file1, header1, nvar1)
data1.grid(h,2000,2000)
data1.label = label1

data2 = Data()
data2.read(file2, header2, nvar2)
data2.grid(h,2000,2000)
data2.label = label2

data3 = Data()
data3.read(file3, header3, nvar3)
data3.grid(h,2000,2000)
data3.label = label3

# -----------------------------------------------------------------------------------------
# Plot U velocity profiles
# -----------------------------------------------------------------------------------------
print('Plotting horizontal velocity profiles... \n')

ui1 = data1.interp(data1.u)
ui2 = data2.interp(data2.u)
ui3 = data3.interp(data3.u)

stations = [0.*C,0.25*C,0.5*C,0.75*C,1.0*C,1.25*C,1.5*C]
norm = 2*Uref
filename = './results/h'+nummesh+'/nn3-profiles-u-h'+str(h)+'.pdf'
xlabel = r"$\overline{u}/2\overline{u}_{ref}+x/c$"
ylabel = r"$y/c$"

plot_profiles(stations, data1, ui1, data2 = data2, vari2 = ui2, data3 = data3, vari3 = ui3, norm = norm, ref_length = C, xlabel = xlabel, ylabel=ylabel, filename=filename)
