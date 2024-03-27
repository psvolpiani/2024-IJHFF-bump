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
# Plot horizontal velocity error
# -----------------------------------------------------------------------------------------
print('\n Plotting velocity error... \n')
error = data3.interp(data3.nut)
filename = './results/h'+nummesh+'/nn3-contour-error-h'+str(h)+'.png'
plot_contour_u(data3, error, ref_length = C, filename = filename, vmin = -0.0, vmax = 0.1, nlevels = 11)

print('\n Plotting velocity error... \n')
error = data2.interp(data2.nut)
filename = './results/h'+nummesh+'/sa-contour-error-h'+str(h)+'.png'
plot_contour_u(data2, error, ref_length = C, filename = filename, vmin = -0.0, vmax = 0.1, nlevels = 11)

# -----------------------------------------------------------------------------------------
# Plot horizontal velocity contour
# -----------------------------------------------------------------------------------------
#print('\n Plotting horizontal velocity contour... \n')
#
#ui1 = data1.interp(data1.u)
#ui2 = data2.interp(data2.u)
#ui3 = data3.interp(data3.u)
#filename = './results/h'+nummesh+'/nn-contour-u-h'+str(h)+'.png'
#
#plot_contour_u(data1, ui1/Uref, data2 = data3, vari2 = ui3/Uref, ref_length = C, filename = filename, vmin = -0.1, vmax = 1.4, nlevels = 16)

# -----------------------------------------------------------------------------------------
# Plot vertical velocity contour
# -----------------------------------------------------------------------------------------
#print('Plotting vertical velocity contour... \n')
#
#vi1 = data1.interp(data1.v)
#vi2 = data2.interp(data2.v)
#vi3 = data3.interp(data3.v)
#filename = './results/h'+nummesh+'/nn-contour-v-h'+str(h)+'.png'
#
#plot_contour_v(data1, vi1/Uref, data2 = data3, vari2 = vi3/Uref, ref_length = C, filename = filename, vmin = -0.2, vmax = 0.2, nlevels = 11)

# -----------------------------------------------------------------------------------------
# Plot U velocity profiles
# -----------------------------------------------------------------------------------------
#print('Plotting horizontal velocity profiles... \n')
#
#ui1 = data1.interp(data1.u)
#ui2 = data2.interp(data2.u)
#ui3 = data3.interp(data3.u)
#
#stations = [0.*C,0.25*C,0.5*C,0.75*C,1.0*C,1.25*C,1.5*C]
#norm = 2*Uref
#filename = './results/h'+nummesh+'/nnExact-profiles-u-h'+str(h)+'.png'
#xlabel = r"$\overline{u}/2\overline{u}_{ref}+x/c$"
#ylabel = r"$y/c$"
#
#plot_profiles(stations, data1, ui1, data2 = data3, vari2 = ui3, norm = norm, ref_length = C, xlabel = xlabel, ylabel=ylabel, filename=filename)
