#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys
sys.path.append("../lib/")
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
Prefs = {'20': 5.54675, 
         '26': 7.49148,
         '31': 8.41009,
         '38': 10.0917,
         '42': 9.50295}
Pref = Prefs[nummesh]
nutmaxvals = {'20': 120, 
              '26': 120,
              '31': 170,
              '38': 170,
              '42': 250}
nutmax = nutmaxvals[nummesh]

directory1 = '../reference/results/h'+nummesh+'/'
file1 = directory1+'reference-Ue-Ve-Pe-nute-h'+str(h)+'-tecplot.dat'
header1 = 3
nvar1 = 7
label1 = 'LES'

#directory2 = '../baseline/results/h'+nummesh+'/'
#file2 = directory2+'baseline-solution-h'+str(h)+'-Re-2500-tecplot.dat'
#header2 = 3
#nvar2 = 7
#label2 = 'RANS - SA (original domain)'

directory3 = './results/h'+nummesh+'/'
file3 = directory3+'baseline-solution-h'+str(h)+'-Re-2500-tecplot.dat'
header3 = 3
nvar3 = 7
label3 = 'RANS - SA (shorter domain)'

# -----------------------------------------------------------------------------------------
# Read files
# -----------------------------------------------------------------------------------------
data1 = Data()
data1.read(file1, header1, nvar1)
data1.grid(h,1600,1000)
data1.label = label1

#data2 = Data()
#data2.read(file2, header2, nvar2)
#data2.grid(h,1600,1000)
#data2.label = label2

data3 = Data()
data3.read(file3, header3, nvar3)
data3.grid(h,1600,1000)
data3.label = label3

# -----------------------------------------------------------------------------------------
# Plot horizontal velocity contour
# -----------------------------------------------------------------------------------------
print('\n Plotting horizontal velocity contour... \n')

ui1 = data1.interp(data1.u)
#ui2 = data2.interp(data2.u)
ui3 = data3.interp(data3.u)
filename = './figures/baseline-shorter-contour-u-h'+str(h)+'.png'

#plot_contour_u(data1, ui1/Uref, data2 = data2, vari2 = ui2/Uref, data3 = data3, vari3 = ui3/Uref, ref_length = C, filename = filename, vmin = -0.1, vmax = 1.2, nlevels = 14)
plot_contour_u(data1, ui1/Uref, data2 = data3, vari2 = ui3/Uref, ref_length = C, filename = filename, vmin = -0.1, vmax = 1.2, nlevels = 14)

# -----------------------------------------------------------------------------------------
# Plot vertical velocity contour
# -----------------------------------------------------------------------------------------
print('Plotting vertical velocity contour... \n')

vi1 = data1.interp(data1.v)
#vi2 = data2.interp(data2.v)
vi3 = data3.interp(data3.v)
filename = './figures/baseline-shorter-contour-v-h'+str(h)+'.png'

#plot_contour_v(data1, vi1/Uref, data2 = data2, vari2 = vi2/Uref, data3 = data3, vari3 = vi3/Uref, ref_length = C, filename = filename, vmin = -0.1, vmax = 0.1, nlevels = 11)
plot_contour_v(data1, vi1/Uref, data2 = data3, vari2 = vi3/Uref, ref_length = C, filename = filename, vmin = -0.1, vmax = 0.1, nlevels = 11)

# -----------------------------------------------------------------------------------------
# Plot U velocity profiles
# -----------------------------------------------------------------------------------------
print('Plotting horizontal velocity profiles... \n')

ui1 = data1.interp(data1.u)
#ui2 = data2.interp(data2.u)
ui3 = data3.interp(data3.u)

stations = [0.*C,0.25*C,0.5*C,0.75*C,1.0*C,1.25*C,1.5*C]
norm = 2*Uref
filename = './figures/baseline-shorter-profiles-u-h'+str(h)+'.pdf'
xlabel = r"$\overline{u}/2\overline{u}_{ref}+x/c$"
ylabel = r"$y/c$"

#plot_profiles(stations, data1, ui1, data2 = data2, vari2 = ui2, data3 = data3, vari3 = ui3, norm = norm, ref_length = C, xlabel = xlabel, ylabel=ylabel, filename=filename)
plot_profiles(stations, data1, ui1, data2 = data3, vari2 = ui3, norm = norm, ref_length = C, xlabel = xlabel, ylabel=ylabel, filename=filename)

