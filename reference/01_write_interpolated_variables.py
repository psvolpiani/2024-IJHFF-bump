import numpy as np
from datetime import datetime as dt

# =============================================================
# Before using this script we need to interpolate the reference
# solution in the FreeFem mesh. We can use either Tecplot or
# Paraview to do so.
# =============================================================

# -----------------
# Read tecplot file
# -----------------

h = input('Insert bump height:')
filename = './results/h'+h+'/reference-interp-tecplot-h'+h
file = filename+'.dat'

nodes = {'20': 23184, 
         '26': 26712, 
         '31': 29232, 
         '38': 33012, 
         '42': 34776}

header = 15
nnodes = nodes[h]
nvar = 9
n = 0
data = []


with open(file) as f:
    
    lines = f.readlines()
    
    for line in lines:
        
        if (n > header-1 and n < nnodes+header):
        
            myarray = np.fromstring(line, dtype=float, sep=' ')
            data = np.concatenate([data , myarray])
            
        n=n+1
        
data  = np.reshape(data, (nnodes,nvar))
xnew  = data[:,0]
ynew  = data[:,1]
uunew = data[:,2]
uvnew = data[:,3]
vvnew = data[:,4]
wwnew = data[:,5]
pnew  = data[:,6]
unew  = data[:,7]
vnew  = data[:,8]

tecplot_file = open(filename+"-X.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(xnew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

tecplot_file = open(filename+"-Y.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(ynew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

tecplot_file = open(filename+"-U.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(unew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

tecplot_file = open(filename+"-V.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(vnew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

tecplot_file = open(filename+"-P.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(pnew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

tecplot_file = open(filename+"-UU.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(uunew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

tecplot_file = open(filename+"-VV.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(vvnew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

tecplot_file = open(filename+"-WW.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(wwnew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

tecplot_file = open(filename+"-UV.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(uvnew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

now = dt.now()
print("Interpolated variables written from Tecplot file.")
#log_file = open("./01_write_interpolated_variables_tec.txt","w")
#log_file.write('Interpolated variables written from Tecplot file on '+\
#               now.strftime("%d/%m/%Y %H:%M:%S")+'.')
#log_file.close()
