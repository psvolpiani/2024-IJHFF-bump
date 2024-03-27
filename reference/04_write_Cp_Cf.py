import numpy as np
from math import *
nummesh = input('Insert bump number:')

# Read files
cp_filename = './database/les-Cp-h'+nummesh+'.dat'
cf_filename = './database/les-Cf-h'+nummesh+'.dat'

x = np.genfromtxt(cp_filename,dtype = float, skip_header = 1, \
                       delimiter = '	', usecols = 0)
cp = np.genfromtxt(cp_filename,dtype = float, skip_header = 1, \
                       delimiter = '	', usecols = 1)
cf = np.genfromtxt(cf_filename,dtype = float, skip_header = 1, \
                       delimiter = '	', usecols = 1)
# Adjust normalization
C    = 305.0 /1000.0
Uref = 16.8
q = 0.5*Uref**2 # Reference dynamic pressure
x = x*C
cp = cp*q
cf = cf*q

# Create y array
def bump(x,nummesh):
    C      = 305.0 /1000.0;
    L      = C - 2.0 * ( C / 12.0 );
    R1     = 0.323 * C;
    hp     = R1 - sqrt( R1**2 - ((C-L)/2.0)**2 );
    hoC    = nummesh / 305.0;
    h      = hoC * C;
    hpp    = h - hp;
    R2     = ( hpp**2 + (L/2.0)**2 ) / ( 2.0 * hpp ) ;
      
    if (x<0. ):
      y = 0.
    elif (x>=0.  and x<C/12.0 ):
      y = (R1-sqrt(R1**2-x**2))
    elif (x>=C/12.0 and x<L+C/12.0 ):
      y = (-R2+h+sqrt(R2**2-(C/2.0-x)**2))
    elif (x>=L+C/12.0 and x<C ):
      x = x-C
      y = (R1-sqrt(R1**2-x**2))
    else: y=0.
    
    return y

n = len(x)
y = np.empty((n,1), dtype = float)

for i in range(n):
    y[i] = bump(x[i], int(nummesh))
    
# Reshape arrays
x = np.reshape(x, newshape = (n,1))
cp = np.reshape(cp, newshape = (n,1))
cf = np.reshape(cf, newshape = (n,1))

# Flip cf
cf = np.flip(cf)

# Append arrays
les = np.append(x, y, axis = 1)
les = np.append(les, cp, axis = 1)
les = np.append(les, cf, axis = 1)

# Save file
np.savetxt('./results/h'+nummesh+'/reference-X-Y-Cp-Cf-h'+nummesh+'.dat', les, fmt='%.18f', delimiter = ' ')