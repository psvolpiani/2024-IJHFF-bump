import numpy as np
import operator
from scipy.interpolate import griddata
from math import *
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import ticker as tk 

# rc('text.latex', preamble= r'\usepackage[dvips]{graphicx}\usepackage{xfrac}')
rc('font', family='serif')
rc('lines', linewidth=1.)
rc('font', size=16)
plt.rc('legend',**{'fontsize':10})
plt.rc({'axes.labelsize': 20})

#===============================================================
# PROFILE FUNCTION
#===============================================================

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
 
        
#===============================================================
# DATASET CLASS
#===============================================================
class Data:
    
    def read(self, file, header_size, nvar, connectivity = False):
        print('Loading data...\n')
        
        data            = []            # Initialize data array
        connect         = []
        
        print('File: '+ file)
        
        eof = False
        i = 0
        nodes = 1
        elements = 1
            
        with open(file) as f:
            while eof == False:
                line = f.readline()
                if (i == header_size - 1):
                    nodes = int(line[line.find('N=')+2:line.find('E=')-1])
                    elements = int(line[line.find('E=')+2:line.find(',',line.find('E='))])
                    
                elif (i > header_size - 1 and i < header_size + nodes):
                    myarray = np.fromstring(line, dtype=float, sep=' ')
                    data = np.concatenate([data , myarray])
                
                elif (i >= header_size + nodes and i < header_size + nodes + elements) :
                    if connectivity:
                        myarray = np.fromstring(line, dtype=int, sep=' ')
                        connect = np.concatenate([connect,myarray])
                    else:
                        break
                elif (i == header_size + nodes + elements):
                    break
                
                i+=1
                    
        # Reshape data so that we have blocks of lines corresponding to each case 
        self.size       = nodes    # Total size of data with all the cases
        self.base       = np.reshape(data, (self.size,nvar))
        if connectivity:
            self.connectivity    = np.reshape(connect,(elements,3))
        self.nvar       = nvar
        
        self.x = self.base[:,0]
        self.y = self.base[:,1]
        self.u = self.base[:,2]
        self.v = self.base[:,3]
        self.p = self.base[:,4]
        self.nut = self.base[:,5]
        if(nvar>6) : self.nuT = self.base[:,6]
        
            
    def read_features_7(self, file, header_size, nvar, connectivity = False):
        print('Loading data...\n')
        
        data            = []            # Initialize data array
        connect         = []
        
        print('File: '+ file)
        
        eof = False
        i = 0
        nodes = 1
        elements = 1
            
        with open(file) as f:
            while eof == False:
                line = f.readline()
                if (i == header_size - 1):
                    nodes = int(line[line.find('N=')+2:line.find('E=')-1])
                    elements = int(line[line.find('E=')+2:line.find(',',line.find('E='))])
                    
                elif (i > header_size - 1 and i < header_size + nodes):
                    myarray = np.fromstring(line, dtype=float, sep=' ')
                    data = np.concatenate([data , myarray])
                
                elif (i >= header_size + nodes and i < header_size + nodes + elements) :
                    if connectivity:
                        myarray = np.fromstring(line, dtype=int, sep=' ')
                        connect = np.concatenate([connect,myarray])
                    else:
                        break
                elif (i == header_size + nodes + elements):
                    break
                
                i+=1
                    
        # Reshape data so that we have blocks of lines corresponding to each case 
        self.size       = nodes    # Total size of data with all the cases
        self.base       = np.reshape(data, (self.size,nvar))
        if connectivity:
            self.connectivity    = np.reshape(connect,(elements,3))
        self.nvar       = nvar
        
        self.x = self.base[:,0]
        self.y = self.base[:,1]
        self.u = self.base[:,2]
        self.v = self.base[:,3]
        self.p = self.base[:,4]
        self.nut = self.base[:,5]
        self.q1 = self.base[:,6]
        self.q2 = self.base[:,7]
        self.q3 = self.base[:,8]
        self.q4 = self.base[:,9]
        self.q5 = self.base[:,10]
        self.q6 = self.base[:,11]
        self.q7 = self.base[:,12]
        self.dnuT_nu = self.base[:,13]
        self.nuT_nu = self.base[:,14]
        self.nuTe_nu = self.base[:,15]
        if(nvar>16) : self.nuTNN_nu = self.base[:,16]
                
            
    def read_features_8(self, file, header_size, nvar, connectivity = False):
        print('Loading data...\n')
        
        data            = []            # Initialize data array
        connect         = []
        
        print('File: '+ file)
        
        eof = False
        i = 0
        nodes = 1
        elements = 1
            
        with open(file) as f:
            while eof == False:
                line = f.readline()
                if (i == header_size - 1):
                    nodes = int(line[line.find('N=')+2:line.find('E=')-1])
                    elements = int(line[line.find('E=')+2:line.find(',',line.find('E='))])
                    
                elif (i > header_size - 1 and i < header_size + nodes):
                    myarray = np.fromstring(line, dtype=float, sep=' ')
                    data = np.concatenate([data , myarray])
                
                elif (i >= header_size + nodes and i < header_size + nodes + elements) :
                    if connectivity:
                        myarray = np.fromstring(line, dtype=int, sep=' ')
                        connect = np.concatenate([connect,myarray])
                    else:
                        break
                elif (i == header_size + nodes + elements):
                    break
                
                i+=1
                    
        # Reshape data so that we have blocks of lines corresponding to each case
        self.size       = nodes    # Total size of data with all the cases
        self.base       = np.reshape(data, (self.size,nvar))
        if connectivity:
            self.connectivity    = np.reshape(connect,(elements,3))
        self.nvar       = nvar
        
        self.x = self.base[:,0]
        self.y = self.base[:,1]
        self.u = self.base[:,2]
        self.v = self.base[:,3]
        self.p = self.base[:,4]
        self.nut = self.base[:,5]
        self.q1 = self.base[:,6]
        self.q2 = self.base[:,7]
        self.q3 = self.base[:,8]
        self.q4 = self.base[:,9]
        self.q5 = self.base[:,10]
        self.q6 = self.base[:,11]
        self.q7 = self.base[:,12]
        self.q8 = self.base[:,13]
        self.dnuT_nu = self.base[:,14]
        self.nuT_nu = self.base[:,15]
        self.nuTe_nu = self.base[:,16]
        if(nvar>17) : self.nuTNN_nu = self.base[:,17]
        
        
    def read_da(self, file, header_size, nvar, connectivity = False):
        print('Loading data...\n')
        
        data            = []            # Initialize data array
        connect         = []
        
        print('File: '+ file)
        
        eof = False
        i = 0
        nodes = 1
        elements = 1
            
        with open(file) as f:
            while eof == False:
                line = f.readline()
                if (i == header_size - 1):
                    nodes = int(line[line.find('N=')+2:line.find('E=')-1])
                    elements = int(line[line.find('E=')+2:line.find(',',line.find('E='))])
                    
                elif (i > header_size - 1 and i < header_size + nodes):
                    myarray = np.fromstring(line, dtype=float, sep=' ')
                    data = np.concatenate([data , myarray])
                
                elif (i >= header_size + nodes and i < header_size + nodes + elements) :
                    if connectivity:
                        myarray = np.fromstring(line, dtype=int, sep=' ')
                        connect = np.concatenate([connect,myarray])
                    else:
                        break
                elif (i == header_size + nodes + elements):
                    break
                
                i+=1
                    
        # Reshape data so that we have blocks of lines corresponding to each case
        self.size       = nodes    # Total size of data with all the cases
        self.base       = np.reshape(data, (self.size,nvar))
        if connectivity:
            self.connectivity    = np.reshape(connect,(elements,3))
        self.nvar       = nvar
        
        self.x = self.base[:,0]
        self.y = self.base[:,1]
        self.u = self.base[:,2]
        self.v = self.base[:,3]
        self.p = self.base[:,4]
        self.nut = self.base[:,5]
        self.nuT = self.base[:,6]
        self.fxda = self.base[:,7]
        self.fyda = self.base[:,8]
        self.fxsa = self.base[:,9]
        self.fysa = self.base[:,10]
        self.fxsum = self.base[:,11]
        self.fysum = self.base[:,12]
        
    def grid(self, h, nx, ny):
        # Create grid values first.
        self.xi = np.linspace(min(self.x), max(self.x), nx)
        self.yi = np.linspace(min(self.y), max(self.y), ny)
        self.h  = h
        
        # Build profile
        self.xprof = np.linspace(min(self.x), max(self.x), nx)
        self.yprof = np.zeros(len(self.xprof))
        for i in range (len(self.xprof)): self.yprof[i] = bump(self.xprof[i],h)

    def interp(self, var):
        # Perform linear interpolation of the data (x,y) on a grid defined by (xi,yi)
        vari = griddata((self.x, self.y), var, (self.xi[None,:], self.yi[:,None]), method='linear')
        
        for i in range (0,len(self.xi)):
          for j in range (0,len(self.yi)):
            if ( self.yi[j] < bump(self.xi[i],self.h)):
                vari[j,i] = np.nan
                
        return vari
    
    def read_cp_cf(self, filename):
        wall = np.genfromtxt(filename,dtype = float, delimiter = ' ')
        self.xw = wall[:,0]
        self.yw = wall[:,1]
        self.cp = wall[:,2]
        self.cf = wall[:,3]
            
  
def plot_pcolormesh(data, vari, ref_length, filename, vmin = None, vmax = None):
    
    fig, ax1 = plt.subplots(figsize=(10,3))
    
    ax1.plot(data.xprof/ref_length,data.yprof/ref_length,linewidth=2.,color='gray')
    
    # plt.contour(data.xi/ref_length,data.yi/ref_length,vari,15,linewidths=0.5,colors='k', linestyles = 'solid')
    
    cntr1 = ax1.pcolormesh(data.xi/ref_length,data.yi/ref_length,vari, cmap="jet", vmin = vmin, vmax = vmax)
    
    ax1.set_ylim([0.0, 0.45])
    ax1.set_xlim([-0.25, 1.5])
    
    fig.colorbar(cntr1, ax=ax1, ticks = tk.MaxNLocator(nbins=8))
    ax1.set_xlabel( r"$x/c$",fontsize=17)
    ax1.set_ylabel( r"$y/c$",fontsize=17)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    
def plot_contour(data1, vari1, data2 = None, vari2 = None, data3 = None, vari3 = None, ref_length = 1.0, filename = 'countour.png', vmin = None, vmax = None):
    fig, ax = plt.subplots(figsize=(10,3))
    
    ax.plot(data1.xprof/ref_length,data1.yprof/ref_length,linewidth=2.,color='gray')
    
    #plt.contour(data1.xi/ref_length,data1.yi/ref_length,vari1,15,linewidths=0.5,colors='k', linestyles = 'solid', vmin = vmin, vmax = vmax)
    if data2 is not None:
        plt.contour(data2.xi/ref_length,data2.yi/ref_length,vari2,15,linewidths=0.5,colors='k', linestyles = 'dashed', vmin = vmin, vmax = vmax)
    if data3 is not None:
        plt.contour(data3.xi/ref_length,data3.yi/ref_length,vari3,15,linewidths=0.5,colors='k', linestyles = 'dashdot', vmin = vmin, vmax = vmax)
        
    cntr = ax.contourf(data1.xi/ref_length,data1.yi/ref_length,vari1, 15, cmap="jet")
    
    ax.set_ylim([0.0, 0.25])
    ax.set_xlim([-0.1, 1.2])
    
    fig.colorbar(cntr, ax=ax)
    ax.set_xlabel( r"$x/c$",fontsize=17)
    ax.set_ylabel( r"$y/c$",fontsize=17)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    
def plot_contour_nuT(data1, vari1, data2 = None, vari2 = None, data3 = None, vari3 = None, ref_length = 1.0, filename = 'countour.png', vmin = 0, vmax = 100, nlevels = 15):
    fig, ax = plt.subplots(figsize=(10,3))
    
    ax.plot(data1.xprof/ref_length,data1.yprof/ref_length,linewidth=2.,color='gray')
    
    #plt.contour(data1.xi/ref_length,data1.yi/ref_length,vari1,15,linewidths=0.5,colors='k', linestyles = 'solid', vmin = vmin, vmax = vmax)
    if data2 is not None:
        plt.contour(data2.xi/ref_length,data2.yi/ref_length,vari2,15,linewidths=0.5,colors='k', linestyles = 'dashed', vmin = vmin, vmax = vmax)
    if data3 is not None:
        plt.contour(data3.xi/ref_length,data3.yi/ref_length,vari3,15,linewidths=0.5,colors='k', linestyles = 'dashdot', vmin = vmin, vmax = vmax)
        
    cntr = ax.contourf(data1.xi/ref_length,data1.yi/ref_length,vari1, np.linspace(vmin,vmax,nlevels), cmap="jet", vmin = vmin, vmax = vmax, extend='both')
    
    ax.set_ylim([0.0, 0.35])
    ax.set_xlim([-0.25, 1.5])
    
    fig.colorbar(cntr, ax=ax)
    ax.set_xlabel( r"$x/c$",fontsize=17)
    ax.set_ylabel( r"$y/c$",fontsize=17)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    
def plot_contour_dnuT(data1, vari1, data2 = None, vari2 = None, data3 = None, vari3 = None, ref_length = 1.0, filename = 'countour.png', vmin = None, vmax = None, nlevels = 15):
    fig, ax = plt.subplots(figsize=(10,3))
    
    ax.plot(data1.xprof/ref_length,data1.yprof/ref_length,linewidth=2.,color='gray')
    
    #plt.contour(data1.xi/ref_length,data1.yi/ref_length,vari1,15,linewidths=0.5,colors='k', linestyles = 'solid', vmin = vmin, vmax = vmax)
    if data2 is not None:
        plt.contour(data2.xi/ref_length,data2.yi/ref_length,vari2,15,linewidths=0.5,colors='k', linestyles = 'dashed', vmin = vmin, vmax = vmax)
    if data3 is not None:
        plt.contour(data3.xi/ref_length,data3.yi/ref_length,vari3,15,linewidths=0.5,colors='k', linestyles = 'dashdot', vmin = vmin, vmax = vmax)
        
    cntr = ax.contourf(data1.xi/ref_length,data1.yi/ref_length,vari1, 15, cmap="jet", vmin = vmin, vmax = vmax, extend='both')
    
    ax.set_ylim([0.0, 0.35])
    ax.set_xlim([-0.25, 1.5])
    
    fig.colorbar(cntr, ax=ax)
    ax.set_xlabel( r"$x/c$",fontsize=17)
    ax.set_ylabel( r"$y/c$",fontsize=17)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    
def plot_contour_u(data1, vari1, data2 = None, vari2 = None, data3 = None, vari3 = None, ref_length = 1.0, filename = 'countour.png', vmin = -0.1, vmax = 1.3, nlevels = 15):
    fig, ax = plt.subplots(figsize=(10,3))
    
    ax.plot(data1.xprof/ref_length,data1.yprof/ref_length,linewidth=2.,color='gray')
    
    #plt.contour(data1.xi/ref_length,data1.yi/ref_length,np.linspace(vmin,vmax,13),linewidths=0.5,colors='k', linestyles = 'solid', vmin = vmin, vmax = vmax)
    if data2 is not None:
        plt.contour(data2.xi/ref_length,data2.yi/ref_length,vari2,np.linspace(vmin,vmax,nlevels),linewidths=0.5,colors='k', linestyles = 'dashed', vmin = vmin, vmax = vmax)
    if data3 is not None:
        plt.contour(data3.xi/ref_length,data3.yi/ref_length,vari3,np.linspace(vmin,vmax,nlevels),linewidths=0.5,colors='k', linestyles = 'dashdot', vmin = vmin, vmax = vmax)
        
    cntr = ax.contourf(data1.xi/ref_length,data1.yi/ref_length,vari1, np.linspace(vmin,vmax,nlevels), cmap="jet", vmin = vmin, vmax = vmax,extend='both')
    
    ax.set_ylim([0.0, 0.45])
    ax.set_xlim([-0.25, 1.5])
    
    fig.colorbar(cntr, ax=ax)
    ax.set_xlabel( r"$x/c$",fontsize=17)
    ax.set_ylabel( r"$y/c$",fontsize=17)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    
def plot_contour_v(data1, vari1, data2 = None, vari2 = None, data3 = None, vari3 = None, ref_length = 1.0, filename = 'countour.png', vmin = -0.12, vmax = 0.12, nlevels = 13):
    fig, ax = plt.subplots(figsize=(10,3))
    
    ax.plot(data1.xprof/ref_length,data1.yprof/ref_length,linewidth=2.,color='gray')
    
    #plt.contour(data1.xi/ref_length,data1.yi/ref_length,np.linspace(vmin,vmax,13),linewidths=0.5,colors='k', linestyles = 'solid', vmin = vmin, vmax = vmax)
    if data2 is not None:
        plt.contour(data2.xi/ref_length,data2.yi/ref_length,vari2,np.linspace(vmin,vmax,nlevels),linewidths=0.5,colors='k', linestyles = 'dashed', vmin = vmin, vmax = vmax)
    if data3 is not None:
        plt.contour(data3.xi/ref_length,data3.yi/ref_length,vari3,np.linspace(vmin,vmax,nlevels),linewidths=0.5,colors='k', linestyles = 'dashdot', vmin = vmin, vmax = vmax)
        
    cntr = ax.contourf(data1.xi/ref_length,data1.yi/ref_length,vari1, np.linspace(vmin,vmax,nlevels), cmap="jet", vmin = vmin, vmax = vmax,extend='both')
    
    ax.set_ylim([0.0, 0.45])
    ax.set_xlim([-0.25, 1.5])
    
    fig.colorbar(cntr, ax=ax)
    ax.set_xlabel( r"$x/c$",fontsize=17)
    ax.set_ylabel( r"$y/c$",fontsize=17)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    
def profiles(data, vari, station, norm = 1.0, ref_length = 1.0):
    x0 = []
    y0 = []
    min_index, min_value = min(enumerate(abs(data.xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
    for j in range (0,len(data.yi)):
      if ( data.yi[j] >= data.yprof[j]):
        y0.append(data.yi[j]/ref_length)
        x0.append(vari[j,I]/norm+station/ref_length)
    return x0, y0
    
def plot_profiles(stations, data1, vari1, data2 = None, vari2 = None, data3 = None, vari3 = None, norm = 1.0, ref_length = 1.0, xlabel = 'x', ylabel = 'y', filename = 'profiles.pdf'):
   
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(data1.xprof/ref_length,data1.yprof/ref_length,linewidth=1.5,color='gray')
    
    for station in stations:
        x1, y1 = profiles(data1, vari1, station, norm = norm, ref_length = ref_length)
        if data2 is not None:
            x2, y2 = profiles(data2, vari2, station, norm = norm, ref_length = ref_length)
        if data3 is not None:
            x3, y3 = profiles(data3, vari3, station, norm = norm, ref_length = ref_length)
        
        if station == stations[0]:
            ax.plot(x1,y1,color='k', linestyle = 'solid', label = data1.label)
            if data2 is not None:
                ax.plot(x2,y2,color='b', linestyle = 'dashed', label = data2.label)
            if data3 is not None:
                ax.plot(x3,y3,color='r', linestyle = 'dashdot', label = data3.label)
        else:
            ax.plot(x1,y1,color='k', linestyle = 'solid')
            if data2 is not None:
                ax.plot(x2,y2,color='b', linestyle = 'dashed')
            if data3 is not None:
                ax.plot(x3,y3,color='r', linestyle = 'dashdot')
                
    plt.legend(loc='upper left')
    
    #ax.set_xlim([0., 2.0])
    ax.set_xlim([0., 2.1])
    ax.set_ylim([0.0, 0.35])
    #ax.xaxis.set_ticks(np.linspace(-1, 10, 12))
    #ax.yaxis.set_ticks(np.linspace(0, 3, 4))
    ax.set_xlabel( xlabel,fontsize=18)
    ax.set_ylabel( ylabel,fontsize=18)
    fig.subplots_adjust(right=0.90)
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    
def plot_cp(data1, data2 = None, data3 = None, data4 = None, ref_length = 1.0, pdyn = 1.0, filename = 'cp.png'):
    
    fig, ax = plt.subplots()
    #ax.set_title('Pressure Coefficient for Bump h'+nummesh, fontsize=12)
    ax.plot(data1.xw/ref_length,data1.cp/pdyn, 'o', label = data1.label, \
             markerfacecolor = 'w', \
             markeredgecolor = 'r', \
             markevery = 5, \
             markersize = 4.)
    if data2 is not None:
        ax.plot(data2.xw/ref_length,data2.cp/pdyn, 'b-', label = data2.label)
    
    if data3 is not None:
        ax.plot(data3.xw/ref_length,data3.cp/pdyn, 'g--', label = data3.label)
        
    if data4 is not None:
        ax.plot(data4.xw/ref_length,data4.cp/pdyn, 'k--', label = data4.label)
        
    ax.legend(borderpad = 0.65)
    ax.set_ylabel(r'$C_p$ [-]')
    ax.set_xlabel(r'$x/c$ [-]')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim([-0.25,1.5])
    
    plt.show()
    fig.savefig(filename, bbox_inches='tight')
    
def plot_cf(data1, data2 = None, data3 = None, data4 = None, ref_length = 1.0 , pdyn = 1.0 , filename = 'cf.png'):
    fig, ax = plt.subplots()
    #ax.set_title('Friction Coefficient for Bump h'+nummesh, fontsize=12)
    ax.plot(data1.xw/ref_length,data1.cf/pdyn, 'o', label = data1.label, \
             markerfacecolor = 'w', \
             markeredgecolor = 'r', \
             markevery = 5, \
             markersize = 4.)
    if data2 is not None:
        ax.plot(data2.xw/ref_length,data2.cf/pdyn, 'b-', label = data2.label)
    
    if data3 is not None:
        ax.plot(data3.xw/ref_length,data3.cf/pdyn, 'g--', label = data3.label)
        
    if data4 is not None:
        ax.plot(data4.xw/ref_length,data4.cf/pdyn, 'k--', label = data4.label)
        
    ax.legend(borderpad = 0.65)
    ax.set_ylabel(r'$C_f$ [-]')
    ax.set_xlabel(r'$x/c$ [-]')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim([-0.25,1.5])
    
    plt.show()
    fig.savefig(filename, bbox_inches='tight')
