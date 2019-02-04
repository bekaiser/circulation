# 

import numpy as np
import math as ma
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fft, dct
from decimal import Decimal
from scipy.stats import chi2
from scipy import signal
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
from scipy.fftpack import fft, fftshift
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc

from mpl_toolkits.mplot3d import Axes3D


figure_path = './figures'


# =============================================================================
# functions


def weights(z,x,m):
# From Bengt Fornbergs (1998) SIAM Review paper.
#  	Input Parameters
#	z location where approximations are to be accurate,
#	x(0:nd) grid point locations, found in x(0:n)
#	n one less than total number of grid points; n must
#	not exceed the parameter nd below,
#	nd dimension of x- and c-arrays in calling program
#	x(0:nd) and c(0:nd,0:m), respectively,
#	m highest derivative for which weights are sought,
#	Output Parameter
#	c(0:nd,0:m) weights at grid locations x(0:n) for derivatives
#	of order 0:m, found in c(0:n,0:m)
#      	dimension x(0:nd),c(0:nd,0:m)

  n = np.shape(x)[0]-1
  c = np.zeros([n+1,m+1])
  c1 = 1.0
  c4 = x[0]-z
  for k in range(0,m+1):  
    for j in range(0,n+1): 
      c[j,k] = 0.0
  c[0,0] = 1.0
  for i in range(0,n+1):
    mn = min(i,m)
    c2 = 1.0
    c5 = c4
    c4 = x[i]-z
    for j in range(0,i):
      c3 = x[i]-x[j]
      c2 = c2*c3
      if (j == i-1):
        for k in range(mn,0,-1): 
          c[i,k] = c1*(k*c[i-1,k-1]-c5*c[i-1,k])/c2
      c[i,0] = -c1*c5*c[i-1,0]/c2
      for k in range(mn,0,-1):
        c[j,k] = (c4*c[j,k]-k*c[j,k-1])/c3
      c[j,0] = c4*c[j,0]/c3
    c1 = c2
  return c


def fornberg1d( xl , x , u , N , m ):
  # Input: the grid, the location to interpolate to, values to interpolate 
  # Output: the interpolated variable at the specified location.
  # N =  an even number of indices for the stencil
  # m = order of the derivative to be interpolated (0 = just interpolation)

  Nx = np.shape(u)[0] # u must be shape = [Nx,1]

  # ensure u is shape = [Nx,1]
  v = np.zeros([Nx,1])
  for j in range(0,Nx):
    v[j] = u[j]
  u = v

  # find the nearest N grid points to xl on x: 
  dist = np.zeros([Nx])
  for j in range(0,Nx):
    dist[j] = np.sqrt( (x[j] - xl)**2. )
  minloc = np.where(dist == np.amin(dist))[0][0]
  # add near-boundary option for stencil here!!!
  xint = x[minloc-int(N/2)+1:minloc+int(N/2)+1] # the nearest N grid points to xl
  # now interpolate with the Fornberg scheme     
  c = np.transpose(weights(xl,xint,N))
  uint = np.dot( c[m,:] , u[minloc-int(N/2)+1:minloc+int(N/2)+1,0] )

  return uint 


def fornberg3d( xl , x , yl , y , zl , z , u , N , m ):
  # Input: the grid, the location to interpolate to, values to interpolate 
  # Output: the interpolated variable at the specified location.
  # N =  an even number of indices for the stencil
  # m = order of the derivative to be interpolated (0 = just interpolation)
  
  #print(np.shape(u))
  Ny = np.shape(u)[0]
  Nx = np.shape(u)[1]
  Nz = np.shape(u)[2]

  # interpolate in x:
  ux = np.zeros([Ny,Nz])
  for i in range(0,Ny):
    for j in range(0,Nz):
      ux[i,j] = fornberg1d( xl , x , u[i,:,j] , N , m ) 

  # interpolate in y:
  uy = np.zeros([Nz])
  for j in range(0,Nz):
    uy[j] = fornberg1d( yl , y , ux[:,j] , N , m ) 

  # interpolate in z:
  uz = fornberg1d( zl , z , uy , N , m ) 

  return uz # interpolated to (xl,yl,zl)


def trajectory( xn , x , yn , y , zn , z , un , u , vn , v , wn , w , N , m , t , dt , Nt , figure_path ):
 # input:
 # xn,yn,zn = trajectory locations (initialized with x0,y0,z0 at time t[0] = 0), shape = [Nt]
 # un,vn,wn = initial velocities, to be rewritten as the scheme advances, shape = [1]
 # x,y,z,u,v,w = grid points & velocity fields, shape = 
 # N = number of points in interp stencil, Fornberg algorithm constant 
 # m = order of derivative (0th = interpolation) Fornberg algorithm constant 
 # dt = time step (must be constant)
 # t = time
 # Nt = length of time vector

 for j in range(0,Nt-1):
  print(j)
  # update Lagrangian locations using the previous steps' location and velocity field:
  xn[j+1] = dt*un + xn[j] # un is at time t[j]
  yn[j+1] = dt*vn + yn[j]
  zn[j+1] = dt*wn + zn[j]
 
  # interpolate the velocites at time t[j+1] to the locations at time t[j+1]
  un = fornberg3d( xn[j+1] , x , yn[j+1] , y , zn[j+1] , z , u[j+1,:,:,:] , N , m )
  vn = fornberg3d( xn[j+1] , x , yn[j+1] , y , zn[j+1] , z , v[j+1,:,:,:] , N , m )
  wn = fornberg3d( xn[j+1] , x , yn[j+1] , y , zn[j+1] , z , w[j+1,:,:,:] , N , m )

  # plot:
  plot_num = '%i' %j
  if j < 10:
    plotname = figure_path +'/00000' + plot_num +'.png' 
  elif j < 100:
    plotname = figure_path +'/0000' + plot_num +'.png' 
  elif j < 1000:
    plotname = figure_path +'/000' + plot_num +'.png' 
  elif j < 10000:
    plotname = figure_path +'/00' + plot_num +'.png' 
  elif j < 100000:
    plotname = figure_path +'/0' + plot_num +'.png' 
  elif j < 1000000:
    plotname = figure_path +'/' + plot_num +'.png'  

  #plottitle = 'trajectory, (x0,y0,z0)=(%.2f,%.2f,%.2f)' %(x0,y0,z0)
  plottitle = 't = %.1f s' %(t[j])
  fig = plt.figure()  
  ax = fig.add_subplot(111, projection='3d') 
  ax.plot(xn[0:j+1],yn[0:j+1], zn[0:j+1], color='b') #, marker='o')
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

  plt.axis([-2.8,-2.64,-0.01,0.035]) 
  plt.title(plottitle);  
  plt.savefig(plotname,format="png"); plt.close(fig);

 return xn,yn,zn


def trapezoid( f , ds ):
 # non-uniform trapezoidal rule
 # f = (scalar) function at midpoint, length Ns
 # ds = increments, length Ns-1

 Np = np.shape(ds)[0] # Ns-1

 I = 0.
 for i in range(0,Np):
  I = (f[i+1]+f[i])/2.*ds[i] + I
 
 return I

# add trapezoidal rule for vectors! fx,fy,fz dotted with dx,dy,dz

def trapezoid_vec( fx , fy , fz , x , y , z ):
 # non-uniform trapezoidal rule
 # f = (scalar) function at midpoint, length Ns

 Np = np.shape(x)[0]-1 # Ns-1

 I = 0.
 for i in range(0,Np): 
  I = (fx[i+1]+fx[i])/2.*(x[i+1]-x[i]) + (fy[i+1]+fy[i])/2.*(y[i+1]-y[i]) + (fz[i+1]+fz[i])/2.*(z[i+1]-z[i]) + I
 
 return I




# =============================================================================
# grid

Lx, Ly, H = (6., 6., 20.)

Nx = 128
Ny = 128
Nz = 128

# computational domain (identical to how dedalus builds)
dx = Lx/Nx; x = np.linspace(0.0, dx*Nx-dx, num=Nx) - Lx/2.
dy = Ly/Ny; y = np.linspace(0.0, dy*Ny-dy, num=Ny) - Ly/2. 
z = -np.cos(((np.linspace(1., Nz, num=Nz))*2.-1.)/(2.*Nz)*np.pi)*H/2.+H/2.
X,Y,Z = np.meshgrid(x,y,z) # shape = Ny, Nx, Nz

dt = 1. # s
Tfinal = 100.
Nt = int(Tfinal/dt)
t = np.linspace(0.0, Tfinal-dt, num=Nt)



# =============================================================================
# a circle on the grid

N = [1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8]
Nerr = np.shape(N)[0]
Cerror = np.zeros([Nerr])
Verror = np.zeros([Nerr])

Cerror_raw = np.zeros([Nerr])
Verror_raw = np.zeros([Nerr])

for q in range(0,Nerr):
 print(q)
 Ns = int(N[q])

 # circle center coordinates
 (x0,y0,z0) = (X[64,64,5],Y[64,64,5],Z[64,64,5]) # shape = Ny, Nx, Nz
 r = 0.5 # m, radius

 # 180 points on the sphere:
 #tht = np.arange(0.,360.,0.01)*np.pi/180.
 tht = np.linspace( 0. , 360. , num=Ns , endpoint=True )*np.pi/180.
 #Ns = np.shape(tht)[0]
 #print(Ns)
 #print(tht[Ns-1]*180./np.pi)
 x = x0 + r*np.cos(tht)
 y = y0 + r*np.sin(tht)
 z = z0 + tht*0.
 C = 2.*np.pi*r # m, true circumference
 #Ns = np.shape(tht)[0]


 # midpoint increments along the curve:
 ds = np.zeros([Ns-1])
 for j in range(0,Ns-1):
  ds[j] = np.sqrt( (x[j+1]-x[j])**2. + (y[j+1]-y[j])**2. + (z[j+1]-z[j])**2. )
 f = np.ones([Ns])

 # integrate:
 I = trapezoid( f , ds )
 #print(I,C)

 Cerror[q] = abs(I-C)/abs(C) 
 Cerror_raw[q] = I-C

 #print(Ns)
 #print(int(Ns/2))

 fx = f
 #fx[0:int(Ns/2)] = fx[0:int(Ns/2)]*0.
 fx[int(Ns/2)+1:Ns] = fx[int(Ns/2)+1:Ns]*0

 I = trapezoid_vec( fx ,  np.zeros([Ns]) ,  np.zeros([Ns]) , x , y , z )
 #print(I,C)
 
 Verror[q] = abs(1.+I)/abs(1.)
 Verror_raw[q] = 1.+I


plotname = figure_path +'/circumference_error.png' 
plottitle = r"relative error, $|C-C_{true}|/|C_{true}|$" 
fig = plt.figure()  
plt.loglog(N,Cerror,'k') #,label=r"x error");  
plt.xlabel(r"N number of grid points",family='serif',fontsize='16');
#plt.ylabel("error"); 
plt.axis('tight') #[5.,15.,-1.5e-9,1.5e-9])
plt.title(plottitle,family='serif',fontsize='16');  
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'/vector_error.png' 
plottitle = r"relative error, $|u-u_{true}|/|u_{true}|$" 
fig = plt.figure()  
plt.loglog(N,Verror,'k') #,label=r"x error");  
plt.xlabel(r"N number of grid points",family='serif',fontsize='16');  
#plt.ylabel("error"); 
plt.axis('tight') #[5.,15.,-1.5e-9,1.5e-9])
plt.title(plottitle,family='serif',fontsize='16');  
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'/circumference_error_raw.png' 
plottitle = r"relative error, $C-C_{true}$" 
fig = plt.figure()  
plt.semilogx(N,Cerror_raw,'k') #,label=r"x error");  
plt.xlabel(r"N number of grid points",family='serif',fontsize='16');
#plt.ylabel("error"); 
plt.axis('tight') #[5.,15.,-1.5e-9,1.5e-9])
plt.title(plottitle,family='serif',fontsize='16');  
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'/vector_error_raw.png' 
plottitle = r"relative error, $u-u_{true}$" 
fig = plt.figure()  
plt.semilogx(N,Verror_raw,'k') #,label=r"x error");  
plt.xlabel(r"N number of grid points",family='serif',fontsize='16');  
#plt.ylabel("error"); 
plt.axis('tight') #[5.,15.,-1.5e-9,1.5e-9])
plt.title(plottitle,family='serif',fontsize='16');  
plt.savefig(plotname,format="png"); plt.close(fig);
