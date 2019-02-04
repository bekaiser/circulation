# computes trajectories on a Dedalus Chebyshev (z) / Fourier (x,y) grid

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

# make the trajectory calculation into a function
# velocity fields must 


# want a function that takes the x0,y0,z0 (which may be more than one 
# point) and a known velocity field and outputs the trajectories into 
# data files.


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


def trajectories( xn , x , yn , y , zn , z , un , u , vn , v , wn , w , N , m , t , dt , Nt , Ns, figure_path ):
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
  print('step = %i' %j)
  for i in range(0,Ns):
   # update Lagrangian locations using the previous steps' location and velocity field:
   xn[j+1,i] = dt*un[i] + xn[j,i] # un is at time t[j]
   yn[j+1,i] = dt*vn[i] + yn[j,i]
   zn[j+1,i] = dt*wn[i] + zn[j,i]
 
   # interpolate the velocites at time t[j+1] to the locations at time t[j+1]
   un[i] = fornberg3d( xn[j+1,i] , x , yn[j+1,i] , y , zn[j+1,i] , z , u[j+1,:,:,:] , N , m )
   vn[i] = fornberg3d( xn[j+1,i] , x , yn[j+1,i] , y , zn[j+1,i] , z , v[j+1,:,:,:] , N , m )
   wn[i] = fornberg3d( xn[j+1,i] , x , yn[j+1,i] , y , zn[j+1,i] , z , w[j+1,:,:,:] , N , m )
   
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
  ax.plot(xn[j,:],yn[j,:], zn[j,:], color='b') #, marker='o')
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

  plt.axis([-1.7,0.3,-0.25,0.25]) 
  plt.title(plottitle);  
  plt.savefig(plotname,format="png"); plt.close(fig);

 return xn,yn,zn



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
Tfinal = 8. #100.
Nt = int(Tfinal/dt)
t = np.linspace(0.0, Tfinal-dt, num=Nt)


# =============================================================================
# initial velocities at one point 

# Fornberg (1998) algorithm constraints
N = 4 # four point stencil
m = 0 # zeroth derivative

# input known velocity fields (entire array, corresponds to the analytical solution):
u0 = 0.001 # m/s
v0 = 0.001 # m/s
k = 2.*np.pi/Lx*10. # wavenumber in x
c = 2.*np.pi/(10*Tfinal)
Xk = np.tile( X[...,None],Nt ) 
T = np.tile(np.tile(np.tile(t[...,None],Ny)[...,None],Nx)[...,None],Nz)
#print(sum(sum(sum(X-Xk[:,:,:,0]))))
u = np.ones(np.shape(T))*u0 # shape = Nt, Ny, Nx, Nz
w = np.zeros(np.shape(T))
v = np.zeros([Nt, Ny, Nx, Nz])
for j in range(0,Nt):
 v[j,:,:,:] = v0*np.cos(k*(Xk[:,:,:,j]-c*T[j,:,:,:])) # shape = Nt, Ny, Nx, Nz


# initial locations and velocities at those locations:
(x0c,y0c,z0c) = (X[64,32,5],Y[64,32,5],Z[64,32,5]) # shape = Ny, Nx, Nz
r = 0.05 # m, radius
tht = np.arange(0.,384.,24.)*np.pi/180. # 12 points
C0 = 2.*np.pi*r # m, true circumference at t=0
Ns = np.shape(tht)[0]
xn = np.zeros([Nt,Ns]) 
yn = np.zeros([Nt,Ns])
zn = np.zeros([Nt,Ns])
un = np.zeros([Ns]) 
vn = np.zeros([Ns])
wn = np.zeros([Ns])
for i in range(0,Ns):
 xn[0,i] = x0c + r*np.cos(tht[i])
 yn[0,i] = y0c + r*np.sin(tht[i])
 zn[0,i] = z0c + tht[i]*0.
 un[i] = fornberg3d( xn[0,i] , x , yn[0,i] , y , zn[0,i] , z , u[0,:,:,:] , N , m )
 vn[i] = fornberg3d( xn[0,i] , x , yn[0,i] , y , zn[0,i] , z , v[0,:,:,:] , N , m )
 wn[i] = fornberg3d( xn[0,i] , x , yn[0,i] , y , zn[0,i] , z , w[0,:,:,:] , N , m )

print(un)
print(vn)
print(wn)

# plot the circle:
plotname = figure_path +'/circle.png' 
plottitle = '(x0,y0,z0)=(%.2f,%.2f,%.2f)' %(x0c,y0c,z0c)
fig = plt.figure()  
plt.plot(xn[0,:],yn[0,:],'b');  
plt.xlabel("x"); #plt.legend(loc=1); 
plt.ylabel("y"); 
plt.axis('tight') #[5.,15.,-1.5e-9,1.5e-9])
plt.title(plottitle);  
plt.savefig(plotname,format="png"); plt.close(fig);






(xn,yn,zn) = trajectories( xn , x , yn , y , zn , z , un , u , vn , v , wn , w , N , m , t , dt , Nt , Ns, figure_path )


print(xn[7,:])
print(yn[7,:])
print(zn[7,:])

