# computes trajectories on a Dedalus Chebyshev (z) / Fourier (x,y) grid

import h5py
from mpi4py import MPI
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


# cat old and new
# fix indices...should go to 3121 to 3128?
# why are shapes so different?


figure_path = './figures'
file_path = './statistics'

# circulation file to read:
initialization = 1 # 0 = start new locations, 1 = load previous locations 
N0r = 3149 # pertain to the file to load locations from
N1r = 3156
filename_initialization = file_path + '/circulation_%i_%i.h5' %(N0r,N1r)

# statistics file to read:
N0 = 3156
N1 = 3160
filename = file_path + '/statistics_%i_%i.h5' %(N0,N1)

# Fornberg (1998) algorithm constraints:
N = 6 # four point stencil
m = 0 # zeroth derivative

nz = 32 # upper grid point in z, truncation of the grid for reduced memory 

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
  #print(Nx)

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


def trajectories_parallel( xn , x , yn , y , zn , z , un , u , vn , v , wn , w , N , m , t , dt , Nt , Ns , n , counts , displs , n2 , counts2 , displs2 ):

 # broadcast N,m?
 # Broadcast the grid and (known) u,v,w fields:
 comm.Bcast(x, root=0)
 comm.Bcast(y, root=0)
 comm.Bcast(z, root=0)
 comm.Bcast(u, root=0)
 comm.Bcast(v, root=0)
 comm.Bcast(w, root=0)
 #comm.Bcast(N, root=0) # number of stencil points for Fornberg algorithm
 #comm.Bcast(m, root=0) # order of derivative for Fornberg algorithm

 # initialize chunks:
 if rank == 0:
  xnc = np.zeros([int(n/Nt),Nt]) # shape = Nchunk, Nt
  ync = np.zeros([int(n/Nt),Nt])
  znc = np.zeros([int(n/Nt),Nt])
  unc = np.zeros([int(n2)]) 
  vnc = np.zeros([int(n2)])
  wnc = np.zeros([int(n2)]) 
  #print(int(n/Nt),int(n2))
 else:
  xnc = np.empty([int(n/Nt),Nt], dtype=np.float64 ) # shape = Nchunk, Nt
  ync = np.empty([int(n/Nt),Nt], dtype=np.float64 )
  znc = np.empty([int(n/Nt),Nt], dtype=np.float64 )
  unc = np.empty([int(n2)], dtype=np.float64 ) 
  vnc = np.empty([int(n2)], dtype=np.float64 )
  wnc = np.empty([int(n2)], dtype=np.float64 ) 
 comm.Bcast(xnc, root=0)
 comm.Bcast(ync, root=0)
 comm.Bcast(znc, root=0)
 comm.Bcast(unc, root=0)
 comm.Bcast(vnc, root=0)
 comm.Bcast(wnc, root=0)

 # scatter variables
 comm.Scatterv([ xn , counts , displs , MPI.DOUBLE ], xnc, root = 0) 
 comm.Scatterv([ yn , counts , displs , MPI.DOUBLE ], ync, root = 0) 
 comm.Scatterv([ zn , counts , displs , MPI.DOUBLE ], znc, root = 0) 
 comm.Scatterv([ un , counts2 , displs2 , MPI.DOUBLE ], unc, root = 0) 
 comm.Scatterv([ vn , counts2 , displs2 , MPI.DOUBLE ], vnc, root = 0) 
 comm.Scatterv([ wn , counts2 , displs2 , MPI.DOUBLE ], wnc, root = 0) 

 # compute: 
 for p in range(0,nprocs): # divide the computation into chunks, per processor
  if rank == p:
   #print('rank = %i' %p)
   for j in range(0,Nt-1): # loop over time
    if rank == 0:
     print('step = %i' %(j+1))
    for i in range(0,int(Ns/nprocs)): # loop over points
     xnc[i,j+1] = dt*unc[i] + xnc[i,j] 
     ync[i,j+1] = dt*vnc[i] + ync[i,j] 
     znc[i,j+1] = dt*wnc[i] + znc[i,j]
     #if rank == 0:
     # print('point = %i' %i)     
      #print(u[j+1,:,:,:])
     #print(N,m)
     #print(np.shape(xnc),Nt,Ns)
     unc[i] = fornberg3d( xnc[i,j+1] , x , ync[i,j+1] , y , znc[i,j+1] , z , u[j+1,:,:,:] , N , m )
     vnc[i] = fornberg3d( xnc[i,j+1] , x , ync[i,j+1] , y , znc[i,j+1] , z , v[j+1,:,:,:] , N , m )
     wnc[i] = fornberg3d( xnc[i,j+1] , x , ync[i,j+1] , y , znc[i,j+1] , z , w[j+1,:,:,:] , N , m )

 # gather xn,yn,zn values into x,y,z: 
 if rank == 0:
  x0 = np.zeros([Ns,Nt], dtype=np.float64)
  y0 = np.zeros([Ns,Nt], dtype=np.float64)
  z0 = np.zeros([Ns,Nt], dtype=np.float64)
 else:
  x0 = None
  y0 = None
  z0 = None
 comm.Gatherv(xnc, [x0, counts, displs, MPI.DOUBLE], root = 0)
 comm.Gatherv(ync, [y0, counts, displs, MPI.DOUBLE], root = 0)
 comm.Gatherv(znc, [z0, counts, displs, MPI.DOUBLE], root = 0)

 return x0,y0,z0


def circulation_parallel( xn , x , yn , y , zn , z , un , u , vn , v , wn , w , fxn , fx , fyn , fy , fzn , fz , N , m , t , dt , Nt , Ns , n , counts , displs , n2 , counts2 , displs2 ):
 # xn  is the computed locations
 # x is the grid
 #print(np.shape(u))

 # broadcast N,m?
 # Broadcast the grid and (known) u,v,w fields:
 """
 comm.Bcast(x, root=0)
 comm.Bcast(y, root=0)
 comm.Bcast(z, root=0)
 comm.Bcast(u, root=0)
 comm.Bcast(v, root=0)
 comm.Bcast(w, root=0)
 comm.Bcast(fx, root=0)
 comm.Bcast(fy, root=0)
 comm.Bcast(fz, root=0)
 """
 #comm.Bcast(N, root=0) # number of stencil points for Fornberg algorithm
 #comm.Bcast(m, root=0) # order of derivative for Fornberg algorithm

 # initialize chunks:
 if rank == 0:
  xnc = np.zeros([int(n/Nt),Nt]) # shape = Nchunk, Nt
  ync = np.zeros([int(n/Nt),Nt])
  znc = np.zeros([int(n/Nt),Nt])
  fxnc = np.zeros([int(n/Nt),Nt]) # shape = Nchunk, Nt
  fync = np.zeros([int(n/Nt),Nt])
  fznc = np.zeros([int(n/Nt),Nt])
  unc = np.zeros([int(n2)]) 
  vnc = np.zeros([int(n2)])
  wnc = np.zeros([int(n2)]) 
  #print(int(n/Nt),int(n2))
 else:
  xnc = np.empty([int(n/Nt),Nt], dtype=np.float64 ) # shape = Nchunk, Nt
  ync = np.empty([int(n/Nt),Nt], dtype=np.float64 )
  znc = np.empty([int(n/Nt),Nt], dtype=np.float64 )
  fxnc = np.empty([int(n/Nt),Nt], dtype=np.float64 ) # shape = Nchunk, Nt
  fync = np.empty([int(n/Nt),Nt], dtype=np.float64 )
  fznc = np.empty([int(n/Nt),Nt], dtype=np.float64 )
  unc = np.empty([int(n2)], dtype=np.float64 ) 
  vnc = np.empty([int(n2)], dtype=np.float64 )
  wnc = np.empty([int(n2)], dtype=np.float64 )  
 comm.Bcast(xnc, root=0)
 comm.Bcast(ync, root=0)
 comm.Bcast(znc, root=0)
 comm.Bcast(unc, root=0)
 comm.Bcast(vnc, root=0)
 comm.Bcast(wnc, root=0)
 comm.Bcast(fxnc, root=0)
 comm.Bcast(fync, root=0)
 comm.Bcast(fznc, root=0)

 # scatter variables into chunks (the variables that are computed)
 comm.Scatterv([ xn , counts , displs , MPI.DOUBLE ], xnc, root = 0) 
 comm.Scatterv([ yn , counts , displs , MPI.DOUBLE ], ync, root = 0) 
 comm.Scatterv([ zn , counts , displs , MPI.DOUBLE ], znc, root = 0) 
 comm.Scatterv([ fxn , counts , displs , MPI.DOUBLE ], fxnc, root = 0) 
 comm.Scatterv([ fyn , counts , displs , MPI.DOUBLE ], fync, root = 0) 
 comm.Scatterv([ fzn , counts , displs , MPI.DOUBLE ], fznc, root = 0) 
 comm.Scatterv([ un , counts2 , displs2 , MPI.DOUBLE ], unc, root = 0) 
 comm.Scatterv([ vn , counts2 , displs2 , MPI.DOUBLE ], vnc, root = 0) 
 comm.Scatterv([ wn , counts2 , displs2 , MPI.DOUBLE ], wnc, root = 0) 

 # compute: 
 for p in range(0,nprocs): # divide the computation into chunks, per processor
  if rank == p:
   #print('rank = %i' %p)
   for j in range(0,Nt-1): # loop over time
    if rank == 0:
     print('step = %i' %(j+1))
    for i in range(0,int(Ns/nprocs)): # loop over points

     # new locations
     xnc[i,j+1] = dt*unc[i] + xnc[i,j] 
     ync[i,j+1] = dt*vnc[i] + ync[i,j] 
     znc[i,j+1] = dt*wnc[i] + znc[i,j]
     #if rank == 0:
     # print('point = %i' %i)     
      #print(u[j+1,:,:,:])
     #print(N,m)
     #print(np.shape(xnc),Nt,Ns)

     # velocities at the new locations
     unc[i] = fornberg3d( xnc[i,j+1] , x , ync[i,j+1] , y , znc[i,j+1] , z , u[j+1,:,:,:] , N , m )
     vnc[i] = fornberg3d( xnc[i,j+1] , x , ync[i,j+1] , y , znc[i,j+1] , z , v[j+1,:,:,:] , N , m )
     wnc[i] = fornberg3d( xnc[i,j+1] , x , ync[i,j+1] , y , znc[i,j+1] , z , w[j+1,:,:,:] , N , m )

     # vector function at the new locations
     fxnc[i,j+1] = fornberg3d( xnc[i,j+1] , x , ync[i,j+1] , y , znc[i,j+1] , z , fx[j+1,:,:,:] , N , m )
     fync[i,j+1] = fornberg3d( xnc[i,j+1] , x , ync[i,j+1] , y , znc[i,j+1] , z , fy[j+1,:,:,:] , N , m )
     fznc[i,j+1] = fornberg3d( xnc[i,j+1] , x , ync[i,j+1] , y , znc[i,j+1] , z , fz[j+1,:,:,:] , N , m )

 # gather xn,yn,zn values into x,y,z: 
 if rank == 0:
  x0 = np.zeros([Ns,Nt], dtype=np.float64)
  y0 = np.zeros([Ns,Nt], dtype=np.float64)
  z0 = np.zeros([Ns,Nt], dtype=np.float64)
  fx0 = np.zeros([Ns,Nt], dtype=np.float64)
  fy0 = np.zeros([Ns,Nt], dtype=np.float64)
  fz0 = np.zeros([Ns,Nt], dtype=np.float64) 
 else:
  x0 = None
  y0 = None
  z0 = None
  fx0 = None
  fy0 = None
  fz0 = None
 comm.Gatherv(xnc, [x0, counts, displs, MPI.DOUBLE], root = 0)
 comm.Gatherv(ync, [y0, counts, displs, MPI.DOUBLE], root = 0)
 comm.Gatherv(znc, [z0, counts, displs, MPI.DOUBLE], root = 0)
 comm.Gatherv(fxnc, [fx0, counts, displs, MPI.DOUBLE], root = 0)
 comm.Gatherv(fync, [fy0, counts, displs, MPI.DOUBLE], root = 0)
 comm.Gatherv(fznc, [fz0, counts, displs, MPI.DOUBLE], root = 0)

 return x0,y0,z0,fx0,fy0,fz0



def trapezoid_vec( fx , fy , fz , x , y , z ):
 # non-uniform trapezoidal rule
 # f = (scalar) function at midpoint, length Ns+1
 # ds = increments, length Ns

 Np = np.shape(x)[0]-1

 I = 0.
 for i in range(0,Np):
  I = (fx[i+1]+fx[i])/2.*(x[i+1]-x[i]) + (fy[i+1]+fy[i])/2.*(y[i+1]-y[i]) + (fz[i+1]+fz[i])/2.*(z[i+1]-z[i]) + I
 
 return I


# =============================================================================

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank() # [0,1,2,...,nprocs-1]


# =============================================================================
# read grid and time series information from statistics file

f = h5py.File(filename, 'r')

# read grid infomration:
x = f['/x'][:]
y = f['/y'][:]
z = f['/z'][:]
Nx = np.shape(x)[0]
Ny = np.shape(y)[0]
Nz = np.shape(z)[0]
Z = f['/Z'][:]
Y = f['/Y'][:]
X = f['/X'][:]
H = (f['/H']).value
Lx = (f['/Lx']).value
Ly = (f['/Ly']).value
#z_Chb = np.cos((np.linspace(1., Nz, num=Nz)*2.-1.)/(2.*Nz)*np.pi)
tht = (f['/tht']).value
nu = (f['/nu']).value
BVf = (f['/N']).value
omg = (f['/omg']).value
thtc = (f['/thtc']).value
Uw = (f['/Uw']).value
kap = (f['/kap']).value

 # time series information
Nt = int((f['/Nt']).value)
Ttotal = (f['/Tt']).value # total time period divided by tide period
P = (f['/T']).value # tide period
t = f['/t'][:] # s
Tfinal = Ttotal*P
dt = t[1]-t[0]
if rank == 0:
 print(t/P)

# =============================================================================
# read initial velocities from statistics file



"""
# input known velocity fields (entire array, corresponds to the analytical solution):
u0 = 0.001 # m/s
v0 = 0.001 # m/s
k = 2.*np.pi/Lx*5. # wavenumber in x

c = 2.*np.pi/Tfinal*2./k
Xk = np.tile( X[...,None],Nt ) 
T = np.tile(np.tile(np.tile(t[...,None],Ny)[...,None],Nx)[...,None],Nz)
#print(sum(sum(sum(X-Xk[:,:,:,0]))))
u = np.ones(np.shape(T))*u0 # shape = Nt, Ny, Nx, Nz
w = np.zeros(np.shape(T))
v = np.zeros([Nt, Ny, Nx, Nz])
for j in range(0,Nt):
 v[j,:,:,:] = v0*np.cos(k*(Xk[:,:,:,j]-c*T[j,:,:,:])) # shape = Nt, Ny, Nx, Nz

# import data:
fx = u #np.ones(np.shape(T))
fy = v #np.zeros(np.shape(T))
fz = w #np.zeros(np.shape(T))
"""

# import velocities:
u = f['/u'][:]
v = f['/v'][:]
w = f['/w'][:]

# baroclinic diffusive vorticity components for integration:
fx = f['/b'][:]*np.sin(tht) + nu*f['/d2u'][:] 
fy = nu*f['/d2v'][:]
fz = f['/b'][:]*np.cos(tht) + nu*f['/d2w'][:] 

# truncating the fields for lower memory load: 
u = u[:,:,:,0:nz]
v = v[:,:,:,0:nz]
w = w[:,:,:,0:nz]
fx = fx[:,:,:,0:nz]
fy = fy[:,:,:,0:nz]
fz = fz[:,:,:,0:nz]
z = z[0:nz]


# =============================================================================
# read initial locations from circulation file or make them up

if initialization == 0:

 # initial locations and velocities at those locations:
 (x0c,y0c,z0c) = (X[64,8,3],Y[64,8,3],Z[64,8,3]) # shape = Ny, Nx, Nz
 if rank == 0:
  print('Starting from a circle, centered at:')
  print('x0/Lx = %.3f, y0/Ly = %.3f, z0/H = %.3f' %(x0c/Lx,y0c/Ly,z0c/H))
 r = 0.125 # m, radius
 #tht = np.arange(0.,384,24.)*np.pi/180. # Ns = 16 points
 tht = np.linspace( 0. , 360. , num=100 , endpoint=True )*np.pi/180.
 #C0 = 2.*np.pi*r # m, true circumference at t=0
 Ns = np.shape(tht)[0]

 if rank == 0:
  print('Ns*Nt = %.f must be divisible by %i' %(Ns*Nt,nprocs) )
  print('Ns = %.f must be divisible by %i' %(Ns,nprocs) )

if initialization != 0:
 
 #if rank == 0:

  # load initial locations
 fi = h5py.File(filename_initialization, 'r') # previous file

 x0ri = fi['/xf'][:] # [Ns,Nt]
 y0ri = fi['/yf'][:]
 z0ri = fi['/zf'][:]
 fx0ri = fi['/fxf'][:]
 fy0ri = fi['/fyf'][:]
 fz0ri = fi['/fzf'][:]
 gami = fi['/gam'][:]
 ti = fi['/t'][:]
 if rank == 0:
  print(gami)
  print(gami[int(np.shape(x0ri)[1]-1)])
  print(ti/P)
 Ns = np.shape(x0ri)[0]
 Nend = np.shape(x0ri)[1]-1 
 x0r = x0ri[:,Nend] # [Ns]
 y0r = y0ri[:,Nend]
 z0r = z0ri[:,Nend]
 fx0r = fx0ri[:,Nend]
 fy0r = fy0ri[:,Nend]
 fz0r = fz0ri[:,Nend]

 if rank == 0:
  plotname = figure_path +'/gamma0.png' 
  plottitle = r"$\Gamma$"
  fig = plt.figure()  
  plt.plot(ti/P,gami,'-*b',linewidth=2) #,label='computed');
  plt.xlabel(r"time/period",family='serif',fontsize='16'); 
  plt.ylabel(r"$m^2s^{-1}$",family='serif',fontsize='16'); 
  plt.title(plottitle,family='serif',fontsize='20');  
  plt.savefig(plotname,format="png"); plt.close(fig);

 #if rank == 1:
 # print(x0r)
 # print(np.shape(x0r))
  #comm.Bcast(x0r, root=0)

n = int(Ns*Nt/nprocs) # number of elements in a space-time chunk
counts = np.ones([nprocs])*n # array describing how many elements to send to each process
displs = np.linspace(0,Ns*Nt-n,nprocs,dtype=int) # array describing the displacements where each segment begins

n2 = int(Ns/nprocs) # number of elements in a space chunk
counts2 = np.ones([nprocs])*n2 # array describing how many elements to send to each process
displs2 = np.linspace(0,Ns-n2,nprocs,dtype=int) # array describing the displacements where each segment begins

if rank == 0:
 xn = np.zeros([Ns,Nt]) 
 yn = np.zeros([Ns,Nt])
 zn = np.zeros([Ns,Nt])
 fxn = np.zeros([Ns,Nt]) 
 fyn = np.zeros([Ns,Nt])
 fzn = np.zeros([Ns,Nt])
 un = np.zeros([Ns]) 
 vn = np.zeros([Ns])
 wn = np.zeros([Ns])

else:
 xn = None
 yn = None
 zn = None
 fxn = None
 fyn = None
 fzn = None
 un = None
 vn = None
 wn = None

# broadcast:
if rank == 0:
 xnc = np.zeros([int(n/Nt),Nt]) # shape = Nchunk, Nt
 ync = np.zeros([int(n/Nt),Nt])
 znc = np.zeros([int(n/Nt),Nt])
 fxnc = np.zeros([int(n/Nt),Nt]) # shape = Nchunk, Nt
 fync = np.zeros([int(n/Nt),Nt])
 fznc = np.zeros([int(n/Nt),Nt])
 unc = np.zeros([int(n2)]) 
 vnc = np.zeros([int(n2)])
 wnc = np.zeros([int(n2)]) 
else:
 xnc = np.empty([int(n/Nt),Nt], dtype=np.float64 ) # shape = Nchunk, Nt
 ync = np.empty([int(n/Nt),Nt], dtype=np.float64 )
 znc = np.empty([int(n/Nt),Nt], dtype=np.float64 )
 fxnc = np.empty([int(n/Nt),Nt], dtype=np.float64 ) # shape = Nchunk, Nt
 fync = np.empty([int(n/Nt),Nt], dtype=np.float64 )
 fznc = np.empty([int(n/Nt),Nt], dtype=np.float64 )
 unc = np.empty([int(n2)], dtype=np.float64 ) 
 vnc = np.empty([int(n2)], dtype=np.float64 )
 wnc = np.empty([int(n2)], dtype=np.float64 ) 

comm.Bcast(xnc, root=0)
comm.Bcast(ync, root=0)
comm.Bcast(znc, root=0)
comm.Bcast(fxnc, root=0)
comm.Bcast(fync, root=0)
comm.Bcast(fznc, root=0)
comm.Bcast(unc, root=0)
comm.Bcast(vnc, root=0)
comm.Bcast(wnc, root=0)

"""
if initialization != 0:
 comm.Bcast(x0r, root=0)
 comm.Bcast(y0r, root=0)
 comm.Bcast(z0r, root=0)
 comm.Bcast(fx0r, root=0)
 comm.Bcast(fy0r, root=0)
 comm.Bcast(fz0r, root=0)
"""

comm.Scatterv( [ xn , counts , displs , MPI.DOUBLE ] , xnc, root = 0) # scattering zeros...
comm.Scatterv( [ yn , counts , displs , MPI.DOUBLE ] , ync, root = 0) 
comm.Scatterv( [ zn , counts , displs , MPI.DOUBLE ] , znc, root = 0) 
comm.Scatterv( [ fxn , counts , displs , MPI.DOUBLE ] , fxnc, root = 0) # scattering zeros...
comm.Scatterv( [ fyn , counts , displs , MPI.DOUBLE ] , fync, root = 0) 
comm.Scatterv( [ fzn , counts , displs , MPI.DOUBLE ] , fznc, root = 0) 
comm.Scatterv( [ un , counts2 , displs2 , MPI.DOUBLE ] , unc, root = 0)  
comm.Scatterv( [ vn , counts2 , displs2 , MPI.DOUBLE ] , vnc, root = 0) 
comm.Scatterv( [ wn , counts2 , displs2 , MPI.DOUBLE ] , wnc, root = 0) 


# 2) compute new xn,yn,zn,un,vn,zn values in chunks: 

for p in range(0,nprocs):
 if rank == p:
  for i in range(0,int(Ns/nprocs)):
   #print(i+p*int(Ns/nprocs))
   if initialization == 0:
    xnc[i,0] = x0c + r*np.cos(tht[i+p*int(Ns/nprocs)])
    ync[i,0] = y0c + r*np.sin(tht[i+p*int(Ns/nprocs)]) #0,i
    znc[i,0] = z0c + tht[i]*0.
    unc[i] = fornberg3d( xnc[i,0] , x , ync[i,0] , y , znc[i,0] , z , u[0,:,:,:] , N , m )
    vnc[i] = fornberg3d( xnc[i,0] , x , ync[i,0] , y , znc[i,0] , z , v[0,:,:,:] , N , m )
    wnc[i] = fornberg3d( xnc[i,0] , x , ync[i,0] , y , znc[i,0] , z , w[0,:,:,:] , N , m )

    # vector function at the new locations
    fxnc[i,0] = fornberg3d( xnc[i,0] , x , ync[i,0] , y , znc[i,0] , z , fx[0,:,:,:] , N , m )
    fync[i,0] = fornberg3d( xnc[i,0] , x , ync[i,0] , y , znc[i,0] , z , fy[0,:,:,:] , N , m )
    fznc[i,0] = fornberg3d( xnc[i,0] , x , ync[i,0] , y , znc[i,0] , z , fz[0,:,:,:] , N , m )
   
   if initialization != 0:
    xnc[i,0] = x0r[i+p*int(Ns/nprocs)]
    ync[i,0] = y0r[i+p*int(Ns/nprocs)]
    znc[i,0] = z0r[i+p*int(Ns/nprocs)]
    unc[i] = fornberg3d( xnc[i,0] , x , ync[i,0] , y , znc[i,0] , z , u[0,:,:,:] , N , m )
    vnc[i] = fornberg3d( xnc[i,0] , x , ync[i,0] , y , znc[i,0] , z , v[0,:,:,:] , N , m )
    wnc[i] = fornberg3d( xnc[i,0] , x , ync[i,0] , y , znc[i,0] , z , w[0,:,:,:] , N , m )
    fxnc[i,0] = fx0r[i+p*int(Ns/nprocs)]
    fync[i,0] = fy0r[i+p*int(Ns/nprocs)]
    fznc[i,0] = fz0r[i+p*int(Ns/nprocs)]

# 3) gather xn,yn,zn,un,vn,zn values: 

if rank == 0:
 x0 = np.zeros([Ns,Nt], dtype=np.float64)
 y0 = np.zeros([Ns,Nt], dtype=np.float64)
 z0 = np.zeros([Ns,Nt], dtype=np.float64)
 fx0 = np.zeros([Ns,Nt], dtype=np.float64)
 fy0 = np.zeros([Ns,Nt], dtype=np.float64)
 fz0 = np.zeros([Ns,Nt], dtype=np.float64)
 u0 = np.zeros([Ns], dtype=np.float64)
 v0 = np.zeros([Ns], dtype=np.float64)
 w0 = np.zeros([Ns], dtype=np.float64)

else:
 x0 = None
 y0 = None
 z0 = None
 fx0 = None
 fy0 = None
 fz0 = None
 u0 = None
 v0 = None
 w0 = None

comm.Gatherv( xnc , [ x0, counts, displs, MPI.DOUBLE ] , root = 0)
comm.Gatherv( ync , [ y0, counts, displs, MPI.DOUBLE ] , root = 0)
comm.Gatherv( znc , [ z0, counts, displs, MPI.DOUBLE ] , root = 0)
comm.Gatherv( fxnc , [ fx0, counts, displs, MPI.DOUBLE ] , root = 0)
comm.Gatherv( fync , [ fy0, counts, displs, MPI.DOUBLE ] , root = 0)
comm.Gatherv( fznc , [ fz0, counts, displs, MPI.DOUBLE ] , root = 0)
comm.Gatherv( unc , [ u0, counts2, displs2, MPI.DOUBLE ] , root = 0)
comm.Gatherv( vnc , [ v0, counts2, displs2, MPI.DOUBLE ] , root = 0)
comm.Gatherv( wnc , [ w0, counts2, displs2, MPI.DOUBLE ] , root = 0)

# 4) plot:

if rank == 0:

 gam0 = trapezoid_vec( fx0[:,0] , fy0[:,0] , fz0[:,0] , x0[:,0] , y0[:,0] , z0[:,0] )
 print('initial gamma = %.14f' %(gam0))
 if initialization != 0:
  print('last gamma value from previous file = %.14f' %(gami[Nend]))

 # plot the circle:
 #print(x[:,0])
 #print(np.nonzero(x))
 """
 plotname = figure_path +'/circle.png' 
 plottitle = '(x0,y0,z0)=(%.2f,%.2f,%.2f)' %(x0c,y0c,z0c)
 fig = plt.figure()  
 plt.plot(x0[:,0],y0[:,0],'b');  #0,:
 plt.xlabel('x'); #plt.legend(loc=1); 
 plt.ylabel('y'); 
 plt.axis('tight') #[5.,15.,-1.5e-9,1.5e-9])
 plt.title(plottitle);  
 plt.savefig(plotname,format='png'); plt.close(fig);
 """

# end of initialization

# =============================================================================



(xf,yf,zf,fxf,fyf,fzf) = circulation_parallel( x0 , x , y0 , y , z0 , z , u0 , u , v0 , v , w0 , w , fx0 , fx , fy0 , fy , fz0 , fz , N , m , t , dt , Nt , Ns , n , counts , displs , n2 , counts2 , displs2 )

# cat these two:
#if rank == 0:
"""
 if initialization != 0:
  print(np.shape(xf))
  print(np.shape(x0ri))
  xf = np.concatenate((xf,x0ri),axis=1)
  yf = np.concatenate((yf,y0ri),axis=1)
  zf = np.concatenate((zf,z0ri),axis=1)
"""

# =============================================================================



if rank == 0:
 """
 print(np.shape(xf))
 print(np.shape(yf))
 print(np.shape(zf))
 print(np.shape(fxf))
 print(np.shape(fyf))
 print(np.shape(fzf))
 """

 gam = np.zeros([Nt]) 

 for j in range(0,Nt):

  gam[j] = trapezoid_vec( fxf[:,j] , fyf[:,j] , fzf[:,j] , xf[:,j] , yf[:,j] , zf[:,j] )
  
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
  plottitle = 't/T = %.3f ' %(t[j]/P)
  fig = plt.figure()  
  ax = fig.add_subplot(111, projection='3d') 
  ax.plot(xf[:,j],yf[:,j], zf[:,j], color='b') #, marker='o')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_xlim(-5., 2.); ax.set_ylim(-1., 1.); ax.set_zlim(0., 0.1);
  #plt.axis([-5.,-2.,-1.,1.]) 
  plt.title(plottitle);  
  plt.savefig(plotname,format='png'); plt.close(fig);
 
 #x0ri = x0ri[:,0:Nend-1] 
 #y0ri = y0ri[:,0:Nend-1] 
 #z0ri = z0ri[:,0:Nend-1]
 #fx0ri = fx0ri[:,0:Nend-1] 
 #fy0ri = fy0ri[:,0:Nend-1] 
 #fz0ri = fz0ri[:,0:Nend-1] 
 #ti = ti[0:Nend-1] 
 #gami = gami[0:Nend-1]
 if initialization != 0:
  gam = np.concatenate((gami,gam[1:int(np.shape(gam)[0]-1)]),0)
  t = np.concatenate((ti,t[1:int(np.shape(t)[0]-1)]),0)
  xf = np.concatenate((x0ri,xf[:,1:int(np.shape(xf)[1]-1)]),1)
  yf = np.concatenate((y0ri,yf[:,1:int(np.shape(yf)[1]-1)]),1)
  zf = np.concatenate((z0ri,zf[:,1:int(np.shape(zf)[1]-1)]),1)
  fxf = np.concatenate((fx0ri,fxf[:,1:int(np.shape(fxf)[1]-1)]),1)
  fyf = np.concatenate((fy0ri,fyf[:,1:int(np.shape(fyf)[1]-1)]),1)
  fzf = np.concatenate((fz0ri,fzf[:,1:int(np.shape(fzf)[1]-1)]),1)
  Nt = np.shape(t)[0]
 
 for j in range(0,Nt):

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
  plottitle = 't/T = %.3f ' %(t[j]/P)
  fig = plt.figure()  
  ax = fig.add_subplot(111, projection='3d') 
  ax.plot(xf[:,j],yf[:,j], zf[:,j], color='b') #, marker='o')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_xlim(-5., 5.); ax.set_ylim(-1., 1.); ax.set_zlim(0., 0.6);
  #plt.axis([-5.,-2.,-1.,1.]) 
  plt.title(plottitle);  
  plt.savefig(plotname,format='png'); plt.close(fig);
  

 #print(gam)
 print(np.shape(gam))
 print(np.shape(t))
 # now plot I
 plotname = figure_path +'/gamma.png' 
 plottitle = r"$\partial\Gamma/\partial{t}$"
 fig = plt.figure()  
 #plt.plot(t,gam_true,'k', label='analytical'); 
 plt.plot(t/P,gam,'-*b',linewidth=2) #,label='computed');
 plt.xlabel(r"time/period",family='serif',fontsize='16'); 
 plt.ylabel(r"$m^2s^{-1}$",family='serif',fontsize='16'); 
 #plt.legend(loc=1);  
 #plt.axis('tight') #[5.,15.,-1.5e-9,1.5e-9])
 plt.title(plottitle,family='serif',fontsize='20');  
 plt.savefig(plotname,format="png"); plt.close(fig);

 # save 
 savename = file_path + '/circulation_%i_%i.h5' %(N0,N1) 
 f2 = h5py.File(savename, "w")

 # output file range information:
 dset = f2.create_dataset('N0', data=N0, dtype='f8')
 dset = f2.create_dataset('N1', data=N1, dtype='f8') 

 # time information:
 dset = f2.create_dataset('Nt', data=Nt, dtype='f8')
 dset = f2.create_dataset('P', data=P, dtype='f8')
 dset = f2.create_dataset('Ttotal', data=Ttotal, dtype='f8')
 dset = f2.create_dataset('t', data=t, dtype='f8') 

 # grid information:
 dset = f2.create_dataset('x', data=x, dtype='f8')
 dset = f2.create_dataset('y', data=y, dtype='f8')
 dset = f2.create_dataset('z', data=z, dtype='f8')
 dset = f2.create_dataset('X', data=X, dtype='f8')
 dset = f2.create_dataset('Y', data=Y, dtype='f8')
 dset = f2.create_dataset('Z', data=Z, dtype='f8')
 dset = f2.create_dataset('Lx', data=Lx, dtype='f8')
 dset = f2.create_dataset('Ly', data=Ly, dtype='f8')
 dset = f2.create_dataset('H', data=H, dtype='f8')

 # flow information
 dset = f2.create_dataset('BVf', data=BVf, dtype='f8')
 dset = f2.create_dataset('omg', data=omg, dtype='f8')
 dset = f2.create_dataset('thtc', data=thtc, dtype='f8')
 dset = f2.create_dataset('tht', data=tht, dtype='f8')
 dset = f2.create_dataset('kap', data=kap, dtype='f8')
 dset = f2.create_dataset('nu', data=nu, dtype='f8')
 dset = f2.create_dataset('Uw', data=Uw, dtype='f8')

 # circulation ingredients
 dset = f2.create_dataset('gam', data=gam, dtype='f8') 
 dset = f2.create_dataset('xf', data=xf, dtype='f8') 
 dset = f2.create_dataset('yf', data=yf, dtype='f8') 
 dset = f2.create_dataset('zf', data=zf, dtype='f8') 
 dset = f2.create_dataset('fxf', data=fxf, dtype='f8') 
 dset = f2.create_dataset('fyf', data=fyf, dtype='f8') 
 dset = f2.create_dataset('fzf', data=fzf, dtype='f8') 

 print('\nCirculation computed and written to file' + savename + '.\n')



