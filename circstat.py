# statistics for circulation estimate
# Bryan Kaiser
# 7/28/18

# file/averaging parameters:
N0 = 3156 # first file to read
N1 = 3160 #3222 # last file to read

# slope angle
upsilon = 1./4.

# simulation path
sim_path = '.' #/sub3d_2'

import h5py
import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import sys
from scipy.fftpack import fft, dct
from decimal import Decimal
from scipy.stats import chi2
from scipy import signal
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
from scipy.fftpack import fft, fftshift
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc

file_path = sim_path + '/snapshots'
figure_path = sim_path + '/figures'
stats_path = sim_path + '/statistics'

# grid parameters:
Nx = 128 # grid points in x
Ny = 128 # grid points in y
Nz = 128 # grid points in z
Lx, Ly, H = (10., 10., 30.) # m, channel length and height z = [0,H]




# =============================================================================
# plot variables

cmap_1 = 'Purples_r'
cmap_2 = 'inferno' #'gist_gray'
cmap_3 = 'seismic'
Nc = 200
plot_size = (7.5,8.)

title_fontsize = '16'
xlabel_fontsize = '14'
ylabel_fontsize = '14'

ntop = 206 # top grid point for contour
nx = 64 
nz = 10
ny = 64

# =============================================================================
# functions

def cgq(u,z,Lz,Nz): # Chebyshev-Gauss Quadrature on a grid z = (0,Lz)
  w = np.pi/Nz # Chebyshev weight
  U = 0.
  for n in range(0,Nz-1):  
    U = w*u[Nz-1-n]*np.sqrt(1.-z[n]**2.) + U
  U = U*Lz/2.
  return U

def dy(u,k):
  U = np.fft.fft(u-np.mean(u))
  du = np.real(np.fft.ifft(np.multiply(U,k)*1j))
  return du

def pad_cheby_2d( u , axis_flag ):
  # zero-padding for 3/2 rule de-aliasing in the axis_flag direction
  Nz = np.shape(u)[0]; Ny = np.shape(u)[1]
  if int(axis_flag) == 0:
    Nzp = int(Nz*3/2);  #int(Nz*3/2+4);           # !!!! 
    Up = np.zeros([Nzp,Ny]); kp = np.zeros([Nzp]); 
    for j in range(0,Ny):
      Up[0:Nz,j] = scipy.fftpack.dct(u[:,j],2)/(2.*Nz) # DCT-II
  if int(axis_flag) == 1:
    Nyp = int(Ny*3/2); Up = np.zeros([Nz,Nyp]); kp = np.zeros([Nyp])
    for j in range(0,Nz):
      Up[j,0:Ny] = scipy.fftpack.dct(u[j,:],2)/(2.*Ny) # DCT-II
  return Up 
 
def pad_fourier_2d( u , axis_flag ):
  # zero-padding for 3/2 rule de-aliasing in the axis_flag direction
  Nz = np.shape(u)[0]; Ny = np.shape(u)[1] 
  if int(axis_flag) == 0:
    Nzp = int(Nz*3/2)
    Up = np.zeros([Nzp,Ny],dtype=complex)  
    Up[0:int(Nz/2+1),:] = np.fft.fft(u,axis=int(axis_flag))[0:int(Nz/2+1),:] 
    Up[int(Nzp)-int(Nz/2-1):int(Nzp),:] = np.fft.fft(u,axis=int(axis_flag))[int(Nz/2+1):Nz,:]
  if int(axis_flag) == 1:
    Nyp = int(Ny*3/2)
    Up = np.zeros([Nz,Nyp],dtype=complex)  
    Up[:,0:int(Ny/2+1)] = np.fft.fft(u,axis=int(axis_flag))[:,0:int(Ny/2+1)] 
    Up[:,int(Nyp)-int(Ny/2-1):int(Nyp)] = np.fft.fft(u,axis=int(axis_flag))[:,int(Ny/2+1):Ny]
  return Up 

def trunc_cheby_2d( Up , axis_flag ): 
  # remove 3/2 Chebyshev padding in the axis_flag direction
  if int(axis_flag) == 0: # padding in axis 0
    Nzp = np.shape(Up)[0]; Ny = np.shape(Up)[1]; 
    Nz = int(Nzp*2./3.); #int((Nzp-4)*2./3.);  # !!!! 
    u = np.zeros([Nz,Ny]) 
    for j in range(0,Ny):
      u[:,j] = scipy.fftpack.dct(Up[0:Nz,j],type=3,axis=axis_flag) # DCT-III in z direction
  if int(axis_flag) == 1: # padding in axis 1
    Nz = np.shape(Up)[0]; Nyp = np.shape(Up)[1]; Ny = int(Nyp*2./3.); u = np.zeros([Nz,Ny]); 
    for j in range(0,Nz):
      u[j,:] = scipy.fftpack.dct(Up[j,0:Ny],type=3,axis=axis_flag) # DCT-III in z direction
  return u

def trunc_fourier_2d( Up , axis_flag ):
  # remove 3/2 Fourier padding in the axis_flag direction
  if int(axis_flag) == 0: # padding in axis 0
    Nzp = np.shape(Up)[0]; Ny = np.shape(Up)[1]; Nz = int(Nzp*2./3.); u = np.zeros([Nz,Ny],dtype=complex)
    u[0:int(Nz/2+1),:] = Up[0:int(Nz/2+1),:]
    u[int(Nz/2+1):Nz,:] = Up[int(Nzp)-int(Nz/2-1):int(Nzp),:]
  if int(axis_flag) == 1: # padding in axis 1
    Nz = np.shape(Up)[0]; Nyp = np.shape(Up)[1]; Ny = int(Nyp*2./3.); u = np.zeros([Nz,Ny],dtype=complex)
    u[:,0:int(Ny/2+1)] = Up[:,0:int(Ny/2+1)]
    u[:,int(Ny/2+1):Ny] = Up[:,int(Nyp)-int(Ny/2-1):int(Nyp)]
    #u[:,int(Ny/2):Ny] = Up[:,int(Nyp)-int(Ny/2):int(Nyp)]
  return np.real(np.fft.ifft(u,axis=axis_flag))

def pad_fc_2d( u ):
  # 1) pad Fourier in axis=1
  UpF = pad_fourier_2d( u , 1 )
  upF = np.real(np.fft.ifft(UpF,axis=1)) 
  # 2) pad Chebyshev in axis=0
  UpFpC = pad_cheby_2d( upF , 0 )
  # 3) back to real space
  upFpC = scipy.fftpack.dct(UpFpC,type=3,axis=0) # DCT-III in z direction
  return upFpC

def trunc_fc_2d( U ):
  # 1) truncate Chebyshev
  upFC = trunc_cheby_2d( U , 0 ) 
  # 2) truncate Fourier
  UpFC = np.fft.fft( upFC , axis=1 ) # forward transform
  uFC = trunc_fourier_2d( UpFC , 1 )
  return uFC

def dealias( u , v , Nz ):
  # 1) pad
  up = pad_fc_2d( u ) 
  vp = pad_fc_2d( v ) 
  # 2) multiply in real space
  uvp = np.multiply( up , vp )
  # 3) truncate 
  uv = trunc_fc_2d( scipy.fftpack.dct(uvp,type=2,axis=0)/(2.*int(Nz*3./2.)) )*3./2.
  return uv

def dz( u , uL, uR, z, ze ):
  # interior point derivatives
  # first derivative by centered finite difference in the grid interior
  # u must be Nz in length (where Nz is the number of cell centers) 
  # z must be the cell edges (length Nz+1)
  Nu = np.shape(u)[0]

  # interpolate u to cell edges
  ue = interp1( u , uL, uR, Nu )
  #ue = np.zeros([Nu+1]) # interpolation to cell edges
  #ue[0] = uL # left BC
  #ue[Nu] = uR # right BC
  #for j in range(1,Nu):
  #  ue[j] = (ze[j]-z[j-1])/(z[j]-z[j-1])*(u[j]-u[j-1])+u[j-1] # 2nd-order

  u_z = np.zeros([Nu]) # cell-center 1st derivative
  #u_zz = np.zeros([Nu]) # cell-center 1st derivative
  for j in range(0,Nu):
    #h1 = z[j]-ze[j]
    #h2 = ze[j+1]-z[j]
    u_z[j] = (ue[j+1]-ue[j])/(ze[j+1]-ze[j]) # 2nd-order
    #u_zz[j] = ( h1*(ue[j+1]-u[j]) - h2*(u[j]-ue[j]) ) / (h1*h2*(h1+h2))

  return u_z #,u_zz

def interp1( u , uL, uR, Nu ):
  # interpolate u to cell edges
  ue = np.zeros([Nu+1]) # interpolation to cell edges
  ue[0] = uL # left BC
  ue[Nu] = uR # right BC
  for j in range(1,Nu):
    ue[j] = (ze[j]-z[j-1])/(z[j]-z[j-1])*(u[j]-u[j-1])+u[j-1] # 2nd-order
  return ue

def dzz( u , uL, uR, z, ze ):
  # interior point derivatives
  # first derivative by centered finite difference in the grid interior
  # u must be Nz in length (where Nz is the number of cell centers) 
  # z must be the cell edges (length Nz+1)
  Nu = np.shape(u)[0]

  # interpolate u to cell edges
  ue = interp1( u , uL, uR, Nu )
  #ue = np.zeros([Nu+1]) # interpolation to cell edges
  #ue[0] = uL # left BC
  #ue[Nu] = uR # right BC
  #for j in range(1,Nu):
  #  ue[j] = (ze[j]-z[j-1])/(z[j]-z[j-1])*(u[j]-u[j-1])+u[j-1] # 2nd-order

  #u_z = np.zeros([Nu]) # cell-center 1st derivative
  u_zz = np.zeros([Nu]) # cell-center 1st derivative
  for j in range(0,Nu):
    h1 = z[j]-ze[j]
    h2 = ze[j+1]-z[j]
    #u_z[j] = (ue[j+1]-ue[j])/(ze[j+1]-ze[j]) # 2nd-order
    u_zz[j] = ( h1*(ue[j+1]-u[j]) - h2*(u[j]-ue[j]) ) / (h1*h2*(h1+h2))

  return u_zz

# =============================================================================
# find time series length for wave averaging

# find the number of time steps in the interval N0 to N1:
count_time = 0
for n in range(N0,N1+1):
  filename = file_path +'/snapshots_s' +str(n) +'.h5'
  print(filename)
  f = h5py.File(filename, 'r')
  time = f['/scales/sim_time'][:] 
  nt = np.size(time,0) # number of time steps / file
  for m in range(0,nt): # for each time step in a file
      count_time = count_time + 1
Nt = int(count_time) # number of time steps
print(Nt)
#Nt = int(53069)


# =============================================================================
# analytical solution 

# flow parameters
N = 1e-3 # 1/s
nu = 2.0e-6 # m^2/s, kinematic viscosity
kap = nu
Pr = nu/kap
T = 44700.
omg = 2.0*np.pi/T # rads/s, M2 tidal period
thtc= ma.asin(omg/N) # radians      #*180./np.pi # degrees
tht = upsilon*thtc # radians
#Uw=0.01
#A = Uw*(N**2.0*np.sin(tht)**2.0-omg**2.0)/omg # m/s^2, M2 tide forcing amplitude
dPW=((4.*nu**2.)/((N**2.)*(np.sin(tht))**2.))**(1./4.)
tau = np.sqrt(dPW**2./nu)

Uw=0.01
thtsb = tht

# analytical solution parameters
d0sb=((4.*nu**2.)/((N**2.)*(np.sin(thtsb))**2.))**(1./4.) # Phillips-Wunsch BL thickness

if tht/thtc <= 1:
 Bw = Uw*(N**2.0)*np.sin(thtsb)/omg
else:
 Bw = Uw*(N**2.0)*np.sin(thtsp)/omg

# fix for super!!!

# subcritical coefficients
d1sb = np.power( omg*(1.+Pr)/(4.*nu) + \
       np.power((omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(thtsb)**2.-omg**2.)/(4.*nu**2.) , 1./2. ), -1./2.)
d2sb = np.power( omg*(1.+Pr)/(4.*nu) - \
       np.power((omg*(1.+Pr)/(4.*nu))**2. + \
       Pr*(N**2.*np.sin(thtsb)**2.-omg**2.)/(4.*nu**2.) , 1./2. ), -1./2.)
#print(d2sb)
Lsb = ((d1sb-d2sb)*(2.*nu/Pr+omg*d1sb*d2sb))/(omg*d1sb*d2sb) # wall-normal buoyancy gradient lengthscale
u1sb = d2sb*(omg*d1sb**2.-2.*nu/Pr)/(Lsb*omg*d1sb*d2sb) # unitless
u2sb = d1sb*(2.*nu/Pr-omg*d2sb**2.)/(Lsb*omg*d1sb*d2sb) # unitless
b1sb = d1sb/Lsb # unitless
b2sb = d2sb/Lsb # unitless

# buoyancy flux scaling
dsubBL = (  omg/(2.*nu) - (N*np.sin(tht))/(2.*nu) )**(-1./2.)
Le = Uw/omg
sig = N*np.sin(tht)*( Le/(dsubBL)*np.cos(tht)/np.sin(tht)-1. )**(1./2.) # = i*sigma (made real)
buoy_time_scale_ratio = sig/N # ratio of pertubation to buoyancy gradient
tide_time_scale_ratio = sig/omg # ratio of pertubation to tide
bscale = Le*N**2.*np.sin(tht)
wscale = dsubBL*sig
bfxscale = bscale*wscale


# =============================================================================
# grid


filename = file_path +'/snapshots_s' +str(N0) +'.h5'
f = h5py.File(filename, 'r')

# import the grid
x = f['/scales/x']['1.0'][:]
y = f['/scales/y']['1.0'][:]
z = f['/scales/z']['1.0'][:]
if np.size(z,0) != Nz:
  print('Error: incorrect Nz specified')
if np.size(y,0) != Ny:
  print('Error: incorrect Ny specified')
if np.size(x,0) != Nx:
  print('Error: incorrect Nx specified')
X,Y,Z = np.meshgrid(y,x,z) # shape = [Nx,Ny,Nz]

dt = f['/scales/timestep'][:]

# wavenumbers
ky = np.zeros([Ny]);
ky[1:int(Ny/2)+1] = np.linspace( 1., Ny/2., num=int(Ny/2) )*(2.*np.pi/Ly) # rad/km    
ky[int(Ny/2)+1:Ny] = -np.fliplr( [np.linspace(1., Ny/2.-1., num=int(Ny/2-1) )*(2.*np.pi/Ly)])[0] # rad/km

# cell edges  
ze = np.zeros([int(Nz+1)]) 
for j in range(1,Nz):
  ze[j] = ( z[j] + z[j-1] ) / 2.0 
ze[Nz] = H

# Chebyshev nodes
z_Chby = np.cos((np.linspace(1., Nz, num=Nz)*2.-1.)/(2.*Nz)*np.pi) # Chebyshev nodes


# =============================================================================
# initialization & compute statistics

t = np.zeros([Nt])

u = np.zeros([Nt,Nx,Ny,Nz])
v = np.zeros([Nt,Nx,Ny,Nz])
w = np.zeros([Nt,Nx,Ny,Nz])
b = np.zeros([Nt,Nx,Ny,Nz])
d2u = np.zeros([Nt,Nx,Ny,Nz])
d2v = np.zeros([Nt,Nx,Ny,Nz])
d2w = np.zeros([Nt,Nx,Ny,Nz])

ct = 0 # time step counter initialization
for n in range(N0,N1+1):
  print(n)

  # Files to read
  filename = file_path +'/snapshots_s' +str(n) +'.h5'
  f = h5py.File(filename, 'r')

  # get the time information
  dt = f['/scales/timestep'][:]
  time = (f['/scales/sim_time'][:])
  nt = np.size(time,0) # number of time steps / file
 

  for m in range(0,nt): # for each time step in a file
      
    t[ct] = time[m]  
    u[ct,:,:,:] = f['/tasks/u'][m,:,:,:] 
    v[ct,:,:,:] = f['/tasks/v'][m,:,:,:] 
    w[ct,:,:,:] = f['/tasks/w'][m,:,:,:] 
    b[ct,:,:,:] = f['/tasks/b'][m,:,:,:]  
    d2u[ct,:,:,:] = f['/tasks/d2u'][m,:,:,:]  
    d2v[ct,:,:,:] = f['/tasks/d2v'][m,:,:,:]  
    d2w[ct,:,:,:] = f['/tasks/d2w'][m,:,:,:]  




    # add movie plot here

    if ct < 10: 
      plotnumber = '/00000%d.png' %ct
    elif ct < 100: 
      plotnumber = '/0000%d.png' %ct 
    elif ct < 1000:
      plotnumber = '/000%d.png' %ct 
    elif ct <10000:
      plotnumber = '/00%d.png' %ct 
    elif ct < 100000:
      plotnumber = '/0%d.png' %ct 
    elif ct < 1000000:
      plotnumber = '/%d.png' %ct 

    #fig = plt.figure(figsize=(20,8));
    #fig = plt.figure(figsize=(7.25,6));

    #title_fontsize_wyz = '16'

    #vel_cb = 0.2
    #ctrs = np.linspace(-vel_cb, vel_cb,num=Nc)
    #plot_title = r"$w_{vertical}/U_\infty$  $u_{tide}/U_\infty$ = %.2f, n = " %(np.mean(np.mean(np.mean(u[m,:,:,:],0),0),0)/Uw) + str(n)  
    #CP = plt.contourf(X[:,:,nz],Y[:,:,nz],(w[ct,:,:,nz]*np.cos(tht)+u[ct,:,:,nz]*np.sin(tht))/Uw,ctrs,cmap=cmap_3,fontsize=ylabel_fontsize); 
    #plt.xlabel(r"y (m)",family='serif',fontsize=ylabel_fontsize); 
    #plt.ylabel(r"x (m)",family='serif',fontsize=ylabel_fontsize); 
    #plt.title(plot_title,family='serif',fontsize=title_fontsize_wyz); 
    #fig.colorbar(CP, orientation="vertical",ticks=[-vel_cb,vel_cb]) 
    #plt.axis([-4.8,4.8,-4.8,4.8])
    
    #plot_title = r"$(b-N^2z\cos\theta)/B_\infty$  t/T = %.2f" %(t[ct]/T)         
    #p2=plt.subplot(1,2,2)
    #CP = plt.contourf(X[:,:,nz],Z[:,:,nz],(b[m,:,:,nz]/Bw),600,cmap=cmap_1); 
    #plt.xlabel(r"y (m)",family='serif',fontsize=ylabel_fontsize); 
    #plt.ylabel(r"z (m)",family='serif',fontsize=ylabel_fontsize); 
    #plt.axis([-4.8,4.8,-4.8,4.8])
    #plt.title(plot_title,family='serif',fontsize=title_fontsize_wyz); 
    #fig.colorbar(CP, orientation="vertical") #,ticks=[-0.25,0.25]) 
    
    #plt.subplots_adjust(wspace=0.1, hspace=0.2)

    #plt.savefig(figure_path + '/w' + plotnumber,format='png'); plt.close(fig);






    print('number of oscillations %.10f' %(t[ct]/T) )
    ct = ct + 1  


print(Nt)


# =============================================================================      
# output .h5 files


time_mean_count = ct # number of time steps averaged over
Tt = (t[Nt-1]-t[0])/T # unitless, time period of averaging

#string = '\nnumber of time steps averaged over for time mean: %d\n' %(int(time_mean_count))
#print(string)
string = '\ntime period of time mean: t/T = %.5f\n' %(Tt)
print(string)
string = '\nnumber of time steps: %d\n' %(int(Nt))
print(string)

h5_filename = stats_path + '/statistics_%i_%i.h5' %(N0,N1)
f2 = h5py.File(h5_filename, "w")

# output file range information:
dset = f2.create_dataset('N0', data=N0, dtype='f8')
dset = f2.create_dataset('N1', data=N1, dtype='f8')

# time information:
dset = f2.create_dataset('Nt', data=Nt, dtype='f8')
dset = f2.create_dataset('time_mean_count', data=time_mean_count, dtype='f8')
dset = f2.create_dataset('T', data=T, dtype='f8')
dset = f2.create_dataset('Tt', data=Tt, dtype='f8')
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
dset = f2.create_dataset('N', data=N, dtype='f8')
dset = f2.create_dataset('omg', data=omg, dtype='f8')
dset = f2.create_dataset('thtc', data=thtc, dtype='f8')
dset = f2.create_dataset('tht', data=tht, dtype='f8')
dset = f2.create_dataset('kap', data=kap, dtype='f8')
dset = f2.create_dataset('nu', data=nu, dtype='f8')
dset = f2.create_dataset('Uw', data=Uw, dtype='f8')

# buoyancy flux components
dset = f2.create_dataset('u', data=u, dtype='f8') 
dset = f2.create_dataset('v', data=v, dtype='f8') 
dset = f2.create_dataset('w', data=w, dtype='f8') 
dset = f2.create_dataset('b', data=b, dtype='f8') 
dset = f2.create_dataset('d2u', data=d2u, dtype='f8') 
dset = f2.create_dataset('d2v', data=d2v, dtype='f8') 
dset = f2.create_dataset('d2w', data=d2w, dtype='f8') 

print('\nStatistics computed and written to file' + h5_filename + '.\n')

