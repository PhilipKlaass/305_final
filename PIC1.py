# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:25:36 2024

@author: Philip Klaassen
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def PIC1(L,dt,NT,NG,IW,EPSI,A1,A2,IPHI,SP1,SP2,SP3):
    '''
    1-D electrostatic particle in cell simulation of an unmagnetized plasma 
    based on the ES1 program described in Bridsall and Langdon (2004). Created 
    to reproduce results found in Koen et. al (2012).
    
    C. K. Birdsall and A. B. Langdon, Plasma Physics via Computer Simulation, 
    Series in Plasma Physics (Taylor & Francis Group, London, 2004)

    E. J. Koen, A. B. Collier, S. K. Maharaj, 
    Particle-in-cell simulations of beam-driven electrostatic waves in a plasma. 
    Phys. Plasmas 1 April 2012; 19 (4): 042101. https://doi.org/10.1063/1.3695402
    
    Parameters
    ----------
    L : float
        Length of system
    dt :  float
        Time step
    NT : integer
        Total number of steps to be run
    NG : integer
        Number of grid points (power of 2)
    IW : integer
        Which order weighting to be used;
            1. zero'th order (NGP) mom-conserving
            2. first order (CIC), mom-conserving
            3. first order for particles and zeor'th order for forces,
               energy-conserving
    EPSI : float
        1/epsilon_0
    A1 : float
        Compensation factor
    A2 : float
        Smoothing Factor
    IPHI : flaot
        Plotting frequencies
    SP1,SP2,SP3: 14-tuple
        Parameters for each species
            1. Number of particles
            2. Plasma frequency
            3. Cyclotron frequency
            4. q/m
            5. VT1 Gaussian velocity distrobution wuth VT1 being thermal speed
               and 6*VT1 being the maximum velocity
    '''
    
    #
    T_c = 1
    
    
    
def init(n,L,wp,qm,vt1,nv2,v0):
    '''
    Places particles of one spieces in x,v phase space.

    Parameters
    ----------
    n : integer
        Number of particles
    wp : float
        Plasma frequency
    qm : float
        Charge-mass ratio
    vt1 : float
        RMS thermal velocity for random velocities
    nv2 : float
        Multiply maxwellian by v**nv2
    v0 : TYPE
        Drift velocity
    '''
    
    q = np.zeros(n)+L*wp*wp/(n*qm)
    
    m = np.zeros(n) + q /qm
    
    #evenly spaced positions for a constant rho
    x = np.linspace(0.0, L, n, endpoint = False)
    
    #set drift velocity
    v = np.zeros(n) + v0
    
    #apply maxwellian ditribution
    v = v+ np.random.normal(0,vt1,size = n)
    
    return x,v,q,m


def calc_rho(x,q,L,NG,IW):
    '''
    Calculate charge density from position and charge

    Parameters
    ----------
    x : numpy array
        Stores position of each particle
    q : numpy array
        Stores charge of each particle
    L : integer
        Length of system
    l : float
        Debye Length of cold electrons
    NG : integer
        Number of grid points
    IW : integer
        Which order weighting to be used;
            1. zero'th order (NGP) mom-conserving
            2. first order (CIC), mom-conserving
            3. first order for particles and zeor'th order for forces,
               energy-conserving
    Returns
    -------
    None.

    '''
    n = len(x)
    
    rho = np.zeros(NG)
    
    dx = L/NG
    
    for i in range(n):
        
        x_0 = int(x[i]//dx)
        
        if IW == 'NGP':
            
            if np.abs(x[i]%dx)<0.5 * dx:
                
                rho[x_0 % NG] += q[i] / dx
                
            else:
                rho[(x_0-+1) % NG] += q[i] / dx
                
        if IW == 'CIC' or IW == 'EC':
            rho[x_0 % NG] += q[i]/dx * (dx - x[i] % dx) / dx
            rho[(x_0+1) % NG] += q[i]/dx * (x[i] % dx) / dx
    
    rho +=  (1.6*10**-19)*L/NG
    
    return rho

def calc_fields(rho,NG,dx,EPSI):
    '''
    Calculates the potential and electric fields at the grid points.
    Uses a fourier transform method, peridoic boundary conditions assumed 
    implicitly.

    Parameters
    ----------
    rho : np.array
        Charge density at grid points
    NG : integer
        Number of grid points
    dx : float
        grid spacing
    EPSI : float
        1/epsilon

    Returns
    -------
    phi_grid : np.array
        Potential at grid points
    E_grid : np.array
        Electric field at grid points
    ECE : float
        Electrostatic energy

    '''
    
    #FFT of charge density
    rho_k = sp.fft.rfft(rho)
    
    #make frequencies
    k = 2*np.pi*np.fft.rfftfreq(NG,dx)
    
    #ensure no division by zero and set dc signal to rho = 0
    k[0],rho_k[0] = 1,0 
    
    K = k*(np.sin(0.5*k*dx)/ (0.5*k*dx))

    phi_k = rho_k *EPSI / K**2

    phi_grid = np.fft.irfft(phi_k)

    grad = (np.diag(np.ones(NG-1),1) - np.diag(np.ones(NG-1),-1))
    
    #Periodic boundary consitions
    grad[0,NG-1] = -1
    grad[NG-1,0] = 1
    
    grad = grad/(2*dx)
    
    E_grid = -1* grad@phi_grid
    
    ECE = 0.5*np.sum(phi_k*np.conjugate(rho_k))
    
    return phi_grid,E_grid, ECE
    

def accel(x,v,q,m,E_grid,IW,NG,L,dt):
    '''
    

    Parameters
    ----------
    x : np.array
        Position of particles
    v : np.array
        Velocity of particles
    q : np.array
        Charge of particles
    m : np.array
        Mass of particles
    E_grid : np.array
        Electric field at grid points
    IW : String
        Weighting to be used 
    NG : integer
        number of grid points
    L : integer
        Length of domain
    dt : float
        Time step

    Returns
    -------
    v : np.array
        Updated velocity at +dt

    '''
    
    dx = L/NG
    #x[i] is between the grid point X[i] and X[i]+1 
    X = np.round(x//dx).astype(int)
    
    E_par = np.zeros(len(x))
    num = np.arange(len(x))
    
    if IW== 'NGP' or IW== 'EC':
        
        #True if particle is closer to grid point X[i]
        mask = np.abs(x%dx)<0.5*dx
        
        E_par[num[mask]] = E_grid[X[num[mask]]%NG]
        E_par[num[~mask]] = E_grid[(X[num[~mask]]+1)%NG]
    
    v_old=v
    v= v+(q/m)*dt*E_par
    
    KE = np.sum(0.5*m*v*v_old)
    
    return v, KE


def move(m,q,x,v,L,dt):
    
    x = x+v*dt
    x = x%L
    
    return x
    
    
    
def main_test():
    NG = 1024
    
    L = 1024
    dt = 10**-6
    
    x,v,q,m = init(n=1000, L = L, wp = 1.6*10**5, qm =-1.758*10**14, vt1 =10, nv2 =0, v0 = 0)
    
    for i in range(10000):
        rho = calc_rho(x, q, L =L,NG = NG, IW = 'EC')
        
        phi_grid,E_grid,ECE = calc_fields(rho = rho, NG= NG, dx= 1.0, EPSI = 1)
        
        if i ==0:
            
            #Set initial v to t=-dt/2 for the leapfrog scheme
            v, KE = accel(x, v, -q, m, E_grid, IW = 'EC', NG=NG, L=L, dt=dt)
            
        else:
            v,KE = accel(x, v, q, m, E_grid, IW = 'EC', NG=NG, L=L, dt=dt)
        
        x = move(m=m, q=q, x=x, v=v, L=L, dt=dt)
        
        if i%1000 ==0:
            plt.scatter(x,v,marker = '.')
            plt.title(str(i))
            plt.show()

    
main_test()