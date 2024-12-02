# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:25:36 2024

@author: Philip Klaassen
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from time import time


def PIC1():
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
    '''
    NT = 150000 #Time steps
    
    Th_Tc = 100 #ratio of hot to cold electron temps
    Th_Tb = 100 #ratio of hot to beam electron temps
    nb_nc = 0.04 #ratio of beam to cold electron densities
    ud_vC = 20.0 #ratio of streaming speed to cold thermal speed
    
    #Constants
    m_e = 1.0
    e = -1.0
    k_B = 1.0
    E0 = 1.0 #vaccuum permititvity
    
    N_c = int(5*10**3) #Number of cold electrons
    N_h = N_c #Number of hot electrons 
    N_b = int(nb_nc*N_c) #Number of beam electrons
    
    TC = 1.0 #cold electron temperature (velocities normallized to cold electron thermal speed)
    TH = Th_Tc*TC 
    TB = TH / ( Th_Tb )
    
    vC = np.sqrt(k_B*TC / m_e)
    vH = np.sqrt(k_B*TH / m_e)
    vB = np.sqrt(k_B*TB / m_e)
    
    ud = ud_vC*vC

    n_c0 = (N_c*e)**2 / (1024**2 * E0 * k_B * TC) 
    
    lamD = np.sqrt(E0*k_B*TC/(n_c0*e**2))
    
    L = 1024*lamD
    
    n_c = N_c / L
    lam_C =  np.sqrt(E0*k_B*TC/(n_c*e**2))
    
    n_h = N_h / L
    lam_H =  np.sqrt(E0*k_B*TC/(n_h*e**2))
    
    
    n_0 = (N_c+N_h+N_b) / L
    
    w_pe = np.sqrt(n_0 *e**2 / (m_e*E0) )
    
    w_pc = np.sqrt(n_c *e**2 / (m_e*E0))
    w_ph = np.sqrt(n_h *e**2 / (m_e*E0))
    
    dt = 0.1*w_pe**-1
    
    NG = 1024
    
    xc,vc = init(n=N_c,L=L, vt1=vC , nv2=1, v0=0)
    xh,vh = init(n=N_h,L=L, vt1=vH , nv2=1, v0=0)
    xb,vb = init(n=N_b,L=L, vt1=vB , nv2=1, v0=ud)
    
    x = np.append(xc,np.append(xh, xb))
    v = np.append(vc,np.append(vh, vb))
    
    K_E = np.array([])
    P_E = np.array([])
    time = np.array([])
    E_hist = np.zeros((1024,1024),dtype = float)
    
    for i in range(NT):
        
        rho = calc_rho(x=x, q=e,L=L, NG=NG, IW="EC")
        
        phi_grid,E_grid,ECE = calc_fields(rho=rho, NG=NG, dx=lamD, EPSI=1/E0,
                                          a1=0.5,a2=100,method = 'FFT')
        
        if i==0:
            v,KE = accel(x=x, v=v,q=-e,m=m_e, E_grid=E_grid, IW="EC", NG=NG, L=L, dt=dt)
        else:
            v,KE = accel(x=x, v=v,q=e,m=m_e, E_grid=E_grid, IW="EC", NG=NG, L=L, dt=dt)
        
        x = move(x=x, v=v, L=L, dt=dt)
        
        P_E = np.append(P_E,ECE)
        K_E = np.append(K_E,KE)
        
        time = np.append(time, i*dt*w_pe)
        
        if i%100 ==0:
            plt.scatter(x[0:N_c]/lamD,v[0:N_c],s= 0.01,color ='k')
            plt.xlabel(r"$x$")
            plt.ylabel(r"$v_x$")
            plt.title(str(i))
            plt.show()
            plt.scatter(x[N_c:N_c+N_h]/lamD,v[N_c:N_c+N_h],s= 0.01,color = 'r')
            plt.title(str(i))
            plt.xlabel(r"$x$")
            plt.ylabel(r"$v_x$")
            plt.show()
            plt.scatter(x[N_c+N_h:]/lamD,v[N_c+N_h:],s= 0.01,color = 'b')
            plt.title(str(i))
            plt.xlabel(r"$x$")
            plt.ylabel(r"$v_x$")
            plt.show()
            plt.plot(E_grid*np.abs(e/(m_e*w_pe*vC)),color = 'k')
            plt.title(str(i))
            plt.xlabel(r"$x$")
            plt.ylabel(r"$E_x$")
            plt.show()
            plt.plot(time,-1*P_E*e/(k_B*TC*E0),color = 'k')
            plt.xlabel(r"$\omega_{pe}t$")
            plt.ylabel(r"Electrostatic Energy")
            plt.title(str(i))
            plt.show()
            plt.plot(time,K_E,color = 'k')
            plt.xlabel(r"$\omega_{pe}t$")
            plt.ylabel(r"Kinetic Energy")
            plt.title(str(i))
            plt.show()
            
        if i%1 ==0:
            if i<1024:
                k = int(i/1)
                E_hist[:,k%1024] = E_grid
                
            else:
                
                E_hist = np.roll(E_hist,1,axis = 0)
                E_hist[:,-1] = E_grid
        
            if i%100 ==0 and i != 0 and i>1000:
                dispersion_relation_graph(E_hist=E_hist, lamD=lamD, w_pe=w_pe,
                                      NG=NG, dt= 0.01*w_pe,ud=ud_vC,w_pc=w_pc,
                                      w_ph=w_ph,lam_H=lam_H,lam_C=lam_C)
                
            if i%100 ==0 and i != 0 and i>1000:
                
                fig,ax=plt.subplots()
                pos=ax.imshow(E_hist[:,:k],cmap = 'jet')
                cbar=fig.colorbar(pos,ax=ax)
                plt.show()
        
            
    dispersion_relation_graph(E_hist = E_hist, lamD=lamD, w_pe=w_pe,
                              NG=NG,  dt= 0.01*w_pe)
    plt.plot(P_E)
    plt.show()
    plt.plot(K_E)
    plt.show()
        
    
    
def dispersion_relation_graph(E_hist, lamD,w_pe,NG,dt,ud,w_pc,w_ph,lam_H,lam_C):
    
    k = 2*np.pi*np.fft.rfftfreq(NG,lamD)
    Dk = k[-1]-k[0]
    
    w = 2*np.pi*np.fft.rfftfreq(NG,dt)
    Dw = w[-1]-w[0]
    
    grid_kt = np.fft.rfft(E_hist,axis = 0,norm= 'forward')
    
    grid_wk = np.fft.rfft2(E_hist,norm = 'forward')
    grid_wk = np.abs(grid_wk)
    
    
    
    grid_wk = grid_wk[:len(k),:]
    
    grid_wk=grid_wk.T # So that the horizontal axis is k and the vertical axis is omega
    grid_wk=grid_wk[::-1,:] # So that (k = 0, omega = 0) is at bottom left corner
    grid_wk=grid_wk[:,1:-1] # first and last col are zero
    
    grid_wk = grid_wk[int(round(.425*NG)):,:int(round(.075*NG))]
    
    M = np.max(grid_wk)
    
    m = int(round(len(k)*0.15))
    
    fig,ax=plt.subplots()
    pos=ax.imshow(grid_wk/M,extent=(k[1],k[m],w[0],w[-m]),aspect='auto',
                  cmap='jet')
    cbar=fig.colorbar(pos,ax=ax)
    
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    #theoretical dispersion relations
    k0 = np.linspace(x_min,10*x_max,1000) #beam-driven
    EA = np.sqrt(w_pc**2*(1+3*(k0*lam_C)**2)/(1+(k0*lam_H)**-2)) #electron acoustic
    EP = np.sqrt(w_pc**2*(1+3*(k0*lam_C)**2)+w_ph**2*(1+3*(k0*lam_H)**2)) #electron plasma
    
    ax.plot(k0/2,ud*k0,'k',lw =0.75)
    ax.plot(k0/10,EA,'k',lw =0.75)
    ax.plot(k0/10,EP,'k',lw =0.75)
    
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    
    plt.show()
    
    #fig,ax=plt.subplots()
    
    #pos=ax.imshow(grid_wk,extent=(k[1],k[-2],w[0],w[-1]),aspect='auto')
    #cbar=fig.colorbar(pos,ax=ax)
    
    #plt.show()

    
    
    
    
    
def init(n,L,vt1,nv2,v0):
    '''
    Places particles of one spieces in x,v phase space.

    Parameters
    ----------
    n : integer
        Number of particles
    vt1 : float
        RMS thermal velocity for random velocities
    nv2 : float
        Multiply maxwellian by v**nv2
    v0 : TYPE
        Drift velocity
    '''
    
    #evenly spaced positions for a constant rho
    x = np.linspace(0.0, L, n, endpoint = False)
    
    #set drift velocity
    v = np.zeros(n) + v0
    
    #apply maxwellian ditribution
    v = v+ np.random.normal(0,vt1,size = n)
    
    return x,v


def calc_rho(x,q,L,NG,IW):
    '''
    Calculate charge density from position and charge

    Parameters
    ----------
    x : numpy array
        Stores position of each particle
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
                
                rho[x_0 % NG] += q / dx
                
            else:
                rho[(x_0+1) % NG] += q / dx
                
        if IW == 'CIC' or IW == 'EC':
            rho[x_0 % NG] += q/dx * (dx - x[i] % dx) / dx
            rho[(x_0+1) % NG] += q/dx * (x[i] % dx) / dx
    
    rho +=  -(n*q)/L
    
    return rho

def calc_fields(rho,NG,dx,EPSI,a1,a2,method):
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
    
    if method == "FFT":
        #FFT of charge density
        rho_k = sp.fft.rfft(rho)
        
        #make frequencies
        k = 2*np.pi*sp.fft.rfftfreq(NG,dx)
        
        k[0]=1.     # to avoid division by 0
        rho_k[0]=0
        
        #Smoothing
        SM = np.exp(a1*np.sin(k*dx/2)**2 - a2*np.tan(k*dx/2)**4)
        SM = SM**2
        #N=8
        #kM = np.max(k)
        #SM = np.exp(-(k/kM)**N)
        
        rho_k = rho_k*SM
        #ensure no division by zero and set dc signal to rho = 0
        k[0],rho_k[0] = 1,0 
        
        K = k*(np.sin(0.5*k*dx)/ (0.5*k*dx))
    
        phi_k = rho_k *EPSI / K**2
    
        phi_grid = sp.fft.irfft(phi_k)
    
        grad = (np.diag(np.ones(NG-1),1) - np.diag(np.ones(NG-1),-1))
        
        #Periodic boundary consitions
        grad[0,NG-1] = -1
        grad[NG-1,0] = 1
        
        grad = grad/(2*dx)
        
        E_grid = -1* grad@phi_grid
        
        
        
    if method == "DPE":
            
        lap = 2*np.diag(np.ones(NG)) - np.diag(np.ones(NG-1),1)-np.diag(np.ones(NG-1),-1)
        
        lap[0,-1] = -1
        lap[-1,0] = -1
        
        lap =lap/dx**2
        
        phi_grid = np.linalg.solve(lap,rho)
        
        grad = (np.diag(np.ones(NG-1),1) - np.diag(np.ones(NG-1),-1))/dx
        
        E_grid =-1*grad@phi_grid
        
    ECE = 0.5*dx*np.sum(phi_grid*rho)
    
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
    
    KE = 0.5*m*np.sum(v*v_old)
    
    return v, KE


def move(x,v,L,dt):
    
    x = x+v*dt
    x = x%L
    
    return x
    
    
    
def main_test():
    NG = 1024
    
    L = 1.024*10**-2
    dt = 10**-1
    
    x1,v1 = init(n=10000, L = L, vt1 =.1, nv2 =0, v0 = 0)
    
    x2,v2 = init(n=10000, L = L, vt1 =1.0, nv2 =0, v0 = 0)
    
    x3,v3 = init(n=500, L = L, vt1 =.1, nv2 =0, v0 = 2.0)
    
    N = 25000
    
    L = np.sqrt(N)
    
    x = np.append(x1, x2)
    v = np.append(v1, v2)
    
    x = np.append(x, x3)
    v = np.append(v, v3)
    
    q=1
    m=1
    T = []
    E = []
    
    t_density = 0
    t_field =0
    t_vel =0
    t_pos = 0    
    
    for i in range(1000):
        
        t1 = time()
        
        rho = calc_rho(x, L =L,NG = NG, IW = 'EC')
        
        t2 = time()
        
        phi_grid,E_grid,ECE = calc_fields(rho = rho,NG= NG,dx= 1.0,EPSI = 1)
        
        t3 = time()
        if i ==0:
            
            #Set initial v to t=-dt/2 for the leapfrog scheme
            v, KE = accel(x, v, -q, m, E_grid, IW = 'EC', NG=NG, L=L, dt=dt)
            
        else:
            v,KE = accel(x, v, q, m, E_grid, IW = 'EC', NG=NG, L=L, dt=dt)
        t4 = time()
        x = move(m=m, q=q, x=x, v=v, L=L, dt=dt)
        t5 = time()
        if i%100 ==0:
            plt.scatter(x[0:10000],v[0:10000],color ='b',marker = '.', alpha = 0.3,s=1)
            plt.scatter(x[10000:20000],v[10000:20000],color ='r',marker = '.', alpha = 0.3,s=1)
            plt.scatter(x[20000:],v[20000:],color ='g',marker = '.', alpha = 0.3,s=1)
            plt.title(str(i))
            plt.xlabel('Position')
            plt.ylabel('Velocity')
            plt.show()
            
        if i % 1000==0:
            plt.plot(E_grid, 'k')
            plt.plot(rho, 'b')
            plt.show()
            
        t_density += t2-t1
        t_field +=t3-t2
        t_vel +=t4-t3
        t_pos += t5-t4
        T.append(KE)
        E.append(E)
    
    print(t_density)
    print(t_field)
    print(t_vel)
    print(t_pos)
    plt.plot(T,color = 'k')


    
PIC1()