import OL18_new
from PP import pebble_predictor
import numpy as np
from astropy import constants as c
import matplotlib.pyplot as plt
import math as mth

def allplanets(location,M_star,R_star,Z,alpha,vfrag,p,q,r_):
    """
    Method:
        2-fold process: 
    - Step 1 (T1): Initial mass $(10^{-6})$ g to Transition mass (M_t)
    - Step 2 (T2): Transition mass (M_t) to Classical isolation mass (Classical_iso)

        Calculating T2:
    - We have obtained $M$ = $\pi \Omega_k \Sigma_{plts} R_p^2 \Delta T$
    - So then we just need $\Delta T$ = $M$ / ($\pi \Omega_k \Sigma_{plts} R_p^2$ )
    - This will be the change in time taken -- giving us the position of T2 on the timegrid.

    """
    epsilon=[]
    iso=0
    # some useful constants in cgs
    year = 365.25*24*3600 # in seconds
    au = c.au.cgs.value
    MS = c.M_sun.cgs.value # mass of the sun in cgs
    ME = c.M_earth.cgs.value # mass of the earth in cgs
    k_b = c.k_B.cgs.value # boltzmann const
    m_p = c.m_p.cgs.value # mass of proton
    Grav = c.G.cgs.value # gravitational const
    ZS=0.012 # metallicity of the sun
    Nr = 1000 # number of grid points      
    rhop = 1.25 # internal density of dust grains 
    Rout = 1000.*au
    
    # variables needed in code
    Z0 = (10**Z)*ZS # solids-to-gas ratio 
    mdisk=0.1*M_star # assuming that disk mass is 10% of the star mass
    Rc=30*au 

    Rin = 1.5*R_star # inner radius of disk is 1au and outer radius is 1000 a (HUH)
    rgrid = np.logspace(np.log10(Rin),np.log10(Rout),Nr)   
    SigmaGas = mdisk / (2.*np.pi*Rc**2.) * (rgrid/Rc)**(-1.) * np.exp(-1.*(rgrid/Rc)) # gas surface density
    SigmaDust = Z0*SigmaGas # dust surface density
    T = 280 * ((au*M_star)/(rgrid*MS))**(0.5) # temperature profile
    cs = np.sqrt(k_b*T/(2.3*m_p)) # sound speed
    OmegaK = np.sqrt(Grav*M_star/rgrid**3.) # Keplerian frequency       #using Mstar to be MS here by default

    # grid building
    Nt = 1000 # how many points on the time grid?
    endtime = 3.e6*year
    timegrid = np.logspace(np.log10(year),np.log10(endtime),Nt) # starts from 1 year and ends at endtime defined above, goes for Nt number of points

    ir = rgrid.searchsorted(location)
    it = timegrid.searchsorted((3.e6) * year)

    # finding stokes number and flux numbers
    stokes,flux_values=pebble_predictor(rgrid=rgrid,tgrid=timegrid,Mstar=M_star,SigmaGas=SigmaGas,T=T,SigmaDust=SigmaDust,alpha=alpha,vfrag=vfrag,rhop=rhop)

    H_g = cs/OmegaK # gas scale height
    rhog = SigmaGas / (np.sqrt(2.*np.pi)*H_g) # gas midplane density
    pressure = rhog * (cs**2.) # gas midplane pressure
    
    # now, the pressure gradient and eta at your chosen location rgrid[ir]:
    dpdr = (pressure[ir+1]-pressure[ir-1])/(rgrid[ir+1]-rgrid[ir-1])
    eta = dpdr / (2.*rhog[ir]*(OmegaK[ir]**2.)*rgrid[ir])
    
    Classical_iso = 0.1*ME* ((SigmaDust[ir]/5)**(3/2)) * (rgrid[ir]/au)**3 * ((M_star/MS)**(-1/2))
    
    # Calculating T1
    M0 = 10**(-6)*ME # initial mass of planetesimal in g
    R0 = ((3*M0)/(4*mth.pi*3))**(1/3) # initial radius of planetesimal in cm

    R_t = (580 * ((R0/(10**6))**(3/7)) * ((rgrid[ir]/(4*au))**(5/7)) * ((SigmaDust[ir]/3)**(2/7)) * (1000*100)) # Transition radius in cm
    M_t = 3 * ((4/3) * mth.pi * (R_t)**3) # Transition Mass in g

    N = mth.log2((abs(M_t)/ M0)) # Time taken for this:
    tau_rg = (0.1 * 3 * R0)/(OmegaK[ir] * SigmaDust[ir])

    T1 = N*tau_rg # value of T1

    it2 = timegrid.searchsorted(abs(T1)) # position at which T1 exists

    totmass=[]
    for i in range(0,it2+1):
        totmass.append(M_t/ME)
    
    # Calculating T2
    for i in range(it2, 1000):
        m = totmass[-1]*ME # At the first iteration this is the value of the transition mass, after that it increases with time
        r = ((3*m)/(4*mth.pi*3))**(1/3)     
        dt=(timegrid[i]-timegrid[i-1])
        dm= mth.pi * OmegaK[ir] * SigmaDust[ir] * r**2 * dt
        totmass.append(((totmass[-1]*ME)+dm)/ME)
        if totmass[-1] >= Classical_iso/ME:
            totmass.append(Classical_iso/ME)
            break

    k = len(totmass)
    R_p = r = ((3*(totmass[-1]*ME))/(4*mth.pi*3))**(1/3)
    delT = (totmass[-1]*ME)/(mth.pi * OmegaK[ir] * SigmaDust[ir] * R_p**2 )
    T2 = T1 + delT # value of T2
    it3 = timegrid.searchsorted(T2)

    for i in range(k,1000):
        totmass.append(0)
    totmass_arr = np.asarray(totmass)

    M_iso=40*ME*(M_star/MS)*((H_g[ir]/(0.05*location))**3)
    hgas=H_g[ir]/rgrid[ir]

    for i in range(it3,1000): # through the entire timegrid
        dt=(timegrid[i]-timegrid[i-1]) # time difference
        R=((((3/(4*mth.pi))*totmass_arr[i-1]*ME/3)**(1/3)))/location # radius of planet in terms of au; density of the planet as it grows is
        QP=totmass_arr[i-1]*ME/MS # QP ratio

        eps=OL18_new.epsilon_general (tau=stokes[i,ir], qp=QP, eta=eta, hgas=hgas, alphaz=alpha, Rp=R)
        epsilon.append(eps)
        totmass_arr[i]=totmass_arr[i-1]+((flux_values[i,ir]*eps*dt)/ME) # this becomes the array of mass as a function of time    
        
        if totmass_arr[i]>=M_iso/ME:
            totmass_arr[i]=M_iso/ME

    if totmass_arr[it]>=M_iso/ME:
        iso+=1
        
    Q=[p,q,r_,iso,Z0,location/au,M_star]
    
    return Q