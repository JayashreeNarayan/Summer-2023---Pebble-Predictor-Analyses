# all libraries
import numpy as np
from astropy import constants as c
import matplotlib.pyplot as plt
from all_planets_2 import allplanets
import pandas as pd
from multiprocessing import Pool, freeze_support
import time

start = time.time()

# Some useful constants in cgsp
year = 365.25*24*3600 # in seconds
au = c.au.cgs.value # value of 'au' in cm
MS = c.M_sun.cgs.value # mass of the sun in cgs
ME = c.M_earth.cgs.value # mass of the earth in cgs
k_b = c.k_B.cgs.value # boltzmann const
m_p = c.m_p.cgs.value # mass of proton
Grav = c.G.cgs.value # gravitational const

ZS = 0.012 # metallicity of the sun
RS = c.R_sun.cgs.value # Radius of sun in cm

# Creating grids
Nr = 1000 # number of grid points      
rhop = 1.25 # internal density of dust grains 
Rout = 1000.*au
Nt = 1000   # how many points on the time grid?
endtime = 1.e8*year
timegrid = np.logspace(np.log10(year),np.log10(endtime),Nt) # starts from 1 year and ends at endtime defined above, goes for Nt number of points

# Getting the data
df = pd.read_csv("with_errors2.csv",index_col = False)
df2 = df.drop_duplicates(subset = ["Planet Name"], keep = 'first')
col_names=df2.columns.values.tolist()
for i in col_names:
    df2.dropna(subset = [i], inplace = True)

# Making values from CSV into lists
MStar_list = df2["mass of star (solar masses)"].values.tolist() # in terms of solar masses
location_array = df2["semi major axis (au)"].values.tolist() # in terms of au
Rstar_array = df2["radius of star (solar radius)"].values.tolist()
Metallicity = df2["metallicity = log(k)*metallicity of sun"].values.tolist() # Making a list of all themetallicity values
total = len(MStar_list)

# Values of the variables being defined - input variables
A = [10**-5,10**-4,10**-3,10**-2] # alpha values
V = np.linspace(100,1000,10) # vfrag values
q = total # number of planets being considered from the list

# Output variables - ones that hold the final output
tot = q*len(A)*len(V)
Metal = np.zeros((len(A),len(V),q)) # list of metallicities of planets that got isolated
Location = np.zeros((len(A),len(V),q))
Mstars = np.zeros((len(A),len(V),q))
T2 = np.zeros((len(A),len(V),q)) # time at which planetary core is planted
core_mass = np.zeros((len(A),len(V),q)) # planetary core masses

# No. of planets that got isolated
Final = np.zeros((len(A),len(V),q))
isol = np.zeros((len(A),len(V)))
isolp=np.zeros((len(A),len(V)))

def main():
    args = []

    # making the arguments to send into starmaps
    for i in range(0,len(A)):
        for j in range(0,len(V)):
            for k in range(0,q):
                t = (location_array[k]*au,MStar_list[k]*MS,Rstar_array[k]*RS,Metallicity[k],A[i],V[j],i,j,k)
                args.append(t)

    with Pool() as pool:
        
        L = pool.starmap(allplanets,args) # function call - runs for all 862 planets * 4 alphas * 4 vfrags
        for i in range(0,tot):
            p = L[i][0]
            l = L[i][1]
            r = L[i][2]
            iso = L[i][3]
            Z = L[i][4]
            Loc = L[i][5]
            M_s = L[i][6]     
            t2 = L[i][7]       
            m = L[i][8]

            if iso != 0:
                Final[p][l][r] = 1
                Metal[p][l][r] = Z
                Location[p][l][r] = Loc
                Mstars[p][l][r] = M_s/MS
            else: 
                T2[p][l][r] = t2
                core_mass[p][l][r] = m
        
        for i in range(0,len(A)): # the first 
            for j in range(0,len(V)):
                for k in range(0,q):                    
                    K = (sum(Final[i,j]))
                    isol[i][j] = K
                    isolp[i][j] = K*100/q # saving percentages in isol
                
        print(isol)
        print(q)

        np.save("NPYs/Metallicity_full",Metal)
        np.save("NPYs/isolp_full",isolp)
        np.save("NPYs/Final_full", Final)
        np.save("NPYs/isol_full",isol)
        np.save("NPYs/location_full",Location)
        np.save("NPYs/Mstars_full",Mstars)
        np.save("NPYs/T_2",T2)
        np.save("NPYs/core_masses",core_mass)

# To print how much time the code takes to run
if __name__ == "__main__": 
    freeze_support()
    main()
    end = time.time()

    if end-start<60:
        print(end-start, "seconds")
    else:
        print((end-start)/60, "minutes")

