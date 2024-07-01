import numpy as np
from astropy import constants as c
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool, freeze_support
from all_planets_models import allplanets
import time

start=time.time()

# Some useful constants in cgs
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
A = [10**-4] # alpha values
V = [400] # vfrag values
q = total # number of planets being considered from the list
times = [0,100,1000,10000,100000]

# Output variables - ones that hold the final output
tot = len(times)
Metal = np.zeros((tot, q)) # list of metallicities of planets that got isolated
Location = np.zeros((tot, q)) # list of locations of planets that got isolated
Mstars = np.zeros((tot, q)) # list of stellar masses of planets that got isolated

# No. of planets that got isolated
Final = np.zeros((tot, q)) 
isol = np.zeros((tot))
isolp = np.zeros((tot))

def main():
    args = []
    # making the arguments to send into starmaps
    for i in range(0,tot):
        for k in range(0,q):
            t = (location_array[k]*au, MStar_list[k]*MS, Rstar_array[k]*RS, Metallicity[k], A[0], V[0], times[i])
            args.append(t)

    # initialising star maps
    with Pool() as pool:
        L = pool.starmap(allplanets,args) # function call
        for i in range(0,tot):
            for j in range(0,q):
                iso = L[i][0]
                Z = L[i][1]
                Loc = L[i][2]
                M_s = L[i][3]            

                if iso != 0:
                    Final[i,j] = 1
                    Metal[i,j] = Z
                    Location[i,j] = Loc
                    Mstars[i, j] = M_s/MS
            
        for i in range(0,tot): # the first 
            K = (sum(Final[i]))
            isol[i] = K
            isolp[i] = K*100/q # saving percentages in isol
        
        print(isol)
        print(q)

        np.save("Model_NPYs/Metallicity_full",Metal)
        np.save("Model_NPYs/isolp_full",isolp)
        np.save("Model_NPYs/Final_full", Final)
        np.save("Model_NPYs/isol_full",isol)
        np.save("Model_NPYs/location_full",Location)
        np.save("Model_NPYs/Mstars_full",Mstars)

# To print how much time the code takes to run
if __name__ == "__main__": 
    freeze_support()
    main()
    end = time.time()

    if end-start<60:
        print(end-start, "seconds")
    else:
        print((end-start)/60, "minutes")

