# all libraries
import numpy as np
from astropy import constants as c
import matplotlib.pyplot as plt
from all_planets_models import allplanets
import pandas as pd
from multiprocessing import Pool, freeze_support
import time

start = time.time()

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
A = [10**-5] # alpha values
V = [400] # vfrag values
times = [100*year,1000*year,10000*year,100000*year]
len_times = len(times)
q = total # number of planets being considered from the list

# Output variables - ones that hold the final output
tot = q*len_times
Metal = np.zeros((len_times,q)) # list of metallicities of planets that got isolated
Location = np.zeros((len_times,q))
Mstars = np.zeros((len_times,q))

# No. of planets that got isolated
Final = np.zeros((len_times,q))
isol = np.zeros((len_times))
isolp = np.zeros((len_times))

def main():
    args = []

    # making the arguments to send into starmaps
    for i in range(0,len(A)):
        for j in range(0,len(V)):
            for k in range(0,q):
                for l in range(0, len_times):
                    t = (location_array[k]*au, MStar_list[k]*MS, Rstar_array[k]*RS, Metallicity[k], A[i], V[j], i, j, k, times[l], l)
                    args.append(t)

    with Pool() as pool:
        
        L = pool.starmap(allplanets, args) # function call
        for i in range(0, tot):
            p = L[i][0]
            l = L[i][1]
            r = L[i][2]
            iso = L[i][3]
            Z = L[i][4]
            Loc = L[i][5]
            M_s = L[i][6]  
            T = L[i][7]          

            if iso != 0:
                Final[T][r] = 1
                Metal[T][r] = Z
                Location[T][r] = Loc
                Mstars[T][r] = M_s/MS
        
        for i in range(0,len_times): # the first 
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

