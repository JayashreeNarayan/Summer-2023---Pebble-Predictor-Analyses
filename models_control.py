#all libraries
import numpy as np
from astropy import constants as c
import matplotlib.pyplot as plt
from all_planets_2 import allplanets
import pandas as pd
from multiprocessing import Pool, freeze_support
import time

start=time.time()

# some useful constants in cgs
year = 365.25*24*3600 # in seconds
au = c.au.cgs.value
MS = c.M_sun.cgs.value # mass of the sun in cgs
ME = c.M_earth.cgs.value # mass of the earth in cgs
k_b = c.k_B.cgs.value # boltzmann const
m_p = c.m_p.cgs.value # mass of proton
Grav = c.G.cgs.value # gravitational const

ZS=0.012 # metallicity of the sun
RS=c.R_sun.cgs.value # Radius of sun in cm

# creating grids
Nr = 1000 # number of grid points      
rhop = 1.25 # internal density of dust grains 
Rout = 1000.*au
Nt = 1000   # how many points on the time grid?
endtime = 1.e8*year
timegrid = np.logspace(np.log10(year),np.log10(endtime),Nt) # starts from 1 year and ends at endtime defined above, goes for Nt number of points

# getting the data
df=pd.read_csv("with_errors2.csv"
               ,index_col=False)
df2 = df.drop_duplicates(subset=["Planet Name"], keep='first')
col_names=df2.columns.values.tolist()
for i in col_names:
    df2.dropna(subset=[i], inplace=True)

MStar_list=df2["mass of star (solar masses)"].values.tolist() # in terms of solar masses
total=len(MStar_list)
location_array=df2["semi major axis (au)"].values.tolist() # in terms of au
Rstar_array=df2["radius of star (solar radius)"].values.tolist()
Metallicity=df2["metallicity = log(k)*metallicity of sun"].values.tolist() # Making a list of all themetallicity values

# CALLING THE FUNCTION
A=[10**-4] # alpha values
V=[400] # vfrag values
q=len(location_array) # number of planets being considered from the list
times=[100,1000,10000,100000]

tot=q*len(A)*len(V)
Metal=np.zeros((len(A),len(V),q)) # list of metallicities of planets that got isolated
Location=np.zeros((len(A),len(V),q))
Mstars=np.zeros((len(A),len(V),q))

Final=np.zeros((len(A),len(V),q))
isol=np.zeros((len(A),len(V)))
isolp=np.zeros((len(A),len(V)))

def main():
    args=[]

    # making the arguments
    for i in range(0,len(times)):
        for k in range(0,q):
            t=(location_array[k]*au,MStar_list[k]*MS,Rstar_array[k]*RS,Metallicity[k],A[0],V[0],times[i])
            args.append(t)

    with Pool() as pool:
        
        L=pool.starmap(allplanets,args) # starmap is the keyword is the type of map being used
        for i in range(0,tot):
            
            iso=L[i][0]
            Z=L[i][1]
            Loc=L[i][2]
            M_s=L[i][3]            

            if iso!=0:
                Final[i]=1
                Metal[i]=Z
                Location[i]=Loc
                Mstars[i]=M_s/MS
        
        for i in range(0,len(A)): # the first 
            for j in range(0,len(V)):
                for k in range(0,q):
                    K=(sum(Final[i,j]))
                    isol[i][j]=K
                    isolp[i][j]=K*100/q # saving percentages in isol
                
        print(isol)
        print(total)

        np.save("Model_NPYs/Metallicity_full",Metal)
        np.save("Model_NPYs/isolp_full",isolp)
        np.save("Model_NPYs/Final_full", Final)
        np.save("Model_NPYs/isol_full",isol)
        np.save("Model_NPYs/location_full",Location)
        np.save("Model_NPYs/Mstars_full",Mstars)

# To print how much time the code takes to run
if __name__=="__main__": 
    freeze_support()
    main()
    end=time.time()

    if end-start<60:
        print(end-start)
    else:
        print((end-start)/60)

