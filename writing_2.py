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
year = 365.25*24*3600   #in seconds
au = c.au.cgs.value
MS = c.M_sun.cgs.value      #mass of the sun in cgs
ME = c.M_earth.cgs.value    #mass of the earth in cgs
k_b = c.k_B.cgs.value       #boltzmann const
m_p = c.m_p.cgs.value       #mass of proton
Grav = c.G.cgs.value        #gravitational const

ZS=0.012                    #metallicity of the sun
RS=c.R_sun.cgs.value                #Radius of sun in cm

#creating grids
Nr = 1000 # number of grid points      
rhop = 1.25 # internal density of dust grains 
Rout = 1000.*au
Nt = 1000   # how many points on the time grid?
endtime = 1.e8*year
timegrid = np.logspace(np.log10(year),np.log10(endtime),Nt)     #starts from 1 year and ends at endtime defined above, goes for Nt number of points

#getting the data
df=pd.read_csv("C:\\Users\\Dell\\OneDrive\\Documents\\IISERM\\INTERNSHIPS\\Joanna Drazkowska\\Code\\pebble-predictor-main\\with_errors2.csv"
               ,index_col=False)
df2 = df.drop_duplicates(subset=["Planet Name"], keep='first')
col_names=df2.columns.values.tolist()
for i in col_names:
    df2.dropna(subset=[i], inplace=True)

MStar_list=df2["mass of star (solar masses)"].values.tolist()        #in terms of solar masses
total=len(MStar_list)
location_array=df2["semi major axis (au)"].values.tolist()           #in terms of au
Rstar_array=df2["radius of star (solar radius)"].values.tolist()
Metallicity=df2["metallicity = log(k)*metallicity of sun"].values.tolist()           #Making a list of all themetallicity values

#CALLING THE FUNCTION
A=[10**-5,10**-4,10**-3,10**-2]                            #alpha values
V=np.linspace(100,1000,10)            #vfrag values
#A=[10**-4]
#V=[700]
q=len(location_array)                               #number of planets being considered from the list
#V=[1000]
#q=1

tot=q*len(A)*len(V)
Metal=np.zeros((len(A),len(V),q))                            #list of metallicities of planets that got isolated
Location=np.zeros((len(A),len(V),q))
Mstars=np.zeros((len(A),len(V),q))

Final=np.zeros((len(A),len(V),q))
isol=np.zeros((len(A),len(V)))
isolp=np.zeros((len(A),len(V)))

def main():
    args=[]

    #making the arguments
    for i in range(0,len(A)):
        for j in range(0,len(V)):
            for k in range(0,q):
                t=(location_array[k]*au,MStar_list[k]*MS,Rstar_array[k]*RS,Metallicity[k],A[i],V[j],i,j,k)
                args.append(t)

    with Pool() as pool:
        
        L=pool.starmap(allplanets,args)     #starmap is the keyword is the type of map being used
        #print(L)
        for i in range(0,tot):
            
            p=L[i][0]
            l=L[i][1]
            r=L[i][2]
            #r=0
            iso=L[i][3]
            Z=L[i][4]
            Loc=L[i][5]
            
            M_s=L[i][6]
            #print(L[0][7])
            

            if iso!=0:
                Final[p][l][r]=1
                Metal[p][l][r]=Z
                Location[p][l][r]=Loc
                Mstars[p][l][r]=M_s/MS
              
                
            """
            else:
                fileA.write(str(A[L[i][0]]))
                fileA.write("\n")

                fileB.write(str(V[L[i][1]]))
                fileB.write("\n")

                fileC.write(str(np.where(df2['semi major axis (au)'] ==L[i][4])))
                fileC.write("\n")

                #print(A[L[i][0]], "  ", V[L[i][1]], "  ", (np.where(df2['semi major axis (au)'] ==L[i][4])))
            """
        
        for i in range(0,len(A)):             #the first 
            for j in range(0,len(V)):
                for k in range(0,q):
                    K=(sum(Final[i,j]))
                    isol[i][j]=K
                    isolp[i][j]=K*100/q          #saving percentages in isol
                
        
        print(isol)
        print(total)
        #print(isolp)
        #print(Final)

        np.save("Metallicity_full",Metal)
        np.save("isolp_full",isolp)
        np.save("Final_full", Final)
        np.save("isol_full",isol)
        np.save("location_full",Location)
        np.save("Mstars_full",Mstars)
        
        
        #print(Metal)
        
        #BASIC GRAPH PRINTING
        #plt.plot(V,isol[0],color='red',label=r'$\alpha = 10^{-5}$')
        #plt.plot(V,isol[1],color='orange',label=r'$\alpha = 10^{-4}$')
        #plt.plot(V,isol[2],color='green',label=r'$\alpha = 10^{-3}$')
        #plt.plot(V,isol[3],color='yellow',label=r'$\alpha = 10^{-2}$')

        #plt.xlabel("vfrag")
        #plt.ylabel("Number of planets isolated")
        #plt.ylim(-1,q+1,1)
        #plt.xlim(50,1050,100)

        #plt.legend()
        #plt.grid()
        #plt.show()

if __name__=="__main__":
    freeze_support()
    main()
    end=time.time()

    if end-start<60:
        print(end-start)
    else:
        print((end-start)/60)

