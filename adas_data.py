#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import pandas as pd
import re
plt.rcParams["figure.figsize"] = (10,10)
plt.rc('figure', titlesize=15)
plt.rc('font',size=14)

# %%

#define differently named variables with a for loop
wavelengths=[]
indices=[]
densities=[]
temperatures=[]
all_wl=[]
with open('/home/gediz/ADAS/pec93#he_pjr#he0.dat', 'r') as f:
  lines=f.readlines()
  densities.append(lines[2]+lines[3])
  temperatures.append(lines[4]+lines[5])
  densities=[float(dens) for dens in (densities[0].replace('\n','')).split() ]
  temperatures=[float(temp) for temp in (temperatures[0].replace('\n','')).split()]

  for line in lines:
    if not line.startswith('C'):
      if 'A' in line:
        wavelengths.append(float(line.split()[0]))
        indices.append(lines.index(line))
  for i,wl,nr in zip(indices, wavelengths,np.arange(0,len(wavelengths)+1)):
    list_name="emis_"+str(nr)
    all_wl.append(list_name)
    globals()[list_name]=[wl]
    for j in [0,2,4,6,8,10,12,14,16,18,20,22]:
      globals()[list_name].append([float(emis) for emis in (lines[i+5:i+29][j]+lines[i+5:i+29][j+1]).split()])


for temperature in np.arange(0,12):
  y=[]
  wl=[]
  density=2
  for e in all_wl:
    y.append(globals()[e][density][temperature])
    wl.append(globals()[e][0])
  plt.plot(wl,y,'o')
  print(y)
plt.show()

# %%
#plt12_h  
#line emission from excitation
#radiated powers in Wcm^3
densities=[10**x for x in [7.69897,8.00000,8.30103,8.69897,9.00000,9.30103,9.69897,10.00000,10.30103,10.69897,11.00000,11.30103,11.69897,12.00000,12.30103,12.69897,13.00000,13.30103,13.69897,14.00000,14.30103,14.69897,15.00000,15.30103]]
temperatures=[10**x for x in [-0.69897,-0.52288,-0.30103,-0.15490,0.00000,0.17609, 0.30103,0.47712,0.69897,0.84510,1.00000,1.17609,1.30103,1.47712,1.69897,1.84510,2.00000,2.1760,2.30103,2.47712,2.69897,2.84510,3.00000,3.17609,3.30103,3.47712,3.69897,3.84510,4.00000]]
all_rad_t=[]
all_rad_d=[]
with open('/home/gediz/ADAS/plt12_h.dat', 'r') as f:
  lines=f.readlines()
  o=0
  for i in np.arange(1,30):
    t_name='rad_t_'+str(i)
    all_rad_t.append(t_name)
    globals()[t_name]=[temperatures[i-1]]
    globals()[t_name].append([10**x for x in [float(emis) for emis in (lines[9+i+o]+lines[9+i+o+1]+lines[9+i+o+2]).split()]])
    o+=2
  for j in np.arange(1,25):
    d_name='rad_d_'+str(j)
    all_rad_d.append(d_name)
    globals()[d_name]=[]
    for t in all_rad_t:
      globals()[d_name].append([globals()[t][1][j-1]])
for t in all_rad_t:
  plt.plot(densities,globals()[t][1],'.',markersize=10,label=(str(round(globals()[t][0],2)))+' eV')
  plt.ylabel('radiated power [Wcm$^3$]')
  plt.xlabel(r'density [cm$^-$$^3$]')
  plt.xscale('log')
plt.legend(loc=1,bbox_to_anchor=(1.3,1))
plt.show()
for d,k in zip(all_rad_d,np.arange(0,24)):
  plt.plot(temperatures, globals()[d],'.',markersize=10,label=('%.2E' % densities[k]) +' cm$^-$$^3$')
  plt.ylabel('radiated power [Wcm$^3$]')
  plt.xlabel('temperature [eV]')
  plt.xscale('log')
plt.legend(loc=1,bbox_to_anchor=(1.3,1))
plt.show

# %%
