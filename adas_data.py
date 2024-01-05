#%%

#For H, He, Ar and Ne the ADF15 and ADF11 data files have been collected (see respective Documentation or Adas website or thesis to find out what data each file contains)
#These files can be analyzed with the following functions.
#The ADAS data is stored in complex files with varying syntax according to gas type and age of the data.
#Therefore individual functions to read out the data files were neccessary.
#Most of the bellow functions can put out a value (photo emmissivity constant) depending on the density and temperature you put in.
#As most files don't resolve density you can request a pec for an averaged density.
#The ADF15 files contain line specific rates which allows the plotting of spectra which is possible with the according functions bellow.
#For context on the files analyzed here, again: check my documentation on the ADAs Data or the website it originates from https://open.adas.ac.uk/

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import pandas as pd
import re
#%% Parameter
Poster=False


if Poster==True:
    plt.rc('font',size=18)
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.rcParams['lines.markersize']=18
else:
    plt.rc('font',size=14)
    plt.rc('figure', titlesize=15)
colors=['#1bbbe9','#023047','#ffb703','#fb8500','#c1121f','#780000','#6969B3','#D81159']

h=6.626E-34
c=299792458
e=1.602E-19
m=1E-6
# %% H ADF15
def h_adf15(T_max=200,density='avrg',res=False,Spectrum=False):
  wavelengths=[]
  indices=[]
  densities=[]
  temperatures=[]
  all_wl=[]
  types=[]
  if res==False:
    data='/home/gediz/ADAS/H/pec12#h_pju#h0.dat'
  if res==True:
    data='/home/gediz/ADAS/H/pec96#h_pjr#h0.dat'
  with open(data, 'r') as f:
    lines=f.readlines()
    densities.append(lines[2]+lines[3]+lines[4])
    temperatures.append(lines[5]+lines[6]+lines[7]+lines[8])
    densities=[float(dens)*1e6 for dens in (densities[0].replace('\n','')).split() ]
    temperatures=[float(temp) for temp in (temperatures[0].replace('\n','')).split()]
    t_n=max(np.argwhere(np.array(temperatures)<=T_max))

    for line in lines:
      if not line.startswith('C'):
        if 'A' in line:
          if res==False:
            wavelengths.append(float(line.split()[0][:-1]))
            types.append(line.split()[8])
          if res ==True:
            wavelengths.append(float(line.split()[0]))
            types.append(line.split()[9])
          indices.append(lines.index(line))
    for i,wl,nr,t in zip(indices, wavelengths,np.arange(0,len(wavelengths)+1),types):
      list_name="emis_"+str(wl)+'_'+t
      all_wl.append(list_name)
      globals()[list_name]=[wl]
      for j in np.arange(0,24):
        globals()[list_name].append([float(emis) for emis in (lines[i+8:i+104][j*4]+lines[i+8:i+104][j*4+1]+lines[i+8:i+104][j*4+2]+lines[i+8:i+104][j*4+3]).split()])
  if density=='avrg':
    pec=[]
    for t in np.arange(0,len(temperatures)):
      x=[]
      a=0
      for i in np.arange(0,len(all_wl)):
        if types[i]=='EXCIT':
          x.append(wavelengths[i])
          y=[]
          for d in np.arange(1,len(densities)):    
            y.append(globals()[all_wl[i]][d][t])
          a+=((h*c/(x[i]*10**(-10)))*np.mean(y))
      pec.append(a/e*m)
    return temperatures[0:t_n[0]], pec[0:t_n[0]]
  if type(density)==float and Spectrum==False:
    pec=[]
    for t in np.arange(0,len(temperatures)):
      x=[]
      a=0
      for i in np.arange(0,len(all_wl)):
        if types[i]=='EXCIT':
          x.append(wavelengths[i])
          y=[]
          d=max(np.argwhere(np.array(densities)<=density))
          y.append(globals()[all_wl[i]][d[0]][t])
          a+=((h*c/(x[i]*10**(-10)))*np.mean(y))
      pec.append(a/e*m)
    return temperatures[0:t_n[0]], pec[0:t_n[0]],densities[d[0]],densities
  if Spectrum==True:
    w_e,s_e=[],[]
    d=max(np.argwhere(np.array(densities)<=density))
    t=t_n
    for i,j in zip(all_wl,np.arange(0,len(all_wl))):
      if globals()[i][0]<=10000:
        s_e.append(globals()[i][d[0]][t[0]])
        w_e.append(float(globals()[i][0])*10**(-1))
    #data = np.column_stack([np.array(w_e), np.array(s_e)])
    #np.savetxt('/home/gediz/Results/ADAS_Data/Spectra/H_Spectrum_excit_T_{t}_eV_D_{d}_m-3.txt'.format(t='%.2E' % t[0],d='%.2E' % d[0]) ,data, delimiter='\t \t',header='PEC from excitation for H \n for density={d} 1/m^3 and temperature={t} eV \n wavelengths [nm] \t pec [cm^3/s]'.format(t='%.2E' % t[0],d='%.2E' % d[0]) )
    return w_e,s_e,densities[d[0]],temperatures[t[0]]

# %% H ADF11

#line emission from excitation energy rate coefficients
#excitation energy rate coefficientss in Wcm^3-->eV/m^3
def h_adf11(T_max=200,wish='rad_t_1'):
  densities=[10**x for x in [7.69897,8.00000,8.30103,8.69897,9.00000,9.30103,9.69897,10.00000,10.30103,10.69897,11.00000,11.30103,11.69897,12.00000,12.30103,12.69897,13.00000,13.30103,13.69897,14.00000,14.30103,14.69897,15.00000,15.30103]]
  temperatures=[10**x for x in [-0.69897,-0.52288,-0.30103,-0.15490,0.00000,0.17609, 0.30103,0.47712,0.69897,0.84510,1.00000,1.17609,1.30103,1.47712,1.69897,1.84510,2.00000,2.1760,2.30103,2.47712,2.69897,2.84510,3.00000,3.17609,3.30103,3.47712,3.69897,3.84510,4.00000]]
  t_n=max(np.argwhere(np.array(temperatures)<=T_max))+1

  all_rad_t=[]
  all_rad_d=[]
  with open('/home/gediz/ADAS/H/plt12_h.dat', 'r') as f:
    lines=f.readlines()
    o=0
    for i in np.arange(1,30):
      t_name='rad_t_'+str(i)
      all_rad_t.append(t_name)
      globals()[t_name]=[temperatures[i-1]]
      globals()[t_name].append([((10**x)/e)*m for x in [float(emis) for emis in (lines[9+i+o]+lines[9+i+o+1]+lines[9+i+o+2]).split()]])
      #globals()[t_name].append([10**x for x in [float(emis) for emis in (lines[9+i+o]+lines[9+i+o+1]+lines[9+i+o+2]).split()]])
      o+=2
    for j in np.arange(1,25):
      d_name='rad_d_'+str(j)
      all_rad_d.append(d_name)
      globals()[d_name]=[]
      for t in all_rad_t:
        globals()[d_name].append([globals()[t][1][j-1]])


  #averaged over density like in the Stroth paper
  average=[]
  for k in np.arange(0,29):
    a=[]
    for d in all_rad_d:
      a.append(globals()[d][k])
    average.append(np.mean(a))

  #print(temperatures[8],f'{densities[23]:.2e}',globals()['rad_d_24'][8])
  return temperatures[0:t_n[0]], average[0:t_n[0]], temperatures, densities, average, globals()[wish]



# %% He ADF15
def he_adf15(data='',T_max=200,density='avrg',Spectrum=False):

  wavelengths=[]
  indices=[]
  densities=[]
  temperatures=[]
  all_wl=[]
  types=[]
  if data=='pec93#he_pjr#he0':
    with open('/home/gediz/ADAS/He/pec93#he_pjr#he0.dat', 'r') as f:
      lines=f.readlines()
      densities.append(lines[2]+lines[3])
      temperatures.append(lines[4]+lines[5])
      densities=[float(dens)*1e6 for dens in (densities[0].replace('\n','')).split() ]
      temperatures=[float(temp) for temp in (temperatures[0].replace('\n','')).split()]
      t_n=max(np.argwhere(np.array(temperatures)<=T_max))
      for line in lines:
        if not line.startswith('C'):
          if 'A' in line:
            wavelengths.append(float(line.split()[0]))
            types.append(line.split()[8])
            indices.append(lines.index(line))
      for i,wl,nr,t in zip(indices, wavelengths,np.arange(0,len(wavelengths)+1),types):
        list_name="emis_"+str(wl)+'_'+t+'_'+str(i)
        all_wl.append(list_name)
        globals()[list_name]=[wl]
        for j in np.arange(0,12):
          globals()[list_name].append([float(emis) for emis in (lines[i+5:i+29][j*2]+lines[i+5:i+29][j*2+1]).split()])
 
  if data=='pec96#he_pju#he0' or data=='pec96#he_pjr#he0' or data=='pec96#he_pjr#he1' or data=='pec96#he_pju#he1':
    with open('/home/gediz/ADAS/He/'+data+'.dat', 'r') as f:
      lines=f.readlines()
      densities.append(lines[2]+lines[3]+lines[4])
      temperatures.append(lines[5]+lines[6]+lines[7])
      densities=[float(dens)*1e6 for dens in (densities[0].replace('\n','')).split() ]
      temperatures=[float(temp) for temp in (temperatures[0].replace('\n','')).split()]
      t_n=max(np.argwhere(np.array(temperatures)<=T_max))+1
      for line in lines:
        if not line.startswith('C'):
          if 'A' in line:
            wavelengths.append(float(line.split()[0]))
            types.append(line.split()[9])
            indices.append(lines.index(line))
      for i,wl,nr,t in zip(indices, wavelengths,np.arange(0,len(wavelengths)+1),types):
        list_name="emis_"+str(wl)+'_'+t+'_'+str(i)
        all_wl.append(list_name)
        globals()[list_name]=[wl]
        for j in np.arange(0,24):
          globals()[list_name].append([float(emis) for emis in (lines[i+7:i+79][j*3]+lines[i+7:i+79][j*3+1]+lines[i+7:i+79][j*3+2]).split()])
  
  if data=='pec96#he_bnd#he1':
    with open('/home/gediz/ADAS/He/pec96#he_bnd#he1.dat', 'r') as f:
      lines=f.readlines()
      densities.append(lines[2]+lines[3]+lines[4])
      temperatures.append(lines[5]+lines[6]+lines[7]+lines[8])
      densities=[float(dens)*1e6 for dens in (densities[0].replace('\n','')).split() ]
      temperatures=[float(temp) for temp in (temperatures[0].replace('\n','')).split()]
      t_n=max(np.argwhere(np.array(temperatures)<=T_max))+1
      for line in lines:
        if not line.startswith('C'):
          if 'A' in line:
            wavelengths.append(float(line.split()[0][:-1]))
            types.append(line.split()[8])
            indices.append(lines.index(line))
      for i,wl,nr,t in zip(indices, wavelengths,np.arange(0,len(wavelengths)+1),types):
        list_name="emis_"+str(wl)+'_'+t+'_'+str(i)
        all_wl.append(list_name)
        globals()[list_name]=[wl]
        for j in np.arange(0,24):
          globals()[list_name].append([float(emis) for emis in (lines[i+8:i+104][j*4]+lines[i+8:i+104][j*4+1]+lines[i+8:i+104][j*4+2]+lines[i+8:i+104][j*4+3]).split()])
    
  if density=='avrg':
    pec=[]
    for t in np.arange(0,len(temperatures)):
      x=[]
      a=0
      for i in np.arange(0,len(all_wl)):
        #if types[i]=='RECOM':
        if types[i]=='EXCIT':
          x.append(wavelengths[i])
          y=[]
          z=int(len(wavelengths)/2)
          for d in np.arange(1,len(densities)):    
            y.append(globals()[all_wl[i]][d][t])
          #a+=((h*c/(x[i-z]*10**(-10)))*np.mean(y))
          a+=((h*c/(x[i]*10**(-10)))*np.mean(y))
      pec.append(a/e*m)
    return temperatures[0:t_n[0]], pec[0:t_n[0]]
  if type(density)==float and Spectrum==False:
    pec=[]
    for t in np.arange(0,len(temperatures)):
      x=[]
      a=0
      for i in np.arange(0,len(all_wl)):
        if types[i]=='EXCIT':
          x.append(wavelengths[i])
          y=[]
          d=max(np.argwhere(np.array(densities)<=density))
          y.append(globals()[all_wl[i]][d[0]][t])
          a+=((h*c/(x[i]*10**(-10)))*np.mean(y))
      pec.append(a/e*m)
    return temperatures[0:t_n[0]], pec[0:t_n[0]],densities[d[0]],densities
  if Spectrum==True:
    w_e,s_e=[],[]
    d=max(np.argwhere(np.array(densities)<=density))
    t=t_n
    for i,j in zip(all_wl,np.arange(0,len(all_wl))):
      if globals()[i][0]<=10000:
        if types[j]=='EXCIT':
          s_e.append(globals()[i][d[0]][t[0]])
          w_e.append(float(globals()[i][0])*10**(-1))
    #data = np.column_stack([np.array(w_e), np.array(s_e)])
    #np.savetxt('/home/gediz/Results/ADAS_Data/Spectra/H_Spectrum_excit_T_{t}_eV_D_{d}_m-3.txt'.format(t='%.2E' % t[0],d='%.2E' % d[0]) ,data, delimiter='\t \t',header='PEC from excitation for H \n for density={d} 1/m^3 and temperature={t} eV \n wavelengths [nm] \t pec [cm^3/s]'.format(t='%.2E' % t[0],d='%.2E' % d[0]) )
    return w_e,s_e,densities[d[0]],temperatures[t[0]]


# %% He ADF11 
#line emission from excitation energy rate coefficients
def he_adf11(T_max=200,data=''):
  densities=[]
  temperatures=[] 
  if data=='plt89_he':
    data0='/home/gediz/ADAS/He/plt89_he.dat'
    all_rad_t_0,all_rad_t_1,all_rad_d_0,all_rad_d_1=[],[],[],[]
  if data=='plt96_he':
    data0='/home/gediz/ADAS/He/plt96_he.dat'
    all_rad_t_0,all_rad_t_1,all_rad_d_0,all_rad_d_1=[],[],[],[]
  if data=='prb96_he':
    data0='/home/gediz/ADAS/He/prb96_he.dat'
    all_rad_t_0,all_rad_t_1,all_rad_d_0,all_rad_d_1=[],[],[],[]
  if data=='plt96r_he':
    data0='/home/gediz/ADAS/He/plt96r_he.dat'
    all_rad_t_1_1,all_rad_t_1_2,all_rad_t_2_2,all_rad_d_1_1,all_rad_d_1_2,all_rad_d_2_2=[],[],[],[],[],[]
  with open(data0, 'r') as f:
    lines=f.readlines()
    if data=='plt96_he' or data=='prb96_he':
      densities.append(lines[2]+lines[3]+lines[4])
      temperatures.append(lines[5]+lines[6]+lines[7]+lines[8])
    if data=='plt89_he':
      densities.append(lines[2]+lines[3]+lines[4]+lines[5])
      temperatures.append(lines[6]+lines[7]+lines[8]+lines[9]+lines[10]+lines[11])
    if data=='plt96r_he':
      densities.append(lines[4]+lines[5]+lines[6])
      temperatures.append(lines[7]+lines[8]+lines[9]+lines[10])

    densities=[10**x*1e6 for x in [float(dens) for dens in (densities[0].replace('\n','')).split() ]]
    temperatures=[10**x for x in [float(temp) for temp in (temperatures[0].replace('\n','')).split()]]
    t_n=max(np.argwhere(np.array(temperatures)<=T_max))+1

    if data=='plt89_he':
      o=0
      for i in np.arange(1,49):
        t_name='rad_t_0_'+str(i)
        all_rad_t_0.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
        o+=3

      o=0
      for i in np.arange(1,49):
        t_name='rad_t_1_'+str(i)
        all_rad_t_1.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[205+i+o]+lines[205+i+o+1]+lines[205+i+o+2]+lines[205+i+o+3]).split()]])
        o+=3

      for j in np.arange(1,27):
        d_name='rad_d_0_'+str(j)
        all_rad_d_0.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_0:
          globals()[d_name].append(globals()[t][1][j-1])

      for j in np.arange(1,27):
        d_name='rad_d_1_'+str(j)
        all_rad_d_1.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_1:
          globals()[d_name].append(globals()[t][1][j-1])

      return temperatures[0:t_n[0]],globals()[all_rad_d_0[0]][0:t_n[0]],globals()[all_rad_d_1[0]][0:t_n[0]]
    
    if data=='plt96_he' or data=='prb96_he':
      o=0
      for i in np.arange(1,31):
        t_name='rad_t_0_'+str(i)
        all_rad_t_0.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[9+i+o]+lines[9+i+o+1]+lines[9+i+o+2]).split()]])
        o+=2
      o=0
      for i in np.arange(1,31):
        t_name='rad_t_1_'+str(i)
        all_rad_t_1.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[100+i+o]+lines[100+i+o+1]+lines[100+i+o+2]).split()]])
        o+=2

      for j in np.arange(1,25):
        d_name='rad_d_0_'+str(j)
        all_rad_d_0.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_0:
          globals()[d_name].append(globals()[t][1][j-1])

      for j in np.arange(1,25):
        d_name='rad_d_1_'+str(j)
        all_rad_d_1.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_1:
          globals()[d_name].append(globals()[t][1][j-1])

      return temperatures[0:t_n[0]],globals()[all_rad_d_0[0]][0:t_n[0]],globals()[all_rad_d_1[0]][0:t_n[0]]

    if data=='plt96r_he':
      o=0
      for i in np.arange(1,31):
        t_name='rad_t_1_1'+str(i)
        all_rad_t_1_1.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        #globals()[t_name].append([10**x for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[11+i+o]+lines[11+i+o+1]+lines[11+i+o+2]).split()]])
        o+=2

      o=0
      for i in np.arange(1,31):
        t_name='rad_t_1_2'+str(i)
        all_rad_t_1_2.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        #globals()[t_name].append([10**x for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[102+i+o]+lines[102+i+o+1]+lines[102+i+o+2]).split()]])
        o+=2

      o=0
      for i in np.arange(1,31):
        t_name='rad_t_2_2'+str(i)
        all_rad_t_2_2.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        #globals()[t_name].append([10**x for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[193+i+o]+lines[193+i+o+1]+lines[193+i+o+2]).split()]])
        o+=2

      for j in np.arange(1,25):
        d_name='rad_d_1_1'+str(j)
        all_rad_d_1_1.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_1_1:
          globals()[d_name].append(globals()[t][1][j-1])

      for j in np.arange(1,25):
        d_name='rad_d_1_2'+str(j)
        all_rad_d_1_2.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_1_2:
          globals()[d_name].append(globals()[t][1][j-1])

      for j in np.arange(1,25):
        d_name='rad_d_2_2'+str(j)
        all_rad_d_2_2.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_2_2:
          globals()[d_name].append(globals()[t][1][j-1])

      return temperatures[0:t_n[0]],globals()[all_rad_d_1_1[0]][0:t_n[0]],globals()[all_rad_d_1_2[0]][0:t_n[0]],globals()[all_rad_d_2_2[0]][0:t_n[0]]


# %% Ar ADF11

#line emission from excitation energy rate coefficients
#excitation energy rate coefficientss in Wcm^3
def ar_adf11(T_max=200):
  all_rad_t_0,all_rad_d_0,all_rad_t_1,all_rad_d_1,densities,temperatures=[],[],[],[],[],[]
  with open('/home/gediz/ADAS/Ar/plt89_ar.dat', 'r') as f:
    lines=f.readlines()
    densities.append(lines[2]+lines[3]+lines[4]+lines[5])
    temperatures.append(lines[6]+lines[7]+lines[8]+lines[9]+lines[10]+lines[11])

    densities=[10**x*1e6 for x in [float(dens) for dens in (densities[0].replace('\n','')).split() ]]
    temperatures=[10**x for x in [float(temp) for temp in (temperatures[0].replace('\n','')).split()]]
    t_n=max(np.argwhere(np.array(temperatures)<=T_max))+1

    o=0
    for i in np.arange(1,49):
      t_name='rad_t_0_'+str(i)
      all_rad_t_0.append(t_name)
      globals()[t_name]=[temperatures[i-1]]
      globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
      o+=3

    o=0
    for i in np.arange(1,49):
      t_name='rad_t_1_'+str(i)
      all_rad_t_1.append(t_name)
      globals()[t_name]=[temperatures[i-1]]
      globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[205+i+o]+lines[205+i+o+1]+lines[205+i+o+2]+lines[205+i+o+3]).split()]])
      o+=3

    for j in np.arange(1,27):
      d_name='rad_d_0_'+str(j)
      all_rad_d_0.append(d_name)
      globals()[d_name]=[]
      for t in all_rad_t_0:
        globals()[d_name].append(globals()[t][1][j-1])

    for j in np.arange(1,27):
      d_name='rad_d_1_'+str(j)
      all_rad_d_1.append(d_name)
      globals()[d_name]=[]
      for t in all_rad_t_1:
        globals()[d_name].append(globals()[t][1][j-1])

    return temperatures[0:t_n[0]],globals()[all_rad_d_0[0]][0:t_n[0]],globals()[all_rad_d_1[0]][0:t_n[0]]
# %% Ar ADF15
def ar_adf15(T_max=200,data='',density='avrg',Spectrum=False):
  wavelengths,indices,densities,temperatures,all_wl=[],[],[],[],[]

  if data=='pec40#ar_ca#ar0' or data=='pec40#ar_cl#ar0' or data=='pec40#ar_ic#ar0' or data=='pec40#ar_ls#ar0'or data=='pec40#ar_ic#ar1':
    with open('/home/gediz/ADAS/Ar/'+data+'.dat', 'r') as f:
      lines=f.readlines()
      densities.append(lines[10])
      temperatures.append(lines[11]+lines[12])
      densities=[float(dens)*1e6 for dens in (densities[0].replace('\n','')).split() ]
      temperatures=[float(temp) for temp in (temperatures[0].replace('\n','')).split()]
      for line in lines:
        if not line.startswith('C'):
          if 'pl' in line:
            wavelengths.append(float(line.split()[0]))
            indices.append(lines.index(line))
      for i,wl,nr in zip(indices, wavelengths,np.arange(0,len(wavelengths)+1)):
        list_name="emis_"+str(wl)+'_'+str(i)
        all_wl.append(list_name)
        globals()[list_name]=[wl]
        for j in np.arange(0,7):
          globals()[list_name].append([float(emis) for emis in (lines[i+4:i+18][j*2]+lines[i+4:i+18][j*2+1]).split()])  
  if density=='avrg':
    pec=[]
    for t in np.arange(0,len(temperatures)):
      x=[]
      a=0
      for i in np.arange(0,len(all_wl)):
          x.append(wavelengths[i])
          y=[]
          for d in np.arange(1,len(densities)):    
            y.append(globals()[all_wl[i]][d][t])
          a+=((h*c/(x[i]*10**(-10)))*np.mean(y))
      pec.append(a/e*m)
    return temperatures, pec
  if type(density)==float and Spectrum==False:
    pec=[]
    for t in np.arange(0,len(temperatures)):
      x=[]
      a=0
      for i in np.arange(0,len(all_wl)):
          x.append(wavelengths[i])
          y=[]
          d=max(np.argwhere(np.array(densities)<=density))
          y.append(globals()[all_wl[i]][d[0]][t])
          a+=((h*c/(x[i]*10**(-10)))*np.mean(y))
      pec.append(a/e*m)
    return temperatures, pec,densities[d[0]],densities
  if Spectrum==True:
    w_e,s_e=[],[]
    d=max(np.argwhere(np.array(densities)<=density))
    t=max(np.argwhere(np.array(temperatures)<=T_max))
    for i,j in zip(all_wl,np.arange(0,len(all_wl))):
      if globals()[i][0]<=10000:
        s_e.append(globals()[i][d[0]][t[0]])
        w_e.append(float(globals()[i][0])*10**(-1))
    #data = np.column_stack([np.array(w_e), np.array(s_e)])
    #np.savetxt('/home/gediz/Results/ADAS_Data/Spectra/H_Spectrum_excit_T_{t}_eV_D_{d}_m-3.txt'.format(t='%.2E' % t[0],d='%.2E' % d[0]) ,data, delimiter='\t \t',header='PEC from excitation for H \n for density={d} 1/m^3 and temperature={t} eV \n wavelengths [nm] \t pec [cm^3/s]'.format(t='%.2E' % t[0],d='%.2E' % d[0]) )
    return w_e,s_e,densities[d[0]],temperatures[t[0]]


# %% Ne ADF11
def ne_adf11(T_max=200,data=''):
  densities=[]
  temperatures=[] 
  if data=='plt89_ne':
    data0='/home/gediz/ADAS/Ne/plt89_ne.dat'
    all_rad_t_0,all_rad_t_1,all_rad_d_0,all_rad_d_1=[],[],[],[]
  if data=='plt96_ne':
    data0='/home/gediz/ADAS/Ne/plt96_ne.dat'
    all_rad_t_0,all_rad_t_1,all_rad_d_0,all_rad_d_1=[],[],[],[]
  if data=='plt96r_ne':
    data0='/home/gediz/ADAS/Ne/plt96r_ne.dat'
    all_rad_t_1_1,all_rad_t_1_2,all_rad_t_2_2,all_rad_d_1_1,all_rad_d_1_2,all_rad_d_2_2=[],[],[],[],[],[]
  with open(data0, 'r') as f:
    lines=f.readlines()
    if data=='plt96_ne':
      densities.append(lines[2]+lines[3]+lines[4])
      temperatures.append(lines[5]+lines[6]+lines[7]+lines[8])
    if data=='plt89_ne':
      densities.append(lines[2]+lines[3]+lines[4]+lines[5])
      temperatures.append(lines[6]+lines[7]+lines[8]+lines[9]+lines[10]+lines[11])
    if data=='plt96r_ne':
      densities.append(lines[4]+lines[5]+lines[6])
      temperatures.append(lines[7]+lines[8]+lines[9]+lines[10])

    densities=[10**x*1e6 for x in [float(dens) for dens in (densities[0].replace('\n','')).split() ]]
    temperatures=[10**x for x in [float(temp) for temp in (temperatures[0].replace('\n','')).split()]]
    t_n=max(np.argwhere(np.array(temperatures)<=T_max))+1

    if data=='plt89_ne':
      o=0
      for i in np.arange(1,49):
        t_name='rad_t_0_'+str(i)
        all_rad_t_0.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
        o+=3

      o=0
      for i in np.arange(1,49):
        t_name='rad_t_1_'+str(i)
        all_rad_t_1.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[205+i+o]+lines[205+i+o+1]+lines[205+i+o+2]+lines[205+i+o+3]).split()]])
        o+=3

      for j in np.arange(1,27):
        d_name='rad_d_0_'+str(j)
        all_rad_d_0.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_0:
          globals()[d_name].append(globals()[t][1][j-1])

      for j in np.arange(1,27):
        d_name='rad_d_1_'+str(j)
        all_rad_d_1.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_1:
          globals()[d_name].append(globals()[t][1][j-1])

      return temperatures[0:t_n[0]],globals()[all_rad_d_0[0]][0:t_n[0]],globals()[all_rad_d_1[0]][0:t_n[0]]
    
    if data=='plt96_ne':
      o=0
      for i in np.arange(1,31):
        t_name='rad_t_0_'+str(i)
        all_rad_t_0.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[9+i+o]+lines[9+i+o+1]+lines[9+i+o+2]).split()]])
        o+=2
      o=0
      for i in np.arange(1,31):
        t_name='rad_t_1_'+str(i)
        all_rad_t_1.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[100+i+o]+lines[100+i+o+1]+lines[100+i+o+2]).split()]])
        o+=2

      for j in np.arange(1,25):
        d_name='rad_d_0_'+str(j)
        all_rad_d_0.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_0:
          globals()[d_name].append(globals()[t][1][j-1])

      for j in np.arange(1,25):
        d_name='rad_d_1_'+str(j)
        all_rad_d_1.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_1:
          globals()[d_name].append(globals()[t][1][j-1])

      return temperatures[0:t_n[0]],globals()[all_rad_d_0[0]][0:t_n[0]],globals()[all_rad_d_1[0]][0:t_n[0]]

    if data=='plt96r_ne':
      o=0
      for i in np.arange(1,31):
        t_name='rad_t_1_1'+str(i)
        all_rad_t_1_1.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        #globals()[t_name].append([10**x for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[11+i+o]+lines[11+i+o+1]+lines[11+i+o+2]).split()]])
        o+=2

      o=0
      for i in np.arange(1,31):
        t_name='rad_t_1_2'+str(i)
        all_rad_t_1_2.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        #globals()[t_name].append([10**x for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[102+i+o]+lines[102+i+o+1]+lines[102+i+o+2]).split()]])
        o+=2

      o=0
      for i in np.arange(1,31):
        t_name='rad_t_2_2'+str(i)
        all_rad_t_2_2.append(t_name)
        globals()[t_name]=[temperatures[i-1]]
        #globals()[t_name].append([10**x for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
        globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[193+i+o]+lines[193+i+o+1]+lines[193+i+o+2]).split()]])
        o+=2

      for j in np.arange(1,25):
        d_name='rad_d_1_1'+str(j)
        all_rad_d_1_1.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_1_1:
          globals()[d_name].append(globals()[t][1][j-1])

      for j in np.arange(1,25):
        d_name='rad_d_1_2'+str(j)
        all_rad_d_1_2.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_1_2:
          globals()[d_name].append(globals()[t][1][j-1])

      for j in np.arange(1,25):
        d_name='rad_d_2_2'+str(j)
        all_rad_d_2_2.append(d_name)
        globals()[d_name]=[]
        for t in all_rad_t_2_2:
          globals()[d_name].append(globals()[t][1][j-1])

      return temperatures[0:t_n[0]],globals()[all_rad_d_1_1[0]][0:t_n[0]],globals()[all_rad_d_1_2[0]][0:t_n[0]],globals()[all_rad_d_2_2[0]][0:t_n[0]]

#%% Ne ADF15
def ne_adf15(data='',T_max=200,density='avrg',Spectrum=False):
  wavelengths=[]
  indices=[]
  densities=[]
  temperatures=[]
  all_wl=[]
  types=[]
 
  if data=='pec96#ne_pju#ne0' or data=='pec96#ne_pjr#ne0' or data=='pec96#ne_pjr#ne1' or data=='pec96#ne_pju#ne1':
    with open('/home/gediz/ADAS/Ne/'+data+'.dat', 'r') as f:
      lines=f.readlines()
      densities.append(lines[2]+lines[3]+lines[4])
      temperatures.append(lines[5]+lines[6]+lines[7])
      densities=[float(dens)*1e6 for dens in (densities[0].replace('\n','')).split() ]
      temperatures=[float(temp) for temp in (temperatures[0].replace('\n','')).split()]
      t_n=max(np.argwhere(np.array(temperatures)<=T_max))+1
      for line in lines:
        if not line.startswith('C'):
          if 'A' in line:
            wavelengths.append(float(line.split()[0]))
            types.append(line.split()[9])
            indices.append(lines.index(line))
      for i,wl,nr,t in zip(indices, wavelengths,np.arange(0,len(wavelengths)+1),types):
        list_name="emis_"+str(wl)+'_'+t+'_'+str(i)
        all_wl.append(list_name)
        globals()[list_name]=[wl]
        for j in np.arange(0,24):
          globals()[list_name].append([float(emis) for emis in (lines[i+7:i+79][j*3]+lines[i+7:i+79][j*3+1]+lines[i+7:i+79][j*3+2]).split()])
  
    
  if density=='avrg':
    pec=[]
    for t in np.arange(0,len(temperatures)):
      x=[]
      a=0
      for i in np.arange(0,len(all_wl)):
        if types[i]=='EXCIT':
          x.append(wavelengths[i])
          y=[]
          for d in np.arange(1,len(densities)):    
            y.append(globals()[all_wl[i]][d][t])
          a+=((h*c/(x[i]*10**(-10)))*np.mean(y))
      pec.append(a/e*m)
    return temperatures[0:t_n[0]], pec[0:t_n[0]]
  if type(density)==float and Spectrum==False:
    pec=[]
    for t in np.arange(0,len(temperatures)):
      x=[]
      a=0
      for i in np.arange(0,len(all_wl)):
        if types[i]=='EXCIT':
          x.append(wavelengths[i])
          y=[]
          d=max(np.argwhere(np.array(densities)<=density))
          y.append(globals()[all_wl[i]][d[0]][t])
          a+=((h*c/(x[i]*10**(-10)))*np.mean(y))
      pec.append(a/e*m)
    return temperatures[0:t_n[0]], pec[0:t_n[0]],densities[d[0]],densities
  if Spectrum==True:
    w_e,s_e=[],[]
    d=max(np.argwhere(np.array(densities)<=density))
    t=t_n
    for i,j in zip(all_wl,np.arange(0,len(all_wl))):
      if globals()[i][0]<=10000:
        s_e.append(globals()[i][d[0]][t[0]])
        w_e.append(float(globals()[i][0])*10**(-1))
    #data = np.column_stack([np.array(w_e), np.array(s_e)])
    #np.savetxt('/home/gediz/Results/ADAS_Data/Spectra/H_Spectrum_excit_T_{t}_eV_D_{d}_m-3.txt'.format(t='%.2E' % t[0],d='%.2E' % d[0]) ,data, delimiter='\t \t',header='PEC from excitation for H \n for density={d} 1/m^3 and temperature={t} eV \n wavelengths [nm] \t pec [cm^3/s]'.format(t='%.2E' % t[0],d='%.2E' % d[0]) )
    return w_e,s_e,densities[d[0]],temperatures[t[0]]



# %%
if __name__ == "__main__":
  T=50
  #H--------------------------
 # plt.plot(h_adf11(T_max=T)[0],h_adf11(T_max=T)[1],'bv--',label='H, ADF11, unresolved, ion charge=0')
  # for de in [1E14,1E15,1E16,1E17,1E18,1E19,1E20,1E21,1E22,1E23]:
  #   plt.plot(h_adf15(density=de)[0],h_adf15(density=de)[1],'v--',label='H, ADF15, unresolved, ion charge=0, density= {d}'.format(d='%.2E'%de))
  # plt.plot(h_adf15(density='avrg')[0],h_adf15(density='avrg')[1],'v--',label='H, ADF15, unresolved, ion charge=0, density= average')
  # plt.plot(h_adf15(res=True)[0],h_adf15(res=True)[1],'v--',label='H, ADF15, resolved, ion charge=0')

  #He------------------------
  # plt.plot(he_adf11(res=True)[0],he_adf11(res=True)[1],'o--',label='He, ADF11, resolved, ion charge=0, metastables=1')
  # plt.plot(he_adf11(res=True)[0],he_adf11(res=True)[2],'o--',label='He, ADF11, resolved, ion charge=0,metastables=2')
  #plt.plot(he_adf11(data='plt89_he')[0],he_adf11(data='plt89_he')[1],'ro--',label='He, ADF11, unresolved, ion charge=0, 89')
  plt.plot(he_adf11(data='plt96_he',T_max=T)[0],he_adf11(data='plt96_he',T_max=T)[1],'ro--',label='He, ADF11, unresolved, ion charge=0, 96, excitation')
  plt.plot(he_adf11(data='prb96_he',T_max=T)[0],he_adf11(data='prb96_he',T_max=T)[1],'ro--',label='He, ADF11, unresolved, ion charge=0, 96, bremsstahlung and recombination')

  # for de in [1e+17, 2e+17, 5e+17, 1e+18, 2e+18, 5e+18, 1e+19, 2e+19, 5e+19, 1e+20, 2e+20, 5e+20]:
  #   plt.plot(he_adf15(data='pec93#he_pjr#he0',density=de)[0],he_adf15(data='pec93#he_pjr#he0',density=de)[1],'s--',label='He, ADF15, resolved, ion charge=0, 93, density= {d}'.format(d='%.2E'%de))
  # print(he_adf15(data='pec93#he_pjr#he0',density=de)[2])
  # plt.plot(he_adf15(data='pec93#he_pjr#he0',density='avrg')[0],he_adf15(data='pec93#he_pjr#he0',density='avrg')[1],'s--',label='He, ADF15, resolved, ion charge=0, 93,density=average')
  # plt.plot(he_adf15(data='pec96#he_pjr#he0')[0],he_adf15(data='pec96#he_pjr#he0')[1],'s--',label='He, ADF15, resolved, ion charge=0, 96')
  plt.plot(he_adf15(data='pec96#he_pju#he0')[0],he_adf15(data='pec96#he_pju#he0')[1],'s--',label='He, ADF15, unresolved, ion charge=0, 96')

  # plt.plot(he_adf11(res=True)[0],he_adf11(res=True)[3],'o--',label='He, ADF11, resolved, ion charge=1,metastables=1')
  #plt.plot(he_adf11(data='plt89_he')[0],he_adf11(data='plt89_he')[2],'rs--',label='He, ADF11, unresolved, ion charge=1, 89')
  #plt.plot(he_adf11(data='plt96_he')[0],he_adf11(data='plt96_he')[2],'rs--',label='He, ADF11, unresolved, ion charge=1, 96')
  # plt.plot(he_adf15(data='pec96#he_pjr#he1')[0],he_adf15(data='pec96#he_pjr#he1')[1],'s--',label='He, ADF15, resolved, ion charge=1')
  # plt.plot(he_adf15(data='pec96#he_pju#he1')[0],he_adf15(data='pec96#he_pju#he1')[1],'s--',label='He, ADF15, unresolved, ion charge=1')
  # plt.plot(he_adf15(data='pec96#he_bnd#he1')[0],he_adf15(data='pec96#he_bnd#he1')[1],'s--',label='He, ADF15, bnd?, ion charge=1')

  #Ar---------------------------------
  #plt.plot(ar_adf11(T_max=T)[0],ar_adf11(T_max=T)[1],'gD--',label='Ar, ADF11, unresolved, ion charge=0')

  # for de in [1e+16, 1e+17, 1e+18, 1e+19, 1e+20, 1e+21, 1e+22]:
  #   plt.plot(ar_adf15(data='pec40#ar_ic#ar0',density=de)[0],ar_adf15(data='pec40#ar_ic#ar0',density=de)[1],'D--',label='Ar, ADF15, ion charge=0, density= {d}'.format(d='%.2E'%de))
  # plt.plot(ar_adf15(data='pec40#ar_ic#ar0',density='avrg')[0],ar_adf15(data='pec40#ar_ic#ar0',density='avrg')[1],'D--',label='Ar, ADF15, ion charge=0')

  # # plt.plot(ar_adf15(data='pec40#ar_ic#ar1')[0],ar_adf15(data='pec40#ar_ic#ar1')[1],'P--',label='Ar, ADF15, ion charge=1')
  #plt.plot(ar_adf11()[0],ar_adf11()[2],'rP--',label='Ar, ADF11, unresolved, ion charge=1')

  #Ne---------------------------------------------------------------------
  #plt.plot(ne_adf11(data='plt96_ne',T_max=T)[0],ne_adf11(data='plt96_ne',T_max=T)[1],'ys--',label='Ne, ADF11, unresolved, ion charge=0, 96')
  #plt.plot(ne_adf11(data='plt89_ne')[0],ne_adf11(data='plt89_ne')[1],'ro--',label='Ne, ADF11, unresolved, ion charge=0, 89')
  #plt.plot(ne_adf11(data='plt96r_ne')[0],ne_adf11(data='plt96r_ne')[1],'go--',label='Ne, ADF11, resolved, ion charge=0, 96')
  #plt.plot(ne_adf11(data='plt96_ne')[0],ne_adf11(data='plt96_ne')[2],'bs--',label='Ne, ADF11, unresolved, ion charge=1, 96')
  #plt.plot(ne_adf11(data='plt89_ne')[0],ne_adf11(data='plt89_ne')[2],'rs--',label='Ne, ADF11, unresolved, ion charge=1, 89')
  #plt.plot(ne_adf11(data='plt96r_ne')[0],ne_adf11(data='plt96r_ne')[2],'gs--',label='Ne, ADF11, resolved, ion charge=1, 96')

  #plt.plot(ne_adf15(data='pec96#ne_pjr#ne0')[0],ne_adf15(data='pec96#ne_pjr#ne0')[1],'go--',label='Ne, ADF15, resolved, ion charge=0, 96')
  #plt.plot(ne_adf15(data='pec96#ne_pju#ne0')[0],ne_adf15(data='pec96#ne_pju#ne0')[1],'yo--',label='Ne, ADF15, unresolved, ion charge=0, 96')
  #plt.plot(ne_adf15(data='pec96#ne_pjr#ne1')[0],ne_adf15(data='pec96#ne_pjr#ne1')[1],'gs--',label='Ne, ADF15, resolved, ion charge=1')
  #plt.plot(ne_adf15(data='pec96#ne_pju#ne1')[0],ne_adf15(data='pec96#ne_pju#ne1')[1],'ys--',label='Ne, ADF15, unresolved, ion charge=1')


  plt.ylabel('collisional radiative coefficients [eVm$^3$/s]')
  plt.xlabel('temperature [eV]')
  plt.yscale('log')
  plt.ylim(1E-20,1E-12)
  plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.8))
  plt.show()



