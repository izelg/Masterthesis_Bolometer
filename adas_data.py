#%%
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
def h_adf15(spectrum=False,rc=False,T_max=200,res=False):
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
    t_n=max(np.argwhere(np.array(temperatures)<=T_max))+1

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

  if spectrum==True:
    ty='EXCIT' #'RECOM', 'ECXIT
    t=0
    d=11
    #for d in np.arange(1,23):
    w_e=[]
    s_e=[]
    w_r=[]
    s_r=[]
    for i,j in zip(all_wl,np.arange(0,len(all_wl))):
      if types[j]=='EXCIT':
        if globals()[i][0] <=10000:
          s_e.append(globals()[i][d][t])
          w_e.append(float(globals()[i][0])*10**(-1))
      if types[j]=='RECOM':
        if globals()[i][0] <=10000:
          s_r.append(globals()[i][d][t])
          w_r.append(float(globals()[i][0])*10**(-1))

      #plt.bar(w_e,s_e,width=10,label=str(temperatures[t])+'eV')
    plt.bar(w_e,s_e,width=10,label=str('%.2E' % densities[d])+'m$^-$$^3$')
      #plt.bar(w_r,s_r,width=10,label=str(temperatures[t])+'eV')
    plt.xlabel('wavelength [nm]')
    plt.ylabel('pec [cm$^3$/s]')
    plt.legend(loc=1,bbox_to_anchor=(1.3,1))
    #plt.suptitle(r'PEC for H from {ty} with $\rho$= {d} cm$^-$$^3$'.format(ty=ty,d='%.2E' % densities[d]))
    plt.suptitle(r'PEC for H from {ty} with T= {t} eV'.format(ty=ty,t='%.2E' % temperatures[t]))
    plt.show()
    data = np.column_stack([np.array(w_e), np.array(s_e)])
    np.savetxt('/home/gediz/Results/ADAS_Data/Spectra/H_Spectrum_excit_T_{t}_eV_D_{d}_m-3.txt'.format(t='%.2E' % temperatures[t],d='%.2E' % densities[d]) ,data, delimiter='\t \t',header='PEC from excitation for H \n for density={d} 1/m^3 and temperature={t} eV \n wavelengths [nm] \t pec [cm^3/s]'.format(t='%.2E' % temperatures[t],d='%.2E' % densities[d]) )
    return(w_e,s_e)

  if rc==True:
    #plt.figure(figsize=(8,5))
    pec_d=[]
    for d in np.arange(1,len(densities)):
      pec=[]
      for t in np.arange(0,len(temperatures)):
        x=[]
        y=[]
        a=0
        for i in np.arange(0,len(all_wl)):
          if types[i]=='EXCIT':
            x.append(wavelengths[i])
            y.append(globals()[all_wl[i]][d][t])
            a+=((h*c/(x[i]*10**(-10)))*y[i])
        pec.append(a/e*m)
      #plt.plot(temperatures[0:t_n[0]],pec[0:t_n[0]],'o--',markersize=10,color=col,alpha=0.4,label=('%.1E' % densities[d-1]) +' m$^-$$^3$')
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
    # plt.plot(temperatures[0:t_n[0]],pec[0:t_n[0]],'o--',label='with averaged \n density',color='#1ba1e9')
    # plt.plot(h_adf11()[0],h_adf11()[1],'bo--',label='data from ADF11')
    # plt.legend(loc='right',bbox_to_anchor=(1.48,0.5),title='Hydrogen')
    # plt.xlabel('temperature [eV]',fontsize=25)
    # plt.ylabel(r'⟨σ$_{ex}$ v$_e$⟩$_{rad}^0$ ⟨E$_{rad}^0$⟩ [eVm$^3$/s]',fontsize=25)
    # #plt.ylabel('excitation energy \n rate coefficients [eVm$^3$/s]',fontsize=25)
    # plt.yscale('log')
    # plt.ylim(8E-15,6E-13)
    # fig1= plt.gcf()
    # plt.show()
    # fig1.savefig('/home/gediz/LaTex/Thesis/Figures/h_adf15
    #.pdf',bbox_inches='tight')
    return temperatures[0:t_n[0]], pec[0:t_n[0]]


# %% H ADF11

#line emission from excitation energy rate coefficients
#excitation energy rate coefficientss in Wcm^3
def h_adf11(T_max=200):
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
  return temperatures[0:t_n[0]], average[0:t_n[0]]



# %% He ADF15
def he_adf15(data='',T_max=200):
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
      t_n=max(np.argwhere(np.array(temperatures)<=T_max))+1
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
  return temperatures[0:t_n[0]],pec[0:t_n[0]]




# %% He ADF11 
#line emission from excitation energy rate coefficients
def he_adf11(T_max=200,res=False):
  densities=[]
  temperatures=[] 
  if res==False:
    data='/home/gediz/ADAS/He/plt89_he.dat'
    all_rad_t_0,all_rad_t_1,all_rad_d_0,all_rad_d_1=[],[],[],[]
  if res==True:
    data='/home/gediz/ADAS/He/plt96r_he.dat'
    all_rad_t_1_1,all_rad_t_1_2,all_rad_t_2_2,all_rad_d_1_1,all_rad_d_1_2,all_rad_d_2_2=[],[],[],[],[],[]
  with open(data, 'r') as f:
    lines=f.readlines()
    if res ==False:
      densities.append(lines[2]+lines[3]+lines[4]+lines[5])
      temperatures.append(lines[6]+lines[7]+lines[8]+lines[9]+lines[10]+lines[11])
    if res==True:
      densities.append(lines[4]+lines[5]+lines[6])
      temperatures.append(lines[7]+lines[8]+lines[9]+lines[10])

    densities=[10**x*1e6 for x in [float(dens) for dens in (densities[0].replace('\n','')).split() ]]
    temperatures=[10**x for x in [float(temp) for temp in (temperatures[0].replace('\n','')).split()]]
    t_n=max(np.argwhere(np.array(temperatures)<=T_max))+1

    if res==False:
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

    if res==True:
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
def ar_adf15(data=''):
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
  return temperatures[0:-1],pec[0:-1]

  # %%
#H--------------------------
#plt.plot(h_adf11(T_max=201)[0],h_adf11(T_max=201)[1],'v--',label='H, ADF11, unresolved, ion charge=0')
# plt.plot(h_adf15(rc=True)[0],h_adf15(rc=True)[1],'v--',label='H, ADF15, unresolved, ion charge=0')
# plt.plot(h_adf15(rc=True,res=True)[0],h_adf15(rc=True,res=True)[1],'v--',label='H, ADF15, resolved, ion charge=0')

#He------------------------
# plt.plot(he_adf11(res=True)[0],he_adf11(res=True)[1],'o--',label='He, ADF11, resolved, ion charge=0, metastables=1')
# plt.plot(he_adf11(res=True)[0],he_adf11(res=True)[2],'o--',label='He, ADF11, resolved, ion charge=0,metastables=2')
#plt.plot(he_adf11(res=False)[0],he_adf11(res=False)[1],'o--',label='He, ADF11, unresolved, ion charge=0')
# plt.plot(he_adf15(data='pec93#he_pjr#he0')[0],he_adf15(data='pec93#he_pjr#he0')[1],'s--',label='He, ADF15, resolved, ion charge=0, 93')
# plt.plot(he_adf15(data='pec96#he_pjr#he0')[0],he_adf15(data='pec96#he_pjr#he0')[1],'s--',label='He, ADF15, resolved, ion charge=0, 96')
# plt.plot(he_adf15(data='pec96#he_pju#he0')[0],he_adf15(data='pec96#he_pju#he0')[1],'s--',label='He, ADF15, unresolved, ion charge=0, 96')

# plt.plot(he_adf11(res=True)[0],he_adf11(res=True)[3],'o--',label='He, ADF11, resolved, ion charge=1,metastables=1')
#plt.plot(he_adf11(res=False)[0],he_adf11(res=False)[2],'o--',label='He, ADF11, unresolved, ion charge=1')
# plt.plot(he_adf15(data='pec96#he_pjr#he1')[0],he_adf15(data='pec96#he_pjr#he1')[1],'s--',label='He, ADF15, resolved, ion charge=1')
# plt.plot(he_adf15(data='pec96#he_pju#he1')[0],he_adf15(data='pec96#he_pju#he1')[1],'s--',label='He, ADF15, unresolved, ion charge=1')
# plt.plot(he_adf15(data='pec96#he_bnd#he1')[0],he_adf15(data='pec96#he_bnd#he1')[1],'s--',label='He, ADF15, bnd?, ion charge=1')

#Ar---------------------------------
plt.plot(ar_adf11()[0],ar_adf11()[1],'D--',label='Ar, ADF11, unresolved, ion charge=0')
plt.plot(ar_adf15(data='pec40#ar_ic#ar0')[0],ar_adf15(data='pec40#ar_ic#ar0')[1],'D--',label='Ar, ADF15, ion charge=0')

plt.plot(ar_adf15(data='pec40#ar_ic#ar1')[0],ar_adf15(data='pec40#ar_ic#ar1')[1],'P--',label='Ar, ADF15, ion charge=1')
plt.plot(ar_adf11()[0],ar_adf11()[2],'P--',label='Ar, ADF11, unresolved, ion charge=1')
plt.ylabel('collisional radiative coefficients [eVm$^3$/s]')
plt.xlabel('temperature [eV]')
plt.yscale('log')
plt.ylim(1E-15,1E-11)
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.8))
plt.show()


# %%
