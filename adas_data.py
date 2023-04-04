#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import pandas as pd
import re
#%% Parameter
Poster=True


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
# %% pec93#he_pjr#he0.dat
def pec_he(spectrum=False,rc=False):
  wavelengths=[]
  indices=[]
  densities=[]
  temperatures=[]
  all_wl=[]
  types=[]
  with open('/home/gediz/ADAS/pec93#he_pjr#he0.dat', 'r') as f:
    lines=f.readlines()
    densities.append(lines[2]+lines[3])
    temperatures.append(lines[4]+lines[5])
    densities=[float(dens)*1e6 for dens in (densities[0].replace('\n','')).split() ]
    temperatures=[float(temp) for temp in (temperatures[0].replace('\n','')).split()]

    for line in lines:
      if not line.startswith('C'):
        if 'A' in line:
          wavelengths.append(float(line.split()[0]))
          types.append(line.split()[8])
          indices.append(lines.index(line))
    for i,wl,nr,t in zip(indices, wavelengths,np.arange(0,len(wavelengths)+1),types):
      list_name="emis_"+str(wl)+t
      all_wl.append(list_name)
      globals()[list_name]=[wl]
      for j in [0,2,4,6,8,10,12,14,16,18,20,22]:
        globals()[list_name].append([float(emis) for emis in (lines[i+5:i+29][j]+lines[i+5:i+29][j+1]).split()])

  if spectrum ==True:
    ty='EXCIT' #'RECOM', 'ECXIT'
    t=3
    d=1
    #for d in np.arange(1,12):
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
    #plt.suptitle(r'PEC for He from {ty} with $\rho$= {d} cm$^-$$^3$'.format(ty=ty,d='%.2E' % densities[d]))
    plt.suptitle(r'PEC for He from {ty} with T= {t} eV'.format(ty=ty,t='%.2E' % temperatures[t]))
    plt.show()
    data = np.column_stack([np.array(w_e), np.array(s_e)])
    np.savetxt('/home/gediz/Results/ADAS_Data/Spectra/He_Spectrum_excit_T_{t}_eV_D_{d}_m-3.txt'.format(t='%.2E' % temperatures[t],d='%.2E' % densities[d]) ,data, delimiter='\t \t',header='PEC from excitation for H \n for density={d} 1/m^3 and temperature={t} eV \n wavelengths [nm] \t pec [cm^3/s]'.format(t='%.2E' % temperatures[t],d='%.2E' % densities[d]) )
    return(w_e,s_e)

  if rc==True:  
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
      plt.plot(temperatures[0:20],pec[0:20],'o--',label=('%.2E' % densities[d-1]) +' cm$^-$$^3$')

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
    plt.plot(temperatures[0:20],pec[0:20],'ro--',label='with averaged density')
    #plt.plot(erc_he()[0][0:24],erc_he()[1][0:24],'bo--',label='data from ADF11')
    plt.legend(loc=1,bbox_to_anchor=(1.4,1))
    plt.xlabel('temperatures [eV]')
    plt.ylabel('excitation energy rate coefficients [eVm$^3$/s]')
    plt.yscale('log')
    #plt.ylim(1E-16,1E-11)
    plt.show()



# %% pec12#h_pju#h0.dat
def pec_h(spectrum=False,rc=False):
  wavelengths=[]
  indices=[]
  densities=[]
  temperatures=[]
  all_wl=[]
  types=[]
  with open('/home/gediz/ADAS/pec12#h_pju#h0.dat', 'r') as f:
    lines=f.readlines()
    densities.append(lines[2]+lines[3]+lines[4])
    temperatures.append(lines[5]+lines[6]+lines[7]+lines[8])
    densities=[float(dens)*1e6 for dens in (densities[0].replace('\n','')).split() ]
    temperatures=[float(temp) for temp in (temperatures[0].replace('\n','')).split()]

    for line in lines:
      if not line.startswith('C'):
        if 'A' in line:
          wavelengths.append(float(line.split()[0][:-1]))
          types.append(line.split()[8])
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
    plt.figure(figsize=(8,5))
    pec_d=[]
    for d,col in zip([2,5,8,11,14,17,20,23],colors):#np.arange(1,len(densities)):
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
      plt.plot(temperatures[0:19],pec[0:19],'o--',markersize=10,color=col,alpha=0.4,label=('%.1E' % densities[d-1]) +' m$^-$$^3$')
      pec_d.append(pec[10])
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
    plt.plot(temperatures[0:19],pec[0:19],'o--',label='with averaged \n density',color='#1ba1e9')
    #plt.plot(erc_h()[0],erc_h()[1],'bo--',label='data from ADF11')
    plt.legend(loc='right',bbox_to_anchor=(1.48,0.5),title='Hydrogen')
    plt.xlabel('temperature [eV]',fontsize=25)
    plt.ylabel(r'⟨σ$_{ex}$ v$_e$⟩$_{rad}^0$ ⟨E$_{rad}^0$⟩ [eVm$^3$/s]',fontsize=25)
    #plt.ylabel('excitation energy \n rate coefficients [eVm$^3$/s]',fontsize=25)
    plt.yscale('log')
    plt.ylim(8E-15,6E-13)
    fig1= plt.gcf()
    plt.show()
    fig1.savefig('/home/gediz/LaTex/Thesis/Figures/pec_h.pdf',bbox_inches='tight')

    # plt.plot(densities[0:23],pec_d,'o--')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    #return temperatures[0:20], pec[0:20]
# %% plt12_h  
#line emission from excitation energy rate coefficients
#excitation energy rate coefficientss in Wcm^3
def erc_h():
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
      globals()[t_name].append([((10**x)/e)*m for x in [float(emis) for emis in (lines[9+i+o]+lines[9+i+o+1]+lines[9+i+o+2]).split()]])
      #globals()[t_name].append([10**x for x in [float(emis) for emis in (lines[9+i+o]+lines[9+i+o+1]+lines[9+i+o+2]).split()]])
      o+=2
    for j in np.arange(1,25):
      d_name='rad_d_'+str(j)
      all_rad_d.append(d_name)
      globals()[d_name]=[]
      for t in all_rad_t:
        globals()[d_name].append([globals()[t][1][j-1]])
  # for t in all_rad_t:
  #   plt.plot(densities,globals()[t][1],'.',markersize=10,label=(str(round(globals()[t][0],2)))+' eV')
  #   plt.ylabel('excitation energy rate coefficients [eVm$^3$/s]')
  #   plt.xlabel(r'density [cm$^-$$^3$]')
  #   plt.xscale('log')
  # plt.legend(loc=1,bbox_to_anchor=(1.3,1))
  # plt.show()

  # for d,k in zip(all_rad_d,np.arange(0,23)):
  #   plt.plot(temperatures[0:20], globals()[d][0:20],'.',markersize=10,label=('%.2E' % densities[k]) +' cm$^-$$^3$')
  #   plt.ylabel('excitation energy rate coefficients [eVm$^3$/s]')
  #   plt.xlabel('temperature [eV]')
  #   #plt.xscale('log')
  # plt.legend(loc=1,bbox_to_anchor=(1.3,1))
  # plt.show

  #averaged over density like in the Stroth paper
  average=[]
  for k in np.arange(0,29):
    a=[]
    for d in all_rad_d:
      a.append(globals()[d][k])
    average.append(np.mean(a))
  plt.plot(temperatures[0:16],average[0:16],'bo--')
  plt.plot([1,1.5,2,3,5,7,10,15,20,30,50,70],[(x/e)*m for x in [1.87E-30,5.14E-29,2.80E-28,1.51E-27,5.81E-27,1.09E-26,1.73E-26,2.56E-26,3.20E-26,4.05e-26,4.90E-26,5.27E-26]],'ro--')
  plt.ylabel('excitation energy rate coefficients [eVm$^3$/s]')
  plt.xlabel('temperature [eV]')
  plt.show
  #print(temperatures[8],f'{densities[23]:.2e}',globals()['rad_d_24'][8])
  return temperatures[0:20], average[0:20]




# %% plt89_he  
#line emission from excitation energy rate coefficients
#excitation energy rate coefficientss in Wcm^3
def erc_he():
  densities=[]
  temperatures=[] 
  all_rad_t_0,all_rad_t_1,all_rad_d_0,all_rad_d_1=[],[],[],[]
  with open('/home/gediz/ADAS/plt89_he.dat', 'r') as f:
    lines=f.readlines()
    densities.append(lines[2]+lines[3]+lines[4]+lines[5])
    temperatures.append(lines[6]+lines[7]+lines[8]+lines[9]+lines[10]+lines[11])
    densities=[10**x for x in [float(dens) for dens in (densities[0].replace('\n','')).split() ]]
    temperatures=[10**x for x in [float(temp) for temp in (temperatures[0].replace('\n','')).split()]]

    o=0
    for i in np.arange(1,49):
      t_name='rad_t_0_'+str(i)
      all_rad_t_0.append(t_name)
      globals()[t_name]=[temperatures[i-1]]
      #globals()[t_name].append([10**x for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
      globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
      o+=3

    o=0
    for i in np.arange(1,49):
      t_name='rad_t_1_'+str(i)
      all_rad_t_1.append(t_name)
      globals()[t_name]=[temperatures[i-1]]
      #globals()[t_name].append([10**x for x in [float(emis) for emis in (lines[12+i+o]+lines[12+i+o+1]+lines[12+i+o+2]+lines[12+i+o+3]).split()]])
      globals()[t_name].append([((10**x)/e*m) for x in [float(emis) for emis in (lines[205+i+o]+lines[205+i+o+1]+lines[205+i+o+2]+lines[205+i+o+3]).split()]])
      o+=3

    for j in np.arange(1,27):
      d_name='rad_d_0_'+str(j)
      all_rad_d_0.append(d_name)
      globals()[d_name]=[]
      for t in all_rad_t_0:
        globals()[d_name].append(globals()[t][1][j-1])

    for j in np.arange(1,27):
      d_name='rad_d_01_'+str(j)
      all_rad_d_1.append(d_name)
      globals()[d_name]=[]
      for t in all_rad_t_1:
        globals()[d_name].append(globals()[t][1][j-1])
  # for t in all_rad_t:
  #   plt.plot(densities,globals()[t][1],'.',markersize=10,label=(str(round(globals()[t][0],2)))+' eV')
  #   plt.ylabel('excitation energy rate coefficients [eVcm$^3$/s]')
  #   plt.xlabel(r'density [cm$^-$$^3$]')
  #   plt.xscale('log')
  # plt.legend(loc=1,bbox_to_anchor=(1.3,1))
  # plt.show()
  # for d,k in zip(all_rad_d,np.arange(0,27)):
  #   plt.plot(temperatures, globals()[d],'.',markersize=10,label=('%.2E' % densities[k]) +' cm$^-$$^3$')
  #   plt.ylabel('excitation energy rate coefficients [eVcm$^3$/s]')
  #   plt.xlabel('temperature [eV]')
  #   plt.xscale('log')
  # plt.legend(loc=1,bbox_to_anchor=(1.3,1))
  # plt.sho
  #averaged over density like in the Stroth paper

  #plt.plot(temperatures[0:20],globals()[all_rad_d[0]][0:20],'bo--',label='He')
  #plt.plot([1,1.5,2,3,5,7,10,15,20,30,50,70],[(x/e)*m for x in [1.87E-30,5.14E-29,2.80E-28,1.51E-27,5.81E-27,1.09E-26,1.73E-26,2.56E-26,3.20E-26,4.05e-26,4.90E-26,5.27E-26]],'ro--',label='H')
  # plt.ylabel('excitation energy rate coefficients [eVm$^3$/s]')
  # plt.xlabel('temperature [eV]')
  # plt.ylim(1E-17,1E-12)
  # plt.yscale('log')
  # plt.legend()
  # plt.show

  #print(rad_d_1)
  #print(temperatures[3],f'{densities[7]:.2e}',globals()['rad_d_8'][3])
  return temperatures, globals()[all_rad_d_0[0]]
  # %%
pec_h(rc=True)
# %%
