#%%

#Written by: Izel Gediz
#Date of Creation: 11.11.2022


from ast import increment_lineno
from pdb import line_prefix
from unicodedata import name
from blinker import Signal
from click import style
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import statistics
import os
import itertools
from scipy.interpolate import interp1d
from scipy import integrate
#%%
Poster=True


if Poster==True:
    plt.rc('font',size=20)
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.rcParams['lines.markersize']=12
else:
    plt.rc('font',size=14)
    plt.rc('figure', titlesize=15)
#colors=['#1bbbe9','#023047','#ffb703','#fb8500','#c1121f','#780000','#6969B3','#D81159','#04E762','#89FC00','#03CEA4','#04A777','#537A5A','#FF9B71','#420039','#D81159']
colors=['#03045E','#0077B6','#00B4D8','#370617','#9D0208','#DC2F02','#F48C06','#FFBA08','#3C096C','#7B2CBF','#C77DFF','#2D6A4F','#40916C','#52B788','#03045E','#0077B6','#00B4D8']
markers=['o','v','s','P','p','D','*','x','o','v','s','P','p','D','*','x','o','v','s']


# %%
def LoadData(location):
    with open(location, "r") as f:
        cols = f.readlines()[3]
        cols = re.sub(r"\t+", ';', cols)[:-2].split(';')
    data = pd.read_csv(location, skiprows=4, sep="\t\t", names=cols, engine='python')
    return data

def Pressure(shotnumber,gas):
    location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
    y= LoadData(location)["Pressure"]
    time = LoadData(location)['Zeit [ms]'] / 1000
    pressure= np.mean(y[0:100])
    d = 9.33                # according to PKR261 manual
    pressure = 10.**(1.667*pressure-d)*1000
    if gas == 'H':
        corr = 2.4
    elif gas == 'D':
        print( '    you have choosen deuterium as gas, no calibration factor exists for this gas' )
        print( '    the same factor as for hydrogen will be used' )
        corr = 2.4
    elif gas == 'He':
        corr =5.9
    elif gas == 'Ne':
        corr =4.1
    elif gas == 'Ar':
        corr =.8
    elif gas == 'Kr':
        corr =.5
    elif gas == 'Xe':
        corr =.4
    return pressure*corr

def GetMicrowavePower(shotnumber):
    location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
    time = np.array(LoadData(location)['Zeit [ms]'] / 1000)[:,None]
    z= LoadData(location)['2 GHz Richtk. forward']
    w= LoadData(location)['8 GHz power']

    height_z = abs(max(z)-min(z))
    height_w = abs(max(w)-min(w))
    if height_w >= 0.07:        #This is the part where the code finds out if 8 or 2GHz MW heating was used. Change the signal height if MW powers used change in the future
        MW = '8 GHz'
    elif height_z >= 0.07: 
        MW = '2.45 GHz'
    else:
        MW = 'none'
    if MW=='2.45 GHz':
        U_in_for=LoadData(location)['2 GHz Richtk. forward']
        U_in_back=LoadData(location)['2 GHz Richtk. backward']
        U_in_for[U_in_for>0]    = -1e-6
        U_in_back[U_in_back>0]    = -1e-6
        signal_dBm_for  = (42.26782054007 + (-28.92407247331 - 42.26782054007) / ( 1. + (U_in_for / (-0.5508373840567) )**0.4255365582241 ))+60.49
        signal_dBm_back  = (42.26782054007 + (-28.92407247331 - 42.26782054007) / ( 1. + (U_in_back / (-0.5508373840567) )**0.4255365582241 ))+60.11
        signalinwatt_for   = 10**(signal_dBm_for/10.) * 1e-3
        signalinwatt_back   =10**(signal_dBm_back/10.) * 1e-3
        start=np.argmax(np.gradient(signalinwatt_for))
        stop=np.argmin(np.gradient(signalinwatt_for))
        return (np.mean(signalinwatt_for[start:stop])-np.mean(signalinwatt_back[start:stop]),MW)
    if MW=='8 GHz':
        U_in=LoadData(location)['8 GHz power']*1E3
        a1  = 17.5637
        a2  = 0.332023
        a3  = 0.458919 
        P= a1 * np.exp(a2 * np.abs(U_in)**a3)

        return(np.mean(np.sort(P)[-100:-1]),MW)
    
#This function extracts the mean values of all timetraces recorded with the 2D probe. 
#This of course can also be achieved by saving the ascii file provided by the 2D Probe scan Labview Programm.
#However if this step was forgotten or didn't work this function can be used.
#It needs however the extracted dat files from the hdf files with all timetraces (use char_hdf2ascii__allout)
def ExtractMeanValues():
    mean_U,mean_I,mean_Isat,mean_bolo,mean_inter,mean_pos=([] for i in range(6))
    for i in ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']:
        char_U, char_I, I_isat, Position,Bolo_sum,Interferometer=np.genfromtxt('/home/gediz/Measurements/Plasma_characteristics/2D_Probe/shot{s}/0000{n}.dat'.format(s=shotnumber,n=i),unpack=True)
        mean_U.append(np.mean(char_U))
        mean_I.append(np.mean(char_I))
        mean_Isat.append(np.mean(I_isat))
        mean_bolo.append(np.mean(Bolo_sum))
        mean_inter.append(np.mean(Interferometer))
        mean_pos.append(np.mean(Position))
    data = np.column_stack([np.array(mean_pos), np.array(mean_U), np.array(mean_I), np.array(mean_Isat), np.array(mean_bolo), np.array(mean_inter)])
    np.savetxt(str(infile)+"shot{}.dat".format(shotnumber) , data, delimiter='\t \t', fmt='%10.6f')

#This function plots the mean values of the 2D Probe measurements.
#Here e.g. the relative density Profiles form Ion satturation currents can be visualized.
#It is also possible to compare a list of shots as specified before calling the function
def PlotMeanValues(compare=False,save=False):
    if compare==True:
        for i in shotnumbers:
            Position, char_U, char_I, I_isat=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=i),unpack=True,usecols=(0,1,2,3))
            plt.plot(Position, I_isat,label='shot n°{s}, P$_m$$_w$= {mw} W,\n p= {p} mPa '.format(s=i,mw=float(f'{GetMicrowavePower(i):.3f}'),p=float(f'{Pressure(i,gas):.3f}')))
        plt.xlabel('position R [m]')
        plt.ylabel('ion satturation current [mA]')
        plt.suptitle('Comparison of Ion saturation currents from 2D-probe scans for {} '.format(gas))
        plt.legend(loc=1, bbox_to_anchor=(1.9,1))    
        fig1= plt.gcf()
        plt.show()
        # for i in shotnumbers:
        #     Position, char_U, char_I, I_isat, Bolo_sum, Interferometer=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=i),unpack=True)
        #     plt.plot(Position,Bolo_sum,label='shot{s}, MW: {mw} Watt , Pressure: {p} mPa '.format(s=i,mw=float(f'{GetMicrowavePower(i):.3f}'),p=float(f'{Pressure(i):.3f}')))
        # plt.xlabel('Position R [m]')
        # plt.ylabel('Bolo_sum Channel Signal [V]')
        # plt.suptitle('Comparison of Bolometer sum channel signals from 2D Probe scans')
        # plt.legend(loc=1, bbox_to_anchor=(1.7,1))    
        # fig2= plt.gcf()
        # plt.show()
        # for i in shotnumbers:
        #     Position, char_U, char_I, I_isat, Bolo_sum, Interferometer=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=i),unpack=True)
        #     plt.plot(Position, Interferometer,label='shot{s}, MW: {mw} Watt , Pressure: {p} mPa '.format(s=i,mw=float(f'{GetMicrowavePower(i):.3f}'),p=float(f'{Pressure(i):.3f}')))
        # plt.xlabel('Position R [m]')
        # plt.ylabel('Interferometer Signal [V]')
        # plt.suptitle('Comparison of Interferometersignals from 2D Probe scans')
        # plt.legend(loc=1, bbox_to_anchor=(1.7,1))    
        # fig3= plt.gcf()
        # plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"comparisons/{g}/comparison_of_shots{s}_Ion_saturation_{g}.pdf".format(s=shotnumbers,g=gas), bbox_inches='tight')
            # fig2.savefig(str(outfile)+"comparisons/comparison_of_shots{}_Bolo_sum.pdf".format(shotnumbers), bbox_inches='tight')
            # fig3.savefig(str(outfile)+"comparisons/comparison_of_shots{}_Interferometer.pdf".format(shotnumbers), bbox_inches='tight')
 
    else:
        Position, char_U, char_I, I_isat,Bolo_sum, Interferometer=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=shotnumber),unpack=True)
        plt.plot(Position, I_isat)
        plt.xlabel('position R [m]')
        plt.ylabel('ion satturation current [mA]')
        plt.suptitle('2D Probe scan of shot {s} // {g}\n Ion saturation current $\propto$ n \n P$_m$$_w$ = {mw} W // p= {p} mPa '.format(s=shotnumber,g=gas,mw=float(f'{GetMicrowavePower(shotnumber):.3f}'),p=float(f'{Pressure(shotnumber,gas):.3f}')),y=1.05)
        fig1= plt.gcf()
        plt.show()
        # plt.plot(Position,Bolo_sum)
        # plt.xlabel('Position R [m]')
        # plt.ylabel('Bolo_sum Channel Signal [V]')
        # plt.suptitle('2D Probe scan of shot {s} // {g} \n Bolometer sum Channel signal\n MW: {mw} Watt // Pressure: {p} mPa '.format(s=shotnumber,g=gas,mw=float(f'{GetMicrowavePower(shotnumber):.3f}'),p=float(f'{Pressure(shotnumber):.3f}')),y=1.05)
        # fig2= plt.gcf()
        # plt.show()
        # plt.plot(Position, Interferometer)
        # plt.xlabel('Position R [m]')
        # plt.ylabel('Interferometer Signal[V]')
        # plt.suptitle('2D Probe scan of shot {s} // {g}\n Interferometer Signal\n MW: {mw} Watt // Pressure: {p} mPa '.format(s=shotnumber,g=gas,mw=float(f'{GetMicrowavePower(shotnumber):.3f}'),p=float(f'{Pressure(shotnumber):.3f}')),y=1.05)
        # fig3= plt.gcf()
        # plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"shot{s}/2D_Probe_Singal_mean_values_of_I_sat_{g}.pdf".format(s=shotnumber,g=gas), bbox_inches='tight')
            #fig2.savefig(str(outfile)+"shot{s}/2D_Probe_Singal_mean_values_of_Bolo_sum_{g}.pdf".format(s=shotnumber,g=gas), bbox_inches='tight')
            #fig3.savefig(str(outfile)+"shot{s}/2D_Probe_Singal_mean_values_of_Interferometer_{g}.pdf".format(s=shotnumber,g=gas), bbox_inches='tight')


#This function plots the Temperatures received from fitting to the characteristics.
#It is also possible to compare a list of shots as specified before calling the function
def TemperatureProfile(s,Type='',ScanType='',save=False):

    if Type=='Compare':
        plt.figure(figsize=(10,7))
        pressure,mw=[],[]
        for i in shotnumbers:
            pressure.append(Pressure(i,gas))
            mw.append(GetMicrowavePower(i)[0])
        if ScanType=='Pressure':
            sortnumbers=[shotnumbers[i] for i in np.argsort(pressure)]
        if ScanType=='Power':
            sortnumbers=[shotnumbers[i] for i in np.argsort(mw)]
        if ScanType=='None':
            sortnumbers=shotnumbers
        for i,c,m in zip(sortnumbers,colors,markers):
            if ScanType=='Pressure':
                label='shot n°{s}, p= {p} mPa'.format(s=i,p=float(f'{Pressure(i,gas):.1f}'))
                title= r'{g}, MW= {m}, P$_M$$_W$ $\approx$ {mw} kW'.format(g=gas,m=GetMicrowavePower(i)[1],mw=float(f'{np.mean(mw)*10**(-3):.2f}'))
            if ScanType=='Power':
                label='shot n°{s}, P$_M$$_W$ = {mw} kW'.format(s=i,mw=float(f'{GetMicrowavePower(i)[0]*10**(-3):.2f}'))
                title= r'{g}, MW= {m}, p $\approx$ {p} mPa'.format(g=gas,m=GetMicrowavePower(i)[1],p=float(f'{np.mean(pressure):.1f}'))
            if ScanType=='None':
                label='shot n°{s}, P$_M$$_W$ = {mw} kW, p= {p} mPa'.format(s=i,mw=float(f'{GetMicrowavePower(i)[0]*10**(-3):.2f}'),p=float(f'{Pressure(i,gas):.1f}'))
                title= r'{g}, MW= {m}'.format(g=gas,m=GetMicrowavePower(i)[1])
            Position, T=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}Te.dat'.format(s=i),unpack=True)
            plt.plot(Position, T,linewidth=3,color=c,marker=m,label=label)#
        plt.ylim(bottom=0)
        plt.xlabel('position R- r$_0$ [m]',fontsize=30)
        plt.ylabel('temperature [eV]',fontsize=30)
        plt.legend(loc=1, bbox_to_anchor=(1.8,1),title=title)  
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"comparisons/{g}/comparison_of_shots{s}_temperatureprofiles_{g}.pdf".format(s=shotnumbers,g=gas), bbox_inches='tight')

    if Type=='Single':
        plt.figure(figsize=(10,6))
        Position, T=np.genfromtxt(infile+'shot{s}Te.dat'.format(s=shotnumber),unpack=True)
        plt.plot(Position, T,label='shot n°{s}, P$_M$$_W$= {mw} W , \n p= {p} mPa '.format(s=shotnumber,mw=float(f'{GetMicrowavePower(shotnumber)[0]:.3f}'),p=float(f'{Pressure(shotnumber,gas):.3f}')))
        plt.xlabel('position R [m]')
        plt.ylabel('temperature [eV]')
        plt.legend(loc=1, bbox_to_anchor=(1.8,1),title='{g}, MW= {m}'.format(g=gas,m=GetMicrowavePower(i)[1]))  
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"shot{s}/Temperatureprofile_from_fits_to_characteristics_{g}.pdf".format(s=shotnumber,g=gas), bbox_inches='tight')

    if Type=='Values' : 
        Position, T=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}Te.dat'.format(s=s),unpack=True)
        return(Position,T,np.mean(T))

def CorrectedDensityProfile(s):
    Position, char_U, char_I, I_isat,Bolo_sum, Interferometer=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=s),unpack=True)
    location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=s)
    I_isat_fit=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}ne.dat'.format(s=s),usecols=1,unpack=True)
    inter_original=LoadData(location)['Interferometer digital']
    inter=savgol_filter((LoadData(location)['Interferometer digital']),100,3)
    time =LoadData(location)['Zeit [ms]'] / 1000
    stop=np.argmin(np.gradient(inter[int(len(inter)*0.2):-1]))+int(len(inter)*0.2)
    mean_1=int(len(inter[0:stop])*0.8)
    offset=np.mean(inter[stop+50:-1])
    mean_density=np.mean(inter[mean_1:stop-50])-offset
    if mean_density<0:
        mean_density=np.mean(inter[mean_1:stop-50])-(3.6-np.mean(inter[stop+50:-1]))
    # plt.plot(np.gradient(inter[int(len(inter)*0.2):-1],time[int(len(inter)*0.2):-1]))
    # plt.show()
    # plt.plot(time,inter_original,alpha=0.5)
    # plt.plot(time[int(len(inter)*0.2):-1],inter[int(len(inter)*0.2):-1])
    # plt.plot(time[stop-50],inter[stop-50],'go')
    # plt.plot(time[mean_1],inter[mean_1],'ro')
    # plt.plot(time[stop+50],inter[stop+50],'bo')
    # plt.show()
    correction,corrected,corrected_fit=[],[],[]
    for i in [a-offset for a in Interferometer]:
        correction.append(mean_density/i)
    for i,j,k in zip(correction, I_isat,I_isat_fit):
        corrected.append(i*j)
        corrected_fit.append(i*k)

    return corrected,mean_density,corrected_fit

def NormDensityProfile():
    Position, char_U, char_I, I_isat,Bolo_sum, Interferometer=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=shotnumber),unpack=True)
    new_pos=np.concatenate((-np.flip(Position)+2*Position[0],Position[1:,]))
    density=np.concatenate((np.flip(I_isat),I_isat[1:,]))
    density_corr=np.concatenate((np.flip(CorrectedDensityProfile()[0]),CorrectedDensityProfile()[0][1:,]))
    d=CorrectedDensityProfile()[1]*3.88E17
    density_interpol=interp1d(new_pos,density*d/integrate.trapezoid(density))
    density_corr_interpol=interp1d(new_pos,density_corr*d/integrate.trapezoid(density_corr))
    #plt.plot(new_pos, density,'ro')
    plt.plot(new_pos,density_interpol(new_pos), label='Mirrored Density Profile')
    plt.plot(new_pos,density_corr_interpol(new_pos),label='Mirrored Density Profile \n corrected signal')
    plt.xlabel('Position [m]')
    plt.ylabel('Density [m$^-$$^3$]')
    plt.legend(loc=1, bbox_to_anchor=(1.5,1))  
    plt.suptitle('Density Profiles from Ion Saturation current \n shot {s} // {g} // MW: {mw} Watt // Pressure: {p} mPa '.format(s=shotnumber,g=gas,mw=float(f'{GetMicrowavePower(shotnumber):.3f}'),p=float(f'{Pressure(shotnumber,gas):.3f}')),y=1.05)  
    plt.show()
    print(integrate.trapezoid(density_interpol(new_pos)))

def DensityProfile(s,Type='',ScanType='',save=False):
    if Type=='Compare':
        plt.figure(figsize=(10,7))
        pressure,mw=[],[]
        for i in shotnumbers:
            pressure.append(Pressure(i,gas))
            mw.append(GetMicrowavePower(i)[0])
        if ScanType=='Pressure':
            sortnumbers=[shotnumbers[i] for i in np.argsort(pressure)]
            density_profiles=[density_profiles_from[i] for i in np.argsort(pressure)]
        if ScanType=='Power':
            sortnumbers=[shotnumbers[i] for i in np.argsort(mw)]
            density_profiles=[density_profiles_from[i] for i in np.argsort(mw)]
        if ScanType=='None':
            sortnumbers=shotnumbers
            density_profiles=density_profiles_from
        for i,c,m,n in zip(sortnumbers,colors,markers,np.arange(0,len(shotnumbers))):
            if ScanType=='Pressure':
                label='shot n°{s}, p= {p} mPa'.format(s=i,p=float(f'{Pressure(i,gas):.1f}'))
                title= r'{g}, MW= {m}, P$_M$$_W$ $\approx$ {mw} kW'.format(g=gas,m=GetMicrowavePower(i)[1],mw=float(f'{np.mean(mw)*10**(-3):.2f}'))
            if ScanType=='Power':
                label='shot n°{s}, P$_M$$_W$ = {mw} kW'.format(s=i,mw=float(f'{GetMicrowavePower(i)[0]*10**(-3):.2f}'))
                title= r'{g}, MW= {m}, p $\approx$ {p} mPa'.format(g=gas,m=GetMicrowavePower(i)[1],p=float(f'{np.mean(pressure):.1f}'))
            if ScanType=='None':
                label='shot n°{s}, P$_M$$_W$ = {mw} kW, p= {p} mPa'.format(s=i,mw=float(f'{GetMicrowavePower(i)[0]*10**(-3):.2f}'),p=float(f'{Pressure(i,gas):.1f}'))
                title= r'{g}, MW= {m}'.format(g=gas,m=GetMicrowavePower(i)[1])
            d=(CorrectedDensityProfile(i)[1]*3.88E17)/2
            if density_profiles[n]=='d':
                Position=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=i),usecols=0)
                norm=integrate.trapezoid(CorrectedDensityProfile(i)[0],Position)/abs(Position[-1]-Position[0])
                Density=[u*d/norm for u in CorrectedDensityProfile(i)[0]]
                plt.plot(Position, Density,linewidth=3,color=c,marker=m,label=label)
            if density_profiles[n]=='f':
                Position=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}ne.dat'.format(s=i),usecols=0,unpack=True)
                norm=integrate.trapezoid(CorrectedDensityProfile(i)[2],Position)/abs(Position[-1]-Position[0])
                Density=[u*d/norm for u in CorrectedDensityProfile(i)[2]]
                plt.plot(Position, Density,linewidth=3,color=c,marker=m,label='* '+label)       
        plt.ylim(bottom=0)
        plt.xlabel('position R - r$_0$ [m]',fontsize=30)
        plt.ylabel('density [m$^-$$^3$]',fontsize=30)
        #plt.suptitle(' Comparison of density-profiles for {}'.format(gas))
        plt.legend(loc=1, bbox_to_anchor=(1.8,1),title=title)  
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"comparisons/{g}/comparison_of_shots{s}_densityprofiles_{g}.pdf".format(s=shotnumbers,g=gas), bbox_inches='tight')

    if Type=='Single':
        plt.figure(figsize=(10,6))
        d=(CorrectedDensityProfile(s)[1]*3.88E17)/2
        Position=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=s),usecols=0)
        norm=integrate.trapezoid(CorrectedDensityProfile(s)[0],Position)/abs(Position[-1]-Position[0])
        Density=[u*d/norm for u in CorrectedDensityProfile(s)[0]]
        Position_fit=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}ne.dat'.format(s=s),usecols=0,unpack=True)
        norm_fit=integrate.trapezoid(CorrectedDensityProfile(s)[2],Position_fit)/abs(Position_fit[-1]-Position_fit[0])
        Density_fit=[u*d/norm_fit for u in CorrectedDensityProfile(s)[2]]
        plt.plot(Position, Density,'bo--',label='shot n°{s}, P$_m$$_w$= {mw} W , \n p={p} mPa '.format(s=s,mw=float(f'{GetMicrowavePower(s)[0]:.3f}'),p=float(f'{Pressure(s,gas):.3f}')))#
        plt.plot(Position_fit, Density_fit,'ro--',label='shot n°{s}, from fits, P$_m$$_w$= {mw} W , \n p={p} mPa '.format(s=s,mw=float(f'{GetMicrowavePower(s)[0]:.3f}'),p=float(f'{Pressure(s,gas):.3f}')))#        
        plt.xlabel('position R [m]')
        plt.ylabel('density [m$^-$$^3$]')
        plt.legend(loc=1, bbox_to_anchor=(1.7,1))    
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"shot{s}/Densityprofile_{g}.pdf".format(s=s,g=gas), bbox_inches='tight')
    if Type=='Values':
        d=(CorrectedDensityProfile(s)[1]*3.88E17)/2
        Position=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=s),usecols=0)
        norm=integrate.trapezoid(CorrectedDensityProfile(s)[0],Position)/abs(Position[-1]-Position[0])
        Density=[u*d/norm for u in CorrectedDensityProfile(s)[0]]

        return Position, Density    
            
def CompareDifferentGases():
    for i,j in zip(shotnumbers,gases):
        gas=j
        Pos, I_isat= np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=i),unpack=True,usecols=(0,3))
        plt.plot(Pos,I_isat,label='shot{s}, {g}, MW: {mw} Watt , Pressure: {p} mPa '.format(s=i,g=gas,mw=float(f'{GetMicrowavePower(i):.3f}'),p=float(f'{Pressure(i,gas):.3f}')))
    plt.legend(loc=1, bbox_to_anchor=(1.5,1))  
    plt.show()
    for i,j in zip(shotnumbers,gases):
        gas=j
        Position, T=np.genfromtxt('/data6/Auswertung/shot{s}/shot{s}Te.dat'.format(s=i),unpack=True)
        plt.plot(Position, T,label='shot{s}, {g}, MW: {mw} Watt, Pressure: {p} mPa '.format(s=i,g=gas,mw=float(f'{GetMicrowavePower(i):.3f}'),p=float(f'{Pressure(i,gas):.3f}')))
    plt.legend(loc=1, bbox_to_anchor=(1.5,1))  
    plt.show()


def FastElectrons():
    sonde=[20,15,10,4]
    T_e=[1019,996,1043,627]
    T_e_2=[2228,1814,1055,809]
    T_e_3=[728,947,899,562]
    P_ges=[49.38721664642027, 48.3411106082018, 51.78224889181524, 43.49598790487404]
    P_ges_2=[42.00941616635303, 42.229649016504304, 38.98121447677319, 36.008070999731174]
    shotnumbers=[13159,13160,13161,13162]
    shotnumbers_2=[13163,13164,13165,13166]
    shotnumbers_3=[13135,13136,13137,13138]
    fig,ax=plt.subplots(figsize=(10,7))
    
    ax2=ax.twinx()
    for s,i in zip(shotnumbers_2,np.arange(0,len(shotnumbers))):
        bolo_p=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot{s}/shot{s}_bolometerprofile_from_radiation_powers.txt'.format(s=s),usecols=1)
        ax.plot(sonde[i],T_e_2[i],'ro')
        ax2.plot(sonde[i],P_ges_2[i],'bo',label='shot n°{s} // P$_m$$_w$= {m} W // p={p} mPa'.format(s=s,m=float(f'{GetMicrowavePower(s):.3f}'),p=float(f'{Pressure(s,gas):.3f}')))
    ax.set(xlabel='probe position [cm]', ylabel='Temperature of hot population [eV]')
    ax2.set(ylabel='total power emitted by plasma [W]')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim([min(P_ges_2)-1,max(P_ges_2)+1])
    ax2.legend(loc='lower center',bbox_to_anchor=(0.5,-0.5))
    ax.tick_params(axis='y', labelcolor='red')
    plt.show()
    
def Densities(s,gas):
    T=290
    k=1.38E-23
    p=Pressure(s,gas)*10**(-3)
    n=p/(k*T)
    n_e=(CorrectedDensityProfile(s)[1]*3.88E17)/2
    n_0=n-n_e
    deg_ion=(n_e/n)*100
    return n,n_e,n_0,deg_ion
    
    
      
# %%
if __name__ == "__main__":
    shotnumbers=[13221,13220,13223,13225]#[13299,13300,13301,13302,13303,13304,13305,13306,13307,13308,13309,13310,13311,13313,13314]#np.arange(13299,13312)
    #density_profiles_from=['d','d','d','d','d','d','d','d','f','f','f','f','f','f','f','f']#['d','d','d','f','f','f','f','f','f','f','f','d','d','d','d']#13280-13291['d','f','d','d','f','f','f','f','f','d','d','f']#13299-13112['d','d','d','f','f','f','f','f','f','f','f','d','d']
    density_profiles_from=['d' for i in range(len(shotnumbers))]
    gas='He' 
    shotnumber=13228
    infile='/data6/shot{s}/kennlinien/auswert/'.format(s=shotnumber)
    #infile='/data6/shot{}/probe2D/'.format(shotnumber)
    outfile='/home/gediz/Results/Plasma_charactersitics/'

    if not os.path.exists(str(outfile)+'shot{}'.format(shotnumber)):
        os.makedirs(str(outfile)+'shot{}'.format(shotnumber))

    #ExtractMeanValues()
    #CompareDifferentGases()
    #GetMicrowavePower(shotnumber)
    print(TemperatureProfile(shotnumber,'Values','Power')[2])
    print(Densities(shotnumber,gas)[1],Densities(shotnumber,gas)[3])
    #PlotMeanValues()
    #FastElectrons()
    #DensityProfile(shotnumbers,'Compare','Power')
    #DensityProfile(shotnumber,'Single')
    #print(Densities(shotnumber,gas)[1])
    #print(TemperatureProfile(shotnumber,'Values')[2])
    # for s in np.arange(13316,13321):
    #     DensityProfile(s,'Single')
    #CorrectedDensityProfile(shotnumber)

    #print(Densities(shotnumber,gas)[3])
# %%
