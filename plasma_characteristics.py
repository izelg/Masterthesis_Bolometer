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
if __name__ == "__main__":
    Poster=False
    Latex=True


    if Poster==True:
        plt.rc('font',size=20)
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)
        plt.rcParams['lines.markersize']=12
        width=10
        height=7
    elif Latex==True:
        width=412/72.27
        height=width*(5**.5-1)/2
        n=1
        plt.rcParams['text.usetex']=True
        plt.rcParams['font.family']='serif'
        plt.rcParams['axes.labelsize']=11*n
        plt.rcParams['font.size']=11*n
        plt.rcParams['legend.fontsize']=11*n
        plt.rcParams['xtick.labelsize']=11*n
        plt.rcParams['ytick.labelsize']=11*n
        plt.rcParams['lines.markersize']=4
        specialfontsize=13
    else:
        w=10
        h=7
        plt.rc('font',size=14)
        plt.rc('figure', titlesize=15)
    colors=['#1bbbe9','#023047','#ffb703','#fb8500','#c1121f','#780000','#6969B3','#D81159','#1bbbe9','#023047','#ffb703','#fb8500','#c1121f']
    markers=['o','v','s','P','p','D','*','x','o','v','s','P','p','D','*','x']
    colors2=['#03045E','#0077B6','#00B4D8','#370617','#9D0208','#DC2F02','#F48C06','#FFBA08','#7B2CBF','#C77DFF','#2D6A4F','#40916C','#52B788','#03045E','#0077B6','#00B4D8']


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
    mean_U,mean_I,mean_Isat,mean_bolo,mean_inter,mean_pos,I_sat_SEM=([] for i in range(7))
    for i in ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']:
        char_U, char_I, I_isat, Position,Bolo_sum,Interferometer=np.genfromtxt('/home/gediz/Measurements/Plasma_characteristics/2D_Probe/shot{s}/0000{n}.dat'.format(s=shotnumber,n=i),unpack=True)
        mean_U.append(np.mean(char_U))
        mean_I.append(np.mean(char_I))
        mean_Isat.append(np.mean(I_isat))
        I_sat_SEM.append(np.std(I_isat,ddof=1))
        mean_bolo.append(np.mean(Bolo_sum))
        mean_inter.append(np.mean(Interferometer))
        mean_pos.append(np.mean(Position))
    data = np.column_stack([np.array(mean_pos), np.array(mean_U), np.array(mean_I), np.array(mean_Isat), np.array(mean_bolo), np.array(mean_inter)])
    print([(a/b)*100 for a,b in zip(I_sat_SEM,mean_Isat)])

    #np.savetxt(str(infile)+"shot{}.dat".format(shotnumber) , data, delimiter='\t \t', fmt='%10.6f')

#This function plots the mean values of the 2D Probe measurements.
#Here e.g. the relative density Profiles form Ion satturation currents can be visualized.
#It is also possible to compare a list of shots as specified before calling the function
def PlotMeanValues(compare=False,save=False):
    if compare==True:
        for i in shotnumbers:
            Position, char_U, char_I, I_isat=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=i),unpack=True,usecols=(0,1,2,3))
            plt.plot(Position, I_isat,label='shot n$^\circ${s}, $P_{\mathrm{MW}}$= {mw} W,\n p= {p} mPa '.format(s=i,mw=float(f'{GetMicrowavePower(i):.3f}'),p=float(f'{Pressure(i,gas):.3f}')))
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
        plt.suptitle('2D Probe scan of shot {s} // {g}\n Ion saturation current $\propto$ n \n $P_{\mathrm{MW}}$ = {mw} W // p= {p} mPa '.format(s=shotnumber,g=gas,mw=float(f'{GetMicrowavePower(shotnumber):.3f}'),p=float(f'{Pressure(shotnumber,gas):.3f}')),y=1.05)
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
def TemperatureProfile(s,Type='',ScanType='',save=False,figurename=''):
    if Type=='Compare':
        plt.figure(figsize=(width/2,height))
        pressure,mw=[],[]
        for i in s:
            pressure.append(Pressure(i,gas))
            mw.append(GetMicrowavePower(i)[0])
        if ScanType=='Pressure':
            sortnumbers=[s[i] for i in np.argsort(pressure)]
        if ScanType=='Power':
            sortnumbers=[s[i] for i in np.argsort(mw)]
        if ScanType=='None':
            sortnumbers=s
        print(sortnumbers)
        for i,c,m in zip(sortnumbers,colors2,markers):
            if ScanType=='Pressure':
                label=r'n$^\circ$'+str(i)+r', p= '+str(f'{Pressure(i,gas):.1f}')+' mPa'
                title= str(gas)+r', MW= '+str(GetMicrowavePower(i)[1])+r', $P_{\mathrm{MW}}$ $\approx$ '+str(f'{np.mean(mw)*10**(-3):.2f}')+' kW'
            if ScanType=='Power':
                label=r'n$^\circ$'+str(i)+r', $P_{\mathrm{MW}}$ = '+str( f'{GetMicrowavePower(i)[0]*10**(-3):.2f}')+' kW'
                title= str(gas)+', MW= '+str(GetMicrowavePower(i)[1])+r', p $\approx$ '+str(f'{np.mean(pressure):.1f}')+' mPa'
            if ScanType=='None':
                label=r'n$^\circ$'+str(i)+r', $P_{\mathrm{MW}}$ = '+str(f'{GetMicrowavePower(i)[0]*10**(-3):.2f}')+' kW, p= '+str(f'{Pressure(i,gas):.1f}')+' mPa'
                title= str(gas)+', MW= '+str(GetMicrowavePower(i)[1])

            Position, T=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}Te.dat'.format(s=i),unpack=True)
            #plt.errorbar(Position*100+60, T,yerr=[0.1*x for x in T], color=c,marker=m,capsize=5,alpha=0.1)#
            plt.plot(Position*100+60, T, color=c,marker=m,label=label)
        plt.ylim(bottom=0)
        plt.xlabel('$R$ [cm]')
        plt.ylabel('$T_{\mathrm{e}}$ [eV]')
        #plt.legend(loc='lower center',title=title,bbox_to_anchor=(0.5,-1.3))  
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig('/home/gediz/LaTex/Thesis/Figures/{}_temperature.pdf'.format(figurename), bbox_inches='tight')
            #fig1.savefig(str(outfile)+"comparisons/{g}/comparison_of_shots{s}_temperatureprofiles_{g}.pdf".format(s=shotnumbers,g=gas), bbox_inches='tight')


    if Type=='Single':
        i=shotnumber
        plt.figure(figsize=(10,6))
        Position, T=np.genfromtxt(infile+'shot{s}Te.dat'.format(s=shotnumber),unpack=True)
        plt.plot(Position, T,label='shot n$^\circ${s}, $P_{\mathrm{MW}}$= {mw} W , \n p= {p} mPa '.format(s=shotnumber,mw=float(f'{GetMicrowavePower(shotnumber)[0]:.3f}'),p=float(f'{Pressure(shotnumber,gas):.3f}')))
        plt.xlabel('position R [m]')
        plt.ylabel('temperature [eV]')
        plt.legend(loc=1, bbox_to_anchor=(1.8,1),title='{g}, MW= {m}'.format(g=gas,m=GetMicrowavePower(i)[1]))  
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"shot{s}/Temperatureprofile_from_fits_to_characteristics_{g}.pdf".format(s=shotnumber,g=gas), bbox_inches='tight')

    if Type=='Values' : 
        Position, T=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}Te.dat'.format(s=s),unpack=True)
        error=[0.1*x for x in T]
        return(Position,T,np.mean(T),error)

def CorrectedDensityProfile(s,Plot=False):
    Position, char_U, char_I, I_isat,Bolo_sum, Interferometer=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=s),unpack=True)
    location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=s)
    I_isat_fit=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}ne.dat'.format(s=s),usecols=1,unpack=True)
    Temperature=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}Te.dat'.format(s=s),usecols=1,unpack=True)
    inter_original=LoadData(location)['Interferometer digital']
    inter=savgol_filter((LoadData(location)['Interferometer digital']),100,3)
    time =LoadData(location)['Zeit [ms]'] / 1000
    stop=np.argmin(np.gradient(inter[int(len(inter)*0.2):-1]))+int(len(inter)*0.2)
    mean_1=int(len(inter[0:stop])*0.8)
    offset=np.mean(inter[stop+50:-1])
    mean_density=np.mean(inter[mean_1:stop-50])-offset
    error_int=np.std(inter_original[mean_1:stop-50],ddof=1)/np.sqrt(len(inter_original[mean_1:stop-50]))
    error_isat=0.01
    error_T=0.1
    if mean_density<0:
        mean_density=np.mean(inter[mean_1:stop-50])-(3.6-np.mean(inter[stop+50:-1]))
    if Plot==True:
        plt.plot(time,inter_original,alpha=0.5)
        #plt.plot(time[int(len(inter)*0.2):-1],inter[int(len(inter)*0.2):-1])
        plt.plot(time[stop-50],inter[stop-50],'go')
        plt.plot(time[mean_1],inter[mean_1],'ro')
        plt.plot(time[stop+50],inter[stop+50],'bo')
        plt.show()
    correction,corrected,corrected_fit,error_corr,error_corr_fit=[],[],[],[],[]
    for i in [a-offset for a in Interferometer]:
        correction.append(mean_density/i)
    for i,j,k,m in zip(correction, I_isat,I_isat_fit,Temperature):
        corrected.append(i*j/np.sqrt(m))
        corrected_fit.append(i*k/np.sqrt(m))
        error_corr.append((j/np.sqrt(m))*error_int+(i/np.sqrt(m))*error_isat*j+(i*j/(2*m**(3/2)))*error_T*m)
        error_corr_fit.append((k/np.sqrt(m))*error_int+(i/np.sqrt(m))*0.1*k+(i*k/(2*m**(3/2)))*error_T*m)
    return corrected,mean_density,corrected_fit,error_corr,error_corr_fit,error_int

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

def DensityProfile(s,df,Type='',ScanType='',save=False,figurename=''):
    if Type=='Compare':
        plt.figure(figsize=(width/2,height))
        pressure,mw=[],[]
        for i in shotnumbers:
            pressure.append(Pressure(i,gas))
            mw.append(GetMicrowavePower(i)[0])
        if ScanType=='Pressure':
            sortnumbers=[shotnumbers[i] for i in np.argsort(pressure)]
            density_profiles=[df[i] for i in np.argsort(pressure)]
        if ScanType=='Power':
            sortnumbers=[shotnumbers[i] for i in np.argsort(mw)]
            density_profiles=[df[i] for i in np.argsort(mw)]
        if ScanType=='None':
            sortnumbers=shotnumbers
            density_profiles=df
        for i,c,m,n in zip(sortnumbers,colors2,markers,np.arange(0,len(shotnumbers))):
            if ScanType=='Pressure':
                label=r'n$^\circ$'+str(i)+r', p= '+str(f'{Pressure(i,gas):.1f}')+' mPa'
                title= str(gas)+r', MW= '+str(GetMicrowavePower(i)[1])+r', $P_{\mathrm{MW}}$ $\approx$ '+str(f'{np.mean(mw)*10**(-3):.2f}')+' kW'
            if ScanType=='Power':
                label=r'n$^\circ$'+str(i)+r', $P_{\mathrm{MW}}$ = '+str( f'{GetMicrowavePower(i)[0]*10**(-3):.2f}')+' kW'
                title= str(gas)+', MW= '+str(GetMicrowavePower(i)[1])+r', p $\approx$ '+str(f'{np.mean(pressure):.1f}')+' mPa'
            if ScanType=='None':
                label=r'n$^\circ$'+str(i)+r', $P_{\mathrm{MW}}$ = '+str(f'{GetMicrowavePower(i)[0]*10**(-3):.2f}')+' kW, p= '+str(f'{Pressure(i,gas):.1f}')+' mPa'
                title= str(gas)+', MW= '+str(GetMicrowavePower(i)[1])

            d=(CorrectedDensityProfile(i)[1]*3.88E17)/2
            if density_profiles[n]=='d':
                Position=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=i),usecols=0)
                norm=integrate.trapezoid(CorrectedDensityProfile(i)[0],Position)/abs(Position[-1]-Position[0])
                Density=[u*d/norm for u in CorrectedDensityProfile(i)[0]]
                plt.plot(Position*100+60, Density,color=c,marker=m,label=label)
            if density_profiles[n]=='f':
                Position=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}ne.dat'.format(s=i),usecols=0,unpack=True)
                norm=integrate.trapezoid(CorrectedDensityProfile(i)[2],Position)/abs(Position[-1]-Position[0])
                Density=[u*d/norm for u in CorrectedDensityProfile(i)[2]]
                plt.plot(Position*100+60, Density,color=c,marker=m,label='* '+label)       
        
        plt.ylim(bottom=0)
        plt.xlabel('$R$ [cm]')
        plt.ylabel('$n_{\mathrm{e}}$ [m$^{-3}$]')
        plt.legend(loc='lower center',title=title,bbox_to_anchor=(0.5,-1))  
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig('/home/gediz/LaTex/Thesis/Figures/{}_density.pdf'.format(figurename), bbox_inches='tight')
            # fig1.savefig(str(outfile)+"comparisons/{g}/comparison_of_shots{s}_densityprofiles_{g}.pdf".format(s=shotnumbers,g=gas), bbox_inches='tight')

    if Type=='Single':
        plt.figure(figsize=(width,height))
        d=(CorrectedDensityProfile(s)[1]*3.88E17)/2
        Position,I_isat=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=s),unpack=True,usecols=(0,3))
        norm=integrate.trapezoid(CorrectedDensityProfile(s)[0],Position)/abs(Position[-1]-Position[0])
        Density=[u*d/norm for u in CorrectedDensityProfile(s)[0]]
        Position_fit=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}ne.dat'.format(s=s),usecols=0,unpack=True)
        norm_fit=integrate.trapezoid(CorrectedDensityProfile(s)[2],Position_fit)/abs(Position_fit[-1]-Position_fit[0])
        Density_fit=[u*d/norm_fit for u in CorrectedDensityProfile(s)[2]]
        plt.plot(Position, Density,'bo--',label=r'shot n$^\circ$'+str(s)+r', $P_{\mathrm{MW}}$ = '+str(f'{GetMicrowavePower(s)[0]*10**(-3):.2f}')+' kW, p= '+str(f'{Pressure(s,gas):.1f}')+' mPa')
        plt.plot(Position_fit, Density_fit,'ro--', label=r'shot n$^\circ$'+str(s)+r' from fits, $P_{\mathrm{MW}}$ = '+str(f'{GetMicrowavePower(s)[0]*10**(-3):.2f}')+' kW, p= '+str(f'{Pressure(s,gas):.1f}')+' mPa')
        plt.xlabel('position R [m]')
        plt.ylabel('density [m$^-$$^3$]')
        plt.legend(loc=1, bbox_to_anchor=(1.7,1))    
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"shot{s}/Densityprofile_{g}.pdf".format(s=s,g=gas), bbox_inches='tight')
    if Type=='Values':
        if df[0]=='d':
            error_int=CorrectedDensityProfile(s)[5]
            error_corr=CorrectedDensityProfile(s)[3]
            d=(CorrectedDensityProfile(s)[1]*3.88E17)/2
            Position=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=s),usecols=0)
            norm=integrate.trapezoid(CorrectedDensityProfile(s)[0],Position)/abs(Position[-1]-Position[0])
            Density=[u*d/norm for u in CorrectedDensityProfile(s)[0]]
            errors=[a+b for a,b in zip([u*d/norm for u in error_corr],[u*error_int*3.88E17/(2*norm) for u in CorrectedDensityProfile(s)[0]])]
        if df[0]=='f':
            error_int=CorrectedDensityProfile(s)[5]
            error_corr=CorrectedDensityProfile(s)[4]
            d=(CorrectedDensityProfile(s)[1]*3.88E17)/2
            Position=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}ne.dat'.format(s=i),usecols=0,unpack=True)
            norm=integrate.trapezoid(CorrectedDensityProfile(i)[2],Position)/abs(Position[-1]-Position[0])
            Density=[u*d/norm for u in CorrectedDensityProfile(i)[2]]

        return Position, Density,errors    
            
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
        ax2.plot(sonde[i],P_ges_2[i],'bo',label='shot n$^\circ${s} // $P_{\mathrm{MW}}$= {m} W // p={p} mPa'.format(s=s,m=float(f'{GetMicrowavePower(s):.3f}'),p=float(f'{Pressure(s,gas):.3f}')))
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
    shotnumbers=np.arange(13098,13107)
    #Ar power   1.3     np.arange(13280,13292)  ['d','f','d','d','f','f','f','f','f','d','d','f']
    #Ar power           [13099,13107,13108,13109] 
    #H power    1.42    np.arange(13215,13228)
    #H power            [13090,13095,13096,13097]
    #He power           np.arange(13069,13073)
    #He power           np.arange(13170,13175)
    #He power   1.1     [13265, 13264, 13263, 13262, 13261, 13260, 13259, 13258, 13257]
    #H pressure 1.45    np.arange(13242,13256)
    #H pressure         np.arange(13088,13095)
    #He pressure  1.3   np.arange(13268,13280)
    #Ar pressure  1.4   np.arange(13299,13312)  ['d','d','d','f','f','f','f','f','f','f','f','d','d']
    #Ar pressure        np.arange(13098,13107)  ['f','d','d','d','f','f','f','d','d']
    #Ne pressure  1      np.arange(13340,13348)  ['f']
    #Ne pressure            np.arange(13079,13085)  ['f','d','f','f','d','d']
    density_profiles_from=['f' for i in range(len(shotnumbers))]
    gas='Ar' 
    shotnumber=13252
    infile='/data6/shot{s}/kennlinien/auswert/'.format(s=shotnumber)
    #infile='/data6/shot{}/probe2D/'.format(shotnumber)
    outfile='/home/gediz/Results/Plasma_charactersitics/'

    if not os.path.exists(str(outfile)+'shot{}'.format(shotnumber)):
        os.makedirs(str(outfile)+'shot{}'.format(shotnumber))

    #DensityProfile(shotnumbers,'Compare','Pressure',save=True,figurename='{g}_pressure'.format(g=gas),density_profiles_from)
    #TemperatureProfile(shotnumbers,'Compare','Pressure',save=True,figurename='{g}_pressure'.format(g=gas))
    for s in shotnumbers:
        DensityProfile(s,'Single')
# %%
