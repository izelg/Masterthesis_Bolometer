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
plt.rc('font',size=14)
plt.rc('figure', titlesize=15)
#%%
# plt.figure(figsize=(10,7))
# for i in (np.arange(10,27,1)):
#     U,I,P=np.genfromtxt('/data6/Auswertung/shot13079/kennlinien/0000{i}.dat'.format(i=i),unpack=True)

#     plt.plot(U*20,-I,linestyle='None',marker='.',markersize=0.1,label='{i}'.format(i=np.mean(P)))
# plt.suptitle('Strom-Spannungs Kennlinien aufgenommen mit Langmuirsonden-Verfahreinheit')
# plt.xlabel('Spannung [V]')
# plt.ylabel('Strom [mA]')
# #plt.legend(loc=1)
# plt.show()

# # plt.plot(U,np.gradient(I),linestyle='None',marker='.')
# # plt.xlim(0.5,1)
# # #plt.plot(U, interp1d(U,np.gradient(I)))
# # plt.show()

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
        print( '    the same factor as for hydrogren will be used' )
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
    #plt.plot(signalinwatt_for[start:stop])
    #plt.plot(signalinwatt_back[start:stop])
    #plt.plot(start,signalinwatt_for[start],'ro')
    #plt.plot(stop,signalinwatt_for[stop],'ro')
    #plt.show()
    print(np.mean(signalinwatt_for[start:stop]),np.mean(signalinwatt_back[start:stop]))
    return (np.mean(signalinwatt_for[start:stop])-np.mean(signalinwatt_back[start:stop]))
    
    
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
            plt.plot(Position, I_isat,label='shot{s}, MW: {mw} Watt,\n Pressure: {p} mPa '.format(s=i,mw=float(f'{GetMicrowavePower(i):.3f}'),p=float(f'{Pressure(i,gas):.3f}')))
        plt.xlabel('Position R [m]')
        plt.ylabel('Ion satturation current [mA]')
        plt.suptitle('Comparison of Ion saturation currents from 2D Probe scans for {} '.format(gas))
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
        plt.xlabel('Position R [m]')
        plt.ylabel('Ion satturation current [mA]')
        plt.suptitle('2D Probe scan of shot {s} // {g}\n Ion saturation current $\propto$ n \n MW: {mw} Watt // Pressure: {p} mPa '.format(s=shotnumber,g=gas,mw=float(f'{GetMicrowavePower(shotnumber):.3f}'),p=float(f'{Pressure(shotnumber,gas):.3f}')),y=1.05)
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
def TemperatureProfile(compare=False,save=False):
    if compare==True:
        for i in shotnumbers:
            Position, T=np.genfromtxt('/data6/Auswertung/shot{s}/shot{s}Te.dat'.format(s=i),unpack=True)
            plt.plot(Position, T,label='shot{s}, MW: {mw} Watt, Pressure: {p} mPa '.format(s=i,mw=float(f'{GetMicrowavePower(i):.3f}'),p=float(f'{Pressure(i,gas):.3f}')))
        plt.xlabel('Position R [m]')
        plt.ylabel('Temperature [eV]')
        plt.suptitle(' Comparison of Temeperatureprofiles for {}'.format(gas))
        plt.legend(loc=1, bbox_to_anchor=(1.9,1))    
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"comparisons/{g}/comparison_of_shots{s}_temperatureprofiles_{g}.pdf".format(s=shotnumbers,g=gas), bbox_inches='tight')

    else:
        Position, T=np.genfromtxt(infile+'shot{s}Te.dat'.format(s=shotnumber),unpack=True)
        plt.plot(Position, T,label='shot{s}, MW: {mw} Watt , \n Pressure: {p} mPa '.format(s=shotnumber,mw=float(f'{GetMicrowavePower(shotnumber):.3f}'),p=float(f'{Pressure(shotnumber,gas):.3f}')))
        plt.xlabel('Position R [m]')
        plt.ylabel('Temperature [eV]')
        plt.suptitle('2D Probe scan of shot {s} // {g} \n Temperatureprofile determined by fits to the characteristics'.format(s=shotnumber,g=gas),y=1.05)
        plt.legend(loc=1, bbox_to_anchor=(1.7,1))    
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"shot{s}/Temperatureprofile_from_fits_to_characteristics_{g}.pdf".format(s=shotnumber,g=gas), bbox_inches='tight')

def CorrectedDensityProfile():
    Position, char_U, char_I, I_isat,Bolo_sum, Interferometer=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=shotnumber),unpack=True)
    location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
    inter=LoadData(location)['Interferometer digital']
    stop=np.argmin(np.gradient(inter))
    #mean_density=np.mean(inter[stop-5000:stop])
    mean_density=np.mean(Interferometer[-8:-1])
    correction=[]
    corrected=[]
    for i in Interferometer:
        correction.append(mean_density/i)
    for i,j in zip(correction, I_isat):
        corrected.append(i*j)
    # plt.plot(I_isat)
    # plt.plot(corrected)
    # plt.show()
    return corrected

def NormDensityProfile():
    Position, char_U, char_I, I_isat,Bolo_sum, Interferometer=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=shotnumber),unpack=True)
    new_pos=np.concatenate((-np.flip(Position)+2*Position[0],Position[1:,]))
    density=np.concatenate((np.flip(I_isat),I_isat[1:,]))
    density_corr=np.concatenate((np.flip(CorrectedDensityProfile()),CorrectedDensityProfile()[1:]))
    density_interpol=interp1d(new_pos,density*1/integrate.trapezoid(density))
    density_corr_interpol=interp1d(new_pos,density_corr*1/integrate.trapezoid(density_corr))
    #plt.plot(new_pos, density,'ro')
    plt.plot(new_pos,density_interpol(new_pos), label='Mirrored Density Profile')
    plt.plot(new_pos,density_corr_interpol(new_pos),label='Mirrored Density Profile \n corrected signal')
    plt.xlabel('Position [m]')
    plt.ylabel('Arb')
    plt.legend(loc=1, bbox_to_anchor=(1.5,1))  
    plt.suptitle('Density Profiles from Ion Saturation current \n shot {s} // {g} // MW: {mw} Watt // Pressure: {p} mPa '.format(s=shotnumber,g=gas,mw=float(f'{GetMicrowavePower(shotnumber):.3f}'),p=float(f'{Pressure(shotnumber,gas):.3f}')),y=1.05)  
    plt.show()
    print(integrate.trapezoid(density_interpol(new_pos)))

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
    
# %%
shotnumbers=(13123,13118,13119) 
gases=('He','He','He')
shotnumber=13123
infile='/data6/Auswertung/shot{s}/'.format(s=shotnumber)
#infile='/data6/shot{}/probe2D/'.format(shotnumber)
outfile='/home/gediz/Results/Plasma_charactersitics/'

if not os.path.exists(str(outfile)+'shot{}'.format(shotnumber)):
    os.makedirs(str(outfile)+'shot{}'.format(shotnumber))
gas='He'   

#CompareDifferentGases()
#GetMicrowavePower(shotnumber)
#TemperatureProfile(compare=True,save=True)
PlotMeanValues(compare=True,save=True)
#%%
