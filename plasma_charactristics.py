#%%

#Written by: Izel Gediz
#Date of Creation: 11.11.2022


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

#%%

# U,I,P=np.genfromtxt('/data6/Auswertung/shot13042/kennlinien/000000.dat',unpack=True)
# def Kenn(x,a,b,T):
#     return a+b*(1-np.exp(-x/T))
# popt,pcov=curve_fit(Kenn,U,I)
# print(*popt)
# plt.plot(U,I,linestyle='None',marker='.')
# plt.plot(U,Kenn(U,*popt))
# #plt.xlim(0.4,0.8)
# plt.show()

# plt.plot(U,np.gradient(I),linestyle='None',marker='.')
# plt.xlim(0.5,1)
# plt.plot(U, interp1d(U,np.gradient(I)))
# plt.show()

# %%
def LoadData(location):
    with open(location, "r") as f:
        cols = f.readlines()[3]
        cols = re.sub(r"\t+", ';', cols)[:-2].split(';')
    data = pd.read_csv(location, skiprows=4, sep="\t\t", names=cols, engine='python')
    return data

def Pressure(shotnumber):
    location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
    y= LoadData(location)["Pressure"]
    time = LoadData(location)['Zeit [ms]'] / 1000
    pressure= np.mean(y[0:100])
    d = 9.33                # according to PKR261 manual
    pressure = 10.**(1.667*pressure-d)*1000
    #plt.plot(time,y)
    #plt.plot(time[500],y[500],'ro')
    #plt.show()
    #print(pressure)
    return pressure

def MeanValues(compare=False,save=False):
    if compare==True:
        for i in shotnumbers:
            Position, char_U, char_I, I_isat, Bolo_sum, Interferometer=np.genfromtxt('/data6/Auswertung/shot{s}/shot{s}.dat'.format(s=i),unpack=True)
            plt.plot(Position, I_isat,label='shot{s}, MW used: , Pressure: {p} mPa '.format(s=i,p=float(f'{Pressure(i):.3f}')))
        plt.xlabel('Position R [m]')
        plt.ylabel('Ion satturation current')
        plt.suptitle('Comparison of Ion saturation currents from 2D Probe scans ')
        plt.legend(loc=1, bbox_to_anchor=(1.7,1))    
        fig1= plt.gcf()
        plt.show()
        for i in shotnumbers:
            Position, char_U, char_I, I_isat, Bolo_sum, Interferometer=np.genfromtxt('/data6/Auswertung/shot{s}/shot{s}.dat'.format(s=i),unpack=True)
            plt.plot(Position,Bolo_sum,label='shot{s}, MW used: , Pressure: {p} mPa '.format(s=i,p=float(f'{Pressure(i):.3f}')))
        plt.xlabel('Position R [m]')
        plt.ylabel('Bolo_sum Channel Signal')
        plt.suptitle('Comparison of Bolometer sum channel signals from 2D Probe scans')
        plt.legend(loc=1, bbox_to_anchor=(1.7,1))    
        fig2= plt.gcf()
        plt.show()
        for i in shotnumbers:
            Position, char_U, char_I, I_isat, Bolo_sum, Interferometer=np.genfromtxt('/data6/Auswertung/shot{s}/shot{s}.dat'.format(s=i),unpack=True)
            plt.plot(Position, Interferometer,label='shot{s}, MW used: , Pressure: {p} mPa '.format(s=i,p=float(f'{Pressure(i):.3f}')))
        plt.xlabel('Position R [m]')
        plt.ylabel('Interferometer Signal')
        plt.suptitle('Comparison of Interferometersignals from 2D Probe scans')
        plt.legend(loc=1, bbox_to_anchor=(1.7,1))    
        fig3= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"comparisons/comparison_of_shots{}_Ion_saturation.pdf".format(shotnumbers), bbox_inches='tight')
            fig2.savefig(str(outfile)+"comparisons/comparison_of_shots{}_Bolo_sum.pdf".format(shotnumbers), bbox_inches='tight')
            fig3.savefig(str(outfile)+"comparisons/comparison_of_shots{}_Interferometer.pdf".format(shotnumbers), bbox_inches='tight')
 
    else:
        Position, char_U, char_I, I_isat, Bolo_sum, Interferometer=np.genfromtxt(infile+'shot{s}.dat'.format(s=shotnumber),unpack=True)
        plt.plot(Position, I_isat)
        plt.xlabel('Position R [m]')
        plt.ylabel('Ion satturation current')
        plt.suptitle('2D Probe scan of shot {s} \n Ion saturation current $\propto$ n \n MW used: // Pressure: {p} mPa '.format(s=shotnumber,p=float(f'{Pressure():.3f}')),y=1.05)
        fig1= plt.gcf()
        plt.show()
        plt.plot(Position,Bolo_sum)
        plt.xlabel('Position R [m]')
        plt.ylabel('Bolo_sum Channel Signal')
        plt.suptitle('2D Probe scan of shot {s} \n Bolometer sum Channel signal\n MW used: // Pressure: {p} mPa '.format(s=shotnumber,p=float(f'{Pressure():.3f}')),y=1.05)
        fig2= plt.gcf()
        plt.show()
        plt.plot(Position, Interferometer)
        plt.xlabel('Position R [m]')
        plt.ylabel('Interferometer Signal')
        plt.suptitle('2D Probe scan of shot {s} \n Interferometer Signal\n MW used: // Pressure: {p} mPa '.format(s=shotnumber,p=float(f'{Pressure():.3f}')),y=1.05)
        fig3= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"shot{}/2D_Probe_Singal_mean_values_of_I_sat.pdf".format(shotnumber), bbox_inches='tight')
            fig2.savefig(str(outfile)+"shot{}/2D_Probe_Singal_mean_values_of_Bolo_sum.pdf".format(shotnumber), bbox_inches='tight')
            fig3.savefig(str(outfile)+"shot{}/2D_Probe_Singal_mean_values_of_Interferometer.pdf".format(shotnumber), bbox_inches='tight')

def TemperatureProfile(compare=False,save=False):
    if compare==True:
        for i in shotnumbers:
            Position, T=np.genfromtxt('/data6/Auswertung/shot{s}/shot{s}Te.dat'.format(s=i),unpack=True)
            plt.plot(Position, T,label='shot{s}, MW used: , Pressure: {p} mPa '.format(s=i,p=float(f'{Pressure(i):.3f}')))
        plt.xlabel('Position R [m]')
        plt.ylabel('Temperature')
        plt.suptitle(' Comparison of Temeperatureprofiles')
        plt.legend(loc=1, bbox_to_anchor=(1.7,1))    
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"comparisons/comparison_of_shots{}_temperatureprofiles.pdf".format(shotnumbers), bbox_inches='tight')

    else:
        Position, T=np.genfromtxt(infile+'shot{s}Te.dat'.format(s=shotnumber),unpack=True)
        plt.plot(Position, T,label='shot{s}, MW used: , Pressure: {p} mPa '.format(s=shotnumber,p=float(f'{Pressure():.3f}')))
        plt.xlabel('Position R [m]')
        plt.ylabel('Temperature')
        plt.suptitle('2D Probe scan of shot {} \n Temperatureprofile determined by fits to the characteristics'.format(shotnumber))
        plt.legend(loc=1, bbox_to_anchor=(1.7,1))    
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"shot{}/Temperatureprofile_from_fits_to_characteristics.pdf".format(shotnumber), bbox_inches='tight')



# %%
shotnumbers=(13065,13058,13068,13059,13060,13062,13063) 
shotnumber=13071
infile='/data6/Auswertung/shot{s}/'.format(s=shotnumber)
outfile='/home/gediz/Results/Plasma_charactersitics/'

if not os.path.exists(str(outfile)+'shot{}'.format(shotnumber)):
    os.makedirs(str(outfile)+'shot{}'.format(shotnumber))
    
#MeanValues(compare=True,save=True)
TemperatureProfile(compare=True,save=True)
# %%
