#%%
#Written by: Izel Gediz
#Date of Creation: 16.08.2022


from pdb import line_prefix
from tracemalloc import start
from unicodedata import name
from unittest import result
from blinker import Signal
from click import style
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import statistics
import os
import collections
from scipy import integrate
from scipy.interpolate import pchip_interpolate

#%%------------------------------------------------------------------------------------------------------
plt.rc('font',family='serif')
plt.rc('figure', titlesize=15)
plt.rc('figure', figsize=(10,5))


def Gold_Abs():
    location =str(golddata)
    x,y= np.loadtxt(location, unpack='true')
    name='Gold absorption'
    return x,y, name

def Spectrum(lightsource=''):
    x,y= np.genfromtxt(str(spectrumdata)+'spectrometer_data_of_lightsource_'+lightsource+'.txt', skip_header=2,unpack='true')  
    name='Spectral Data'
    return x,y, name


#Use this function to fuse the data of the three spectrometerchannels, plot them and save the plot as well as the new fused datafile
#They should be saved with names similar to those in this folder
#/home/gediz/Measurements/Spectrometer/Spectra_of_lamps_17_08_2022
def Spectrometer_Data(lightsource='', save=False):
    x1,y1=np.genfromtxt(str(spectrumdata)+lightsource+'_linkes_Spektrum.txt', skip_header=17, skip_footer=1,unpack='true')    
    x2,y2=np.genfromtxt(str(spectrumdata)+lightsource+'_mittleres_Spektrum.txt', skip_header=17, skip_footer=1,unpack='true')    
    x3,y3=np.genfromtxt(str(spectrumdata)+lightsource+'_rechtes_Spektrum.txt', skip_header=17, skip_footer=1,unpack='true')    
    y2=y2[int(np.argwhere(x2<x1[-1])[-1]+1):-1]
    y3=y3[int(np.argwhere(x3<x2[-1])[-1]+1):-1]
    x2=x2[int(np.argwhere(x2<x1[-1])[-1]+1):-1]
    x3=x3[int(np.argwhere(x3<x2[-1])[-1]+1):-1]
    x=list(x1)+list(x2)+list(x3)
    y=list(y1)+list(y2)+list(y3)
    plt.plot(x,y)
    plt.xlabel('wavelength [nm]')
    plt.ylabel('Counts')
    plt.suptitle('Spectrometer Data of Lightsource {}'.format(lightsource))
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"spectrometer_data_of_lightsource_{}.pdf".format(lightsource))
        data = np.column_stack([np.array(x), np.array(y)])#, np.array(z), np.array(abs(y-z))])
        np.savetxt(str(outfile)+"spectrometer_data_of_lightsource_{}.txt".format(lightsource), data, delimiter='\t \t', header='Data of all three Spectrometer-channels for the lightsource {} \n wavelength [nm] \t counts'.format(lightsource))
    return x,y

def Gold_Fit(lightsource=''):
    gold_interpolation=pchip_interpolate(Gold_Abs()[0],Gold_Abs()[1], Spectrum(lightsource)[0])
    reduced_spectrum=[]
    for i1, i2 in zip(Spectrum(lightsource)[1], gold_interpolation):
        reduced_spectrum.append(i1*i2)
    #plt.plot(Spectrum(lightsource)[0], Spectrum(lightsource)[1])
    #plt.plot(Spectrum(lightsource)[0], reduced_spectrum)
    #plt.show()
    spec_int_trap =integrate.trapezoid(Spectrum(lightsource)[1], Spectrum(lightsource)[0])
    new_spec_int_trap=integrate.trapezoid(reduced_spectrum, Spectrum(lightsource)[0])
    absorbed_percentage=(new_spec_int_trap/spec_int_trap)*100
    return reduced_spectrum, absorbed_percentage


def Log_Plot(data):
    x=data[0]
    y=data[1]
    fig,ax = plt.subplots()
    ax.semilogx(x,y)
    plt.show()

def Double_Plot(lightsource='', save=False):
    x1=Gold_Abs()[0]
    y1=Gold_Abs()[1]
    label1=Gold_Abs()[2]
    x2=Spectrum(lightsource)[0]
    y2=Spectrum(lightsource)[1]
    label2=Spectrum(lightsource)[2]
    y3=Gold_Fit(lightsource)[0]
    percentage=Gold_Fit(lightsource)[1]
    fig,ax = plt.subplots()
    ax2=ax.twinx()
    ax2=ax.twiny()
    ax3=ax.twinx()
    #ax3=ax.twiny()
    lns1=ax.semilogx(x2,y2, label=label2, color='red', alpha=0.5)
    lns2=ax2.semilogx(x2,y3,label=label2+' reduced to {}%'.format(float(f'{percentage:.2f}')), color='green')
    lns3=ax3.semilogx(x1,y1, label=label1)
    ax.set(xlabel='wavelength [nm]', ylabel='Counts')
    ax3.set(ylabel='Absorption')
    ax3.tick_params(axis='y', labelcolor='blue')
    leg = lns1 + lns2 +lns3
    labs = [l.get_label() for l in leg]
    ax.legend(leg, labs, loc=1)
    plt.suptitle('Gold absorption and the Spectrum of lightsource {}'.format(lightsource))
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig("/home/gediz/Results/Goldfoil_Absorption/Lightsources_Gold_Absorption/reduced_spectrum_absorbed_by_gold_of_lightsource_{}.pdf".format(lightsource))
    
    

#%%

if __name__ == "__main__":
    infile ='/scratch.mv3/koehn/backup_Anne/zilch/Bolo/Absorption_AU/'
    #outfile='/home/gediz/Results/Goldfoil_Absorption/'
    #outfile='/home/gediz/Results/Spectrometer/Spectra_of_lamps_17_08_2022/'
    spectrumdata='/home/gediz/Results/Spectrometer/Spectra_of_lamps_17_08_2022/'
    golddata= '/home/gediz/Results/Goldfoil_Absorption/Golddata_interpolated_for_Spectrometer.txt'


    Double_Plot('UV_Taschenlampe_snapshot_weitweg', save=True)
    #Spectrometer_Data('UV_Taschenlampe_snapshot_weitweg', save=True)

# %%
