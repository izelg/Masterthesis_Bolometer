#%%
#Written by: Izel Gediz
#Date of Creation: 16.08.2022
#Here you'll find functions to plot the Absorption curve of Gold, different spectra and their percentage absorbed by goldfoil.
#It uses the gold absorption data from Anne and takes spectrometer data in 2 column .txt format
#For my measurements the oceanview yaz spectrometer was used which saves the data in three different documents. there is also a funciton here to fuse the data.


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
import scipy.signal as sig

#%%------------------------------------------------------------------------------------------------------
plt.rc('figure', titlesize=15)
plt.rc('figure', figsize=(10,5))

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
    U_in=LoadData(location)['2 GHz Richtk. forward']
    U_in[U_in>0]    = -1e-6
    signal_dBm  = 42.26782054007 + (-28.92407247331 - 42.26782054007) / ( 1. + (U_in / (-0.5508373840567) )**0.4255365582241 )
    signal_dBm  += 60.49 #for foreward signal
    #signal_dBm  += 60.11 #for backward signal
    signalinwatt   = 10**(signal_dBm/10.) * 1e-3
    start=np.argmax(np.gradient(signalinwatt))
    stop=np.argmin(np.gradient(signalinwatt))
    # plt.plot(start,signalinwatt[start],'ro')
    # plt.plot(stop,signalinwatt[stop],'ro')
    # plt.show()
    return (np.mean(signalinwatt[start:stop]))

def Gold_Abs():
    location =str(golddata)
    x,y= np.loadtxt(location, unpack='true')
    name='Gold absorption'
    return x,y, name

def Spectrum(lightsource=''):
    x,y= np.genfromtxt(str(spectrumdata)+'spectrometer_data_of_lightsource_'+lightsource+'.txt', skip_header=2,unpack='true')  
    name='Spectral Data'
    print (x[np.argmax(y)])
    return x,y, name

#Use this function to plot the Absportion line of Gold using the literature data and an interpolation
def GoldAbsorptionPlot(save=False):
    wavelength, golddata= np.loadtxt('/home/gediz/Results/Goldfoil_Absorption/gold_abs_Anne.txt', unpack='true')
    wavelength_new_array=np.arange(0,10E5,1)
    fittedgolddata= pchip_interpolate(wavelength, golddata, wavelength_new_array)
    fig,ax=plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(7)
    plt.rcParams.update({'font.family':'sans-serif'})
    ax.set(xlabel='wavelength [nm]', ylabel='relative Absorption')
    plt.suptitle('Absorption of Goldfoil')
    ax.semilogx(wavelength_new_array, fittedgolddata, color='Red', label='Interpolated Data')
    ax.semilogx(wavelength, golddata, '.', label='Literature Data')
    ax.axvspan(400,750,facecolor='green', alpha=0.3)
    plt.annotate('visible light', (780, 0.8), color='green')
    plt.grid(True,linestyle='dotted')
    plt.legend()
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig("/home/gediz/Results/Goldfoil_Absorption/Golddata_and_Interpolation_with_visible_light.pdf")

#Use this function to fuse the data of the three spectrometerchannels, plot them and save the plot as well as the new fused datafile
#They should be saved with names similar to those in this folder
#/home/gediz/Measurements/Spectrometer/Spectra_of_lamps_17_08_2022
def Spectrometer_Data(lightsource='', save=False,analyze=False):
    x1,y1=np.genfromtxt(str(spectrumdata)+lightsource+'_linkes_spektrum.txt', skip_header=17, skip_footer=1,unpack='true')    
    x2,y2=np.genfromtxt(str(spectrumdata)+lightsource+'_mittleres_spektrum.txt', skip_header=17, skip_footer=1,unpack='true')    
    x3,y3=np.genfromtxt(str(spectrumdata)+lightsource+'_rechtes_spektrum.txt', skip_header=17, skip_footer=1,unpack='true')    
    y2=y2[int(np.argwhere(x2<x1[-1])[-1]+1):-1]
    y3=y3[int(np.argwhere(x3<x2[-1])[-1]+1):-1]
    x2=x2[int(np.argwhere(x2<x1[-1])[-1]+1):-1]
    x3=x3[int(np.argwhere(x3<x2[-1])[-1]+1):-1]
    x=list(x1)+list(x2)+list(x3)
    y=list(y1)+list(y2)+list(y3)
    plt.plot(x,y)
    plt.xlabel('wavelength [nm]')
    plt.ylabel('Counts')
    plt.suptitle('Spectrometer Data of {l} \n {e}'.format(l=lightsource,e=extratitle))
    if analyze==True:
        peaks=sig.find_peaks(y,300,10)
        for i in peaks[0]:
            plt.plot(x[i],y[i],'ro')
            plt.annotate(str(x[i])+'nm',(x[i]+5,y[i]),color='red')
        print(peaks)
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"spectrometer_data_of_lightsource_{}.pdf".format(lightsource))
        data = np.column_stack([np.array(x), np.array(y)])#, np.array(z), np.array(abs(y-z))])
        np.savetxt(str(outfile)+"spectrometer_data_of_lightsource_{}.txt".format(lightsource), data, delimiter='\t \t', header='Data of all three Spectrometer-channels for the lightsource {l} \n {e} \nwavelength [nm] \t counts'.format(l=lightsource,e=extratitle))
    return x,y

def Peak_Analyzer(lightsource=''):
    x=Spectrum(lightsource)[0]
    y=Spectrum(lightsource)[1]

    plt.xlabel('wavelength [nm]')
    plt.ylabel('Counts')
    plt.suptitle('Spectrometer Data of {l} \n {e}'.format(l=lightsource,e=extratitle))
    peaks=sig.find_peaks(y,300,10)
    prominences=sig.peak_prominences(x,peaks[0])
    #plt.xlim(x[peaks[0][0]-50],x[peaks[0][3]+50])
    plt.plot(x,y,'r.')
    x_peaks=[]
    y_peaks=[]
    for i,j in zip(peaks[0],prominences[2]):
        x_peaks.extend(x[i-5:i+5])
        y_peaks.extend(y[i-5:i+5])
        plt.plot(x[i],y[i],'ro')
        plt.annotate(str(x[i])+'nm',(x[i]+5,y[i]),color='red')
    plt.plot(x_peaks,y_peaks,'b.')
    plt.show()
    print(peaks[1])
    return peaks[0]

    
    
#This function creates an interpolated curve of the golddata with the same x(wavelength)-data
#then each spectral point is multiplyed by the relative absorption of gold at that wavelength
#it returns the reduced spectrum that is absorbed by gold and the percentage of the original spectrum 
#by deriving both integrals.
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
    absorbed_percentage_integral=(new_spec_int_trap/spec_int_trap)*100

    spec_int_trap =np.sum(Spectrum(lightsource)[1])
    new_spec_int_trap=np.sum(reduced_spectrum)
    absorbed_percentage_points=(new_spec_int_trap/spec_int_trap)*100
    return reduced_spectrum, absorbed_percentage_integral, absorbed_percentage_points, spec_int_trap, new_spec_int_trap

#Plot spectral or gold data quickly on a log scale
def Log_Plot(data):
    x=data[0]
    y=data[1]
    fig,ax = plt.subplots()
    ax.semilogx(x,y)
    plt.show()

#This function plots and saves the original spectrum, the gold absorption curve and the reduces spectrum as 
#absorbed by gold all together and prints the absorbed percentage in the legend
def Reduced_Spectrum(lightsource='', save=False):
    x1=Gold_Abs()[0]
    y1=Gold_Abs()[1]
    label1=Gold_Abs()[2]
    x2=Spectrum(lightsource)[0]
    y2=Spectrum(lightsource)[1]
    label2=Spectrum(lightsource)[2]
    y3=Gold_Fit(lightsource)[0]
    percentage_integral=Gold_Fit(lightsource)[1]
    percentage_points=Gold_Fit(lightsource)[2]
    fig,ax = plt.subplots()
    ax2=ax.twinx()
    #ax2=ax.twiny()
    ax3=ax.twinx()
    #ax3=ax.twiny()
    lns1=ax.semilogx(x2,y2,marker='.',linestyle='-', label=label2, color='red', alpha=0.5)
    lns2=ax.semilogx(x2,y3,marker='.',linestyle='-', label=label2+' reduced to {i}% using integral \n and {p}% using points'.format(i=float(f'{percentage_integral:.2f}'),p=float(f'{percentage_points:.2f}')), color='green')
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
    
def CompareSpectra():
    z=0
    for i in lightsources: 
        x,y=np.genfromtxt(str(outfile)+'spectrometer_data_of_lightsource_'+str(i)+'.txt',unpack=True,skip_header=3)
        title=open(str(outfile)+'spectrometer_data_of_lightsource_'+str(i)+'.txt', 'r').readlines()[1]
        plt.plot(x+z,y,label=title)
        z=z+5
    plt.ylabel('Counts')
    plt.xlabel('wavelength [nm]')
    plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.7))
    plt.show()
#%%
#Down here insert the data you want to analyze and one or several of the above functions and run the script

if __name__ == "__main__":
    infile ='/scratch.mv3/koehn/backup_Anne/zilch/Bolo/Absorption_AU/'
    #outfile='/home/gediz/Results/Goldfoil_Absorption/'
    outfile='/home/gediz/Results/Spectrometer/Spectra_of_He_plasma_15_12_2022/'
    #spectrumdata='/home/gediz/Measurements/Spectrometer/Spectra_of_Helium_Plasma_15_12_2022/'
    spectrumdata='/home/gediz/Results/Spectrometer/Spectra_of_He_plasma_15_12_2022/'
    golddata= '/home/gediz/Results/Goldfoil_Absorption/Golddata_interpolated_for_Spectrometer.txt'
    shotnumber=13120
    gas='He' 
    extratitle='{g} // p={p} mPa// MW={m} W'.format(g=gas,m=float(f'{GetMicrowavePower(shotnumber):.3f}'),p=float(f'{Pressure(shotnumber):.3f}'))
    lightsources=('shot13122_sonde_raus','shot13121_sonde_raus','shot13120_sonde_raus')
    
    #CompareSpectra()
    Reduced_Spectrum('shot13118_sonde_raus')
    #Spectrometer_Data('shot13123',analyze=True)
    #Peak_Analyzer('shot13118_sonde_raus')
    #Gold_Fit('shot13118_sonde_raus_peaks')
    
    
    
    
    
    
# %%
peak1=[1433,3046,4710,5012,2444,3678,1473,2526,1127,780,2028]
peak2=[467,878,1375,1359,873,927,499,750,485,320,655]
peak3=[1080,2138,3260,3748,1882,746,1354,2075,943,657,1436]
peak4=[579,797,1048,1134,771,836,650,664,439,336,708]

p1_p2=[]
p2_p3=[]
p3_p4=[]
for i in np.arange(0,len(peak1)):
    p1_p2.append(peak1[i]/peak2[i])
    p2_p3.append(peak2[i]/peak3[i])
    p3_p4.append(peak3[i]/peak4[i])

print(p1_p2)
print(p2_p3)
print(p3_p4)


# %%
