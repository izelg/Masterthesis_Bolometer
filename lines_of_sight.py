#%%
#Written by: Izel Gediz
#Date of Creation: 16.09.2022
#This code uses data collected in the measurement of the lines of sight of the bolometerchannels which was produced using a 2D Stepping Motor.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import re
from scipy.interpolate import pchip_interpolate
#from bolo_radiation import LoadData, PlotAllTimeseries, PlotAllTimeseriesTogether, PlotSingleTimeseries

#%%

def MotorData(save=False):
    x,a,p =np.genfromtxt(motordata, unpack=True)    #x is the position data in mm, a is the "amplitude" data stored and processed by the software, p is the "Phase" data stored and processed by the software
    x_=np.arange(x[0],x[-1],0.00001)
    fit=pchip_interpolate(x,a,x_)
    amp=list(i-min(a) for i in fit)
    plt.plot(x_,amp,'r.--', label='Interpolated signal "Amplitude"')
    signal_edge_list=[np.argwhere(amp>max(amp)/np.e), np.argwhere(amp>max(amp)/10)]
    for signal_edge in signal_edge_list:
        fwhm1, fwhm2=x_[signal_edge[0]],x_[signal_edge[-1]]
        plt.plot(fwhm1,amp[int(signal_edge[0])],'bo')
        plt.plot(fwhm2,amp[int(signal_edge[-1])],'bo')
        fwhm=float(fwhm2-fwhm1)
        plt.plot([fwhm1,fwhm2],[amp[int(signal_edge[0])],amp[int(signal_edge[-1])]], color='blue', label='Width of channel: {} m'.format(float(f'{fwhm:.4f}')))
        
    #plt.plot(x,a,'b.--', label='"Amplitude Data"'.format(float(f'{np.mean(a):.4f}')))
    #plt.plot(x,p, label='"Phase Data": {} V'.format(float(f'{np.mean(p):.4f}')))
    plt.xlabel('Position [m]')
    plt.ylabel('Preprocessed Signal [V]')
    plt.suptitle(motordatatitle)
    plt.legend(loc=1, bbox_to_anchor=(1.4,1))
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(motordataoutfile)+str(filename[:-4])+".pdf", bbox_inches='tight')


#importing and using functions from bolo_radiation.py doesn't work yet so I copied them here
def LoadData(location):
    with open(location, "r") as f:
        cols = f.readlines()[3]
        cols = re.sub(r"\t+", ';', cols)[:-2].split(';')
    data = pd.read_csv(location, skiprows=4, sep="\t\t", names=cols, engine='python')
    return data

#This Function plots a timeseries of your choosing
#-->use these channelnames: Zeit [ms]		8 GHz power		2 GHz Richtk. forward	I_Bh			U_B			Pressure		2 GHz Richtk. backward	slot1			I_v			Interferometer (Mueller)	Interferometer digital	8 GHz refl. power	Interferometer (Zander)	Bolo_sum		Bolo1			Bolo2			Bolo3			Bolo4			Bolo5			Bolo6			Bolo7			Bolo8			optDiode		r_vh			Coil Temperature
def PlotSingleTimeseries(i=1, save=False):
    if Datatype=='Data':
        y= LoadData(location)["Bolo{}".format(i)]
        time = LoadData(location)['Zeit [ms]'] / 1000
        title='Shot n° {s} // Channel "Bolo {n}" \n {e}'.format(s=shotnumber, n=i, e=extratitle)
    elif Datatype=='Source':
        time=np.genfromtxt(str(source), usecols=(0), unpack=True, skip_footer=200)
        y=np.genfromtxt(str(source), usecols=(i), unpack=True, skip_footer=200)
        title='Raw signal data of {s} // Channeln° {n}\n {e}'.format(s=sourcetitle, n=i, e=extratitle)
    
    plt.figure(figsize=(10,5))
    plt.plot(time, y)
    plt.suptitle(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Signal [V]')
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_channel_{c}_raw_signal.pdf".format(n=shotnumber, c=i), bbox_inches='tight')
    return time,y

def MotorAndBoloData(i=1):
    x,a,p =np.genfromtxt(motordata, unpack=True)    #x is the position data in mm, a is the "amplitude" data stored and processed by the software, p is the "Phase" data stored and processed by the software
    # x_=np.arange(x[0],x[-1],0.00001)
    # fit=pchip_interpolate(x,a,x_)
    # amp=list(i+20 for i in fit)
    y= LoadData(location)["Bolo{}".format(i)]
    time = LoadData(location)['Zeit [ms]'] / 1000
    
    print(len(time),len(x))


#%%
motordata='/home/gediz/Measurements/Lines_of_sight/motor_data/shot60037_bolo8_y.dat'
motordatatitle='y-Sweep of Channel 8'
motordataoutfile='/home/gediz/Results/Lines_of_sight/motor_data/'
path,filename=os.path.split(motordata)

#for the bolo_ratdiation functions:
Datatype='Data'
shotnumber=60037
location='/home/gediz/Measurements/Lines_of_sight/shot_data/shot{}.dat'.format(shotnumber)
outfile='/home/gediz/Results/Lines_of_sight/shot_data/'
extratitle=motordatatitle
if not os.path.exists(str(outfile)+'shot{}'.format(shotnumber)):
    os.makedirs(str(outfile)+'shot{}'.format(shotnumber))


PlotSingleTimeseries(8, save=True)
MotorData(save=True)
#MotorAndBoloData(8)

# %%
