#%%
#Written by: Izel Gediz
#Date of Creation: 14.09.2022
#This code takes Powermeterdata of a Thorlabs Powermeter, assembeled with the corresponding software and plots it

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


#%%

#This function reads the power data and transforms it into mW!!!!!!
#The power data is plotted in a row, independent of the timestamp because it is irrelevant
#If SignalHeight is true the mean background and signal values are plotted
#The signal height itself is derived from the value of linear fits to the background and signal data, at the right edge of the signal
#This is the case (rather than it beeing derived by the difference of the mean values) because sometimes adjustments were made in the begining of the measurement.

def PlotPowermeterData(SignalHeight=False, save=False):
    power_=np.genfromtxt(source, delimiter='\t', skip_header=2, usecols=(1), dtype=str)
    power=[]
    x=np.arange(0,len(power_))
    for i in x:
        power.append(float(power_[i].replace(',','.'))*1000)
    print(power)
    plt.plot(x,power,'bo--')
    plt.ylabel('Power [mW]')
    plt.xlabel('Time [arb]')
    plt.suptitle(title)
    if SignalHeight==True:
        def lin (x,a,b):
            return a*x + b
        meanvalue=np.mean(power)
        light_on=[]
        light_on_x=[]
        light_off=[]
        light_off_x=[]
        for i in power:
            if i>meanvalue:
                light_on.append(i)
                light_on_x.append(power.index(i))
            if i<meanvalue:
                light_off.append(i)
                light_off_x.append(power.index(i))
        steps=[]
        for i in np.arange(0, len(light_off_x)-1):
            step= (light_off_x[i]-light_off_x[i+1])
            steps.append(abs(step))                     
        right_edge=light_off_x[int(np.argwhere(np.array(steps)>1))+2]
        popt1, pcov1 = curve_fit(lin,light_off_x,light_off)
        popt2, pcov2 = curve_fit(lin,light_on_x,light_on)
        signalheight=float(f'{abs(lin(right_edge-1,*popt2)-lin(right_edge-1,*popt1)):.4f}')
        background=float(f'{np.mean(light_off):.4f}') 
        signal=float(f'{np.mean(light_on):.4f}') 
        plt.plot(x, lin(x, *popt1), label=('Average background radiation {} mW'.format(background)))
        plt.plot(x, lin(x, *popt2), label=('Average signal {} mW'.format(signal)))
        plt.plot([right_edge-1,right_edge-1],[lin(right_edge-1,*popt1),lin(right_edge-1,*popt2)], color='red', label='Signal height {} mW'.format(signalheight))
        plt.legend(loc=1, bbox_to_anchor=(1.8,1))
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+str(filename[:-4])+".pdf", bbox_inches='tight')

#%%
source='/home/gediz/Measurements/Powermeter/weißlicht_quarzfenster_pos_4.txt'
title= 'White light through quarzglas \n at height of bolometerhead, sensor height ~channel 7 and 8'
outfile='/home/gediz/Results/Powermeter/'
path,filename=os.path.split(source)


PlotPowermeterData(SignalHeight=True,save=True)

# %%
