#%%


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re



#%% --------------------------------------------------------------------------------------------------------


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
    return (np.mean(signalinwatt_for[start:stop])-np.mean(signalinwatt_back[start:stop]))
    
#%%
shotnumber=13122
gas='He'
location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)

print('shot n°{s} // {g} //MW Power was {mw} W, pressure was {p} mPa'.format(s=shotnumber,g=gas,mw=float(f'{GetMicrowavePower(shotnumber):.3f}'),p=float(f'{Pressure(shotnumber):.3f}')))

# %%
