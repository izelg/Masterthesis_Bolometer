#%%
#Written by: Izel Gediz
#Date of Creation: 11.08.2022
#This skript takes the calibration Data of the omic calibration and determines tau and kapa
#It needs the Oscilloscope Timeseries of the excitation Square Wave and the Measured Voltage at the Bolometerchannels
#In September I added the parts for the relative and absolut calibration with the green laser


from pdb import line_prefix
from tracemalloc import start
from unicodedata import name
from blinker import Signal
from click import style
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import statistics
import os


#%%----------------------------------------------------------------------------------------
def LoadData(location):
    with open(location, "r") as f:
        cols = f.readlines()[0]
        cols = re.sub(r",+", ';', cols)[:-1].split(';')
    data = pd.read_csv(location, skiprows=1, sep=",", names=cols, engine='python')
    return data


def Analyze_U_sq(channelnumber, Plot=True):
    location =str(infile)+'TEK0000{}.CSV'.format(channelnumber)
    time=LoadData(location)['"TIME"']
    U_sq=LoadData(location)['"CH1"'] 
    start= np.argmax(np.gradient(U_sq, time))    #Start of the Square Signal
    stop= np.argmin(np.gradient(U_sq, time))     #End of the Square Signal
    signal_high_time = time[start+10:stop-10]
    signal_low_time = np.concatenate((time[0:start],time[stop:-1]))
    signal_high = U_sq[start+10:stop-10]
    signal_low = np.concatenate((U_sq[0:start],U_sq[stop:-1]))
    U_cal=(np.mean(signal_high)-np.mean(signal_low))
    if Plot==True:
        plt.plot(time, U_sq, alpha=0.5)
        plt.hlines(np.mean(signal_high),time[0], time[-1:],  color='red', linestyle='dotted')
        plt.hlines(np.mean(signal_low),time[0], time[-1:],  color='red', linestyle='dotted')
        plt.vlines(0, np.mean(signal_low), np.mean(signal_high), color='green', label='mean signal height is {} V'.format(f'{U_cal:.2f}' ))
        plt.plot(time[start], U_cal/2, marker='o', linestyle='None', color='red', label='Signal rising edge')
        plt.plot(time[stop], U_cal/2, marker='o', linestyle='None', color='red', label='Signal falling edge')  
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.legend(loc=1, bbox_to_anchor=(1.6,1))
        plt.show()
    return start, stop, U_cal



#This function fits to the calibration Data a function of the form of Equation
#4.31 of Anne Zilchs Diploma Thesis 'Untersuchung von Strahlungsverlusten mittels Bolometrie an einem toroidalen Niedertemperaturplasma' from 2011 
# to determine the constant TAU for a given Bolometerchannel 
def Get_Tau(channelnumber, Plot=False):
    location =str(infile)+'TEK0000{}.CSV'.format(channelnumber)
    def I_func(t,I_0, Delta_I, tau):
        return I_0+Delta_I*(1-np.exp(-t/tau))
    time=LoadData(location)['"TIME"']
    U_sq=LoadData(location)['"CH1"']                #Square Signal to warm the Resistors
    U_b=LoadData(location)['"CH2"']                 #Response Voltage of the Resistors
    I_b=U_b/100                                     #Response Current  through Test Resistor 100 Ohm
    start= np.argmax(np.gradient(U_sq, time))+10    #Start of the Square Signal
    stop= np.argmin(np.gradient(U_sq, time))-10     #End of the Square Signal
    time_cut=time[start:stop]-time[start]           #Time array, shortened to the periode during Square Signal on, and start set to 0 s
    I_b_cut= I_b[start:stop]*1000                   #Current array, shortened equally and Values in mA

    popt, pcov = curve_fit(I_func, time_cut, I_b_cut)
    if Plot==True:
        plt.plot(time_cut, I_b_cut)
        plt.xlabel('Time [s]')
        plt.ylabel('I_b [mA]')
        plt.plot(time_cut, I_func(time_cut, *popt))
        plt.show()
    return popt

#This function fits to the calibration Data a function of the form of Equation
#4.32 and 4.23 of Anne Zilchs Diploma Thesis 'Untersuchung von Strahlungsverlusten mittels Bolometrie an einem toroidalen Niedertemperaturplasma' from 2011 
# to determine the constant KAPPA for a given Bolometerchannel 
def Get_Kappa(channelnumber):
    location =str(infile)+'TEK0000{}.CSV'.format(channelnumber)
    def K_func():
        I_0=Get_Tau(channelnumber, Plot=False)[0]/1000
        Delta_I=Get_Tau(channelnumber, Plot=False)[1]/1000
        U_sq=LoadData(location)['"CH1"'] 
        U_cal=Analyze_U_sq(channelnumber, Plot=False)[2]
        R_M=2*((U_cal/I_0)-100)
        #print('I_0=', f'{I_0:.4f}', 'A','// Delta_I = ',f'{Delta_I:.4f}','A',' // U_cal = ', f'{U_cal:.4f}', 'V',' // R_M = ' ,f'{R_M:.4f}','O')
        return(R_M**2*I_0**4)/(4*U_cal*Delta_I), R_M
        
    return K_func()

def Figure(input, Save=False):
    if input== tau:
        name='tau'
    if input== kappa:
        name='kappa'
    if input==R_M:
        name=' the derived Channel-Resistivities'
    plt.plot(x,input, linestyle='None', marker='o')
    plt.xlabel('Bolometerchannel')
    plt.ylabel('Signal [Arb.]')
    plt.suptitle('Values of {}'.format(name))
    fig1 = plt.gcf()
    plt.show()
    if Save==True:
        fig1.savefig(str(outfile)+"old_calibration_of_Anne_{}.pdf".format(name))
    return

#This function derives all kappas and taus from a measurement series and saves their plots and values.
def GetAllOmicCalibration():
    infile ='/scratch.mv3/koehn/backup_Anne/zilch/measurements/Cal/Bolo_cal_vak/Messwerte_2010_10_08/'
    outfile='/home/gediz/Results/Calibration/old_calibration/'
    x=[]
    tau=[]
    kappa=[]
    R_M=[]
    for channelnumber in [0,1,2,3,4,5,6,7]:
        x.append(channelnumber+1)
        tau.append(Get_Tau(channelnumber, Plot=False)[2])
        kappa.append(abs(Get_Kappa(channelnumber)[0]))
        R_M.append(Get_Kappa(channelnumber)[1])
    Figure(tau, Save=True)
    Figure(kappa, Save=True)
    Figure(R_M, Save=True)
    data = np.column_stack([np.array(x), np.array(tau), np.array(kappa), np.array(R_M)])
    np.savetxt(str(outfile)+"old_calibration_of_Anne.txt" , data, delimiter='\t \t', fmt=['%d', '%10.3f', '%10.3f', '%10.3f'], header='Values for tau \t kappa \t \R_M (derived Resistance of each channel in Ohm)')

#This function derives relative correction constants based on bolometerprofiles derived by raw data.
#Use bolo_radiation.py to create such profiles from your bolometerdata
#Type=mean uses the mean value of all signals as reference, so by multiplying each channel with the resulting correction constant you equalize all signals to the mean signal
#Type=value uses the measured value of 0.419 mW (old batteries) or 0.487 mW (new batteries) to calculate the correction constants and can consequently only be used when Powerprofiles created with the green laser are investigated
def RelativeOpticalCalibration(Type='',save=False):
    x,y=np.genfromtxt(boloprofile, unpack=True)
    if Type=='mean':
        mean=np.mean(y)
    if Type=='value':
        mean=419 
    corr_abs=[]
    corr_rel=[]
    for (i,j) in zip(y,np.arange(0,8)):
        corr_abs.append(mean-i)
        corr_rel.append(mean/i)
        plt.plot([j+1,j+1],[mean,mean-corr_abs[j]], alpha=0.5, label='Relative correction channel°{b}: {c}'.format(b=j+1,c=float(f'{corr_rel[j]:.3f}')))
        
    plt.suptitle(open(boloprofile, 'r').readlines()[2][3:-1])
    plt.plot(x,y,'bo--')
    plt.plot([1,8],[mean,mean], label='Relative to {t}: {m}'.format(t=Type,m=float(f'{mean:.3f}')), color='r')
    plt.ylabel(open(boloprofile, 'r').readlines()[3][14:-1])
    plt.xlabel('Bolometerchannel')
    plt.legend(loc=1,bbox_to_anchor=(1.7,1))
    fig1 = plt.gcf()
    plt.show()
    if save==True:
        data = np.column_stack([np.array(x), np.array(corr_rel)])#, np.array(z), np.array(abs(y-z))])
        np.savetxt(outfile+'relative_calibration_constants_from_'+filename[:-4]+'_using_{}.txt'.format(Type), data, delimiter='\t \t', fmt=['%d', '%10.3f'], header='Correction constants to be multiplied with each channel signal to equalize to {v} V\n relative correction constants from {f}\nchanneln° \t relative correction'.format(v=float(f'{mean:.3f}'), f=filename[:-4]))
        fig1.savefig(outfile+'relative_calibration_constants_from_'+filename[:-4]+'_using_{}.pdf'.format(Type), bbox_inches='tight')


# %%

infile ='/scratch.mv3/koehn/backup_Anne/zilch/measurements/Cal/Bolo_cal_vak/Messwerte_2010_10_08/'
outfile='/home/gediz/Results/Calibration/Calibration_Bolometer_September_2022/relative_correction_constants/'
boloprofile='/home/gediz/Results/Calibration/Calibration_Bolometer_September_2022/combined_shots/shots_60004_to_60011/bolometerprofile_from_radiation_powers_of_calibration_with_green_laser_vacuum.txt'
path,filename=os.path.split(boloprofile)


RelativeOpticalCalibration(Type='value',save=True)

# %%
