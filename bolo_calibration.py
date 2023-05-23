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
from scipy.signal import savgol_filter
import sympy as sp


#%%----------------------------------------------------------------------------------------
def LoadData(location):
    with open(location, "r") as f:
        cols = f.readlines()[0]
        cols = re.sub(r",+", ';', cols)[:-1].split(';')
    data = pd.read_csv(location, skiprows=1, sep=",", names=cols, engine='python')
    return data

#This function analyzes the Square function used to heat the resistors for the ohmic calibration
#It returns the duty cycle and height of the pulse
def Analyze_U_sq(documentnumber, Plot=False):
    location =str(infile)+'NewFile{}.csv'.format(documentnumber)
    time, U_sq= np.genfromtxt(location,delimiter=',',unpack=True, usecols=(0,1),skip_header=2)
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

#For a ohmic calibration this function reproduces the Oscilloscope Picture
#In this case the Oscilloscope used could only store the data of the two channels in two different documents
def OscilloscopePicture(documentnumber):
    #time, U_sq= np.genfromtxt(str(infile)+'NewFile{}.csv'.format(documentnumber_U_sq),delimiter=',',unpack=True, usecols=(0,1))    #Square Signal to warm the Resistors
    time,U_sq,U_b_n,U_b= np.genfromtxt(str(infile)+'NewFile{}.csv'.format(documentnumber),delimiter=',',unpack=True, usecols=(0,1,2,3))    #Square Signal to warm the Resistors
    #U_b=savgol_filter(U_b0,10,3)
    fig,ax1=plt.subplots()
    ax2=ax1.twinx()
    ax3=ax1.twinx()
    lns1=ax1.plot(time, U_sq,color='blue',label='Square Pulse')
    lns3=ax3.plot(time,U_b_n,color='red',label='Bolometer Response',alpha=0.5)
    lns2=ax2.plot(time,U_b,color='red',label='Bolometer Response with Filter')
    ax1.set(ylabel='Voltage [V]')
    ax1.set(xlabel='Time [s]')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    ax3.set_yticks([], [])
    #ax2.set_ylim([0.53,0.55])
    leg = lns1 + lns2 +lns3
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=1,bbox_to_anchor=(1.7,1))

    plt.suptitle('Oscilloscope Signal for a square pulse heating Bolometerchannel 1 \n and the Channels Response Signal')
    plt.show()
        
        
        
#This function fits to the calibration Data a function of the form of Equation
#4.31 of Anne Zilchs Diploma Thesis 'Untersuchung von Strahlungsverlusten mittels Bolometrie an einem toroidalen Niedertemperaturplasma' from 2011 
# to determine the constant TAU for a given Bolometerchannel 
def Get_Tau( documentnumber, Plot=False):
    def I_func(t,I_0, Delta_I, tau):
        return I_0+Delta_I*(1-np.exp(-t/tau))
    #time, U_sq= np.genfromtxt(str(infile)+'NewFile{}.csv'.format(documentnumber_U_sq),delimiter=',',unpack=True, usecols=(0,1))    #Square Signal to warm the Resistors
    time,U_sq,U_b= np.genfromtxt(str(infile)+'NewFile{}.csv'.format(documentnumber),delimiter=',',unpack=True, usecols=(0,1,3),skip_header=2)    #Square Signal to warm the Resistors
    I_b=U_b/100                                     #Response Current  through Test Resistor 100 Ohm
    start= np.argmax(np.gradient(U_sq, time))+3    #Start of the Square Signal
    stop= np.argmin(np.gradient(U_sq, time))-3   #End of the Square Signal
    time_cut=time[start:stop]-time[start]           #Time array, shortened to the periode during Square Signal on, and start set to 0 s
    I_b_cut= I_b[start:stop]*1000                   #Current array, shortened equally and Values in mA
    print(start,stop)
    print(len(I_b_cut))
    popt, pcov = curve_fit(I_func, time_cut,I_b_cut)
    #I_b_cut_sav=savgol_filter(I_b_cut,10,3)
    if Plot==True:
        #plt.plot(time[start],np.gradient(U_b)[start],'ro')
        plt.plot(time_cut, I_b_cut, color='red', alpha=0.5,label='Bolometer Response')
        plt.xlabel('Time [s]')
        plt.ylabel('I_b [mA]')
        plt.plot(time_cut, I_func(time_cut, *popt),color='darkred', label='Exponential Fit \n tau={}'.format(float(f'{popt[2]:.4f}')))
        plt.legend(loc=1, bbox_to_anchor=(1.4,1))
        plt.show()
    print (popt)
    return popt

#This function fits to the calibration Data a function of the form of Equation
#4.32 and 4.23 of Anne Zilchs Diploma Thesis 'Untersuchung von Strahlungsverlusten mittels Bolometrie an einem toroidalen Niedertemperaturplasma' from 2011 
# to determine the constant KAPPA for a given Bolometerchannel 
def Get_Kappa(documentnumber):
    def K_func():
        I_0=Get_Tau(documentnumber, Plot=False)[0]/1000
        Delta_I=Get_Tau(documentnumber, Plot=False)[1]/1000
        #U_sq= np.genfromtxt(str(infile)+'TEK00{}.CSV'.format(documentnumber_U_sq),delimiter=',',unpack=True, usecols=(4))    #Square Signal to warm the Resistors
        U_cal=Analyze_U_sq(documentnumber)[2]
        R_M=2*((U_cal/I_0)-100)
        #print('I_0=', f'{I_0:.4f}', 'A','// Delta_I = ',f'{Delta_I:.4f}','A',' // U_cal = ', f'{U_cal:.4f}', 'V',' // R_M = ' ,f'{R_M:.4f}','O')
        return(R_M**2*I_0**4)/(4*U_cal*Delta_I), R_M
        
    return K_func()



#This function derives all kappas and taus from a measurement series and saves their plots and values.
def GetAllOmicCalibration(save=False):
    for n in [0,1,2,3,4,5,6,7,8,9]:
        x=[]
        tau=[]
        kappa=[]
        R_M=[]
        for i in [str(n),'1'+str(n),'2'+str(n),'3'+str(n),'4'+str(n),'5'+str(n),'6'+str(n),'7'+str(n)]:
            x=[1,2,3,4,5,6,7,8]
            tau.append(Get_Tau(i, Plot=True)[2])
            kappa.append(abs(Get_Kappa(i)[0]))
            R_M.append(Get_Kappa(i)[1])
        #print(tau)
        plt.plot(x,tau,'bo')
        plt.xlabel('Bolometerchannel')
        plt.ylabel('tau [s]')
        fig1=plt.gcf()
        plt.show()
        plt.plot(x,kappa,'ro')
        plt.xlabel('Bolometerchannel')
        plt.ylabel('kappa [10^-4 A^2]')
        fig2=plt.gcf()
        plt.show()
        if save ==True:
            data = np.column_stack([np.array(x), np.array(tau), np.array(kappa), np.array(R_M)])
            np.savetxt(str(outfile)+"ohmic_calibration_air_tau_and_kappa_reduced_noise_measurement_0{}.txt".format(n) , data, delimiter='\t \t', fmt=['%d', '%10.3f', '%10.3f', '%10.3f'], header='Values for tau \t kappa \t \R_M (derived Resistance of each channel in Ohm)')
            fig1.savefig(str(outfile)+"ohmic_calibration_tau_air_reduced_noise_measurement_0{}.pdf".format(n))
            fig1.savefig(str(outfile)+"ohmic_calibration_kappa_air_reduced_noise_measurement_0{}.pdf".format(n))


#This function plots a comparison of different Ohmic calibration measurements
def CompareTauAndKappa(save=False):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10 = ([] for i in range(30))
    for i,j,k,n in zip([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10],[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10],[k1,k2,k3,k4,k5,k6,k7,k8,k9,k10],[0,1,2,3,4,5,6,7,8,9]):
        i.append(np.genfromtxt('/home/gediz/Results/Calibration/Ohmic_Calibration/Ohmic_Calibration_Vacuum_November/ohmic_calibration_vacuum_tjk_tau_and_kappa_reduced_noise_measurement_0{}.txt'.format(n), unpack=True, usecols=(0)))
        j.append(np.genfromtxt('/home/gediz/Results/Calibration/Ohmic_Calibration/Ohmic_Calibration_Vacuum_November/ohmic_calibration_vacuum_tjk_tau_and_kappa_reduced_noise_measurement_0{}.txt'.format(n), unpack=True, usecols=(1)))
        k.append(np.genfromtxt('/home/gediz/Results/Calibration/Ohmic_Calibration/Ohmic_Calibration_Vacuum_November/ohmic_calibration_vacuum_tjk_tau_and_kappa_reduced_noise_measurement_0{}.txt'.format(n), unpack=True, usecols=(2))) 
    mean_t,sd_t,sem_t=[],[],[]
    for i,j,k,n in zip([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10],[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10],['red','blue','orange','green','darkcyan','gold','blueviolet','magenta','grey','yellow'],[0,1,2,3,4,5,6,7,8,9]):
        plt.plot(i,j,label='Measurement {} TJ-K'.format(n),marker='o',color=k,alpha=0.3)
    for m in [0,1,2,3,4,5,6,7]:
        val=[t1[0][m],t2[0][m],t3[0][m],t4[0][m],t5[0][m],t6[0][m],t7[0][m],t8[0][m],t9[0][m],t10[0][m]]
        mean_t.append(np.mean(val))
        sd_t.append(np.std(val,ddof=1))
        sem_t.append(np.std(val,ddof=1)/np.sqrt(len(val)))
        plt.errorbar(m+1,mean_t[m],yerr=sem_t[m],marker='o',linestyle='None', capsize=5,color='red')
    plt.xlabel('Bolometerchannel')
    plt.ylabel('tau [s]')
    #plt.legend(loc=1,bbox_to_anchor=(1.5,1))
    plt.suptitle('Ohmic calibration in air// Results for Tau')
    fig1=plt.gcf()
    plt.show()
    mean_k,sd_k,sem_k=[],[],[]
    for i,j,k,n in zip([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10],['red','blue','orange','green','darkcyan','gold','blueviolet','magenta','grey','yellow'],[k1,k2,k3,k4,k5,k6,k7,k8,k9,k10],[0,1,2,3,4,5,6,7,8,9]):
        plt.plot(i,k,label='Measurement {} TJ-K'.format(n),marker='o',color=j,alpha=0.3)
    for m in [0,1,2,3,4,5,6,7]:
        val=[k1[0][m],k2[0][m],k3[0][m],k4[0][m],k5[0][m],k6[0][m],k7[0][m],k8[0][m],k9[0][m],k10[0][m]]
        mean_k.append(np.mean(val))
        sd_k.append(np.std(val,ddof=1))
        sem_k.append(np.std(val,ddof=1)/np.sqrt(len(val)))
        plt.errorbar(m+1,mean_k[m],yerr=sem_k[m],marker='o',linestyle='None', capsize=5,color='red')
    plt.xlabel('Bolometerchannel')
    plt.ylabel('kappa [A^2]')
    #plt.legend(loc=1,bbox_to_anchor=(1.5,1))
    plt.suptitle('Ohmic calibration in air// Results for Kappa')
    fig2=plt.gcf()
    plt.show()
    print(sd_k)
    if save==True:
        data = np.column_stack([np.array([1,2,3,4,5,6,7,8]),np.array(mean_t), np.array(sem_t), np.array(mean_k), np.array(sem_k)])
        np.savetxt(str(outfile)+"ohmic_calibration_air_tau_and_kappa_mean_and_sem.txt" , data, delimiter='\t \t', fmt=['%d', '%10.5f', '%10.5f', '%10.5f', '%10.5f'], header='Values for tau \t sem sau \t kappa \t sem kappa')
        fig1.savefig(str(outfile)+"ohmic_calibration_tau_air_mean_and_sem.pdf")
        fig2.savefig(str(outfile)+"ohmic_calibration_kappa_air_mean_and_sem.pdf")

        

#This function derives the actual resistances using the measured values by solving a set of linear equations
def DeriveResistances():
    ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8=np.genfromtxt('/home/gediz/Measurements/Calibration/Channel_resistances_September_2022/All_resistor_values_bolometer_sensor_third_Measurement.txt',unpack=True,usecols=(1,2,3,4,5,6,7,8),skip_header=6)
    ch1_cal=[]
    ch2_cal=[]
    ch3_cal=[]
    ch4_cal=[]
    ch5_cal=[]
    ch6_cal=[]
    ch7_cal=[]
    ch8_cal=[]
    for i,j in zip([ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8],[ch1_cal,ch2_cal,ch3_cal,ch4_cal,ch5_cal,ch6_cal,ch7_cal,ch8_cal]):
        #print(i)
        m1,m2,r1,r2=sp.symbols(list("abcd"))
        m1_val=i[0]
        m2_val=i[1]
        r1_val=i[2]
        r2_val=i[3]
        eqns=[1/m1+1/(m2+r1+r2)-1/m1_val,1/m2+1/(m1+r1+r2)-1/m2_val,1/r1+1/(r2+m1+m2)-1/r1_val,1/r2+1/(r1+m1+m2)-1/r2_val]
        #print(sp.solve(eqns,m1,m2,r1,r2)[3])
        j.append(sp.solve(eqns,m1,m2,r1,r2)[3])
        print(j)

#This funtion is comparing the results of the calculated resistances for different measurements using the value of RM1
def CompareResistances():
    rm1,rm2,rr1,rr2=np.genfromtxt('/home/gediz/Results/Calibration/Channel_resistances_September_2022/all_resistor_values_bolometer_sensors_calculated.txt',unpack=True,delimiter=',',usecols=(1,2,3,4))
    rm1_2,rm2_2,rr1_2,rr2_2=np.genfromtxt('/home/gediz/Results/Calibration/Channel_resistances_September_2022/all_resistor_values_bolometer_sensors_calculated_second_set.txt',unpack=True,delimiter=',',usecols=(1,2,3,4))
    rm1_3,rm2_3,rr1_3,rr2_3=np.genfromtxt('/home/gediz/Results/Calibration/Channel_resistances_September_2022/all_resistor_values_bolometer_sensors_calculated_third_set.txt',unpack=True,delimiter=',',usecols=(1,2,3,4))
    for x,y in zip([1,2,3,4,5,6,7,8],[2,1,4,3,6,5,8,7]):     
        plt.plot(x,rm1[x-1],'bo',label='First Definite Measurement')
        plt.plot(x,rm1_2[x-1],'ro',label='Second Measurement to compare')
        plt.plot(x,rm1_3[x-1],'go',label='Third Measurement to compare')

    plt.xlabel('channels')
    plt.ylabel('Resistance in Ohm')
    plt.suptitle('Values for -Measurement Resistor 1- of different Measurements')
    plt.show()      

#This function was written to plot the results of the wavelength dependency investigation
#A documentation and resulsts can be found here /home/gediz/Results/Calibration/Wavelength_dependency_study
def WavelengthDependency():
    x,y,z=np.genfromtxt('/home/gediz/Results/Calibration/Wavelength_dependency_study/absorbed_percentages.txt', unpack=True)
    def lin(x,a,b):
        return a*x+b
    popt,pcov=curve_fit(lin,x,y)
    plt.plot(x,y,'o',color='red',label='Absorbed red light percentage\n form white light')
    plt.plot(x,lin(x,*popt),color='red',alpha=0.5, label='Fit to three shots')
    plt.plot([1,8],[0.23,0.23],color='red',label='Expected ratio 23%')
    popt,pcov=curve_fit(lin,x,z)
    plt.plot(x,z,'o',color='green',label='Absorbed green light percentage \n form white light')
    plt.plot(x,lin(x,*popt),color='green',alpha=0.5, label='Fit to three shots')
    plt.plot([1,8],[0.25,0.25],color='green',label='Expected ratio 25%')
    plt.xlabel('Channelnumber')
    plt.ylabel('Absorbed Percentage')
    plt.legend(loc=1,bbox_to_anchor=(1.6,1))
    plt.suptitle('Wavelength dependency investigation')
    plt.show()
 
    
#This function derives relative correction constants based on bolometerprofiles derived by raw data.
#Use bolo_radiation.py to create such profiles from your bolometerdata
#Type=mean uses the mean value of all signals as reference, so by multiplying each channel with the resulting correction constant you equalize all signals to the mean signal
#Type=value uses the measured value of 0.419 mW (old batteries) or 0.487 mW (new batteries) to calculate the correction constants and can consequently only be used when Powerprofiles created with the green laser are investigated
def RelativeOpticalCalibration(Type='',save=False):
    #x,y=np.genfromtxt(boloprofile, unpack=True, usecols=(0,1))
    x=[1,2,3,4,5,6,7,8]
    y=boloprofile
    #y[-1]=y[-1]*2.6
    #print(y)
    if Type=='mean':
        mean=np.mean(y)
    if Type=='value':
        mean=193#419 
    corr_abs=[]
    corr_rel=[]
    for (i,j) in zip(y,np.arange(0,8)):
        corr_abs.append(mean-i)
        corr_rel.append(mean/i)
        plt.plot([j+1,j+1],[mean,mean-corr_abs[j]], alpha=0.5, label='Relative correction channel°{b}: {c}'.format(b=j+1,c=float(f'{corr_rel[j]:.3f}')))
    corr_y=[]
    for a,b in zip(y, corr_rel):
        corr_y.append(a*b)
    #plt.suptitle(open(boloprofile, 'r').readlines()[2][3:-1])
    plt.suptitle(t)
    plt.plot(x,y,'bo--')
    plt.plot(x,corr_y,'ro--', label='Relative to {t}: {m}'.format(t=Type,m=float(f'{mean:.3f}')))
    #plt.plot([1,8],[mean,mean], label='Relative to {t}: {m}'.format(t=Type,m=float(f'{mean:.3f}')), color='r')
    #plt.ylabel(open(boloprofile, 'r').readlines()[3][14:-1])
    plt.xlabel('Bolometerchannel')
    plt.legend(loc=1,bbox_to_anchor=(1.7,1))
    fig1 = plt.gcf()
    plt.show()
    if save==True:
        data = np.column_stack([np.array(x), np.array(corr_rel)])#, np.array(z), np.array(abs(y-z))])
        np.savetxt(outfile+'relative_calibration_constants_from_'+t+'_using_{}.txt'.format(Type), data, delimiter='\t \t', fmt=['%d', '%10.3f'], header='Correction constants to be multiplied with each channel signal to equalize to {v} V\n relative correction constants from {f}\nchanneln° \t relative correction'.format(v=float(f'{mean:.3f}'), f=filename[:-4]))
        fig1.savefig(outfile+'relative_calibration_constants_from_'+t+'_using_{}.pdf'.format(Type), bbox_inches='tight')

#This function was used to plot all different experiments to calibrate the bolometer together
#Since more  experiments were added all the time and each needed a descriptive title the function is not very elegant
def CompareBolometerProfiles(save=False):
    x=[1,2,3,4,5,6,7,8]
    y0=np.genfromtxt(boloprofile_0, unpack=True, usecols=1)
    y1=np.genfromtxt(boloprofile_1, unpack=True, usecols=1)
    y2=np.genfromtxt(boloprofile_2, unpack=True, usecols=1)
    y3=np.genfromtxt(boloprofile_3, unpack=True, usecols=1)
    y4=np.genfromtxt(boloprofile_4, unpack=True, usecols=1)
    y5=np.genfromtxt(boloprofile_5, unpack=True, usecols=1)
    y6=np.genfromtxt(boloprofile_6, unpack=True, usecols=1)
    y7=np.genfromtxt(boloprofile_7, unpack=True, usecols=1)
    y8=np.genfromtxt(boloprofile_8, unpack=True, usecols=1)
    for i in [y0,y1,y2,y3,y4,y5,y6,y7,y8]:
        i[-1]=i[-1]*2.6
    for k in [y0,y1,y2,y3,y4,y5,y6,y7,y8]:
        #for i,j in zip([0,1,2,3,4,5,6,7],[0.621,0.836,0.965,0.635,1.669,1.307,1.748,1.657]):
        for i,j in zip([0,1,2,3,4,5,6,7],[0.703,0.930,1.104,0.728,1.325,1.042,1.438,1.313]):  
            k[i]=k[i]*j
    ch1=[]
    ch2=[]
    ch3=[]
    ch4=[]
    ch5=[]
    ch6=[]
    ch7=[]
    ch8=[]
    mean=[]
    sd=[]
    sem=[]
    mean_err=[]
    for i,j,k,p in zip([0,1,2,3,4,5,6,7],[ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8],[0.016,0.019,0.024,0.015,0.043,0.029,0.045,0.050],[0.132*0.621,0.076*0.836,0.076*0.965,0.069*0.635,0.041*1.669,0.051*1.307,0.052*1.748,0.05*1.647]):
        j.extend([y0[i],y1[i],y2[i],y3[i],y4[i],y5[i],y6[i],y7[i],y8[i]])
        sd.append(np.std(j,ddof=1))
        sem.append(np.std(j,ddof=1)/np.sqrt(9))
        mean.append(np.mean(j))
        mean_err.append(np.mean(j)*k+p)
    plt.ylim(200,425)
    plt.plot(x,y0,alpha=al,label='by hand, batteries, after reassemblance', marker='o')
    plt.plot(x,y1,alpha=al,label='with step motor, batteries, by hand, 1mm steps', marker='o')
    plt.plot(x,y2,alpha=al, label='2D scan, 3VDC, angle upwards',  marker='o')
    plt.plot(x,y3,alpha=al,label='2D scan, 3VDC, angle downwards', marker='o')
    plt.plot(x,y4 ,alpha=al,label='by Hand, 3VDC, horizontal', marker='o')    
    plt.plot(x,y5,alpha=al,label='by Hand, 3VDC, angle upwards',  marker='o')
    plt.plot(x,y6 ,alpha=al,label='by Hand, 3VDC, angle downwards (breakdown)', marker='o')
    plt.plot(x,y7,alpha=al,label='by Hand, new batteries, angle downwards', marker='o')
    plt.plot(x,y8 ,alpha=al,label='by Hand, new batteries, angle downwards', marker='o')
    plt.plot(x,mean,'ro',label='mean value of all measurements')
    plt.errorbar(x,mean,yerr=mean_err, ecolor='red',fmt='none',capsize=5,label='Error due to correction constant')
    plt.suptitle('Radiation Powers // Green laser in vacuum')
    plt.xlabel('Bolometerchannel')
    plt.ylabel('Power [\u03bcW]')
    plt.legend(loc=1,bbox_to_anchor=(1.9,1))
    fig1=plt.gcf()
    plt.show()  
    print(mean,mean_err)
    if save ==True:
        data = np.column_stack([np.array(x), np.array(mean), np.array(sd), np.array(sem)])
        np.savetxt(str(outfile)+"calibration_green_laser_vacuum_radiation_power__by_mean_correction_constants_mean_sd_sem.txt" , data, delimiter='\t \t', fmt=['%d', '%10.3f', '%10.3f', '%10.3f'], header='From all Calibration measurements with the green laser in vacuum the mean value of the original signals for each channel and their sd and sem. \n channelnumber   mean[V]    sd  sem')
        fig1.savefig(str(outfile)+"calibration_green_laser_all_radiation_power_profiles_with_mean_and_error.pdf",bbox_inches='tight')




#This function plots the resulting Relative Corrections derived with RelativeOpticalCAlibration()
def CompareRelativeCorrections(save=False):
    x=[1,2,3,4,5,6,7,8]
    y0=np.genfromtxt(relativecorrection_0, unpack=True, usecols=1)
    y1=np.genfromtxt(relativecorrection_1, unpack=True, usecols=1)
    y2=np.genfromtxt(relativecorrection_2, unpack=True, usecols=1)
    y3=np.genfromtxt(relativecorrection_3, unpack=True, usecols=1)
    y4=np.genfromtxt(relativecorrection_4, unpack=True, usecols=1)
    y5=np.genfromtxt(relativecorrection_5, unpack=True, usecols=1)
    y6=np.genfromtxt(relativecorrection_6, unpack=True, usecols=1)
    y7=np.genfromtxt(relativecorrection_7, unpack=True, usecols=1)
    y8=np.genfromtxt(relativecorrection_8, unpack=True, usecols=1)
    y9=np.genfromtxt(relativecorrection_9, unpack=True, usecols=1)
    y10=np.genfromtxt(relativecorrection_10, unpack=True, usecols=1)
    y11=np.genfromtxt(relativecorrection_11, unpack=True, usecols=1)
    ch1=[]
    ch2=[]
    ch3=[]
    ch4=[]
    ch5=[]
    ch6=[]
    ch7=[]
    ch8=[]
    mean=[]
    sd=[]
    sem=[]
    for i,j in zip([0,1,2,3,4,5,6,7],[ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8]):
        j.extend([y0[i],y1[i],y2[i],y3[i],y4[i],y5[i],y6[i],y7[i],y8[i]])
        sd.append(np.std(j,ddof=1))
        sem.append(np.std(j,ddof=1)/np.sqrt(9))
        mean.append(np.mean(j))
    al=1
    #plt.plot(x,y0,alpha=al,label='air,  1k$\Omega$, sine, simulated wheatstone', marker='o')
    #plt.plot(x,y1,alpha=al,label='Air, 1k$\Omega$', marker='o')
    #plt.plot(x,y2,alpha=al, label='air, 1k$\Omega$, changed V2,V3',  marker='o')
    #plt.plot(x,y3,alpha=al,label='UV air 254', marker='o')
    #plt.plot(x,y4 ,alpha=al,label='UV air 350', marker='o')    
    #plt.plot(x,y5,alpha=al,label='UV air klein',  marker='o')
    plt.plot(x,y6 ,alpha=al,label='vac,  1k$\Omega$', marker='o')
    plt.plot(x,y7,alpha=al,label='vac,  100$\Omega$', marker='o')
    #plt.plot(x,y8 ,alpha=al,label='vac, sine 1k$\Omega$', marker='o')
    plt.plot(x,y9,alpha=al,label='Res 1', marker='o')
    plt.plot(x,y10,alpha=al,label='Res 2', marker='o')
    plt.plot(x,y11,alpha=al,label='Res 3', marker='o')
    #plt.plot(x,mean,'ro',label='mean value of all measurements')
    #plt.errorbar(x,mean,yerr=sem, ecolor='red',fmt='none',capsize=5,label='Standard error of the mean')
    plt.suptitle('Relative correction constants for Radiation Powers // Green laser in vacuum')
    plt.xlabel('Bolometerchannel')
    plt.ylabel('[arb]')
    plt.legend(loc=1,bbox_to_anchor=(2,1))
    fig1=plt.gcf()
    plt.show()  
    if save ==True:
        data = np.column_stack([np.array(x), np.array(mean), np.array(sd), np.array(sem)])
        np.savetxt(str(outfile)+"calibration_green_laser_vacuum_power_signals_relative_corrections_mean_sd_sem.txt" , data, delimiter='\t \t', fmt=['%d', '%10.3f', '%10.3f', '%10.3f'], header='From all Calibration measurements with the green laser in vacuum the mean value of the relative correction constant for each channel and their sd and sem. \n channelnumber   mean[V]    sd  sem')
        fig1.savefig(str(outfile)+"calibration_green_laser_power_signals_all_relative_corrections_with_mean_and_error.pdf",bbox_inches='tight')

# %%

infile ='/home/gediz/Measurements/Calibration/Ohmic_Calibration/Ohmic_Calibration_Vacuum_November/10_11_2022/'
outfile='/home/gediz/Results/Calibration/Ohmic_Calibration/Ohmic_Calibration_Air_December/07_12_2022/'
boloprofile='/home/gediz/Results/Bolometer_Profiles/shot70032/shot70032_bolometerprofile_from_radiation_powers.txt'

##Bolometerprofile from which to calculate the relative correction constants:
boloprofile_0='/home/gediz/Results/Calibration/Calibration_Bolometer_September_2022/combined_shots/shots_60004_to_60011/bolometerprofile_from_radiation_powers_of_calibration_with_green_laser_vacuum.txt'
boloprofile_1='/home/gediz/Results/Calibration/Calibration_Bolometer_September_2022/bolometerprofiles/shot60039/shot60039_bolometerprofile_from_radiation_powers.txt'
boloprofile_2='/home/gediz/Results/Calibration/Calibration_Bolometer_September_2022/bolometerprofiles/shots_60042_60043/bolometerprofile_from_radiation_powers_of_calibration_with_green_laser_vacuum_2D_scan_upwards_angle.txt'
boloprofile_3='/home/gediz/Results/Calibration/Calibration_Bolometer_September_2022/bolometerprofiles/shots_60044_to_60046/bolometerprofile_from_radiation_powers_of_calibration_with_green_laser_vacuum_2D_scan_downwards_angle.txt'
boloprofile_4='/home/gediz/Results/Calibration/Calibration_Bolometer_September_2022/bolometerprofiles/shots_60048_60050/bolometerprofile_from_radiation_powers_of_calibration_with_green_laser_vacuum_by hand_horizontal_beam_3VDC.txt'
boloprofile_5='/home/gediz/Results/Calibration/Calibration_Bolometer_September_2022/bolometerprofiles/shot60052/bolometerprofile_from_radiation_powers_of_calibration_with_green_laser_vacuum_by hand_upwards_beam_3VDC.txt'
boloprofile_6='/home/gediz/Results/Calibration/Calibration_Bolometer_September_2022/bolometerprofiles/shots60053_60054/bolometerprofile_from_radiation_powers_of_calibration_with_green_laser_vacuum_by hand_downwards_beam_3VDC.txt'
boloprofile_7='/home/gediz/Results/Calibration/Calibration_Bolometer_September_2022/bolometerprofiles/shot60055/bolometerprofile_from_radiation_powers_of_calibration_with_green_laser_vacuum_by hand_downwards_beam_new_batteries.txt'
boloprofile_8='/home/gediz/Results/Calibration/Calibration_Bolometer_September_2022/bolometerprofiles/shot60056/bolometerprofile_from_radiation_powers_of_calibration_with_green_laser_vacuum_by hand_downwards_beam_new_batteries_02.txt'

path,filename=os.path.split(boloprofile)

##Path of the derived correction constants to compare with each other:
# relativecorrection_0='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/relative_correction_constants_for_power_signals/relative_calibration_constants_from_shot70024_bolometerprofile_from_radiation_powers_using_mean.txt'
# relativecorrection_1='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/relative_correction_constants_for_power_signals/relative_calibration_constants_from_shot70025_bolometerprofile_from_radiation_powers_using_mean.txt'
# relativecorrection_2='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/relative_correction_constants_for_power_signals/relative_calibration_constants_from_shot70026_bolometerprofile_from_radiation_powers_using_mean.txt'
# relativecorrection_3='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/relative_correction_constants_for_power_signals/relative_calibration_constants_from_shot70027_bolometerprofile_from_radiation_powers_using_mean.txt'
# relativecorrection_4='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/relative_correction_constants_for_power_signals/relative_calibration_constants_from_shot70028_bolometerprofile_from_radiation_powers_using_mean.txt'
# relativecorrection_5='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/relative_correction_constants_for_power_signals/relative_calibration_constants_from_shot70029_bolometerprofile_from_radiation_powers_using_mean.txt'
# relativecorrection_6='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/relative_correction_constants_for_power_signals/relative_calibration_constants_from_shot70030_bolometerprofile_from_radiation_powers_using_mean.txt'
# relativecorrection_7='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/relative_correction_constants_for_power_signals/relative_calibration_constants_from_shot70031_bolometerprofile_from_radiation_powers_using_mean.txt'
# relativecorrection_8='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/relative_correction_constants_for_power_signals/relative_calibration_constants_from_shot70032_bolometerprofile_from_radiation_powers_using_mean.txt'
relativecorrection_0='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_air,  1k$\Omega$, sine, simulated wheatstone_using_mean.txt'
relativecorrection_1='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_Air, 1k$\Omega$_using_mean.txt'
relativecorrection_2='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_air, 1k$\Omega$, changed V2,V3_using_mean.txt'
relativecorrection_3='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_UV air 254_using_mean.txt'
relativecorrection_4='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_UV air 350_using_mean.txt'
relativecorrection_5='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_UV air klein_using_mean.txt'
relativecorrection_6='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_vac,  1k$\Omega$_using_mean.txt'
relativecorrection_7='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_vac, 100$\Omega$_using_mean.txt'
relativecorrection_8='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_vac, sine,1k$\Omega$_using_mean.txt'
relativecorrection_9='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_Res 1_using_mean.txt'
relativecorrection_10='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_Res 2_using_mean.txt'
relativecorrection_11='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/relative_calibration_constants_from_Res 3_using_mean.txt'

CompareTauAndKappa()
# %%
outfile='/home/gediz/Results/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistors_additional_to_AC/'
plt.rc('font',size=14)
b,a1,a2,v1,v2,v3,a3=np.genfromtxt('/home/gediz/Measurements/Calibration/Calibration_Bolometer_December_2022/calibration_with_resistor_additional_to_AC_in.txt',unpack=True,skip_header=5)
uv1=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot70025/shot70025_bolometerprofile_from_raw_data.txt',usecols=1)
uv2=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot70034/shot70034_bolometerprofile_from_raw_data.txt',usecols=1)
uv3=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot70035/shot70035_bolometerprofile_from_raw_data.txt',usecols=1)
res1=[x*(-1) for x in np.genfromtxt('/home/gediz/Results/Calibration/Channel_resistances_September_2022/all_resistor_values_bolometer_sensors_calculated.txt',usecols=3,delimiter=',')]
res2=[x*(-1) for x in np.genfromtxt('/home/gediz/Results/Calibration/Channel_resistances_September_2022/all_resistor_values_bolometer_sensors_calculated_second_set.txt',usecols=3,delimiter=',')]
res3=[x*(-1) for x in np.genfromtxt('/home/gediz/Results/Calibration/Channel_resistances_September_2022/all_resistor_values_bolometer_sensors_calculated_third_set.txt',usecols=3,delimiter=',')]
blind=[832,412,1920,632,1800,1700,1270,3040]
out=[16,8,80,8,40,60,40,160]
prozent=[16/832,8/412,80/1920,8/632,40/1800,60/1700,40/1270,160/3040]
tau = [0.1204,0.1195,0.1204,0.1214,0.0801,0.0792,0.0779,0.0822]
kappa =  [0.460,0.465,0.466,0.469,0.649,0.649,0.637,0.638]

fig,ax1=plt.subplots()
fig.set_figheight(7)
fig.set_figwidth(10)
ax2=ax1.twinx()
ax3=ax1.twinx()
ax4=ax1.twinx()
ax5=ax1.twinx()
ax6=ax1.twinx()
ax7=ax1.twinx()
ax8=ax1.twinx()
ax9=ax1.twinx()
ax10=ax1.twinx()
ax11=ax1.twinx()
ax12=ax1.twinx()
values=[a1,a2,a3,v1,v2,v3,uv1,uv2,uv3,res1,res2,res3,blind,out,prozent]
colors=['red','darkred','orange','blue','darkblue','lightblue','green','darkgreen','lightgreen','pink','violet','magenta']
labels=[r'Air, 1k$\Omega$',r'air, 1k$\Omega$, changed V2,V3',r'air,  1k$\Omega$, sine, simulated wheatstone',r'vac,  1k$\Omega$',r'vac, 100$\Omega$',r'vac, sine,1k$\Omega$','UV air klein','UV air 350','UV air 254','Res 1','Res 2', 'Res 3']
axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]
n=1
for p in [3,4,9,10,11]:
    for i,k,t,a in zip([values[p]],[colors[p]],[labels[p]],[axes[p]]):
        j=[x*2*kap*ta for x,kap,ta in zip(i,kappa,tau)]
        leg=a.plot(b,j,marker='o',color=k,label=t)
        #a.plot(b,i,marker='x',color=k)
        a.set_yticks([], [])
        labs =[l.get_label() for l in leg] 
        a.legend(leg, labs, loc=1,bbox_to_anchor=(1.7,n))
        n=n-0.1
        #RelativeOpticalCalibration('mean')
ax1.set(ylabel='Signal normalized')
ax1.set(xlabel='Bolometerchannel')
# leg = l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12

plt.show()
# %%
