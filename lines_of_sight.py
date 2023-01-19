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
from scipy.signal import savgol_filter
#from bolo_radiation import LoadData, PlotAllTimeseries, PlotAllTimeseriesTogether, PlotSingleTimeseries

#%%

#This function plots the Motordata collected with the stepping motor
#It can extract the width of the  signal which is needed to reconstruct the positional information in the data collected by TJ-K
def MotorData(save=False):
    x,a,p =np.genfromtxt(motordata, unpack=True)    #x is the position data in mm, a is the "amplitude" data stored and processed by the software, p is the "Phase" data stored and processed by the software
    x_=np.arange(x[0],x[-1],0.00001)
    fit=pchip_interpolate(x,a,x_)
    amp=list(i-min(a) for i in fit)
   # amp=savgol_filter(amp0,1000,3)
    plt.plot(x_,amp,'r.--', label='Interpolated signal "Amplitude"', alpha=0.5)
    #plt.plot(x_,amp,'r.--')
    signal_edge_list=[np.argwhere(amp>max(amp)/np.e)]#, np.argwhere(amp>max(amp)/10)]
    for signal_edge in signal_edge_list:
        fwhm1, fwhm2=x_[signal_edge[0]],x_[signal_edge[-1]]
        plt.plot(fwhm1,amp[int(signal_edge[0])],'bo')
        plt.plot(fwhm2,amp[int(signal_edge[-1])],'bo')
        fwhm=float(fwhm2-fwhm1)
        plt.plot([fwhm1,fwhm2],[amp[int(signal_edge[0])],amp[int(signal_edge[-1])]], color='blue', label='Width of channel: {} m'.format(float(f'{fwhm:.6f}')))
        
    #plt.plot(x,a,'b.--', label='"Amplitude Data"'.format(float(f'{np.mean(a):.4f}')))
    #plt.plot(x,p, label='"Phase Data": {} V'.format(float(f'{np.mean(p):.4f}')))
    plt.xlabel('Position [m]')
    plt.ylabel('Preprocessed Signal [V]')
    plt.suptitle(motordatatitle)
    plt.legend(loc=1, bbox_to_anchor=(1.4,1))
    fig1= plt.gcf()
    plt.show()
    print(fwhm1,fwhm2)
    if save==True:
        fig1.savefig(str(motordataoutfile)+str(filename[:-4])+".pdf", bbox_inches='tight')


#This function takes one channelsignal aquired during a sweep with a lightsource across it and determines
#The width of the signal at different heights after inverting it and substracting the backgroundsignal
def BoloDataWidths(i=1, save=False):
    cut=0
    y0= LoadData(location)["Bolo{}".format(i)][cut:]    #aprox. the first 10 seconds are ignored because due to the motormovement a second peak appeared there
    y=savgol_filter(y0,1000,3)
    time = LoadData(location)['Zeit [ms]'][cut:] / 1000
    title='Shot n° {s} // Channel "Bolo {n}" \n {e}'.format(s=shotnumber, n=i, e=extratitle)

    ##finding background mean value:
    steps=[]
    def lin (x,a,b):
        return a*x + b
    for j in np.arange(cut, len(y)-1000):
        step= (y[j]-y[j+1000])
        steps.append(abs(step))
    start=(np.argwhere(np.array([steps])>0.005)[0][1]+cut)
    stop=(np.argwhere(np.array([steps])>0.005)[-1][1]+cut)
    background_x = np.concatenate((time[0:start-cut],time[stop-cut:-1]))
    background_y=np.concatenate((y[0:start-cut],y[stop-cut:-1]))
    popt,pcov=curve_fit(lin,background_x,background_y)
    amp=list((y[j]-lin(time[j],*popt))*(-1) for j in np.arange(cut,len(y)))

    ##enable these plots to see how the signal was manipulated
    plt.plot(time, y0,color='red', alpha=0.5)
    plt.plot(time,y,color='red')
    plt.plot(time, lin (time,*popt), color='black')
    plt.plot(time[start],y[start],'ro')
    plt.plot(time[stop],y[stop],'ro')

    ##Signal width like in MotorData()
    signal_edge_list=[np.argwhere(amp>max(amp)/np.e), np.argwhere(amp>max(amp)/10)]
    for signal_edge in signal_edge_list:
        fwhm1, fwhm2=time[cut+int(signal_edge[0])],time[cut+int(signal_edge[-1])]
        print(fwhm1,fwhm2)
        plt.plot(fwhm1,amp[int(signal_edge[0])],'bo')
        plt.plot(fwhm2,amp[int(signal_edge[-1])],'bo')
        fwhm=float(fwhm2-fwhm1)
        plt.plot([fwhm1,fwhm2],[amp[int(signal_edge[0])],amp[int(signal_edge[-1])]], color='blue', label='Width of channel: {} s'.format(float(f'{fwhm:.4f}')))
        

    plt.legend(loc=1, bbox_to_anchor=(1.4,1))
    plt.plot(time,amp,color='blue',alpha=0.5)
    plt.suptitle(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Signal [V]')
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_channel_{c}_raw_signal_and_widths_in_s.pdf".format(n=shotnumber, c=i), bbox_inches='tight')

#This function does the same as BoloDataWidths() but for all channels at once
#It then saves the widths of all signals and their heigths together in one file
def BoloDataWholeSweep(save=False):
    plt.figure(figsize=(10,5))
    plt.suptitle ('All Bolometer Signals of shot n°{n} together. \n {e}'.format(n=shotnumber,  e=extratitle))
    cut=0
    time = LoadData(location)['Zeit [ms]'][cut:] / 1000
    width=[]
    height=[]
    position=[]
    fwhm1_list=[]
    fwhm2_list=[]
    c=[1,2,3,4,5,6,7,8]
    color=['blue','red','green','orange','magenta','gold','darkcyan','blueviolet']
    def lin (x,a,b):
        return a*x + b
    for (i,b) in zip(c,color):
        y0= LoadData(location)["Bolo{}".format(i)][cut:]    #aprox. the first 10 seconds are ignored because due to the motormovement a second peak appeared there
        y=savgol_filter(y0,1000,3)
        steps=[]
        for j in np.arange(cut, len(y)-1000):
            step= (y[j]-y[j+1000])
            steps.append(abs(step))
        start=(np.argwhere(np.array([steps])>0.005)[0][1]+cut)
        stop=(np.argwhere(np.array([steps])>0.005)[-1][1]+cut)
        background_x = np.concatenate((time[0:start-cut],time[stop-cut:-1]))
        background_y=np.concatenate((y[0:start-cut],y[stop-cut:-1]))
        popt,pcov=curve_fit(lin,background_x,background_y)
        amp_origin=list((y[j]-lin(time[j],*popt))*(-1) for j in np.arange(cut,len(y)))
        maximum=max(amp_origin)
        amp=list(amp_origin[j]/maximum for j in np.arange(0,len(y)))
        plt.plot(time,  amp, label="Bolo{}".format(i),color=b,alpha=0.7 )
        
        signal_edge=np.argwhere(amp>max(amp)/np.e)
        fwhm1, fwhm2=time[cut+int(signal_edge[0])],time[cut+int(signal_edge[-1])]
        plt.plot(fwhm1,amp[int(signal_edge[0])],'o',color=b)
        plt.plot(fwhm2,amp[int(signal_edge[-1])],'o',color=b)
        fwhm=float(fwhm2-fwhm1)
        plt.plot([fwhm1,fwhm2],[amp[int(signal_edge[0])],amp[int(signal_edge[-1])]], color=b, label='Width of channel: {} s'.format(float(f'{fwhm:.4f}')))
        plt.plot(time[int(np.argwhere(amp==max(amp))[0])+cut], max(amp),'o',color=b, label='Maximum: {} V'.format(float(f'{max(amp_origin):.4f}')))
        width.append(fwhm)
        position.append(time[int(np.argwhere(amp==max(amp))[0])+cut])
        height.append(max(amp_origin))
        fwhm1_list.append(fwhm1)
        fwhm2_list.append(fwhm2)
    print(fwhm1_list,fwhm2_list)
    plt.xlabel('Time [s]')
    plt.ylabel('Signal [V]/ Maximum')
    plt.legend(loc=1, bbox_to_anchor=(1.3,1) )
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_all_bolo_channels_raw_signals_together_analyzed.pdf".format(n=shotnumber), bbox_inches='tight')
        data = np.column_stack([np.array(c), np.array(position),np.array(height),np.array(width),np.array(fwhm1_list),np.array(fwhm2_list)])
        np.savetxt(outfile+'shot{n}/shot{n}_all_bolo_channels_raw_signals_together_analyzed.txt'.format(n=shotnumber), data, delimiter='\t \t', fmt=['%d', '%10.4f', '%10.4f', '%10.4f', '%10.4f', '%10.4f'], header='Analysis of the Bolometersignals from shot°{s} \n Title for Boloprofileplot: \n shot n°{s}, {m}\n channelnumber \t Position Peak Max [s] \t Height [V] \t Width [s] \t Position left fwhm [s] \t Position right fwhm [s]'.format(s=shotnumber,m=motordatatitle))


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

#This was a first attempt to plot all lines of sight measurements together
#The next level would be to reconstruct their positions and plot them in 3D
def VisualizeLinesOfSight():

    def lin(x,a,b):
        return a*x+b
    x=[(0,13.7,17.7)]
    x_val=[(14,71.7,83.99)]
    x_err=[(0,3.33,4.11)]
    y=[(0,12.4,19.5,22.9)]
    y_val=[(5,41.71,44.1,54.17)]
    y_err=[(0,1.63,1.84,2.97)]
    poptx,pcovx=curve_fit(lin,x[0],list(h/2 for h in x_val[0]))
    popty,pcovy=curve_fit(lin,y[0],list(h/2 for h in y_val[0]))
    range=np.arange(0,25,1)
    for j,i,n in zip(x,x_val,x_err):
        plt.errorbar(j,list(h/2 for h in i),yerr=n,xerr=0.4,marker='o', linestyle='None',capsize=5,color='red')
        plt.errorbar(j,list(-h/2 for h in i),yerr=n,xerr=0.4,marker='o',linestyle='None', capsize=5,color='red')
        plt.plot(range,lin(range,*poptx),color='red')
        #plt.plot(range,lin(range,2.010,7),color='green')
        plt.plot(range,lin(range,-poptx[0],-poptx[1]),color='red')
    for j,i,n in zip(y,y_val,y_err):
        plt.errorbar(j,list(h/2 for h in i),yerr=n,xerr=0.4,marker='o', linestyle='None',capsize=5,color='blue')
        plt.errorbar(j,list(-h/2 for h in i),yerr=n,xerr=0.4,marker='o', linestyle='None',capsize=5,color='blue')
        plt.plot(range,lin(range,*popty),color='blue')
        #plt.plot(range,lin(range,1.15,2.5),color='green')
        plt.plot(range,lin(range,-popty[0],-popty[1]),color='blue')
    #plt.plot([0,22.9],[2.5,27.05],color='blue')
    #plt.plot([0,22.9],[-2.5,-27.05],color='blue')
    plt.xlabel('Distance from slit [cm]')
    plt.ylabel('line of sight widht [mm]')
    plt.legend()
    plt.grid(True)
    plt.suptitle('Widhts of the lines of sight from all 8 channels, vertical and horizontal')
    plt.show()
    #print(lin(0,*popty),lin(12.4,*popty),lin(19.5,*popty),lin(22.9,*popty))
    print(poptx)
    
def TwoDimensional_LinesofSight():
    def lin(x,a,b):
        return a*x+b
    range=np.arange(0,25,0.1)

    plt.grid(True)
    #plt.plot(range,lin(range,2.01,7),color='red')
    #plt.plot(range,lin(range,-2.01,-7),color='red')
    #plt.plot(range,lin(range,1.15,2.5),color='blue')
    #plt.plot(range,lin(range,-1.15,-2.5),color='blue')
    plt.plot([0,22.9],[-2.5,82.45],color='gold')
    plt.plot([0,22.9],[2.5,136.65],color='gold')
    
    plt.plot([0,22.9],[-2.5,51.15],color='darkcyan')
    plt.plot([0,22.9],[2.5,105.35],color='darkcyan')
    
    plt.plot([0,22.9],[-2.5,19.85],color='blueviolet')
    plt.plot([0,22.9],[2.5,74.05],color='blueviolet')
    
    plt.plot([0,22.9],[-2.5,-11.45],color='blue')
    plt.plot([0,22.9],[2.5,42.75],color='blue')
    
    plt.plot([0,22.9],[2.5,11.45],color='red')
    plt.plot([0,22.9],[-2.5,-42.75],color='red')
    
    plt.plot([0,22.9],[2.5,-19.85],color='green')
    plt.plot([0,22.9],[-2.5,-74.05],color='green')
    
    plt.plot([0,22.9],[2.5,-51.15],color='orange')
    plt.plot([0,22.9],[-2.5,-105.35],color='orange')
    
    plt.plot([0,22.9],[2.5,-82.45],color='magenta')
    plt.plot([0,22.9],[-2.5,-136.65],color='magenta')
    
    plt.plot([12.4,12.4],[72.3,72.3],'ro')
    plt.plot([19.5,19.5],[101,101],'ro')
    plt.plot([22.9,22.9],[136.7,136.7],'ro')

    plt.show()
    #print(lin(12.4,1.15,2.5)-lin(12.4,-1.15,2.5))

def DeriveLinesofSight():
    alpha=14    #Angle of Bolometerhead
    b=2         #Height of bolometerhead
    a=4         #Distance to slit
    c=0.5       #height of slit
    d=10.2      #distance to Torsatron
    x=20        #distance to plasma
    h=[-1.72,-1.59,-1.29,-1.16,-0.86,-0.73,-0.43,-0.3,0.3,0.43,0.73,0.86,1.16,1.29,1.59,1.72]
    def lin(x,a,b):
        return a*x+b
    range=np.arange(0,30,0.1)
    x_b=[]
    y_b=[]
    for i in h:
        x_b.append(abs(np.cos((90-alpha)*np.pi/180)*i))
        y_b.append(np.sin((90-alpha)*np.pi/180)*i)
    print(x_b,y_b)
    plt.figure(figsize=(10,10))
    #plt.xlim(0,5)
    #plt.ylim(-2,2)
    plt.vlines([(0,4,14.2,16.4,23.5,26.9)],-10,10,linestyle='dotted',alpha=0.5)
    y_exp=[13.665,8.245,10.535,5.115,7.405,1.985,4.275,-1.145,1.145,-4.275,-1.985,-7.405,-5.115,-10.535,-8.245,-13.665]
    for i,j in zip([0,2,4,6,8,10,12,14],['red','blue','green','gold','magenta','darkcyan','blueviolet','orange']):
        popt1,pcov1=curve_fit(lin,[x_b[i],4],[y_b[i],-0.25])
        popt2,pcov2=curve_fit(lin,[x_b[i+1],4],[y_b[i+1],0.25])
        plt.plot([x_b[i],x_b[i+1]],[y_b[i],y_b[i+1]],color='red')
        plt.plot(range,lin(range,*popt1),color=j,alpha=0.5,linestyle='dashed')
        plt.plot(range,lin(range,*popt2),color=j,alpha=0.5,linestyle='dashed')
        popt3,pcov3=curve_fit(lin,[4,26.9],[0.25,y_exp[i]])
        popt4,pcov4=curve_fit(lin,[4,26.9],[-0.25,y_exp[i+1]])
        plt.plot(range,lin(range,*popt3),color=j)
        plt.plot(range,lin(range,*popt4),color=j)
    plt.xlabel('Distance to Bolometerhead [cm]')
    plt.ylabel('Distance from middle of slit [cm]')
    plt.show()   
    
    
#For different scans that have the same conditions the standard derivation of the aquired values can be determiend
#e.g. for different y-sweeps at the same distance one can find out how accurate the position heith and width of the Signals can be extracted
def ErrorAnalysis(shot1,shot2,shot3):
    x,p1,h1,w1=np.genfromtxt('/home/gediz/Results/Lines_of_sight/shot_data/shot{n}/shot{n}_all_bolo_channels_raw_signals_together_analyzed.txt'.format(n=shot1), unpack=True,usecols=(0,1,2,3))
    x,p2,h2,w2=np.genfromtxt('/home/gediz/Results/Lines_of_sight/shot_data/shot{n}/shot{n}_all_bolo_channels_raw_signals_together_analyzed.txt'.format(n=shot2), unpack=True,usecols=(0,1,2,3))
    x,p3,h3,w3=np.genfromtxt('/home/gediz/Results/Lines_of_sight/shot_data/shot{n}/shot{n}_all_bolo_channels_raw_signals_together_analyzed.txt'.format(n=shot3), unpack=True,usecols=(0,1,2,3))
    d1=[]
    d2=[]
    d3=[]
    for i in [0,1,2,3,4,5,6]:       #extract peak distances from peak positions
        d1.append(p1[i+1]-p1[i])
        d2.append(p2[i+1]-p2[i])
        d3.append(p3[i+1]-p3[i])
    d=[d1,d2,d3]
    h=[h1,h2,h3]
    w=[w1,w2,w3]
    sd_w=[]
    for i in [0,1,2,3,4,5,6]:
        sd_w.append(np.std([w[0][i],w[1][i],w[2][i]],ddof=1))
    #print(np.mean([0.38,1.17,3.9,.83,1.42,0.4,1.26,9.27]))


#%%
motordata='/home/gediz/Measurements/Lines_of_sight/motor_data/shot60032_bolo3_y.dat'
motordatatitle='Motordata of shot60078 // Lines of Sight measurement //channel 1'
motordataoutfile='/home/gediz/Results/Lines_of_sight/motor_data/'
path,filename=os.path.split(motordata)

#for the bolo_ratdiation functions:
Datatype='Data'
shotnumber=60071
#location='/home/gediz/Measurements/Calibration/Calibration_Bolometer_September_2022/Bolometer_calibration_vacuum_and_air_different_sources_09_2022/shot{name}.dat'.format(name=shotnumber) #location of calibration measurement
location='/home/gediz/Measurements/Lines_of_sight/shot_data/shot{}_cropped.dat'.format(shotnumber)
outfile='/home/gediz/Results/Lines_of_sight/shot_data/'
extratitle='Lines of sight // air // UV-Lamp y-scan//distance 2.2cmcm// amplif. x5, x100'
if not os.path.exists(str(outfile)+'shot{}'.format(shotnumber)):
    os.makedirs(str(outfile)+'shot{}'.format(shotnumber))

#DeriveLinesofSight()
#VisualizeLinesOfSight()
BoloDataWholeSweep()
#TwoDimensional_LinesofSight()
#val=[3.74,3.75,3.72,3.77,3.74,3.93,3.72,3.68,3.71,3.71,3.71,3.76,3.73,4.48,3.83,3.68,3.7,3.72,3.77,3.79,3.72,4.15,3.7,3.7,4.02,3.71,3.73,3.75,3.7,3.8,3.73]
#vol=list(x**(-1) for x in val)
# vol=[117.6,114.7,123.9,110.9,107.9,107.6,108.4,128.2,116.4,118.8,117.9,116.8,103.8,108.9,110.7,96.4,116.5,116.6,110.4,116.5,108.2,107.8,112.7,108.8]
# vol=[4.7,1.7,2.05,1.33,2.2,0.98,4.6,4.5,1.7,1.3,1.6,3.7,3.72,0.85]
# print(np.mean(vol))
# print(np.std(vol,ddof=1))#/np.sqrt(len(val)))
# print(np.std(vol,ddof=1)/np.sqrt(len(vol)))
# print(((np.std(vol,ddof=1)/np.sqrt(len(vol)))/np.mean(vol))*100)
# ov1,ov2,ov3,ov4,ov5,ov6,ov7=np.genfromtxt('/home/gediz/Results/Lines_of_sight/overlap_y_scans.txt', usecols=(1,2,3,4,5,6,7),unpack=True,delimiter=',',skip_header=11)
# for j,i in zip([1,2,3,4,5,6,7],[ov1,ov2,ov3,ov4,ov5,ov6,ov7]):
#     plt.plot(j,i[0],'ro--')
#     plt.plot(j,i[1],'bo--')
#     plt.plot(j,i[2],'go--')
# plt.show()


# %%
