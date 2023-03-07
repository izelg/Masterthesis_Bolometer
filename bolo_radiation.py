#%%

#Written by: Izel Gediz
#Date of Creation: 01.08.2022
#This code takes Bolometer data in Voltage Form and derives the Power in Watt
#It can Plot Timeseries in different layouts
#It derives the Signal height for each Bolo channel and creates a Bolometerprofile
#You can also compare different Bolometerprofiles


from pdb import line_prefix
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
import itertools



#%% --------------------------------------------------------------------------------------------------------
# Important Functions 
# (change nothing here)
#Choose a function to create the plot/data you desire

#Parametersettings for all figures plottet

#plt.rc('text',usetex=True)
#plt.rc('font',family='serif')
plt.rc('font',size=20)
#plt.rc('axes',titlesize=11)
#plt.rc('figure', figsize=(10,5))
plt.rc('figure', titlesize=15)

#plt.rc('savefig', pad_inches=0.5)


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
    time = np.array(LoadData(location)['Zeit [ms]'] / 1000)[:,None]
    z= LoadData(location)['2 GHz Richtk. forward']
    w= LoadData(location)['8 GHz power']
    height_z = abs(max(z)-min(z))
    height_w = abs(max(w)-min(w))
    if height_w >= 0.1:        #This is the part where the code finds out if 8 or 2GHz MW heating was used. Change the signal height if MW powers used change in the future
        MW = '8 GHz'
    elif height_z >= 0.1: 
        MW = '2.45 GHz'
    else:
        MW = 'none'
    if MW=='2.45 GHz':
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
        return (np.mean(signalinwatt_for[start:stop])-np.mean(signalinwatt_back[start:stop]),MW)
    if MW=='8 GHz':
        U_in=LoadData(location)['8 GHz power']*1E3
        a1  = 17.5637
        a2  = 0.332023
        a3  = 0.458919 
        P= a1 * np.exp(a2 * np.abs(U_in)**a3)
        return(np.mean(np.sort(P)[-100:-1]),MW)

#This Function plots a timeseries of your choosing
#-->use these channelnames: Zeit [ms]		8 GHz power		2 GHz Richtk. forward	I_Bh			U_B			Pressure		2 GHz Richtk. backward	slot1			I_v			Interferometer (Mueller)	Interferometer digital	8 GHz refl. power	Interferometer (Zander)	Bolo_sum		Bolo1			Bolo2			Bolo3			Bolo4			Bolo5			Bolo6			Bolo7			Bolo8			optDiode		r_vh			Coil Temperature
def PlotSingleTimeseries(i=1, save=False):
    if Datatype=='Data':
        y= np.array(LoadData(location)["Bolo{}".format(i)])[:,None]
        time = np.array(LoadData(location)['Zeit [ms]'] / 1000)[:,None]
        title='Shot n° {s} // Channel "Bolo {n}" \n {e}'.format(s=shotnumber, n=i, e=extratitle)
    elif Datatype=='Source':
        time,y=np.loadtxt(str(sourcefolder)+str(sourcefile), usecols=(0,i), unpack=True)
        title='Raw signal data of {s} // Channeln° {n}\n {e}'.format(s=sourcetitle, n=i, e=extratitle)
    print(len(y))
    plt.figure(figsize=(10,5))
    plt.plot(time, y)
    plt.suptitle(title)
    plt.xlabel('time [s]')
    plt.ylabel('signal [V]')
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_channel_{c}_raw_signal.pdf".format(n=shotnumber, c=i), bbox_inches='tight')

#This Function plots the Timeseries of All 8 Bolometer Channels in a Grid separately
def PlotAllTimeseries (figheight=None, figwidth=None, save=False):
    if figheight is None:
        print("You didn't choose a figureheight so I set it to 10")
        figheight = 10
    if figwidth is None:
        print("You didn't choose a figurewidth so I set it to 10")
        figwidth=10
    time = np.array(LoadData(location)['Zeit [ms]'] / 1000)[:,None]
    fig, axs = plt.subplots(4,2)
    fig.set_figheight(figheight)
    fig.set_figwidth(figwidth)
    fig.suptitle ('All Bolometer Signals of shot n°{n}. MW used: {m} \n {e}'.format(n=shotnumber, m=MW, e=extratitle))
    for i in [0,1,2,3]:
        bolo_raw_data = np.array(LoadData(location)["Bolo{}".format(i+1)])[:,None]
        axs[i,0].plot(time, bolo_raw_data)
        axs[i,0].set_title ('Bolometerchannel {}'.format(i+1))
        bolo_raw_data = np.array(LoadData(location)["Bolo{}".format(i+5)])[:,None]
        axs[i,1].plot(time, bolo_raw_data)
        axs[i,1].set_title ('Bolometerchannel {}'.format(i+5))
    for ax in axs.flat:
        ax.set(xlabel='Time [s]', ylabel='Signal [V]')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_all_bolo_channels_raw_signals.pdf".format(n=shotnumber), bbox_inches='tight')


#This Function plots the Timeseries of all 8 Bolometer Channels together in one Figure
def PlotAllTimeseriesTogether (figheight=None, figwidth=None, save=False):
    if figheight is None:
        print("You didn't choose a figureheight so I set it to 5")
        figheight = 5
    if figwidth is None:
        print("You didn't choose a figurewidth so I set it to 10")
        figwidth=10
    plt.figure(figsize=(figwidth, figheight))
    plt.suptitle ('All bolometer signals of shot n°{n} together. MW used: {m} \n {e}'.format(n=shotnumber, m=GetMicrowavePower()[1], e=extratitle))
    time = np.array(LoadData(location)['Zeit [ms]'] / 1000)[:,None]
    colors=['red','blue','green','gold','magenta','darkcyan','blueviolet','orange','darkblue']
    for i,c in zip(np.arange(1,9),colors):
        bolo_raw_data = np.array(LoadData(location)["Bolo{}".format(i)])[:,None]
        m=min(bolo_raw_data)
        bolo_raw_data=[(k-m)+i*0.05 for k in bolo_raw_data]
        plt.plot(time,  bolo_raw_data, label="Bolo{}".format(i),color=c )
        #plt.plot([time [u],time[u]],[bolo_raw_data[u],bolo_raw_data[u]],'o')
        #print(bolo_raw_data[u])
    plt.xlabel('time [s]')
    plt.ylabel('signal [V]')
    plt.legend(loc=1, bbox_to_anchor=(1.2,1) )
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_all_bolo_channels_raw_signals_together.pdf".format(n=shotnumber), bbox_inches='tight')

#You need this function if you did a measurement where you shone light on only one channel per measurement; so if you want to combine 8 measurements 
#It saves the time series of the 8 channels from the 8 measurements in one document
#It also plots all time series together even if they have different lengths
#So for shot1 enter the shotnumber where you collected data only on channel 1 etc. example: CombinedTimeSeries('50018', '50019',...)
def CombinedTimeSeries (shot1,shot2,shot3, shot4, shot5, shot6, shot7, shot8, Plot=False, save =False):
    path='/home/gediz/Measurements/Measurements_LOG/shot{name}.dat'
    a=[shot1,shot2,shot3,shot4,shot5,shot6,shot7,shot8]
    bolo=[[],[],[],[],[],[],[],[]]
    time=[[],[],[],[],[],[],[],[]]
    c=[1,2,3,4,5,6,7,8]
    for (i,j,k,l) in zip(a,bolo,time,c):
        location=path.format(name=i)
        j.extend(np.array(LoadData(location)['Bolo{}'.format(l)])[:,None])
        k.extend(np.array(LoadData(location)['Zeit [ms]'])[:,None])
    time_longest=max(time, key=len)
    time_longest=[x/1000 for x in time_longest]
    for (j,l) in zip(bolo,c):
        length_extend=len(time_longest)-len(j)
        j.extend([j[-1]]*length_extend)
        plt.plot(time_longest,j, label='Channel {}'.format(l))
    if Plot==True:
        plt.suptitle('{s} \n shots {a} to {b}'.format(s=sourcetitle,a=shot1, b=shot8))
        plt.legend(loc=1, bbox_to_anchor=(1.4,1))
        plt.xlabel('time [s]')
        plt.ylabel('Signal [V]')
        plt.show()
    if save==True:
        if not os.path.exists(str(outfile)+'combined_shots/shots_{a}_to_{b}'.format(a=shot1, b=shot8)):
            os.makedirs(str(outfile)+'combined_shots/shots_{a}_to_{b}'.format(a=shot1, b=shot8))
        fig1= plt.gcf()
        fig1.savefig(str(outfile)+'combined_shots/shots_{a}_to_{b}/All_channels_from_shots_{a}_to_{b}.pdf'.format(a=shot1, b=shot8), bbox_inches='tight')
        data = np.column_stack([np.array(time_longest), np.array(bolo[0]), np.array(bolo[1]), np.array(bolo[2]), np.array(bolo[3]), np.array(bolo[4]), np.array(bolo[5]), np.array(bolo[6]), np.array(bolo[7])])
        #data=list(itertools.zip_longest([time, bolo]))#, time[2], time[3],time[4], time[5], time[6], time[7], bolo[0], bolo[1], bolo[2], bolo[3],bolo[4], bolo[5], bolo[6], bolo[7]], fillvalue=''))
        np.savetxt(str(outfile)+'combined_shots/shots_{a}_to_{b}/All_channels_from_shots_{a}_to_{b}.txt'.format(a=shot1, b=shot8) , data, header='Signals of all Bolometerchannels combined from shots {a} to {b}. \n time1//time2//time3//time4//time5//time6//time7//time8 // Bolo1 //Bolo2 // Bolo3 // Bolo4 //Bolo5 / Bolo6 // Bolo7 // Bolo8'.format(a=shot1, b=shot8))
    return time, bolo

#This is a Function to derive the Time (indices) in which the Plasma was on
#It needs to be fed the MW Power data.
#Down in the running part of the script the code finds out if 8 or 2 GHZ MW power was used. search for errors there if this step fails
def SignalHighLowTime(Plot= False, save=False):
    if MW == '8 GHz':
        MW_n = '8 GHz power'
    if MW== '2.45 GHz':
        MW_n= '2 GHz Richtk. forward'
    y= np.array(LoadData(location)[MW_n])[:,None]
    time= np.array(LoadData(location)['Zeit [ms]'])[:,None]

    steps=[]
    for i in np.arange(0, len(y)-10):
        step= (y[i]-y[i+10])
        steps.append(abs(step))
    start=np.argwhere(np.array([steps])>0.1)[0][1]
    stop=np.argwhere(np.array([steps])>0.1)[-1][1]
    if Plot== True:
        plt.show()
        plt.plot(time,y)
        plt.plot(time[start], y[start], marker='o', color='red')
        plt.plot(time[stop], y[stop], marker='o', color='red')
        plt.suptitle('The MW Power Data of "{}" with markers on the Signal edges'.format(MW))
        plt.xlabel('Time [s]')
        plt.ylabel('Power [Arb.]')
        fig1= plt.gcf()
        plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_signal_edge_with_{m}.pdf".format(n=shotnumber, m=MW), bbox_inches='tight')
    return start, stop

#This function derives the signal heights by determining the background signal and substracting it from the maximum
#It is useful for calibration measurements where the signal was maximized in several steps
def SignalHeight_max(Type='',i=1,Plot=False, save=False):
    cut=0
    Type_types =['Bolo', 'Power']
    if Type not in Type_types:
        raise ValueError("Invalid Type input. Insert one of these arguments {}".format(Type_types))
    if Type == 'Bolo':
        if Datatype=='Data':
            y= np.array(LoadData(location)["Bolo{}".format(i)])[:,None]
            time= np.array(LoadData(location)['Zeit [ms]'])[:,None]
        elif Datatype=='Source':
            time,y=np.loadtxt(str(sourcefolder)+str(sourcefile), usecols=(0,i), unpack=True)
        ylabel= 'Signal [V]'
        unit = 'V'
    if Type == 'Power':
        if Datatype=='Data':
            y= PowerTimeSeries(i)
            time= np.array(LoadData(location)['Zeit [ms]'])[:,None]
        elif Datatype=='Source':
            time=np.loadtxt(str(sourcefolder)+str(sourcefile), usecols=(0), unpack=True)
            y=PowerTimeSeries(i)
        ylabel= 'Power [\u03bcW]'
        unit='\u03bcW'
    # steps=[]
    
    # for j in np.arange(cut, len(y)-100):
    #     step= (y[j]-y[j+100])
    #     steps.append(abs(step))
    # start=(np.argwhere(np.array([steps])>0.05)[0][1]+cut)
    # stop=(np.argwhere(np.array([steps])>0.05)[-1][1]+cut)
    # background_x =np.concatenate((time[0:start-100-cut],time[stop+100-cut:-1]))
    # background_y=np.concatenate((y[0:start-100-cut],y[stop+100-cut:-1]))
    # background=np.mean(background_y)

    # print('last values:',y[stop+100])
    # print('background:',background)
    # print('minimum:',min(y))
    # print('last values signal:',y[stop-1000])
    # maximum=abs(min(y))+background
    maximum=max(y)-min(y)
    if Plot==True:
        plt.plot(time,y)
        plt.plot(time[np.argmax(y)],max(y),'bo')
        #plt.plot(time[start-100],y[start-100],'bo')
        #plt.plot(time[stop+100],y[stop+100],'bo')
        #plt.plot(time[int(np.argwhere(y==min(y))[0]+cut)],min(y),'ro', label='Signal height: {} V'.format(float(f'{max:.3f}')))
        plt.legend(loc=1, bbox_to_anchor=(1.3,1) )
        plt.suptitle('Bolometerdata channel {} with markers for the signal height'.format(i))
        plt.xlabel('Time [s]')
        plt.ylabel('Signal [V]')
        fig1= plt.gcf()
        plt.show()
        if save==True:
            fig1.savefig(str(outfile)+"shot{n}/shot{n}_signal_height_max.pdf".format(n=shotnumber), bbox_inches='tight')
    return (maximum)#,background)

#This function derives the signal heights without fits to account for the drift
#It just takes the right edge of the signal and the mean value of 100 datapoints to the left and right to derive the Signalheight
#It is useful for noisy measurements where the fits don't work or for calibrationmeasurements with no reference MW data
def SignalHeight_rough(Type='', i=1, Plot=False, save=False):
    Type_types =['Bolo', 'Power']
    if Type not in Type_types:
        raise ValueError("Invalid Type input. Insert one of these arguments {}".format(Type_types))
    if Type == 'Bolo':
        if Datatype=='Data':
            y= np.array(LoadData(location)["Bolo{}".format(i)])[:,None]
            time= np.array(LoadData(location)['Zeit [ms]'])[:,None]
        elif Datatype=='Source':
            time,y=np.loadtxt(str(sourcefolder)+str(sourcefile), usecols=(0,i), unpack=True)
        ylabel= 'Signal [V]'
        unit = 'V'
    if Type == 'Power':
        if Datatype=='Data':
            y= PowerTimeSeries(i)
            time= np.array(LoadData(location)['Zeit [ms]'])[:,None]
        elif Datatype=='Source':
            time=np.loadtxt(str(sourcefolder)+str(sourcefile), usecols=(0), unpack=True)
            y=PowerTimeSeries(i)
        ylabel= 'Power [\u03bcW]'
        unit='\u03bcW'

        

    jump=((max(y)-min(y))/4)
    

    steps=[]
    for s in np.arange(0, len(y)-50):
        step= (y[s]-y[s+50])
        steps.append(abs(step))
    start=np.argwhere(np.array([steps])>jump)[0][1]
    stop=np.argwhere(np.array([steps])>jump)[-1][1]
    #print(jump, start, stop)
    signal_off=np.mean(list(y[stop+100:-1]))
    signal_on=np.mean(list(y[stop-400:stop-250]))
    div=signal_off-signal_on

    if Plot== True:
        plt.show()
        plt.plot(time,y, alpha=0.5)
        plt.plot(time[start], y[start],'bo')
        plt.plot(time[stop], y[stop], 'bo')
        plt.plot(time[stop+100], y[stop+100], marker='o', color='green')
        plt.plot(time[-1:], y[-1:], marker='o', color='green')
        plt.plot(time[stop-250], y[stop-250], marker='o', color='red')
        plt.plot(time[stop-400], y[stop-400], marker='o', color='red')
        plt.suptitle('Bolometerdata channel {} with markers for the signal height data'.format(i))
        plt.xlabel('Time [s]')
        plt.ylabel(ylabel)
        fig1= plt.gcf()
        plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_signal_height_rough.pdf".format(n=shotnumber), bbox_inches='tight')

    return (div, jump)

#This is a Function that takes one Bolometersignal (i) and gives you the Signal Height Difference
#It takes the SignalHighLowTime indices to split the Timeseries into two parts
#Two fits are made to the part of the signal with and without  Plasma respectively
#The difference of these Linear fits is determined
#-->Type is either 'Bolo' or 'Power' depending on if you want to plot raw Voltage data or Power data
#--> i is the number of the Bolometerchannel

def SignalHeight(Type="", i=1,  Plot=False, save=False):
    def lin (x,a,b):
        return a*x + b
    time = LoadData(location)['Zeit [ms]'] / 1000
    x=time

    Type_types =['Bolo', 'Power']
    if Type not in Type_types:
        raise ValueError("Invalid Type input. Insert one of these arguments {}".format(Type_types))
    if Type == 'Bolo':
        y= LoadData(location)["Bolo{}".format(i)]
        ylabel= 'Signal [V]'
        unit = 'V'
    if Type == 'Power':
        y= PowerTimeSeries(i)
        ylabel= 'Power [\u03bcW]'
        unit='\u03bcW'


    start= SignalHighLowTime(Plot=False)[0]
    stop= SignalHighLowTime(Plot= False)[1]
    x1 = x[start:stop]
    y1 = y[start:stop]
    x2 = np.concatenate((x[0:start],x[stop:-1]))
    y2 = np.concatenate((y[0:start],y[stop:-1]))

    popt1, pcov1 = curve_fit(lin,x1,y1)
    popt2, pcov2 = curve_fit(lin,x2,y2)
    div1 = lin(x[start], *popt2)-lin(x[start], *popt1)             #Takes the Signal height based on the axial intercept of the fit
    div2 = lin(x[stop], *popt2)-lin(x[stop], *popt1)        #Takes the Signal height a 100 points in front of the falling Signal edge
    div_avrg = float(f'{(div1+div2)/2:.4f}')        #Takes amean value for the Signal height based on the two linear fits

    if Plot==True:
        plt.plot(time, y, alpha=0.5, label='Bolometerdata of channel {c} shot n°{s}'.format(c=i, s= shotnumber))
        plt.plot(x, lin(x, *popt1), color='orange', label= 'Fit to the values with Plasma')
        plt.plot(x, lin(x, *popt2), color='green', label= 'Fit to the values without Plasma')
        plt.plot(x[start-100],lin(x[start-100], *popt1), marker='o', color='blue')
        plt.plot(x[start-100], lin(x[start-100], *popt2), marker='o', color='blue')
        plt.plot(x[stop-100],lin(x[stop-100], *popt1), marker='o', color='blue')
        plt.plot(x[stop-100], lin(x[stop-100], *popt2), marker='o', color='blue')
        plt.plot([x[start-100],x[start-100] ], [lin(x[start-100], *popt1), lin(x[start-100], *popt2)], color='blue', label='Height Difference of the \n Signal with and without Plasma')
        plt.plot([x[stop-100],x[stop-100] ], [lin(x[stop-100], *popt1), lin(x[stop-100], *popt2)], color='blue')
        plt.annotate(float(f'{div1:.4f}'), (x[start-100], lin(x[start-100], *popt1)+div1/2), color='blue')
        plt.annotate(float(f'{div2:.4f}'), (x[stop-100], lin(x[stop-100], *popt1)+div2/2), color='blue')
        plt.plot(x[start], y[start], marker='o', color='red')
        plt.plot(x[stop], y[stop], marker='o', color='red', label='signal edge, derived using the {} Data'.format(MW))
        plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.9))
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        plt.suptitle('Linear Fits to determine the Signal Height \n The average signal height is {v}{u}'.format(v=abs(div_avrg), u=unit))
        fig1= plt.gcf()
        plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_signal_height_channel_Bolo{c}.pdf".format(n=shotnumber, c=1), bbox_inches='tight')
            
    return (div1 , div2, (div1+div2)/2)

#This function derives the Power time series from the raw Bolometer Voltage Time Series of your choosing
#It uses the formula (4.21) of Anne Zilchs Diploma Thesis 'Untersuchung von Strahlungsverlusten mittels Bolometrie an einem toroidalen Niedertemperaturplasma' from 2011
#--> i is the number of the Bolometerchannel
def PowerTimeSeries(i=1, Plot=False, save=False):
    def power(g,k,U_ac, t, U_Li):
        return (np.pi/g) * (2*k/U_ac) * (t* np.gradient(U_Li,time*1000 )+U_Li)
    kappa =  [0.460,0.465,0.466,0.469,0.649,0.649,0.637,0.638]
    kappa_air=[1.226,1.286,1.291,1.381,1.408,1.38,1.369,1.437]
    tau = [0.1204,0.1195,0.1204,0.1214,0.0801,0.0792,0.0779,0.0822]
    tau_air=[0.045,0.045,0.044,0.045,0.036,0.037,0.037,0.036]
    corr=[0.703,0.930,1.104,0.728,1.325,1.042,1.438,1.313]
    g1= (10,30,50)
    g2= Bolometer_amplification_1
    g3= Bolometer_amplification_2
    #g2=(20,50,78.5)
    #g3=(1,2.04,4.88)
    g=2*g1[1]*g2*g3
    U_ac=8
    k= kappa[i-1]
    t = tau[i-1]
    c=corr[i-1]
    if Datatype=='Data':
        U_Li= LoadData(location)["Bolo{}".format(i)]
        time = LoadData(location)['Zeit [ms]'] / 1000
        title='Power data of Shot n° {s} // Channel "Bolo{n}" \n {e}'.format(s=shotnumber, n=i,e=extratitle)
    elif Datatype=='Source':
        time=np.genfromtxt(str(sourcefolder)+str(sourcefile), usecols=(0), unpack=True)
        U_Li=np.genfromtxt(str(sourcefolder)+str(sourcefile), usecols=(i), unpack=True)
        title='Power data of {s} // Channeln° {n} \n {e}'.format(s=sourcetitle, n=i,e=extratitle)

    if Plot==True:
        plt.figure(figsize=(10,5))
        plt.plot(np.array(time)[:,None],np.array(power(g,k,U_ac, t, U_Li)*1000000)[:,None])
        plt.suptitle(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Power [\u03bcW]')
        fig1= plt.gcf()
        plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_channel_Bolo{c}_power_signal.pdf".format(n=shotnumber, c=i), bbox_inches='tight')
    return power(g,k,U_ac, t, U_Li)*1000000

#This function derives the Signal height of all 8 Time Series using the SignalHeight function
#It then Plots the Height Values in a row to show the Bolometerprofile
#Tipp: This Routine doesn't show you the Fit Plots of SignalHeight that were used to derive the signal heights. 
#      So if you want to make sure the code used the good Fits to derive the signal heights change 'Plot' to 'True' at the --><--
#-->Type is either 'Bolo' or 'Power' depending on if you want to plot the signal heights of raw Voltage data or Power data
#Use Type= Cali if you have a combined file like created with CombinedTimeSeries stored outside of the normal shot folders
def BolometerProfile(Type="", save=False):
    print('This could take a second')
    x=[]
    y=[]
    
    #Activate the parts with z to compare your data with Annes Data of the same shot
    #z= np.loadtxt('/scratch.mv3/koehn/backup_Anne/zilch/results/7680/7680_BCh_jump.dat', usecols=(1,))*1000000
    Type_types =['Bolo', 'Power']
    if Type not in Type_types:
        raise ValueError("Invalid Type input. Insert one of these arguments {}".format(Type_types))
    for i in [1,2,3,4,5,6,7,8]:
        x.append(i)
        if MW == 'none':
            y.append(abs(SignalHeight_max(Type,i,Plot=True)[0])) 
            #y.append(abs(SignalHeight_rough(Type,i,Plot=True)[0]))  
        else:
            y.append(abs(SignalHeight(Type, i)[2])) #--><--
    if Type == 'Bolo':
        ylabel1= 'signal [V]'
        name='raw data'
        name_='raw_data'
        #corr=[0.621,0.836,0.965,0.635,1.669,1.307,1.748,1.647] #from calibration in vacuum with wrong AC signal
        corr=[0.947,1.234,0.779,1.06,1.156,0.852,1.271,0.925]   #from calibration with right signal in air
        sem_corr=[0.016,0.019,0.024,0.015,0.043,0.029,0.045,0.05]
    if Type == 'Power':
        ylabel1= 'power [\u03bcW]'
        name= 'radiation powers'
        name_='radiation_powers'
        #corr=[0.703,0.930,1.104,0.728,1.325,1.042,1.438,1.313] #from calibration in vacuum with wron AC signal
        corr=[0.989,0.924,0.875,0.932,1.335,1.034,0.988,1.042]  #from calibration with right signal in air
        #sem_corr=[0.021,0.033,0.024,0.023,0.033,0.023,0.044,0.033]
        sem_corr=[0.005,0.004,0.007,0.005,0.010,0.006,0.004,0.006]  #from calibration with right signal in air
    if Datatype=='Data':
        title= 'Signals of the bolometer channels from {n} of shot n°{s} \n MW used: {m} \n {e}'.format(n=name, s= shotnumber, m=MW, e=extratitle)
    if Datatype=='Source':
        title='Signals of the Bolometerchannels from {n} \n of {e}'.format(n=name,e=extratitle)
    z=[]
    zerr=[]
    for i,j,k in zip(corr,[0,1,2,3,4,5,6,7],sem_corr):
        z.append(y[j]*i)
        zerr.append(y[j]*k)
    plt.figure(figsize=(10,5))
    plt.plot(x,y, marker='o', linestyle='dashed', label="Original Bolometerprofile")
    # colors=['red','blue','green','gold','magenta','darkcyan','blueviolet','orange','darkblue']
    # for i,j,c in zip(x,y,colors):
    #     plt.plot(i,j,marker='o',color=c,markersize=10)
    #plt.errorbar(x,z,yerr=zerr, marker='o',capsize=5, linestyle='dashed',label="Corrected Bolometerprofile \n with relative correction")
    plt.ylabel(ylabel1)
    plt.xlabel('bolometer channel')
    plt.ylim(bottom=0)
    plt.suptitle(title, y=1.05)
    #plt.legend(loc=1, bbox_to_anchor=(1.3,1))
    fig1 = plt.gcf()
    plt.show()
    if save == True:
        data = np.column_stack([np.array(x), np.array(y)])#, np.array(z), np.array(abs(y-z))])
        if Datatype=='Data':
            datafile_path = str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_{t}.txt".format(n=shotnumber, t=name_)
            np.savetxt(datafile_path , data, delimiter='\t \t', fmt=['%d', '%10.3f'], header='Signals of the Bolometerchannels from {n} of shot n°{s}. \n Label for plot \n shot n°{s}// {e}\n channeln° \t {u}'.format(n=name, s= shotnumber, m=GetMicrowavePower()[1], u =ylabel1,e=extratitle))
            fig1.savefig(str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_{t}.pdf".format(n=shotnumber, t=name_), bbox_inches='tight')
        if Datatype=='Source':
            np.savetxt(str(sourcefolder)+'bolometerprofile_from_{t}_of_{n}.txt'.format(t=name_,n=sourcetitlesave) , data, delimiter='\t \t', fmt=['%d', '%10.3f'], header='Signals of the Bolometerchannels from {n} of {s} \n  Label for plot \nshot n°{s}, {n}, MW power: {m}, {e}\nchanneln° // {l}'.format(n=name, s= sourcetitle,m=GetMicrowavePower()[1],e=extratitle,l=ylabel1))
            fig1.savefig(str(sourcefolder)+'bolometerprofile_from_{t}_of_{n}.pdf'.format(t=name_,n=sourcetitlesave), bbox_inches='tight')

    return x, y#, z, y-z

#This function can compare the Bolometerprofiles of 4 different shots
#There must already be a .txt file with the Signals for each channel as created with the function BolometerProfile()
def CompareBolometerProfiles(Type="" ,save=False,normalize=False): 
    x=[1,2,3,4,5,6,7,8]
    #z=[8,7,6,5,4,3,2,1]
    if Type=='Bolo':
        type='raw_data'
        name_='raw data'
        ylabel='Signal [V]'
    if Type=='Power':
        type='radiation_powers'
        name_='radiation powers'
        ylabel='power [\u03bcW]'
    plt.figure(figsize=(13,10))
    #colors=['navy','blue','royalblue','cornflowerblue','indigo','rebeccapurple','darkorchid','mediumorchid','darkgreen','forestgreen','green','limegreen']
    colors=['#1bbbe9','#023047','#ffb703','#fb8500','#c1121f','#780000']
    markers=['o','v','s','P','p','D']
    for i,c,m in zip(shotnumbers,colors,markers):
        shot1=np.loadtxt(str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_{t}.txt".format(n=i, t=type),usecols=1)
        title='shot n°{s}, MW: {mw}, P$_M$$_W$= {m} W, p={p} mPa'.format(s=i,mw=GetMicrowavePower(i)[1],m=float(f'{GetMicrowavePower(i)[0]:.3f}'),p=float(f'{Pressure(i):.3f}'))
        #title='shot n°{s}, P$_M$$_W$= {m} W'.format(s=i,m=float(f'{GetMicrowavePower(i):.3f}'))
        #title='shot n°{s}, p={p} mPa'.format(s=i,p=float(f'{Pressure(i):.3f}'))
        if normalize==True:
            norm='normalized values'
            mean=np.mean(shot1)
            shot1=list(i/mean for i in shot1)
        else:
            norm=''
        #norm='corrected values'
        plt.plot(x,shot1, marker=m, markersize=18,linestyle='dashed', label=title,color=c)#open(str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_{t}.txt".format(n=i, t=type), 'r').readlines()[2][3:-1])
    plt.xlabel('bolometer channel',fontsize=25)
    plt.ylabel(ylabel,fontsize=25)
    #plt.suptitle('Comparison of bolometer-profiles from {n} of {g} {u}'.format(n=name_,g=gas,u=norm))#, c=shot_number_3, d=shot_number_4))
    plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.3),title=gas)#,title=r'He, p$\approx$ 13 mPa')
    plt.ylim(0)
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"comparisons_of_shots/{g}/comparison_{a}_{t}_{g}_{u}.pdf".format(a=shotnumbers, t=type,g=gas,u=norm), bbox_inches='tight')
   
    
#%% -------------------------------------------------------------------------------------------------------- 
#Enter the shotnumber you want to analyze, if you must, change the locations of data (but please don't erase the original ones)
#Then enter one or several of the above functions according to what you want to analyze and run the script

if __name__ == "__main__":
    #shotnumber = str(input('Enter a shotnumber here: '))
    shotnumber=13199
    shotnumbers=(13196,13094) 
    Datatype= 'Data' #'Data' if it is saved with TJ-K software like 'shotxxxxx.dat' or 'Source' if it is a selfmade file like 'combined_shots_etc'

    location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
    #location=  '/data6/Bolo_Calibration_December/shot{name}.dat'.format(name=shotnumber) #location of calibration measurement
    #time = np.array(LoadData(location)['Zeit [ms]'] / 1000)[:,None] # s

    gas='H'
    Bolometer_amplification_1=100
    Bolometer_amplification_2=1
    Bolometer_timeresolution=100
    extratitle='{g} // Bolometer: x{a}, x{b}, {c} ms // P$_M$$_W$= {mw} W // p= {p} mPa'.format(g=gas,a=Bolometer_amplification_2,b=Bolometer_amplification_1,c=Bolometer_timeresolution,mw=float(f'{GetMicrowavePower(shotnumber)[0]:.3f}'),p=float(f'{Pressure(shotnumber):.3f}'))      #As a title for your plots specify what the measurement was about. If you don' use this type ''
    #extratitle=''

    #if the datatype is source because you want to analyze data not saved direclty from TJ-K use:
    sourcefolder= '/home/gediz/Results/Bolometer_Profiles/combined_shots/shots_50025_to_50018/'   #the folder where the combined shots data should be stored
    sourcefile='All_channels_from_shots_50025_to_50018.txt'     #the name of the combined shots file
    sourcetitle='calibration with green laser in air'
    sourcetitlesave='calibration_with_green_laser_air'
    
    outfile='/home/gediz/Results/Bolometer_Profiles/'
    outfile_2='/home/gediz/LaTex/DPG/'
    


    if not os.path.exists(str(outfile)+'shot{}'.format(shotnumber)):
        os.makedirs(str(outfile)+'shot{}'.format(shotnumber))
    
    # for s in (13193,13194,13195,13196,13197,13198,13199,13200):
    #     shotnumber=s
    #     if not os.path.exists(str(outfile)+'shot{}'.format(shotnumber)):
    #         os.makedirs(str(outfile)+'shot{}'.format(shotnumber))
    #     location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
    #     extratitle='{g} // Bolometer: x{a}, x{b}, {c}ms // P$_m$$_w$={mw} W // p={p} mPa'.format(g=gas,a=Bolometer_amplification_2,b=Bolometer_amplification_1,c=Bolometer_timeresolution,mw=float(f'{GetMicrowavePower(shotnumber):.3f}'),p=float(f'{Pressure(shotnumber):.3f}'))      #As a title for your plots specify what the measurement was about. If you don' use this type ''
    #     PlotAllTimeseriesTogether(save=True)
    #     BolometerProfile('Bolo',save=True)
    #BolometerProfile('Power')
    CompareBolometerProfiles('Power',save=True)
    #CompareBolometerProfiles('Bolo')
    #CombinedTimeSeries('50025','50024','50023','50022','50021','50020','50019','50018',Plot=True,save=True)
# %%
