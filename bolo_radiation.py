#%%

#Written by: Izel Gediz
#Date of Creation: 01.08.2022
#This code takes Bolometer data in Voltage Form and derives the Power in Watt
#It can Plot Timeseries in different layouts
#It derives the Signal height for each Bolo channel and creates a Bolometerprofile
#You can also compare different Bolometerprofiles


from click import style
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os
from scipy.signal import savgol_filter


import plasma_characteristics as pc

#%% Parameter
Poster=True


if Poster==True:
    plt.rc('font',size=20)
    plt.rc('xtick',labelsize=25)
    plt.rc('ytick',labelsize=25)
    plt.rcParams['lines.markersize']=12
else:
    plt.rc('font',size=14)
    plt.rc('figure', titlesize=15)
#colors=['#1bbbe9','#023047','#ffb703','#fb8500','#c1121f','#780000','#6969B3','#D81159','#04E762','#89FC00','#03CEA4','#04A777','#537A5A','#FF9B71','#420039','#D81159']
colors=['#03045E','#0077B6','#00B4D8','#370617','#9D0208','#DC2F02','#F48C06','#FFBA08','#3C096C','#7B2CBF','#C77DFF','#2D6A4F','#40916C','#52B788','#03045E','#0077B6','#00B4D8']
markers=['o','v','s','P','p','D','*','x','o','v','s','P','p','D','*','x','o','v','s']

#%% --------------------------------------------------------------------------------------------------------
# Important Functions 
# (change nothing here)
#Choose a function to create the plot/data you desire




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
    plt.suptitle ('All bolometer signals of shot n°{n} together. MW used: {m} \n {e}'.format(n=shotnumber, m=pc.GetMicrowavePower(shotnumber)[1], e=extratitle))
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
    y= savgol_filter(np.ravel(np.array(LoadData(location)[MW_n])[:,None]),10,3)
    time= np.array(LoadData(location)['Zeit [ms]'])[:,None]
    steps=[]
    for i in np.arange(0, len(y)-10):
        step= (y[i]-y[i+10])
        steps.append(abs(step))
    start=np.argwhere(np.array([steps])>0.05)[0][1]
    stop=np.argwhere(np.array([steps])>0.05)[-1][1]
    if Plot== True:
        print(start)
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
            y= PowerTimeSeries(i)[0]
            time= np.array(LoadData(location)['Zeit [ms]'])[:,None]
        elif Datatype=='Source':
            time=np.loadtxt(str(sourcefolder)+str(sourcefile), usecols=(0), unpack=True)
            y=PowerTimeSeries(i)[0]
        unit='\u03bcW'
        ylabel= 'Power [\u03bcW]'
        unit='\u03bcW'
    elif Type == 'Error':
        y= PowerTimeSeries(i)[1]
        time= np.array(LoadData(location)['Zeit [ms]'])[:,None]



        

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

    Type_types =['Bolo', 'Power','Error']
    if Type not in Type_types:
        raise ValueError("Invalid Type input. Insert one of these arguments {}".format(Type_types))
    if Type == 'Bolo':
        y= LoadData(location)["Bolo{}".format(i)]
        ylabel= 'Signal [V]'
        unit = 'V'
    if Type == 'Power':
        y= PowerTimeSeries(i)[0]
        ylabel= 'Power [\u03bcW]'
        unit='\u03bcW'
    if Type == 'Error':
        y= PowerTimeSeries(i)[1]
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
    sd=np.std([div1,div2],ddof=1)
    sem=sd/np.sqrt(2)
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
            
    return (div1 , div2, (div1+div2)/2,sd,sem)

#This function derives the Power time series from the raw Bolometer Voltage Time Series of your choosing
#It uses the formula (4.21) of Anne Zilchs Diploma Thesis 'Untersuchung von Strahlungsverlusten mittels Bolometrie an einem toroidalen Niedertemperaturplasma' from 2011
#--> i is the number of the Bolometerchannel
def PowerTimeSeries(i=1, Plot=False, save=False):
    def power(g,k,U_ac, t, U_Li):
        return (np.pi/g) * (2*k/U_ac) * (t* np.gradient(U_Li,time*1000 )+U_Li)
    def error(g,k,U_ac, t, U_Li,delta_t,delta_k):
        return ((np.pi/g) * (2/U_ac) * (t* np.gradient(U_Li,time*1000 )+U_Li))*delta_k+(np.pi/g) * (2*k/U_ac) * np.gradient(U_Li,time*1000 )*delta_t
    if vacuum==True:
        kappa =  [0.460,0.465,0.466,0.469,0.649,0.649,0.637,0.638]
        kappa_sem=[0.0015,0.0018,0.0025,0.0022,0.0026,0.0074,0.0067,0.0044]
        tau = [0.1204,0.1195,0.1204,0.1214,0.0801,0.0792,0.0779,0.0822]
        tau_sem=[0.00072,0.00058,0.00086,0.00075,0.00057,0.00073,0.00078,0.00094]
    if vacuum==False:
        tau,tau_sem,kappa,kappa_sem=np.genfromtxt('/home/gediz/Results/Calibration/Ohmic_Calibration/Ohmic_Calibration_Air_December/07_12_2022/ohmic_calibration_air_tau_and_kappa_mean_and_sem.txt',unpack=True,usecols=(1,2,3,4))
    corr=[0.703,0.930,1.104,0.728,1.325,1.042,1.438,1.313]
    g1= (10,30,50)
    g2= Bolometer_amplification_1
    g3= Bolometer_amplification_2
    #g2=(20,50,78.5)
    #g3=(1,2.04,4.88)
    g=g1[1]*g2*g3
    U_ac=8
    k= kappa[i-1]
    t = tau[i-1]
    c=corr[i-1]
    delta_t=tau_sem[i-1]
    delta_k=kappa_sem[i-1]
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
        #plt.plot(np.array(time)[:,None],np.array(error(g,k,U_ac, t, U_Li,delta_t,delta_k)*1000000)[:,None])
        plt.suptitle(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Power [\u03bcW]')
        fig1= plt.gcf()
        plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_channel_Bolo{c}_power_signal.pdf".format(n=shotnumber, c=i), bbox_inches='tight')
    return power(g,k,U_ac, t, U_Li)*1000000, error(g,k,U_ac, t, U_Li,delta_t,delta_k)*1000000

#This function derives the Signal height of all 8 Time Series using the SignalHeight function
#It then Plots the Height Values in a row to show the Bolometerprofile
#Tipp: This Routine doesn't show you the Fit Plots of SignalHeight that were used to derive the signal heights. 
#      So if you want to make sure the code used the good Fits to derive the signal heights change 'Plot' to 'True' at the --><--
#-->Type is either 'Bolo' or 'Power' depending on if you want to plot the signal heights of raw Voltage data or Power data
#Use Type= Cali if you have a combined file like created with CombinedTimeSeries stored outside of the normal shot folders
def BolometerProfile(Type="", save=False):
    print('This could take a second')
    x,y,error=[],[],[]
    #Activate the parts with z to compare your data with Annes Data of the same shot
    #z= np.loadtxt('/scratch.mv3/koehn/backup_Anne/zilch/results/7680/7680_BCh_jump.dat', usecols=(1,))*1000000
    Type_types =['Bolo', 'Power']
    if Type not in Type_types:
        raise ValueError("Invalid Type input. Insert one of these arguments {}".format(Type_types))
    for i in [1,2,3,4,5,6,7,8]:
        x.append(i)
        if MW == 'none':
            #y.append(abs(SignalHeight_max(Type,i,Plot=True)[0])) 
            y.append(abs(SignalHeight_rough(Type,i,Plot=False)[0]))
            error.append(abs(SignalHeight_rough('Error', i,Plot=False)[0]))
  
        else:
            y.append(abs(SignalHeight(Type, i,Plot=False)[2])) #--><--
            error.append(abs(SignalHeight('Error', i,Plot=False)[2])+abs(SignalHeight(Type, i,Plot=False)[4]))
           # print('error from tau, kappa:',abs(SignalHeight('Error', i,Plot=False)[2]))
            #print('error from Signal Height:',abs(SignalHeight(Type, i,Plot=False)[3]))
    if Type == 'Bolo':
        ylabel1= 'signal [V]'
        name='raw data'
        name_='raw_data'
    if Type == 'Power':
        ylabel1= 'power [\u03bcW]'
        name= 'radiation powers'
        name_='radiation_powers'
    if Datatype=='Data':
        title= 'Signals of the bolometer channels from {n} of shot n°{s} \n MW used: {m} \n {e}'.format(n=name, s= shotnumber, m=MW, e=extratitle)
    if Datatype=='Source':
        title='Signals of the Bolometerchannels from {n} \n of {e}'.format(n=name,e=extratitle)
    
    plt.figure(figsize=(10,5))
    plt.plot(x,y, marker='o', linestyle='dashed', label="Original Bolometerprofile")
    plt.errorbar(x,y,yerr=error,capsize=5)
    plt.ylabel(ylabel1)
    plt.xlabel('bolometer channel')
    plt.ylim(bottom=0)
    plt.suptitle(title, y=1.05)
    #plt.legend(loc=1, bbox_to_anchor=(1.3,1))
    fig1 = plt.gcf()
    plt.show()
    if save == True:
        data = np.column_stack([np.array(x), np.array(y),np.array(error)])#, np.array(z), np.array(abs(y-z))])
        if Datatype=='Data':
            datafile_path = str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_{t}.txt".format(n=shotnumber, t=name_)
            np.savetxt(datafile_path , data, delimiter='\t \t', fmt=['%d', '%10.3f', '%10.5f'], header='Signals of the Bolometerchannels from {n} of shot n°{s}. \n Label for plot \n shot n°{s}// {e}\n channeln° \t {u} \t error'.format(n=name, s= shotnumber,  u =ylabel1,e=extratitle))
            fig1.savefig(str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_{t}.pdf".format(n=shotnumber, t=name_), bbox_inches='tight')
        if Datatype=='Source':
            np.savetxt(str(sourcefolder)+'bolometerprofile_from_{t}_of_{n}.txt'.format(t=name_,n=sourcetitlesave) , data, delimiter='\t \t', fmt=['%d', '%10.3f', '%10.5f'], header='Signals of the Bolometerchannels from {n} of {s} \n  Label for plot \nshot n°{s}, {n},  {e}\nchanneln° // {l}'.format(n=name, s= sourcetitle,e=extratitle,l=ylabel1))
            fig1.savefig(str(sourcefolder)+'bolometerprofile_from_{t}_of_{n}.pdf'.format(t=name_,n=sourcetitlesave), bbox_inches='tight')

    return x, y,error

#This function can compare the Bolometerprofiles of 4 different shots
#There must already be a .txt file with the Signals for each channel as created with the function BolometerProfile()
def CompareBolometerProfiles(Type="", ScanType='',save=False,normalize=False): 
    x=[1,2,3,4,5,6,7,8]
    #z=[8,7,6,5,4,3,2,1]
    if Type=='Bolo':
        type='raw_data'
        name_='raw data'
        ylabel='Signal [V]'
    if Type=='Power':
        type='radiation_powers'
        name_='radiation powers'
        ylabel='P$_r$$_a$$_d$ [\u03bcW]'
    plt.figure(figsize=(10,7))
    pressure,mw=[],[]
    for i in shotnumbers1:
        pressure.append(pc.Pressure(i,gas))
        mw.append(pc.GetMicrowavePower(i)[0])
    if ScanType=='Pressure':
        sortnumbers=[shotnumbers1[i] for i in np.argsort(pressure)]
    if ScanType=='Power':
        sortnumbers=[shotnumbers1[i] for i in np.argsort(mw)]
    if ScanType=='None':
        sortnumbers=shotnumbers1
    for i,c,m in zip(sortnumbers,colors,markers):
        if ScanType=='Pressure':
            label='shot n°{s}, p= {p} mPa'.format(s=i,p=float(f'{pc.Pressure(i,gas):.1f}'))
            title= r'{g}, MW= {m}, P$_M$$_W$ $\approx$ {mw} kW'.format(g=gas,m=pc.GetMicrowavePower(i)[1],mw=float(f'{np.mean(mw)*10**(-3):.2f}'))
        if ScanType=='Power':
            label='shot n°{s}, P$_M$$_W$ = {mw} kW'.format(s=i,mw=float(f'{pc.GetMicrowavePower(i)[0]*10**(-3):.2f}'))
            title= r'{g}, MW= {m}, p $\approx$ {p} mPa'.format(g=gas,m=pc.GetMicrowavePower(i)[1],p=float(f'{np.mean(pressure):.1f}'))
        if ScanType=='None':
            label='shot n°{s}, P$_M$$_W$ = {mw} kW, p= {p} mPa'.format(s=i,mw=float(f'{pc.GetMicrowavePower(i)[0]*10**(-3):.2f}'),p=float(f'{pc.Pressure(i,gas):.1f}'))
            title= r'{g}, MW= {m}'.format(g=gas,m=pc.GetMicrowavePower(i)[1])
        shot1,error=np.loadtxt(str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_{t}.txt".format(n=i, t=type),unpack=True,usecols=(1,2))
        if normalize==True:
            norm='normalized values'
            mean=np.mean(shot1)
            shot1=list(i/mean for i in shot1)
        else:
            norm=''
        plt.plot(x,shot1,linewidth=3,marker=m, linestyle='dashed', label=label,color=c)#open(str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_{t}.txt".format(n=i, t=type), 'r').readlines()[2][3:-1])
        #plt.errorbar(x,shot1,yerr=error, capsize=5,linestyle='None',color=c)
    plt.xlabel('bolometer channel',fontsize=30)
    plt.ylabel(ylabel,fontsize=30)
    plt.legend(loc=1, bbox_to_anchor=(1.8,1),title=title)
    plt.ylim(0)
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"comparisons_of_shots/{g}/comparison_{a}_{t}_{g}_{u}.pdf".format(a=shotnumbers1, t=type,g=gas,u=norm), bbox_inches='tight')
   
def CompareBolometerProfiles_two_Series(save=False): 
    x=[1,2,3,4,5,6,7,8]
    name_='radiation powers'
    ylabel='P$_r$$_a$$_d$ [\u03bcW]'
    fig=plt.figure(figsize=(13,10))
    ax=fig.add_subplot(111)
    ax2=ax.twiny()
    for i,c,m in zip(shotnumbers1,colors,markers):
        shot1=np.loadtxt(str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_radiation_powers.txt".format(n=i, t=type),usecols=1)
        #title='shot n°{s}, MW: {mw}, P$_M$$_W$= {m} W, p={p} mPa'.format(s=i,mw=pc.GetMicrowavePower(i)[1],m=float(f'{pc.GetMicrowavePower(i)[0]:.1f}'),p=float(f'{pc.Pressure(i,gas):.1f}'))
        #title='shot n°{s}, P$_M$$_W$= {m} W'.format(s=i,m=float(f'{pc.GetMicrowavePower(i)[0]:.1f}'))
        title='shot n°{s}, p={p} mPa'.format(s=i,p=float(f'{pc.Pressure(i,gas):.1f}'))
        ax.plot(x,shot1, marker=m, linestyle='dashed', label=title,color=c)#open(str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_{t}.txt".format(n=i, t=type), 'r').readlines()[2][3:-1])
    for j,c,m in zip(shotnumbers2,colors[len(shotnumbers1):],markers[len(shotnumbers1):]):
        shot1=np.loadtxt(str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_radiation_powers.txt".format(n=j, t=type),usecols=1)
        #title='shot n°{s}, MW: {mw}, P$_M$$_W$= {m} W, p={p} mPa'.format(s=j,mw=GetMicrowavePower(j)[1],m=float(f'{pc.GetMicrowavePower(j)[0]:.1f}'),p=float(f'{pc.Pressure(j,gas):.1f}'))
        #title='shot n°{s}, P$_M$$_W$= {m} W'.format(s=j,m=float(f'{pc.GetMicrowavePower(j)[0]:.1f}'))
        title='shot n°{s}, p={p} mPa'.format(s=j,p=float(f'{pc.Pressure(j,gas):.1f}'))
        ax2.plot(x,shot1, marker=m, linestyle='dashed', label=title,color=c)#open(str(outfile)+"shot{n}/shot{n}_bolometerprofile_from_{t}.txt".format(n=i, t=type), 'r').readlines()[2][3:-1])
    ax.set_xlabel('bolometer channel',fontsize=35)
    ax.set_ylabel(ylabel,fontsize=35)
    ax.legend(loc='lower center',bbox_to_anchor=(0.2,-0.4),title='{g}, MW: {mw}, '.format(g=gas,mw=pc.GetMicrowavePower(i)[1])+r'P$_M$$_W$$\approx$ 2.8 kW')#+r'p$\approx$ 7.5 mPa')
    ax2.legend(loc='lower center',bbox_to_anchor=(0.75,-0.4),title='{g}, MW: {mw}, '.format(g=gas,mw=pc.GetMicrowavePower(j)[1])+r'P$_M$$_W$$\approx$ 2.0 kW')#+r'p$\approx$ 7.5 mPa')
    ax.set_ylim(0)
    ax2.set_xticks([])
    fig1= plt.gcf()
    plt.show()
    if save==True:
        fig1.savefig(str(outfile)+"comparisons_of_shots/{g}/comparison_of_two_series_{a}_{t}_{g}.pdf".format(a=shotnumbers1, t=shotnumbers2,g=gas), bbox_inches='tight')

#%% -------------------------------------------------------------------------------------------------------- 
#Enter the shotnumber you want to analyze, change the locations of data 
#Then enter one or several of the above functions according to what you want to analyze and run the script

if __name__ == "__main__":
    for shotnumber in np.arange(13212,13215):
        #shotnumber=13088
        shotnumbers1=np.arange(13098,13107)#(13221,13220,13223,13222,13224,13218,13225,13226,13217,13216,13219,13227,13215)
        shotnumbers2=(13098,13104,13106) 
        Datatype= 'Data' #'Data' if it is saved with TJ-K software like 'shotxxxxx.dat' or 'Source' if it is a selfmade file like 'combined_shots_etc'

        location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
        #location=  '/data6/Bolo_Calibration_December/shot{name}.dat'.format(name=shotnumber) #location of calibration measurement
        #time = np.array(LoadData(location)['Zeit [ms]'] / 1000)[:,None] # s
        vacuum=True
        gas='He'
        gases=('H','He','Ar','Ne')
        #MW='none'
        MW=pc.GetMicrowavePower(shotnumber)[1]
        Bolometer_amplification_1=100
        Bolometer_amplification_2=1
        Bolometer_timeresolution=100
        #extratitle=''
        extratitle='{g} // Bolometer: x{a}, x{b}, {c} ms // P$_M$$_W$= {mw} W // p= {p} mPa'.format(g=gas,a=Bolometer_amplification_2,b=Bolometer_amplification_1,c=Bolometer_timeresolution,mw=float(f'{pc.GetMicrowavePower(shotnumber)[0]:.3f}'),p=float(f'{pc.Pressure(shotnumber,gas):.3f}'))      #As a title for your plots specify what the measurement was about. If you don' use this type ''

        #if the datatype is source because you want to analyze data not saved direclty from TJ-K use:
        sourcefolder= '/home/gediz/Results/Bolometer_Profiles/combined_shots/shots_50025_to_50018/'   #the folder where the combined shots data should be stored
        sourcefile='All_channels_from_shots_50025_to_50018.txt'     #the name of the combined shots file
        sourcetitle='calibration with green laser in air'
        sourcetitlesave='calibration_with_green_laser_air'
        
        outfile='/home/gediz/Results/Bolometer_Profiles/'
        outfile_2='/home/gediz/LaTex/DPG/'

        if not os.path.exists(str(outfile)+'shot{}'.format(shotnumber)):
            os.makedirs(str(outfile)+'shot{}'.format(shotnumber))
        
        #SignalHeight('Power',1,Plot=True)
        #PlotAllTimeseriesTogether()
        #PowerTimeSeries(1,Plot=True)
        BolometerProfile('Power',save=True)
# %%
