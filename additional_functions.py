
#Created by Izel Gediz
#Date:08.08.2022
#Here i just store functions that don't fit my routines anymore but which might be 
#usefull again later on.
#Feel free to browse but don't expet much, this is a gravejard.


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










#This is a Function that takes one Bolometersignal (i) and gives you the Signal Height Difference
#By fitting two lines to the Signalpart without and With Plasma and taking
#the Difference of their vertical position
#The Parts of the Signal with and Without Plasma determined with the Derivative
#of the Interferometersignal. 
#This function tries using the three different Interferometersignals to determine
#the Sigal High/Low time. The red dots in the plot should give an idea
#if the method succeeded or not. Only use the Signalheight if the red points
#lye on the Signal edges. Use SignalHighLowTime function to check the shape of the
#used interferometer signal
#-->Type is either 'Bolo' or 'Power' depending on if you want to plot raw Voltage data or Power data
#--> i is the number of the Bolometerchannel
#-->Choose Plot=Ture to see the Plot and the Fits being made
def SignalHeight(Type, i, Plot=False):
    def lin (x,a,b):
        return a*x + b
    tries=4
    channelname= 'Interferometer (Zander)'
    while tries>0:
        try:
            tries-=1
            if channelname == 'Interferometer (Zander)' and tries == 2:
                channelname = 'Interferometer (Mueller)'
            if channelname == 'Interferometer (Mueller)' and tries == 1:
                channelname = 'Interferometer digital'
            elif tries==0:
                break
            print ('Channel{x}: I try with {n}'.format(x=i,n=channelname))  
            x= time
            if Type == 'Bolo':
                y= LoadData(location)["Bolo{}".format(i)]
                ylabel= 'Signal [V]'
                unit = 'V'
            if Type == 'Power':
                y= PowerTimeSeries(i)
                ylabel= 'Power [\u03bcW]'
                unit='\u03bcW'
                
            y_interfero = LoadData(location)[channelname]
            yderiv= np.gradient(y_interfero)
            max_index = list(yderiv).index(max(yderiv))
            min_index = list(yderiv).index(min(yderiv))
            x1 = x[max_index:min_index]
            y1 = y[max_index:min_index]
            x2 = np.concatenate((x[0:max_index],x[min_index:-1]))
            y2 = np.concatenate((y[0:max_index],y[min_index:-1]))
            
            popt1, pcov1 = curve_fit(lin,x1,y1)
            popt2, pcov2 = curve_fit(lin,x2,y2)
            div1 = popt2[1]-popt1[1]
            div2 = lin(x[min_index-1000], *popt2)-lin(x[min_index-1000], *popt1)
            div_avrg = float(f'{(div1+div2)/2:.4f}')
            
        except ValueError:
            # channelname = 'Interferometer (Mueller)'
            # tries-=1
            # print ('a Value Error occured. Tried is at{}'.format(tries))
            if tries == 3:
                channelname = 'Interferometer (Mueller)'
                print ('A Value Error occured. There are {} tries left'.format(tries-1))
            if tries == 2:
                channelname=  'Interferometer digital'
                print ('A Value Error occuredThere are {} tries left'.format(tries-1))
            if tries == 1:
                print ('I tried everything')
        else: 
            if Plot==True:
                plt.plot(time, y, alpha=0.5, label='Bolometerdata of channel {c} shot n°{s}'.format(c=i, s= shotnumber))
                plt.plot(x, lin(x, *popt1), color='orange', label= 'Fit to the values with Plasma')
                plt.plot(x, lin(x, *popt2), color='green', label= 'Fit to the Values without Plasma')
                plt.plot(1,popt2[1], marker='o', color='blue')
                plt.plot(1, popt1[1], marker='o', color='blue')
                plt.plot(x[min_index-1000],lin(x[min_index-1000], *popt1), marker='o', color='blue')
                plt.plot(x[min_index-1000], lin(x[min_index-1000], *popt2), marker='o', color='blue')
                plt.plot([1,1],[popt1[1],popt2[1]], color='blue', label='Height Difference of the \n Signal with and without Plasma')
                plt.plot([x[min_index-1000],x[min_index-1000] ], [lin(x[min_index-1000], *popt1), lin(x[min_index-1000], *popt2)], color='blue')
                plt.annotate(float(f'{div1:.4f}'), (1, popt1[1]+div1/2), color='blue')
                plt.annotate(float(f'{div2:.4f}'), (x[min_index-1000], lin(x[min_index-1000], *popt1)+div2/2), color='blue')
                plt.plot(x[max_index], y[max_index], marker='o', color='red')
                plt.plot(x[min_index], y[min_index], marker='o', color='red', label='signal edge using {}'.format(channelname))
                plt.legend(loc=1, bbox_to_anchor=(1.8,1))
                plt.xlabel('Time (s)')
                plt.ylabel(ylabel)
                plt.title('Linear Fits to determine the Height Difference of the Signal \n The average signal height is {v}{u}'.format(v=div_avrg, u=unit))
                plt.show()
            
        finally:
            pass
    return (div1 , div2, (div1+div2)/2)


#%%------------------------------------------------------------------------------

#Belongs to goldfoil_absorption
#This was my attempt to sort the goldfoil absorption data and spectral data in a way to find
# and access both types of values.
#Instead now i try to fit to the gold Absorption curve and multiply the spectrum with that
#Double_Plot(Spectrum(), Gold_Abs())

l_spec= Spectrum()[0]
l_gold=Gold_Abs()[0]
spec=Spectrum()[1]
gold=Gold_Abs()[1]
spec_array=np.column_stack((l_spec,spec,np.array(np.full((len(spec)), ['s'],dtype=str))))
gold_array=np.column_stack((l_gold,gold, np.array(np.full((len(gold)), ['g'],dtype=str))))
z= np.concatenate((spec_array, gold_array), axis=0)
z2=z[z[:,0].argsort()]
gold_indices=np.where(z2=='g')[0]  #gives the indices of the gold absorption values
steps=[]
for i in np.arange(0, len(gold_indices)-1):
    step= (gold_indices[i]-gold_indices[i+1])
    steps.append(abs(step))                     #gives the stepsize between the indices of the gold absorption values. If the stepsize is 1 there are no spectral values inbetween
gaps_indices_lower=np.argwhere(np.array(steps)>1)   #gives the lower indices where gold_indices has  a gap greater than 1
gaps_indices_upper=np.argwhere(np.array(steps)>1)+1

#     #print(np.argwhere(np.array(steps)>1))
#     gaps_indices_upper=[]
#     for i in list(gaps_indices_lower):
#         gaps_indices_upper.append(gold_indices[gaps_indices_lower+1])           #collects the upper indices of the gaps
gaps_indices_both=np.concatenate((gold_indices[gaps_indices_lower],gold_indices[gaps_indices_upper]), axis=1) 
#     #  
# #now i want to combine these lists in a form ([lower1, upper1],[lower2,upper2]...) so that I can collect all spectral values between two absorption values and multiply them by the absorption value (either of upper lower or mean)
print(gold_indices, gaps_indices_both)


#%%------------------------------------------------------------

#Interpolate Gold Absorption on x Axis of your choosing

x=np.arange(0,10000, 0.1)
y=pchip_interpolate(Gold_Abs()[0],Gold_Abs()[1],x)
#data = np.column_stack([np.array(x), np.array(y)])#, np.array(z), np.array(abs(y-z))])
#np.savetxt('/home/gediz/Results/Goldfoil_Absorption/Golddata_interpolated_for_even_spaced_0_1000_nm.txt', data, delimiter='\t \t', header='Interpolation of the Golddata gold_Anne_abs.txt using the x-Axis of 0 to 1000nm \n wavelength [nm] \t Absorption')
#plt.plot(Gold_Abs()[0],Gold_Abs()[1], 'bo')
fig,ax=plt.subplots()
ax.semilogx(Gold_Abs()[0],Gold_Abs()[1], 'bo', label='Gold Absorption Data')
#ax.semilogx(x,y)
plt.plot(x,y, color='red', label='Gold Absorption interpolated function')
plt.xlabel('wavelength [nm]')
plt.ylabel('relative Absorption')
plt.suptitle('Gold Absorption')
plt.legend()
fig1= plt.gcf()
plt.show()
fig1.savefig("/home/gediz/Results/Goldfoil_Absorption/Golddata_and_Interpolation.pdf")


#%%
#used to test the smoothing with the savgol_filter
#originaly from lines_of_sight.py
def SmoothSignal(i=1):
    cut=0
    y= LoadData(location)["Bolo{}".format(i)][cut:]        #aprox. the first 10 seconds are ignored because due to the motormovement a second peak appeared there
    time = LoadData(location)['Zeit [ms]'][cut:] 
    title='Shot n° {s} // Channel "Bolo {n}" \n {e}'.format(s=shotnumber, n=i, e=extratitle)

    y_smooth=savgol_filter(y,1000,3)

    steps=[]
    for j in np.arange(cut, len(y_smooth)-1000):
        step= (y_smooth[j]-y_smooth[j+1000])
        steps.append(abs(step))
    #start=(np.argwhere(np.array([steps])>0.005)[0][1]+cut)
    #stop=(np.argwhere(np.array([steps])>0.005)[-1][1]+cut)
    #print(start,stop)
    plt.plot(np.arange(0,len(steps)),steps,'bo')
    plt.hlines(0.005,0,len(steps))
    #plt.plot([start-cut,start-cut],[steps[start-cut],steps[start-cut]],'ro')
    #plt.plot([stop-cut,stop-cut],[steps[stop-cut],steps[stop-cut]],'ro')
    plt.show()

    #print(start,stop)
    plt.plot(time, y,color='red', alpha=0.5)
    plt.plot(time,y_smooth)
    #plt.plot(time[start],y_smooth[start],'bo')
    #plt.plot(time[stop],y_smooth[stop],'ro')
    #plt.plot(time[stop],y[stop],'go')
    #print(time[stop], y_smooth[stop],y[stop])
    print(len(y),len(y_smooth),len(steps))
    plt.show()
    return (y_smooth)