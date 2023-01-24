#%%

#Written by: Izel Gediz
#Date of Creation: 14.11.2022
#This code takes the simulated data of the cross section of the fluxsurfaces
#It also takes the modeled and exmerimentaly determined lines of sight
#Then it derives the area covered by e.g. channel 5 line of sight and fluxsurface 0 to 5 with a given resolution

from pdb import line_prefix
from unicodedata import name
from blinker import Signal
from click import style
import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import statistics
import os
import itertools
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import csv
from scipy.interpolate import pchip_interpolate
import math
from datetime import datetime

#%%
plt.rc('font',size=14)
plt.rc('figure', titlesize=15)

#The important distances are defined
a=60+32.11+3.45 #Position of Bolometerheadmiddle [cm]
b=3.45 #Distance of Bolometerhead Middle to  Slit [cm]
s_w=1.4 #Width of the slit [cm]
s_h=0.5 #Height of the slit [cm]
alpha=14 #Angle of the Bolometerhead to plane [°]
c_w=0.38 #Channelwidth of Goldsensor [cm]
c_h=0.13 #HChannelheight of Goldsensor [cm]
c_d=0.225 #depth of Goldsensor [cm]
h=2 #height of Bolometerhead [cm]
z_0=63.9    #middle of flux surfaces
r_ves=17.5 #radius of vessel [cm]
#err=0.4     #error of lines of sight in x and y direction [cm]


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
   
# Pixelmethod
def Pixelmethod():
    start=datetime.now()
    #-----------------------------------------------------#-
    #!!! Enter here which channels lines of sight you want to have analyzed(1 to 8), what pixel-resolution you need (in cm) and  which flux surfaces (0 to 7) should be modeled. 
    #note that more channels, a smaller resolution and more fluxsurfaces result in longer computation times
    bolo_channel=[5] #1 to 8
    bolo_power=[0,0,0,3.4E-6,3.4E-6,0,0,0]
    res=0.5
    fluxsurfaces=[0,1,2,3,4,5,6,7] #0 to 7
    err='None'
    #-----------------------------------------------------#-
    if err=='None':
        x_err=0
        y_err=0
    if err=='Min':
        x_err=-0.2
        y_err=-0.2
    if err=='Max':
        x_err=0.2
        y_err=0.2
        
        

    for bol in bolo_channel:
        fig=plt.figure(figsize=(10,10))
        ax=fig.add_subplot(111)
        x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
        y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')


        plt.xlabel('R [cm]')
        plt.ylabel('r [cm]')

        #Derive the exact positions of the bolometerchannels
        #I derive the x and y positions of the four upper channels lower and upper edge
        #I consider the 8 Bolometerchannels distributed over the 4cm with equal distance resulting in a distance of 0.33cm
        f1=0.123 #Distance first channel to edge [cm]
        f2=0.35 #Distance between channels [cm]
        h=[-2+f1,-2+f1+c_h,-2+f1+c_h+f2,-2+f1+c_h*2+f2,-2+f1+c_h*2+f2*2,-2+f1+c_h*3+f2*2,-2+f1+c_h*3+f2*3,-2+f1+c_h*4+f2*3,f1,f1+c_h,f1+c_h+f2,f1+c_h*2+f2,f1+c_h*2+f2*2,f1+c_h*3+f2*2,f1+c_h*3+f2*3,f1+c_h*4+f2*3,f1*2+c_h*4+f2*3]
        x_b=[]
        y_b=[]
        for i in h:
            x_b.append(-abs(np.sin((alpha)*np.pi/180)*i)+a+c_d)
            y_b.append(-np.cos((alpha)*np.pi/180)*i)

        def lin(x,d,e):
            return d*x+e
        ex_1=[-7.23, -3.0600000000000005, -5.760000000000001, -1.5900000000000007, -4.290000000000001, -0.120000000000001, -2.820000000000002, 1.3499999999999979, -1.3500000000000014, 2.8200000000000003, 0.120000000000001, 4.289999999999999, 1.5899999999999963, 5.7599999999999945, 3.059999999999995, 7.229999999999997]
        ex_2=[-10.115, -5.705, -7.855, -3.445, -5.595, -1.1849999999999996, -3.334999999999999, 1.075000000000001, -1.0749999999999993, 3.335000000000001, 1.1850000000000005, 5.594999999999999, 3.4450000000000003, 7.855000000000004, 5.705000000000004, 10.115]
        ex_3=[-13.65, -8.23, -10.52, -5.1000000000000005, -7.390000000000001, -1.9700000000000024, -4.2600000000000025, 1.1599999999999993, -1.1300000000000008, 4.290000000000001, 2.0000000000000018, 7.419999999999998, 5.129999999999997, 10.549999999999999, 8.259999999999998, 13.68]

        #this function fits a linear fit to the space between two points of a flux surface thereby estimating the function describing a surface
        #It then draws a line from the center of the fluxsurface to the point you want to analyze
        #It finds the intersection of that line with the fluxsurface by calculating the difference of this line and each of the 64 lines connecting the surfacepoints and determines the minimal value
        #It is used further down and its use is explained there
        def intersections(g):
            diff_=[]
            for i in g:
                popt,pcov=curve_fit(lin,[x[i],x[i+1]],[y[i],y[i+1]])
                range=np.arange(np.sort([x[i],x[i+1]])[0],np.sort([x[i],x[i+1]])[1],0.0001)
                poptm,pcovm=curve_fit(lin,[z_0,m],[0,n])
                diff=abs(lin(range,*popt)-lin(range,*poptm))
                diff_.append(min(diff))
            return (g[np.argmin(diff_)],np.argmin(diff),range,popt,poptm)
            
        x=np.arange(54,94,1)
        m_=list(p+0.01 for p in (np.arange(int(min(x_.iloc[8])),int(max(x_.iloc[8]))+1,res)))
        n_=list(o+0.01 for o in (np.arange(int(min(y_.iloc[8])),int(max(y_.iloc[8]))+1,res)))
        inside_line=[[],[],[],[],[],[],[],[]]
        lines=[0,2,4,6,8,10,12,14]
        colors=['red','blue','green','gold','magenta','darkcyan','blueviolet','orange','darkblue']
        #here the desired line of sight is plotted from the experimental data. To see the calculate lines of sight activate the dashed lines plot
        #now the points of interest(the ones in a square of the rough size of the outer most fluxsurface) are tested for their position relative to the line of sight
        #If the point lies inside the two lines describing the line of sight of that channel, its coordinates are added to "inside_line"
        plt.plot([x_b[lines[bol-1]],x_b[lines[bol-1]+1]],[y_b[lines[bol-1]],y_b[lines[bol-1]+1]],color='red')
        #popt3,pcov3=curve_fit(lin,[a-b+x_err,a-b-12.4+x_err,a-b-19.5+x_err,a-b-22.9+x_err],[-s_h/2,ex_1[lines[bol-1]]-y_err,ex_2[lines[bol-1]]-y_err,ex_3[lines[bol-1]]-y_err])
        #popt4,pcov4=curve_fit(lin,[a-b+x_err,a-b-12.4+x_err,a-b-19.5+x_err,a-b-22.9+x_err],[s_h/2,ex_1[lines[bol-1]+1]+y_err,ex_2[lines[bol-1]+1]+y_err,ex_3[lines[bol-1]+1]+y_err])
        popt3,pcov3=curve_fit(lin,[a-b+x_err,a-b-22.9+x_err],[-s_h/2,ex_3[lines[bol-1]]-y_err])
        popt4,pcov4=curve_fit(lin,[a-b+x_err,a-b-22.9+x_err],[s_h/2,ex_3[lines[bol-1]+1]+y_err])
        plt.plot(x,lin(x,*popt3),color=colors[bol-1])
        plt.plot(x,lin(x,*popt4),color=colors[bol-1])
        plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[lines[bol-1]],ex_2[lines[bol-1]],ex_3[lines[bol-1]]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=colors[bol-1])
        plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[lines[bol-1]+1],ex_2[lines[bol-1]+1],ex_3[lines[bol-1]+1]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=colors[bol-1])

        for m in m_:
            for n in n_:
                if n< lin(m,*popt4) and n>(lin(m,*popt3)):
                    #ax.add_patch(mpl.patches.Rectangle((m-res/2,n-res/2),res,res,color=colors[bol-1],alpha=0.4,linewidth=0))
                    inside_line[bol-1].append((m,n))
        plt.plot([60+a-b,60+a-b],[s_h/2,-s_h/2],color='blue')

        #Now starting from the points in "inside-line" it is tested weather the point also lies between two fluxsurfaces
        #Therefore the "intersections" function is used on each point.
        #Since there are always two intersections with a fluxsurface if you draw an infinite line through the center of the surface and a point of interest 
        # i restrict the search to the intersection with the lower half of the flux surface for y<0 points and vise versa.
        #The function is then used to find the closest point to it on the fluxsurface. --> ideal is the index of one of the 64 lines describing the fluxsurface that is closest to the desired point
        #Now we use the funciton again but only feed it this particular small fit between the two points on the fluxsurface that are closest and extract the "intersection" point.
        #Now we can compare if our investigated point is smaller or higher in absolut numbers meaning if it lies insight or outside the surface.
        #In this manner we add the points that are between flux surfaces to an 8 dimensional array. Later we count the points in each section of the array and by multiplying the number with the resolution squared we gain information about the space that is covered by a line of sight and a particular fluxsurface.
        #Note that the code only uses points that are not already marked as 'inside two smaller flux surfaces' to be faster.
        inside=[[],[],[],[],[],[],[],[],[]]
        inside_=[]
        vol=[[],[],[],[],[],[],[],[],[]]
        v_i=[[0],[0],[0],[0],[0],[0],[0],[0],[0]]
        for f in fluxsurfaces:
        #m_=list(p+0.01 for p in (np.arange(int(min(x_.iloc[i+1])),int(max(x_.iloc[i+1]))+1,res)))
        #n_=list(b+0.01 for b in (np.arange(int(min(y_.iloc[i+1])),int(max(y_.iloc[i+1]))+1,res)))
            for (m,n) in inside_line[bol-1]:
                if (m,n) not in inside_:
                    plt.plot(m,n,marker='o',color=colors[f])
                    if n<0:
                        half=np.concatenate((np.arange(-1,0),np.arange(32,63)))
                    else:
                        half=np.arange(0,32)
                    x=np.array(x_.iloc[f])
                    y=np.array(y_.iloc[f])
                    plt.plot(x,y,marker='.',color=colors[f])
                    ideal=intersections(half)[0]
                    x_inner=intersections([ideal])[2][intersections([ideal])[1]]
                    y_inner=lin(x_inner,*(intersections([ideal])[4]))
                    x=np.array(x_.iloc[f+1])
                    y=np.array(y_.iloc[f+1])
                    plt.plot(x,y,marker='.',color=colors[f+1])
                    ideal=intersections(half)[0]
                    x_outer=intersections([ideal])[2][intersections([ideal])[1]]
                    y_outer=lin(x_outer,*(intersections([ideal])[4]))
                    if f==0:
                        if abs(y_inner)>=abs(n) and abs(x_inner-z_0)>=abs(m-z_0):
                            ax.add_patch(mpl.patches.Rectangle((m-res/2,n-res/2),res,res,color='grey',alpha=0.4,linewidth=0))
                            inside[f].append((m,n))
                            plt.plot(m,n,marker='o',color='Grey')
                        if abs(y_outer)>=abs(n) and abs(x_outer-z_0)>=abs(m-z_0) and abs(y_inner)<=abs(n) and abs(x_inner-z_0)<=abs(m-z_0):
                            ax.add_patch(mpl.patches.Rectangle((m-res/2,n-res/2),res,res,color=colors[f],alpha=0.4,linewidth=0))
                            inside[f+1].append((m,n))
                    else:
                        if abs(y_outer)>=abs(n) and abs(x_outer-z_0)>=abs(m-z_0) and abs(y_inner)<=abs(n) and abs(x_inner-z_0)<=abs(m-z_0):
                            ax.add_patch(mpl.patches.Rectangle((m-res/2,n-res/2),res,res,color=colors[f],alpha=0.4,linewidth=0))
                            inside[f+1].append((m,n))
                inside_.extend(inside[f])
        
        for v in [0,1,2,3,4,5,6,7,8]:  
            #derive the volume in each fluxsurface the channel occupies with the horizontal line of sight       
            x_horiz=[a-b,a-b-13.7,a-b-17.7]
            y_horiz=[s_w,7.165,8.399]
            popt5,pcov5=curve_fit(lin,[x+x_err for x in x_horiz],[x/2+y_err for x in y_horiz])
            popt6,pcov6=curve_fit(lin,[x+x_err for x in x_horiz],[-x/2-y_err for x in y_horiz])
            
            len_=[]
            for g in np.arange(0,len(inside[v])):
                x_test=inside[v][g][0]
                len_.append(lin(x_test,*popt5)-lin(x_test,*popt6))
                v_i[v]+=(len_[g]*res**2)/(a-inside[v][g][0])**2
        
            vol[v]=np.sum(len_)*res**2
            print("The {n} Fluxsurface covers a space of ~ {s} cm\u00b3 in channel {c} line of sight.".format(n=v,s=float(f'{vol[v]:.2f}'),c=bol))
            
        plt.ylim(min(y_.iloc[f+1])-res,max(y_.iloc[f+1])+res)
        plt.xlim(z_0+min(y_.iloc[f+1])-res,80)
        plt.show()
        vol_ges=((vol[0]+vol[1]+vol[2]+vol[3]+vol[4]+vol[5]+vol[6]+vol[7]+vol[8])/1000000)
        v_i_ges=((v_i[0]+v_i[1]+v_i[2]+v_i[3]+v_i[4]+v_i[5]+v_i[6]+v_i[7]+v_i[8])/100)[0]
        P_ges=((4*np.pi*bolo_power[bol-1])/(((c_w*c_h)/10000)*v_i_ges))*0.1198
        print('Volume Observed by channel {c}: {v}m\u00b3'.format(c=bol,v=float(f'{vol_ges:.5f}')))
        print('Total Plasmaradiationpower: {p} Watt'.format(p=float(f'{P_ges:.2f}')))
    print(datetime.now()-start)



#Total Power of Channel 4

def Totalpower(Type=''):
    #--------------------------------------------------------------#
    mesh=1/0.75     #multiply with this factor to account for 25% absorbance of mesh
    gold=1      #multiply with this factor to account for the amount of the spectrum absorbed by gold
    #--------------------------------------------------------------#
    v_i_ges_middle=0.01476
    v_i_ges_min=0.012895
    v_i_ges_max=0.016405
    P_ges_middle=[]
    P_ges_min=[]
    P_ges_max=[]
    pressures=[]
    mw=[]
    plt.figure(figsize=(10,7))
    colors=['navy','blue','royalblue','lightsteelblue','indigo','rebeccapurple','darkorchid','mediumorchid']
    for s,i in zip(shotnumbers,np.arange(0,len(shotnumbers))):
        bolo_p=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot{s}/shot{s}_bolometerprofile_from_radiation_powers.txt'.format(s=s),usecols=1)[3]*10**(-6)
        pressures.append(Pressure(s))
        mw.append(GetMicrowavePower(s))
        P_ges_middle.append(((4*np.pi*bolo_p)/(((c_w*c_h)/10000)*v_i_ges_middle))*0.1198*mesh*gold)
        P_ges_min.append(((4*np.pi*bolo_p)/(((c_w*c_h)/10000)*v_i_ges_min))*0.1198*mesh*gold)
        P_ges_max.append(((4*np.pi*bolo_p)/(((c_w*c_h)/10000)*v_i_ges_max))*0.1198*mesh*gold)
        title='shot n°{s} // P$_m$$_w$= {m} W // p={p} mPa'.format(s=s,m=float(f'{GetMicrowavePower(s):.3f}'),p=float(f'{Pressure(s):.3f}'))
        if Type=='Pressure':
            plt.errorbar(pressures[i],P_ges_middle[i],yerr=0.1*P_ges_middle[i],marker='o',capsize=5,label=title,color=colors[i])
            plt.xlabel('pressure [mPa]')
        if Type=='Power':
            plt.errorbar(mw[i],P_ges_middle[i],yerr=0.1*P_ges_middle[i],marker='o',capsize=5,label=title,color=colors[i])
            plt.annotate(str(float(f'{(P_ges_middle[i]/mw[i])*100:.1f}'))+'%',(mw[i]+10,P_ges_middle[i]),color=colors[i])
            plt.xlabel('microwave power [W]')
    plt.suptitle('Total emitted radiation Power calculated from channel 4 \n {t} scan'.format(t=Type),y=1.05,fontsize=20)
    plt.ylabel('total power [W]')
    plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.7))
    plt.show()
    #print('Total Plasmaradiationpower: {p} Watt'.format(p=float(f'{P_ges_middle:.2f}')))
    #print('Total Plasmaradiationpower min: {p} Watt'.format(p=float(f'{P_ges_min:.2f}')))
    #print('Total Plasmaradiationpower max: {p} Watt'.format(p=float(f'{P_ges_max:.2f}')))




# The Top View Calculations
def TopView():
    r_ves_min=60-r_ves
    r_ves_max=60+r_ves
    rad_min=54.16       #Minimum radius of outer most fluxsurface
    rad_max=72.32       #Maximum radius of outer most fluxsurface

    theta=np.linspace(0,2*np.pi,100)

    def c1(r,theta):
        return r*np.cos(theta)
    def c2(r,theta):
        return r*np.sin(theta)
    plt.figure(figsize=(10,10))
    plt.plot(c1(rad_min,theta),c2(rad_min,theta),color='blue')
    plt.plot(c1(rad_max,theta),c2(rad_max,theta),color='blue')
    plt.plot(c1(r_ves_min,theta),c2(r_ves_min,theta),color='grey')
    plt.plot(c1(r_ves_max,theta),c2(r_ves_max,theta),color='grey')


    #horizontal lines of sight
    def lin(x,d,e):
        return d*x+e
    x_horiz=[a-b,a-b-13.7,a-b-17.7]
    y_horiz=[s_w,7.165,8.399]

    # popt1,pcov1=curve_fit(lin,[a,a-b],[-c_w/2,-s_w/2])
    # popt2,pcov2=curve_fit(lin,[a,a-b],[c_w/2,s_w/2])
    # plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt1),color='red',linestyle='dashed')
    # plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt2),color='red',linestyle='dashed')
    popt3,pcov3=curve_fit(lin,[x-0.4 for x in x_horiz],[(x/2)+0.4 for x in y_horiz])
    popt4,pcov4=curve_fit(lin,[x-0.4 for x in x_horiz],[(-x/2)-0.4 for x in y_horiz])
    plt.errorbar(x_horiz,[x/2 for x in y_horiz],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color='red')
    plt.errorbar(x_horiz,[-x/2 for x in y_horiz],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color='red')
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt3),color='red')
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt4),color='red')


    # x_test=rad_min
    # plt.plot(x_test,0,'go')
    # plt.plot([x_test,x_test],[lin(x_test,*popt3),lin(x_test,*popt4)],color='green')
    # plt.annotate(str(float(f'{lin(x_test,*popt3)-lin(x_test,*popt4):.2f}'))+'cm',[x_test+1,0],color='green')
    # print(rad_max-rad_min)


    plt.xlim(70,80)
    plt.ylim(-5,5)
    plt.grid('True')
    plt.show() 

# %%

shotnumber=13119
shotnumbers=(13072,13071,13070,13069)
gas='He'

location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)

Totalpower('Power')
#Pixelmethod()
# %%
