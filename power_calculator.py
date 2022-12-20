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

plt.rc('font',size=14)
plt.rc('figure', titlesize=15)


# %%
#The important distances are defined
start=datetime.now()
a=32.11 #Distance of Bolometerhead Middle to Torsatron center [cm]
b=3.45 #Distance of Bolometerhead Middle to  Slit [cm]
s_w=1.4 #Width of the slit [cm]
s_h=0.5 #Height of the slit [cm]
alpha=14 #Angle of the Bolometerhead to plane [°]
c_w=0.38 #Channelwidth of Goldsensor [cm]
c_h=0.17 #HChannelheight of Goldsensor [cm]
h=2 #height of Bolometerhead [cm]
z_0=63.9    #middle of flux surfaces

#-----------------------------------------------------#-
#!!! Enter here which channels lines of sight you want to have analyzed(1 to 8), what pixel-resolution you need (in cm) and  which flux surfaces (0 to 7) should be modeled. 
#note that more channels, a smaller resolution and more fluxsurfaces result in longer computation times
bolo_channel=[5] #1 to 8
res=0.5
fluxsurfaces=[0,1,2,3,4,5,6,7] #0 to 7
#-----------------------------------------------------#-


for b in bolo_channel:
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111)
    x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
    y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')


    plt.xlabel('R [cm]')
    plt.ylabel('r [cm]')

    #Derive the exact positions of the bolometerchannels
    #I derive the x and y positions of the four upper channels lower and upper edge
    #I consider the 8 Bolometerchannels distributed over the 4cm with equal distance resulting in a distance of 0.33cm
    f=0.33
    h=[-2+f/2,-2+f/2+c_h,-2+f/2+c_h+f,-2+f*1.5+c_h*2,-2+f*2.5+c_h*2,-2+f*2.5+c_h*3,-2+f*3.5+c_h*3,-2+f*3.5+c_h*4,f*0.5,f*0.5+c_h,f*1.5+c_h,f*1.5+c_h*+2,f*2.5+c_h*2,f*2.5+c_h*3,f*3.5+c_h*3,f*3.5+c_h*4]
    #h=[-1.6-c_h/2,-1.6+c_h/2,-1.2-c_h/2,-1.2+c_h/2,-0.8-c_h/2,-0.8+c_h/2,-0.4-c_h/2,-0.4+c_h/2,0.4-c_h/2,0.4+c_h/2,0.8-c_h/2,0.8+c_h/2,1.2-c_h/2,1.2+c_h/2,1.6-c_h/2,1.6+c_h/2]
    x_b=[]
    y_b=[]
    for i in h:
        x_b.append(-abs(np.cos((90-alpha)*np.pi/180)*i)+60+a)
        y_b.append(-np.sin((90-alpha)*np.pi/180)*i)

    def lin(x,d,e):
        return d*x+e
    y_exp=np.flip((13.665,8.245,10.535,5.115,7.405,1.985,4.275,-1.145,1.145,-4.275,-1.985,-7.405,-5.115,-10.535,-8.245,-13.665))
    
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
    n_=list(b+0.01 for b in (np.arange(int(min(y_.iloc[8])),int(max(y_.iloc[8]))+1,res)))
    inside_line=[[],[],[],[],[],[],[],[]]
    lines=[0,2,4,6,8,10,12,14]
    colors=['red','blue','green','gold','magenta','darkcyan','blueviolet','orange']
    channels=[0,1,2,3,4,5,6,7]

    #here the desired line of sight is plotted from the experimental data. To see the calculate lines of sight activate the dashed lines plot
    #now the points of interest(the ones in a square of the rough size of the outer most fluxsurface) are tested for their position relative to the line of sight
    #If the point lies inside the two lines describing the line of sight of that channel, its coordinates are added to "inside_line"
    for i,j,k in zip([lines[b-1]],[colors[b-1]],[channels[b-1]]):
        plt.plot([x_b[i],x_b[i+1]],[y_b[i],y_b[i+1]],color='red')
        popt1,pcov1=curve_fit(lin,[x_b[i],60+a-b],[y_b[i],-s_h/2])
        popt2,pcov2=curve_fit(lin,[x_b[i+1],60+a-b],[y_b[i+1],s_h/2])
        #plt.plot(x,lin(x,*popt1),color=j,linestyle='dashed',alpha=0.4)
        #plt.plot(x,lin(x,*popt2),color=j,linestyle='dashed',alpha=0.5)
        popt3,pcov3=curve_fit(lin,[60+a-b,60+a-26.9],[-s_h/2,y_exp[i]])
        popt4,pcov4=curve_fit(lin,[60+a-b,60+a-26.9],[s_h/2,y_exp[i+1]])
        plt.plot(x,lin(x,*popt3),color=j)
        plt.plot(x,lin(x,*popt4),color=j)
        for m in m_:
            for n in n_:
                popt3,pcov3=curve_fit(lin,[60+a-b,60+a-26.9],[-s_h/2,y_exp[i]])
                popt4,pcov4=curve_fit(lin,[60+a-b,60+a-26.9],[s_h/2,y_exp[i+1]])
                if n< lin(m,*popt4) and n>(lin(m,*popt3)):
                    #ax.add_patch(mpl.patches.Rectangle((m-res/2,n-res/2),res,res,color=j,alpha=0.4,linewidth=0))
                    inside_line[k].append((m,n))
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
    for f in fluxsurfaces:
    #m_=list(p+0.01 for p in (np.arange(int(min(x_.iloc[i+1])),int(max(x_.iloc[i+1]))+1,res)))
    #n_=list(b+0.01 for b in (np.arange(int(min(y_.iloc[i+1])),int(max(y_.iloc[i+1]))+1,res)))
        for i,j in zip([fluxsurfaces[f]],[colors[f]]):
            inside_.extend(inside[i]) 
            #for m in m_:
                #for n in n_:
            for (m,n) in inside_line[b-1]:
                if (m,n) not in inside_:
                    plt.plot(m,n,marker='o',color=j)
                    if n<0:
                        half=np.concatenate((np.arange(-1,0),np.arange(32,63)))
                    else:
                        half=np.arange(0,32)
                    x=np.array(x_.iloc[i])
                    y=np.array(y_.iloc[i])
                    plt.plot(x,y,marker='.')
                    ideal=intersections(half)[0]
                    x_inner=intersections([ideal])[2][intersections([ideal])[1]]
                    y_inner=lin(x_inner,*(intersections([ideal])[4]))
                    x=np.array(x_.iloc[i+1])
                    y=np.array(y_.iloc[i+1])
                    plt.plot(x,y,marker='.')
                    ideal=intersections(half)[0]
                    x_outer=intersections([ideal])[2][intersections([ideal])[1]]
                    y_outer=lin(x_outer,*(intersections([ideal])[4]))
                    if i==0:
                        if abs(y_inner)>=abs(n) and abs(x_inner-z_0)>=abs(m-z_0):
                            ax.add_patch(mpl.patches.Rectangle((m-res/2,n-res/2),res,res,color='grey',alpha=0.4,linewidth=0))
                            inside[i].append((m,n))
                        if abs(y_outer)>=abs(n) and abs(x_outer-z_0)>=abs(m-z_0) and abs(y_inner)<=abs(n) and abs(x_inner-z_0)<=abs(m-z_0):
                            ax.add_patch(mpl.patches.Rectangle((m-res/2,n-res/2),res,res,color=j,alpha=0.4,linewidth=0))
                            inside[i+1].append((m,n))
                    else:
                        if abs(y_outer)>=abs(n) and abs(x_outer-z_0)>=abs(m-z_0) and abs(y_inner)<=abs(n) and abs(x_inner-z_0)<=abs(m-z_0):
                            ax.add_patch(mpl.patches.Rectangle((m-res/2,n-res/2),res,res,color=j,alpha=0.4,linewidth=0))
                            inside[i+1].append((m,n))
        print("The {n} Fluxsurface covers a space of ~ {s} cm\u00b2 in channel {c} line of sight.".format(n=i,s=len(inside[i])*res**2,c=b))
    plt.ylim(min(y_.iloc[i+1])-res,max(y_.iloc[i+1])+res)
    plt.xlim(z_0+min(y_.iloc[i+1])-res,z_0+max(y_.iloc[i+1])+res)
    plt.show()
print(datetime.now()-start)
# %%
plt.figure(figsize=(10,10))
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')
m=63
n=-1
diff_=[]
def lin(x,d,e):
    return d*x+e
if n<0:
    half=np.concatenate((np.arange(-1,0),np.arange(32,63)))
else:
    half=np.arange(0,32)
x=np.array(x_.iloc[2])
y=np.array(y_.iloc[2])
plt.plot(x,y,marker='.',linestyle='None')
plt.plot(m,n,'go')

for i in half:
    popt,pcov=curve_fit(lin,[x[i],x[i+1]],[y[i],y[i+1]])
    range=np.arange(np.sort([x[i],x[i+1]])[0],np.sort([x[i],x[i+1]])[1],0.0001)
    plt.plot(range,lin(range,*popt))
    poptm,pcovm=curve_fit(lin,[z_0,m],[0,n])
    plt.plot(range,lin(range,*poptm))
    diff=abs(lin(range,*popt)-lin(range,*poptm))
    diff_.append(min(diff))
    plt.plot(range[np.argmin(diff)],min(diff))
    #return (g[np.argmin(diff_)],np.argmin(diff),range,popt,poptm)
plt.grid(True)

#x_inner=intersections([ideal])[2][intersections([ideal])[1]]
#y_inner=lin(x_inner,*(intersections([ideal])[4]))
#plt.plot(x_inner,y_inner,'ro')
plt.show()
# %%
