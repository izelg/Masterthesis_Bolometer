#%%

#Written by: Izel Gediz
#Date of Creation: 14.11.2022


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
plt.rc('font',size=14)
plt.rc('figure', titlesize=15)

#%%
a=32.11 #Distance of Bolometerhead Middle to Torsatron center [cm]
b=3.45 #Distance of Bolometerhead Middle to  Slit [cm]
s_w=1.4 #Width of the slit [cm]
s_h=0.5 #Height of the slit [cm]
alpha=14 #Angle of the Bolometerhead to plane [°]
c_w=0.38 #Channelwidth of Goldsensor [cm]
c_h=0.17 #HChannelheight of Goldsensor [cm]
h=2 #height of Bolometerhead [cm]
m=63.55 #Middle of Plasma measured from the middle of the TJ-K [cm]


res=0.5
y=np.arange(-15,15,res)
x=np.arange(50,95,res)
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)
plt.hlines(y,50,95,alpha=0.1)
plt.vlines(x,-15,15,alpha=0.1)
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

for i,j in zip([0,2,4,6,8,10,12,14],['red','blue','green','gold','magenta','darkcyan','blueviolet','orange']):
    plt.plot([x_b[i],x_b[i+1]],[y_b[i],y_b[i+1]],color='red')
    popt1,pcov1=curve_fit(lin,[x_b[i],60+a-b],[y_b[i],-s_h/2])
    popt2,pcov2=curve_fit(lin,[x_b[i+1],60+a-b],[y_b[i+1],s_h/2])
    #plt.plot(x,lin(x,*popt1),color=j,linestyle='dashed',alpha=0.4)
    #plt.plot(x,lin(x,*popt2),color=j,linestyle='dashed',alpha=0.5)
    popt3,pcov3=curve_fit(lin,[60+a-b,60+a-26.9],[-s_h/2,y_exp[i]])
    popt4,pcov4=curve_fit(lin,[60+a-b,60+a-26.9],[s_h/2,y_exp[i+1]])
    plt.plot(x,lin(x,*popt3),color=j)
    plt.plot(x,lin(x,*popt4),color=j)

plt.plot([60+a-b,60+a-b],[s_h/2,-s_h/2],color='blue')


p_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
r_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')
for i in [0,1,2,3,4,5,6,7,8]:
    p=p_.iloc[i]
    r=r_.iloc[i]
    plt.plot(np.array(p)[:,None],np.array(r)[:,None],'r.--')
    f=interp1d(p,r)
    plt.plot(np.array(p)[:,None],np.array(f(p))[:,None])
over=[]
for h in x:
    for i in y:
        for j,k in zip([8],['blue']):
            popt3,pcov3=curve_fit(lin,[60+a-b,60+a-26.9],[-s_h/2,y_exp[j]])
            popt4,pcov4=curve_fit(lin,[60+a-b,60+a-26.9],[s_h/2,y_exp[j+1]])
            if i< lin(h,*popt4) and i>=(lin(h,*popt3)-res):
                if (h,i) not in over:
                    over.append((h,i))
                ax.add_patch(mpl.patches.Rectangle((h,i),res,res,color=k,alpha=0.1,linewidth=0))
# # under=[]
# # for h in x:
# #     for i in y:
# #         if i< (lin(h,-np.tan(np.radians(angle1)),x0,y0)-res) and i>=lin(h,np.tan(np.radians(angle2)),x0,y0):
# #             if (h,i) not in under:
# #                 under.append((h,i))
# #                 ax.add_patch(mpl.patches.Rectangle((h,i),res,res,color='red',alpha=0.1,linewidth=0))

d=0.01
o=3
outside=[]
for i in [o]:
    f=interp1d(p_.iloc[i],r_.iloc[i])
    g=interp1d(p_.iloc[i-1],r_.iloc[i-1])
    for p,p2 in zip(p_.iloc[i],p_.iloc[i-1]):
        for (h,i) in over:
            if abs(h-m)-d< abs(p-m) and abs(h-m)>abs(p2-m):
            #if abs(i)-d< abs(f(p)) and abs(h-m)-d< abs(p-m) and abs(i)>abs(g(p2)) and abs(h-m)>abs(p2-m):
                if (h,i) not in outside:
                    outside.append((h,i))
                    ax.add_patch(mpl.patches.Rectangle((h,i),res,res,color='blue',linewidth=0,alpha=0.5))

# inside=[]
# for i in [o]:
 #       m=np.mean(p_.iloc[i])
#     n=np.mean(p_.iloc[i-1])
#     f=interp1d(p_.iloc[i],r_.iloc[i])
#     g=interp1d(p_.iloc[i-1],r_.iloc[i-1])
#     for p,p2 in zip(p_.iloc[i],p_.iloc[i-1]):
#         for (h,i) in under:
#             if abs(i)< abs(f(p)+d) and abs(h-m)< abs(p-m)+d and abs(i)+d>abs(g(p2)) and abs(h-n)+d>abs(p2-n):
#                 if (h,i) not in inside:
#                     inside.append((h,i))
#                     ax.add_patch(mpl.patches.Rectangle((h-res/2,i),res,res,color='red',linewidth=0,alpha=0.5))
        
# p=np.array(p_.iloc[3])
# r=np.array(r_.iloc[3])

# popt_m,pcov_m=curve_fit(lin,[60,m],[0.5,0])
# range=np.arange(min(p_.iloc[3]),max(p_.iloc[3]),0.001)
# f=interp1d(p,r,)
# intersect=np.argwhere(np.diff(np.sign(f(p)-lin(p,*popt_m)))).flatten()
# plt.plot(range,lin(range,*popt_m),color='red')
# plt.plot(60,0.5,'ro')
# plt.plot(m,0,'ro')

# plt.plot(p[intersect[0]],f(p[intersect[0]]),'bo')
print((len(outside)*res**2))#+len(inside)*res**2)/2)
plt.xlim(50,80)
plt.ylim(-15,15)
plt.show()




# %%

plt.figure(figsize=(10,10))
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')
m=61
n=-0.1
print(len(x))
for i in [3]:
    x=np.array(x_.iloc[i])
    y=np.array(y_.iloc[i])
    plt.plot(x,y,marker='.',linestyle='None')
def lin(x,a,b):
    return a*x+b
if n<0:
    half=np.concatenate((np.arange(-1,0),np.arange(32,63)))
else:
    half=np.arange(0,32)
for i in half:
    popt,pcov=curve_fit(lin,[x[i],x[i+1]],[y[i],y[i+1]])
    plt.plot(x[i],y[i],'ro')
    plt.plot(x[i+1],y[i+1],'ro')
    range=np.arange(np.sort([x[i],x[i+1]])[0],np.sort([x[i],x[i+1]])[1],0.0001)
    plt.plot(range,lin(range,*popt))

    poptm,pcovm=curve_fit(lin,[63.55,m],[0,n])
    plt.plot(m,n,'go')
    diff=abs(lin(range,*popt)-lin(range,*poptm))
    intersect=np.argmin(diff)
    intersections=[]
    if abs(lin(range[intersect],*popt)-lin(range[intersect],*poptm))<1e-2:
        plt.plot(range,lin(range,*poptm),color='green')    
        plt.plot(range[intersect],lin(range,*poptm)[intersect],'bo')
        intersections.append((range[intersect],lin(range,*poptm)[intersect]))
        print(intersections)
plt.grid(True)
plt.show()

# %%
