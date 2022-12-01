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
    #plt.plot(x,lin(x,*popt3),color=j)
    #plt.plot(x,lin(x,*popt4),color=j)

plt.plot([60+a-b,60+a-b],[s_h/2,-s_h/2],color='blue')


p_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
r_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')
for i in [0,1,2,3,4,5,6,7,8]:
    p=p_.iloc[i]
    r=r_.iloc[i]
    plt.plot(np.array(p)[:,None],np.array(r)[:,None],'r.--')
    f=interp1d(p,r)
    plt.plot(np.array(p)[:,None],np.array(f(p))[:,None])
# over=[]
# for h in x:
#     for i in y:
#         for j,k in zip([8],['blue']):
#             popt3,pcov3=curve_fit(lin,[60+a-b,60+a-26.9],[-s_h/2,y_exp[j]])
#             popt4,pcov4=curve_fit(lin,[60+a-b,60+a-26.9],[s_h/2,y_exp[j+1]])
#             if i< lin(h,*popt4) and i>=(lin(h,*popt3)-res):
#                 if (h,i) not in over:
#                     over.append((h,i))
#                 ax.add_patch(mpl.patches.Rectangle((h,i),res,res,color=k,alpha=0.1,linewidth=0))
# # under=[]
# # for h in x:
# #     for i in y:
# #         if i< (lin(h,-np.tan(np.radians(angle1)),x0,y0)-res) and i>=lin(h,np.tan(np.radians(angle2)),x0,y0):
# #             if (h,i) not in under:
# #                 under.append((h,i))
# #                 ax.add_patch(mpl.patches.Rectangle((h,i),res,res,color='red',alpha=0.1,linewidth=0))

# d=0.01
# o=3
# outside=[]
# for i in [o]:
#     f=interp1d(p_.iloc[i],r_.iloc[i])
#     g=interp1d(p_.iloc[i-1],r_.iloc[i-1])
#     for p,p2 in zip(p_.iloc[i],p_.iloc[i-1]):
#         for (h,i) in over:
#             if abs(i)-d< abs(f(p)) and abs(h-m)-d< abs(p-m) and abs(i)>abs(g(p2)) and abs(h-m)>abs(p2-m):
#                 if (h,i) not in outside:
#                     outside.append((h,i))
#                     ax.add_patch(mpl.patches.Rectangle((h,i),res,res,color='blue',linewidth=0,alpha=0.5))
#                     over=[]
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
        
p=p_.iloc[3]
r=r_.iloc[3]    
popt_m,pcov_m=curve_fit(lin,[60,m],[0.5,0])
#range=np.arange(min(p_.iloc[3]),max(p_.iloc[3]),0.001)
f=interp1d(p,r)
intersect=np.argwhere(np.diff(np.sign(f(p)-lin(p,*popt_m)))).flatten()
print(p[intersect[0]],f(p[intersect[0]]))
plt.plot(p,lin(p,*popt_m),color='red')
plt.plot(60,0.5,'ro')
plt.plot(m,0,'ro')
plt.plot(p[intersect[1]],f(p[intersect[1]]),'bo')
#print((len(outside)*res**2))#+len(inside)*res**2)/2)
#!!!!!!!!!!!!!!!!Die Gereade die durch 0 geht und ein Viereck muss die flussfläche Schneiden
plt.xlim(min(p_.iloc[3])-1,max(p_.iloc[3])+1)
plt.ylim(-5,5)
plt.show()


#%%

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')

(theta, phi) = np.meshgrid(np.linspace(0, 2 * np.pi, 41),
                           np.linspace(0, 2 * np.pi, 41))

# x = (3 + np.cos(phi)) * np.cos(theta)
# y = (3 + np.cos(phi)) * np.sin(theta)
# z = np.sin(phi)
# ax.plot_surface(x, y, z,alpha=0.4)#, facecolors=cm.jet(fun(theta, phi)))



x = np.linspace(0, 1, 10)  # [0, 2,..,10] : 6 distinct values
y = np.linspace(1, 2, 10)  # [0, 5,..,20] : 5 distinct values
z = np.linspace(0, 10, 100)  # 6 * 5 = 30 values, 1 for each possible combination of (x,y)

X, Y = np.meshgrid(x, y)
Z = np.reshape(z, X.shape)
ax.plot_surface(X, Y, Z)

fig.tight_layout()
plt.show()

# %%


x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')
for i in [0,1,2,3,4,5,6,7,8]:
    x=x_.iloc[i]
    y=y_.iloc[i]
    plt.plot(x,y,'r.--')
    f=interp1d(x,y)
    plt.plot(x,f(x))
plt.show()
# %%
