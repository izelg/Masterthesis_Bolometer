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

#%%
res=0.5
y=np.arange(-15,15,res)
x=np.arange(50,75,res)
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)
plt.hlines(y,50,75,alpha=0.2)
plt.vlines(x,-15,15,alpha=0.2)
def lin(z,a,c,b):
    return a*(z-c)+b
angle1=15
angle2=15
y0=0
x0=50
plt.plot(x,lin(x,np.tan(np.radians(angle1)),x0,y0),x,lin(x,-np.tan(np.radians(angle2)),x0,y0),color='red')
p_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
r_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')
for i in [0,1,2,3,4,5,6,7,8]:
    p=p_.iloc[i]
    r=r_.iloc[i]
    plt.plot(p,r,'r.--')
    f=interp1d(p,r)
    plt.plot(p,f(p))
over=[]
for h in x:
    for i in y:
        if i< lin(h,np.tan(np.radians(angle1)),x0,y0) and i>=(lin(h,-np.tan(np.radians(angle2)),x0,y0)-res):
            if (h,i) not in over:
                over.append((h,i))
                ax.add_patch(mpl.patches.Rectangle((h,i),res,res,color='blue',alpha=0.2,linewidth=0))
under=[]
for h in x:
    for i in y:
        if i< (lin(h,np.tan(np.radians(angle1)),x0,y0)-res) and i>=lin(h,-np.tan(np.radians(angle2)),x0,y0):
            if (h,i) not in under:
                under.append((h,i))
                ax.add_patch(mpl.patches.Rectangle((h,i),res,res,color='red',alpha=0.2,linewidth=0))


theta=np.linspace(0,2*np.pi,100)
r=np.sqrt(15)
def c1(r,theta):
    return r*np.cos(theta)
def c2(r,theta):
    return r*np.sin(theta)
#plt.plot(c1(r,theta),c2(r,theta))
r2=np.sqrt(7)
#plt.plot(c1(r2,theta),c2(r2,theta))
outside=[]
d=0.05
for t in theta:
    for (h,i) in over:
        if abs(i)-d<=abs(r*np.sin(t)) and abs(h)-d<=abs(r*np.cos(t)) and abs(i)+d>=abs(r2*np.sin(t)) and abs(h)+d>=abs(r2*np.cos(t)):
            if (h,i) not in outside:
                outside.append((h,i))
                ax.add_patch(mpl.patches.Rectangle((h-res/2,i),res,res,color='blue',linewidth=0))
            #over=[]
inside=[]
e=0.01
for t in theta:
    for (h,i) in under:
        if abs(i)+0.01<=abs(r*np.sin(t)) and abs(h)+0.011<=abs(r*np.cos(t)) and abs(i)-0.012>=abs(r2*np.sin(t)) and abs(h)-0.013>=abs(r2*np.cos(t)):
            if (h,i) not in inside:
                inside.append((h,i))
                ax.add_patch(mpl.patches.Rectangle((h-res/2,i),res,res,color='red',linewidth=0,alpha=0.5))
            over=[]
print((np.pi*r**2)-np.pi*r2**2)
print((len(outside)*res**2+len(inside)*res**2)/2)
#print(outside)
#plt.xlim(-10,10)
#plt.ylim(-10,10)
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
