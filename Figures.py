#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib.patches import Rectangle
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.patches as patches
from scipy.signal import savgol_filter
from scipy.interpolate import pchip_interpolate
from matplotlib.ticker import ScalarFormatter
from scipy import signal
from scipy import integrate
import math

import plasma_characteristics as pc
import bolo_radiation as br
import adas_data as adas
import power_calculator as poca
#%% Parameter
Poster=False
Latex=False
PPP=True
a=60+32.11+3.45 #Position of Bolometerheadmiddle [cm]
b=3.45 #Distance of Bolometerhead Middle to  Slit [cm]
s_w=1.4 #Width of the slit [cm]
s_h=0.5 #Height of the slit [cm]
alpha=13 #Angle of the Bolometerhead to plane [°]
c_w=0.38 #Channelwidth of Goldsensor [cm]
c_h=0.13 #HChannelheight of Goldsensor [cm]
c_d=0.225 #depth of Goldsensor [cm]
h=2 #height of Bolometerhead [cm]
z_0=63.9    #middle of flux surfaces
t=17.5 #radius of vessel [cm]
h1=4.135E-15
c=299792458
if Poster==True:
    plt.rc('font',size=20)
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.rcParams['lines.markersize']=12
    width=10
    height=7
elif Latex==True:
    width=412/72.27
    height=width*(5**.5-1)/2
    n=1.5
    plt.rcParams['text.usetex']=True
    plt.rcParams['font.family']='serif'
    plt.rcParams['axes.labelsize']=11*n
    plt.rcParams['font.size']=11*n
    plt.rcParams['legend.fontsize']=11*n
    plt.rcParams['xtick.labelsize']=11*n
    plt.rcParams['ytick.labelsize']=11*n
    plt.rcParams['lines.markersize']=4
elif PPP==True:
    width=(412/72.27)*1.5
    height=(width*(5**.5-1)/2)*1.5
    plt.rcParams['text.usetex']=True
    plt.rcParams['font.family']='sans-serif'
    plt.rcParams['axes.labelsize']=20
    plt.rcParams['font.size']=15
    plt.rcParams['legend.fontsize']=15
    plt.rcParams['xtick.labelsize']=20
    plt.rcParams['ytick.labelsize']=20
    plt.rcParams['lines.markersize']=10
    plt.rcParams['lines.linewidth']=3


else:
    w=10
    h=7
    plt.rc('font',size=14)
    plt.rc('figure', titlesize=15)
colors=['#1bbbe9','#023047','#ffb703','#fb8500','#c1121f','#780000','#6969B3','#D81159','#1bbbe9','#023047','#ffb703','#fb8500','#c1121f']
markers=['o','v','s','P','p','D','*','x','o','v','s','P','p','D','*','x','o','v','s','P','p','D','*','x','o','v','s','P','p','D','*','x']
colors2=['#03045E','#0077B6','#00B4D8','#370617','#9D0208','#DC2F02','#F48C06','#FFBA08','#7B2CBF','#C77DFF','#2D6A4F','#40916C','#52B788','#03045E','#0077B6','#00B4D8']
#%% Plasmatypes and regimes

#n=np.arange(10E5, 10E35)
#T=np.arange(10E-2,10E6)

plt.figure(figsize=(width,height))
plt.hlines(6,5,37,linestyle='dashed',alpha=0.7,color='red')
plt.plot([25,37],[-2,5.5],linestyle='dashed',alpha=0.7,color='blue')
plt.plot([22.5,37],[-2,2.1],linestyle='dashed',alpha=0.7,color='green')
plt.xticks([5,10,15,20,25,30,35],[r'10$^5$',r'10$^{10}$',r'10$^{15}$',r'10$^{20}$',r'10$^{25}$',r'10$^{30}$',r'10$^{35}$'])
plt.yticks([-2,0,2,4,6,8],[r'10$^{-2}$',r'10$^0$',r'10$^2$',r'10$^4$',r'10$^6$',''])
plt.xlabel('density [m$^{-3}$]')
plt.ylabel('temperature [eV]')
plt.annotate('interstellar\n plasmas',(7,-1))
plt.annotate('interplanetar\n   plasmas ',(9,1))
plt.annotate('flames',(14,-1.5))
plt.annotate(' solar\ncenter',(27,3))
plt.annotate('magnetic\n  fusion',(22,4.5))
plt.annotate('lightning',(23.5,0.5))
plt.annotate(' solar\ncorona',(14,1.9))
plt.annotate('e$^-$ gas\nin metal',(28.2,-1.9))
plt.annotate(' white\ndwarfs',(33,-0.8))
plt.annotate('supernovae',(28,5.5))
plt.annotate('TJ-K',(18,1),color='red')
plt.annotate('fluorescent\n     light ',(15,-0.5))

plt.annotate('relativistic plasmas',(16,6.1),color='red', alpha=0.7)
plt.annotate('degenerated plasmas', (29.1,0.3), rotation=52,color='blue',alpha=0.7)
plt.annotate('non-ideal plasmas',(27,-1.1),rotation=29,color='green',alpha=0.7)
plt.grid(True,alpha=0.2)
plt.xlim(5,37)
plt.ylim(-2,7.5)
fig1= plt.gcf()
plt.show()
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/plasmas_in_nature_and_laboratory.pdf',bbox_inches='tight')

# %% Lines of sight setup


fig=plt.figure(figsize=(width,width))
ax=fig.add_subplot(111)
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')

x=np.arange(40,a,0.1)


plt.xlabel('R [cm]')
plt.ylabel('z [cm]')
f1=0.14 #Distance first channel to edge [cm]
f2=0.40 #Distance between channels [cm]
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

lines=[0,2,4,6,8,10,12,14]
channels=[0,1,2,3,4,5,6,7]
#fluxsurfaces

for i in [0,1,2,3,4,5,6,7,8]:
    x=np.array(x_.iloc[i])
    y=np.array(y_.iloc[i])
    plt.plot(np.append(x,x[0]),np.append(y,y[0]),color='grey',linewidth=3,alpha=0.5)

pos_9,pos_10,pos_11,pos_12,rad_9,rad_10,rad_11,rad_12=[],[],[],[],[],[],[],[]
for d,pos,rad in zip((1,2,3,4),(pos_9,pos_10,pos_11,pos_12),(rad_9,rad_10,rad_11,rad_12)):
    for j in np.arange(0,len(x)):
        r=np.sqrt((x[j]-z_0)**2+y[j]**2)
        beta=np.arctan(y[j]/(x[j]-z_0))
        if (x[j]-z_0)<=0:
            x_neu=(r+d)*-np.cos(beta)+z_0
            y_neu=(r+d)*-np.sin(beta)
        else:
            x_neu=(r+d)*np.cos(beta)+z_0
            y_neu=(r+d)*np.sin(beta)
        pos.append(x_neu)
        rad.append(y_neu)
    plt.plot(np.append(pos,pos[0]),np.append(rad,rad[0]),linewidth=2,linestyle='dashed',color='grey',alpha=0.5)
    print(pos)
    print(rad)

#lines of sight
for i,j in zip(lines,np.arange(0,9)):
    popt1,pcov1=curve_fit(lin,[x_b[i],a-b],[y_b[i],-s_h/2])
    popt2,pcov2=curve_fit(lin,[x_b[i+1],a-b],[y_b[i+1],s_h/2])
    popt3,pcov3=curve_fit(lin,[a-b,a-b-12.4,a-b-19.5,a-b-22.9],[-s_h/2,ex_1[i],ex_2[i],ex_3[i]])
    popt4,pcov4=curve_fit(lin,[a-b,a-b-12.4,a-b-19.5,a-b-22.9],[s_h/2,ex_1[i+1],ex_2[i+1],ex_3[i+1]])
    plt.plot(np.arange(40,x_b[i],0.1),lin(np.arange(40,x_b[i],0.1),*popt3),color=colors[j],linewidth=2)
    plt.plot(np.arange(40,x_b[i+1],0.1),lin(np.arange(40,x_b[i+1],0.1),*popt4),color=colors[j],linewidth=2)






c='black'#fontcolor inside plot
#torsatron
plt.annotate('plasma vessel',(50,25),color=c) 
vessel=plt.Circle((60,0),t,fill=False,color='grey',linewidth=3,alpha=0.5)
#port
plt.plot([a-b-20,a-b-10.3],[-12.5,-12.5],[a-b-20,a-b-10.3],[12.5,12.5],[a-b-10.3,a-b-10.3],[-12.5,12.5],color='grey',linewidth=3,alpha=0.5)
plt.annotate('outer\n port',(73,22),color=c)
#slit
plt.plot([a-b,a-b],[-12,-s_h/2],[a-b,a-b],[12,s_h/2],color='grey',linewidth=3,alpha=0.5,linestyle='dashed')
plt.annotate('slit',(a-b-0.1,-0.5),xytext=(a-b-20,-25),arrowprops=dict(facecolor=c,edgecolor='none',alpha=0.5,width=2),color=c)
bolovessel=patches.Rectangle((60+21.8,-12),20.8,24,edgecolor='grey',facecolor='none',linewidth=3, alpha=0.5)
plt.annotate('bolometer\n  vessel',(83,22),color=c)
#bolometerhead
ts=ax.transData
coords1=[-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2]
coords2=[-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0]
tr1 = matplotlib.transforms.Affine2D().rotate_deg_around(coords1[0],coords1[1], -alpha)
tr2 = matplotlib.transforms.Affine2D().rotate_deg_around(coords2[0],coords2[1],alpha)
bolohead1=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr1+ts)
bolohead2=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr2+ts)
plt.annotate('bolometer\n   heads',(a,-2),xytext=(a-15,-27),arrowprops=dict(facecolor=c,edgecolor='none',alpha=0.5,width=2),color=c)
ax.add_patch(vessel)
ax.add_patch(bolovessel)
ax.add_patch(bolohead1)
ax.add_patch(bolohead2)

plt.xlim(40,100)
plt.ylim(-30,30)
#plt.xlim(a-3,a+3)
#plt.ylim(-3,3)
#plt.grid(True)
fig1= plt.gcf()
plt.show()
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/lines_of_sight_in_TJ-K_with_fluxsurfaces.pdf',bbox_inches='tight')

# %% Lines of sight measurement vertical

fig=plt.figure(figsize=(width,width))
plt.rc('xtick')
plt.rc('ytick')
ax=fig.add_subplot(111)
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')

x=np.arange(40,a,0.1)


plt.xlabel('R [cm]')
plt.ylabel('z [cm]')
f1=0.14 #Distance first channel to edge [cm]
f2=0.40 #Distance between channels [cm]
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
lines=[0,2,4,6,8,10,12,14]
channels=[1,2,3,4,5,6,7,8]

#lines of sight
for i,j in zip(lines,channels):
    plt.plot([x_b[i],x_b[i+1]],[y_b[i],y_b[i+1]],color='red')
    popt3,pcov3=curve_fit(lin,[a-b,a-b-12.4,a-b-19.5,a-b-22.9],[-s_h/2,ex_1[i],ex_2[i],ex_3[i]])
    popt4,pcov4=curve_fit(lin,[a-b,a-b-12.4,a-b-19.5,a-b-22.9],[s_h/2,ex_1[i+1],ex_2[i+1],ex_3[i+1]])
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt3),color=colors[j])
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt4),color=colors[j])
    plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[i],ex_2[i],ex_3[i]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=colors[j])
    plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[i+1],ex_2[i+1],ex_3[i+1]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=colors[j])

#measurements
plt.vlines([a-b-22.9,a-b-19.5,a-b-12.4],-30,30,color='grey',linestyle='dotted',alpha=0.7,linewidth=3)
plt.annotate('',xy=(a-b,-19.6),xymath=(a-b-22.9,-19.6), arrowprops=dict(arrowstyle='<->',color='grey',alpha=0.7,linewidth=2))
plt.annotate('',xy=(a-b,-17.3),xymath=(a-b-19.5,-17.3), arrowprops=dict(arrowstyle='<->',color='grey',alpha=0.7,linewidth=2))
plt.annotate('',xy=(a-b,-15),xymath=(a-b-12.4,-15), arrowprops=dict(arrowstyle='<->',color='grey',alpha=0.7,linewidth=2))
plt.annotate('$z_3$',xy=(a-b-11,-19.4),color='grey')
plt.annotate('$z_2$',xy=(a-b-10,-17.1),color='grey')
plt.annotate('$z_1$',xy=(a-b-9,-14.8),color='grey')

plt.plot([a-b,a-b],[-12,-s_h/2],[a-b,a-b],[12,s_h/2],color='grey',linewidth=3,alpha=0.5,linestyle='dashed')
bolovessel=patches.Rectangle((60+21.8,-12),20.8,24,edgecolor='grey',facecolor='none',linewidth=3, alpha=0.5)
vessel=plt.Circle((60,0),t,fill=False,color='grey',linewidth=3,alpha=0.5)
plt.plot([a-b-20,a-b-10.3],[-12.5,-12.5],[a-b-20,a-b-10.3],[12.5,12.5],[a-b-10.3,a-b-10.3],[-12.5,12.5],color='grey',linewidth=3,alpha=0.5)
ts=ax.transData
coords1=[-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2]
coords2=[-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0]
tr1 = matplotlib.transforms.Affine2D().rotate_deg_around(coords1[0],coords1[1], -alpha)
tr2 = matplotlib.transforms.Affine2D().rotate_deg_around(coords2[0],coords2[1],alpha)
bolohead1=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr1+ts)
bolohead2=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr2+ts)
ax.add_patch(vessel)
ax.add_patch(bolovessel)
ax.add_patch(bolohead1)
ax.add_patch(bolohead2)

plt.xlim(60,100)
plt.ylim(-20,20)
fig1= plt.gcf()
plt.show()
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/lines_of_sight_measurement_vertical.pdf',bbox_inches='tight')

# %% Lines of sight measurement horizontal

fig=plt.figure(figsize=(width,width))
plt.rc('xtick')
plt.rc('ytick')
ax=fig.add_subplot(111)
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')
x=np.arange(40,a,0.1)
plt.xlabel('R [cm]')
plt.ylabel('z [cm]')

def lin(x,d,e):
    return d*x+e
x_horiz=[a-b,a-b-13.7,a-b-17.7]
y_horiz=[s_w,7.165,8.399]

c=colors[3]
popt3,pcov3=curve_fit(lin,[x-0.4 for x in x_horiz],[(x/2)+0.4 for x in y_horiz])
popt4,pcov4=curve_fit(lin,[x-0.4 for x in x_horiz],[(-x/2)-0.4 for x in y_horiz])
plt.errorbar([a-b-13.7,a-b-17.7],[x/2 for x in [7.165,8.399]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=c)
plt.errorbar([a-b-13.7,a-b-17.7],[-x/2 for x in [7.165,8.399]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=c)
plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt3),color=c)
plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt4),color=c)



#measurements
plt.vlines([a-b-13.7,a-b-17.7],-30,30,color='grey',linestyle='dotted',alpha=0.7,linewidth=3)
plt.annotate('',xy=(a-b,-15),xymath=(a-b-13.7,-15), arrowprops=dict(arrowstyle='<->',color='grey',alpha=0.7,linewidth=2))
plt.annotate('',xy=(a-b,-17.2),xymath=(a-b-17.7,-17.2), arrowprops=dict(arrowstyle='<->',color='grey',alpha=0.7,linewidth=2))
plt.annotate('$z_4$',xy=(a-b-10,-14.8),color='grey')
plt.annotate('$z_5$',xy=(a-b-11,-17),color='grey')

plt.plot([a-b,a-b],[-12,-s_h/2],[a-b,a-b],[12,s_h/2],color='grey',linewidth=3,alpha=0.5,linestyle='dashed')
rad_max=60+t       #Maximum radius of outer most fluxsurface
theta=np.linspace(0,2*np.pi,100)
def c1(r,theta):
    return r*np.cos(theta)
def c2(r,theta):
    return r*np.sin(theta)
plt.plot(c1(rad_max,theta),c2(rad_max,theta),color='grey',linewidth=3, alpha=0.5)
bolovessel=patches.Rectangle((60+21.8,-12),20.8,24,edgecolor='grey',facecolor='none',linewidth=3, alpha=0.5)
plt.plot([a-b-15.5,a-b-10.3],[-12.5,-12.5],[a-b-15.5,a-b-10.3],[12.5,12.5],[a-b-10.3,a-b-10.3],[-12.5,12.5],color='grey',linewidth=3,alpha=0.5)
bolohead1=patches.Rectangle((a,-1.65),2,3.3,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5)
ax.add_patch(bolovessel)
ax.add_patch(bolohead1)

plt.xlim(60,100)
plt.ylim(-20,20)
fig1= plt.gcf()
plt.show()
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/lines_of_sight_measurement_horizontal.pdf',bbox_inches='tight')
# %% Lines of sight calculation
alpha=13
b=3.45 #Distance of Bolometerhead Middle to  Slit [cm]
zoom=False
fig=plt.figure(figsize=(width,width))
plt.rc('xtick')
plt.rc('ytick')
ax=fig.add_subplot(111)
x=np.arange(40,a,0.1)
plt.xlabel('R [cm]')
plt.ylabel('z [cm]')
f1=0.14 #Distance first channel to edge [cm]
f2=0.40 #Distance between channels [cm]
h=[-2+f1,-2+f1+c_h,-2+f1+c_h+f2,-2+f1+c_h*2+f2,-2+f1+c_h*2+f2*2,-2+f1+c_h*3+f2*2,-2+f1+c_h*3+f2*3,-2+f1+c_h*4+f2*3,f1,f1+c_h,f1+c_h+f2,f1+c_h*2+f2,f1+c_h*2+f2*2,f1+c_h*3+f2*2,f1+c_h*3+f2*3,f1+c_h*4+f2*3,f1*2+c_h*4+f2*3]
x_b=[]
y_b=[]
for i in h[0:8]:
    x_b.append(-abs(np.sin((alpha)*np.pi/180)*i)+a+c_d)
    y_b.append(-np.cos((alpha)*np.pi/180)*i+np.sin(13*np.pi/180)*c_d)
for i in h[8:16]:
    x_b.append(-abs(np.sin((alpha)*np.pi/180)*i)+a+c_d)
    y_b.append(-np.cos((alpha)*np.pi/180)*i-np.sin(13*np.pi/180)*c_d)

def lin(x,d,e):
    return d*x+e

lines=[0,2,4,6,8,10,12,14]
i=0
#variation of slit distance
for j,var in zip(colors2[12:-1],[0.1,0.2,-0.1,-0.2]):
    plt.plot([x_b[i],x_b[i+1]],[y_b[i],y_b[i+1]],color=j)
    b=b*(1-var)
    popt1,pcov1=curve_fit(lin,[x_b[i],a-b],[y_b[i],-s_h/2])
    popt2,pcov2=curve_fit(lin,[x_b[i+1],a-b],[y_b[i+1],s_h/2])
    plt.plot(np.arange(40,x_b[i],0.01),lin(np.arange(40,x_b[i],0.01),*popt1),color=j,linestyle='dashed')
    plt.plot(np.arange(40,x_b[i+1],0.01),lin(np.arange(40,x_b[i+1],0.01),*popt2),color=j,linestyle='dashed')
    plt.plot([a-b,a-b],[-12,-s_h/2],[a-b,a-b],[12,s_h/2],color=j,linewidth=2,alpha=0.5,linestyle='dashed')
    print(var,np.arcsin((lin(a-b,*popt1)-lin(80,*popt1))/(a-b-80))*180/np.pi)
i=14
if zoom==True:
    vars=[-0.2,0,0.5]
else:
    vars=[-0.3,0,0.3]
#variation of sensor height
for j,var in zip(colors2[8:-1],vars):
    delta_x=-(np.sin((alpha)*np.pi/180)*var*c_h)
    delta_y=-(np.cos((alpha)*np.pi/180)*var*c_h)
    plt.plot([x_b[i]-delta_x,x_b[i+1]+delta_x],[y_b[i]-delta_y,y_b[i+1]+delta_y],color=j)
    popt1,pcov1=curve_fit(lin,[x_b[i]-delta_x,a-b],[y_b[i]-delta_y,-s_h/2])
    popt2,pcov2=curve_fit(lin,[x_b[i+1]+delta_x,a-b],[y_b[i+1]+delta_y,s_h/2])
    plt.plot(np.arange(40,x_b[i]-delta_x,0.01),lin(np.arange(40,x_b[i]-delta_x,0.01),*popt1),color=j,linestyle='dashed')
    plt.plot(np.arange(40,x_b[i+1]+delta_x,0.01),lin(np.arange(40,x_b[i+1]+delta_x,0.01),*popt2),color=j,linestyle='dashed')


#slit
bolovessel=patches.Rectangle((60+21.8,-12),20.8,24,edgecolor='grey',facecolor='none',linewidth=3, alpha=0.5)
#bolometerhead
ts=ax.transData
coords1=[-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2]
coords2=[-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0]
tr1 = matplotlib.transforms.Affine2D().rotate_deg_around(coords1[0],coords1[1], -alpha)
tr2 = matplotlib.transforms.Affine2D().rotate_deg_around(coords2[0],coords2[1],alpha)
bolohead1=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr1+ts)
bolohead2=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr2+ts)

ax.add_patch(bolovessel)
ax.add_patch(bolohead1)
ax.add_patch(bolohead2)

if zoom==True:
    plt.xlim(a-2,a+2)
    plt.ylim(-2,2)
else:
    plt.xlim(70,100)
    plt.ylim(-15,15)

fig1= plt.gcf()
plt.show()
if zoom==True:
    fig1.savefig('/home/gediz/LaTex/Thesis/Figures/lines_of_sight_calculation_zoom.pdf',bbox_inches='tight')
else:
    fig1.savefig('/home/gediz/LaTex/Thesis/Figures/lines_of_sight_calculation.pdf',bbox_inches='tight')


# %%  Fluxsurfaces and Temperature, Density
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
t=17.5 #radius of vessel [cm]
gas='H'
fig=plt.figure(figsize=(width,height))

ax_t_f=fig.add_subplot(122)
ax_t=ax_t_f.twinx()
ax_d=fig.add_subplot(121)
ax_d_f=ax_d.twinx()
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position_extended.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii_extended.csv',sep=',',engine='python')
a=0
colors=['#1bbbe9','#023047','#ffb703','#c1121f','#780000']
markers=['o','v','s','P','p','D']
#fluxsurfaces
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
    x=[u-60 for u in np.array(x_.iloc[i])]
    y=np.array(y_.iloc[i])
    ax_d_f.plot(np.append(x,x[0]),np.append(y,y[0]),color='grey',linewidth=5,alpha=0.2)
    ax_t_f.plot(np.append(x,x[0]),np.append(y,y[0]),color='grey',linewidth=5,alpha=0.2)

for shot,m,c in zip((13090,13095,13096,13097),markers,colors):
    P_t,T=pc.TemperatureProfile(shot,'Values')[0],pc.TemperatureProfile(shot,'Values')[1]
    P_d,D=pc.DensityProfile(shot,'Values')
    print(D)
    d_mean=pc.CorrectedDensityProfile(shot)[1]*3.88E17/2
    b=[0,5,10,15,20,25]
    l='shot n°{s}, P$_M$$_W$= {m} W'.format(s=shot,m=float(f'{br.GetMicrowavePower(shot)[0]:.1f}'))
    ax_t.plot(P_t*100, T,linewidth=3,color=c,marker=m,markersize=5)
    ax_t.plot([e for i,e in enumerate(P_t*100) if i in b],[e for i,e in enumerate(T) if i in b],marker=m,linestyle='None',color=c,markersize=18,label=l)
    ax_d.plot(P_d*100,D,marker=m,markersize=5,color=c,linewidth=3)#,(P_d-P_d[-1]+P_d[0])*100,np.flip(D)
    ax_d.plot([e for i,e in enumerate(P_d*100) if i in b],[e for i,e in enumerate(D) if i in b],marker=m,linestyle='None',color=c,markersize=18)
    ax_d.hlines(d_mean,P_d[0]*100,P_d[-1]*100,color=c)
ax_d.set_xlabel('R - R$_0$ [cm]',fontsize=25)
ax_t_f.set_xlabel('R - R$_0$ [cm]',fontsize=25)
ax_t.set_xlim(3,20)
ax_d.set_xlim(3,20)
ax_d_f.set_ylim(-13,13)
ax_t_f.set_ylim(-13,13)
ax_t.set_ylim(0)

ax_d.set_ylabel('density [m$^-$$^3$]',fontsize=25)
ax_d.set_yscale('log')
ax_t.set_ylabel('temperature [eV]',fontsize=25)

ax_d_f.set_yticks([])
ax_t_f.set_yticks([])



ax_t.legend(loc='lower center',bbox_to_anchor=(0,-0.62),title=r'H, p$\approx$ 7.5 mPa')


fig1= plt.gcf()
plt.show()
#fig1.savefig('/home/gediz/LaTex/Thesis/Figures/fluxsurfaces_with_temperatureprofiles.pdf',bbox_inches='tight')

# %%  LR, RR, BR
T=60
T_LR,He_0_LR,He_1_LR=adas.he_adf11(data='plt96_he',T_max=T)[0],adas.he_adf11(data='plt96_he',T_max=T)[1],adas.he_adf11(data='plt96_he',T_max=T)[2]
T_BR_RR,He_0_BR_RR,He_1_BR_RR=adas.he_adf11(data='prb96_he',T_max=T)[0],adas.he_adf11(data='prb96_he',T_max=T)[1],adas.he_adf11(data='prb96_he',T_max=T)[2]
T_RR_0,He_0_RR=adas.he_adf15(data='pec96#he_pju#he0',T_max=T)[0],adas.he_adf15(data='pec96#he_pju#he0',T_max=T)[1]
T_RR_1,He_1_RR=adas.he_adf15(data='pec96#he_pju#he1',T_max=T)[0],adas.he_adf15(data='pec96#he_pju#he1',T_max=T)[1]

plt.figure(figsize=(width*0.7,height))
plt.plot(T_LR,He_0_LR,color=colors[0],marker=markers[0],label='He$^0$, excitation')
plt.plot(T_LR,He_1_LR,color=colors[0],marker=markers[0],label='He${^+1}$, excitation',alpha=0.5)

plt.plot(T_RR_0,He_0_RR,color=colors[3],marker=markers[3],label=r'He$^{+1} \rightarrow$ He$^0$, recombination')
plt.plot(T_RR_1,He_1_RR,color=colors[3],marker=markers[3],label=r'He$^{+2} \rightarrow$ He$^{+1}$, recombination',alpha=0.5)

#plt.plot(T_BR_RR,He_0_BR_RR,color=colors[1],marker=markers[1],label='He$^0$,Bremsstrahlung and recombination, source: ADF11 ')
#plt.plot(T_BR_RR,He_1_BR_RR,color=colors[1],marker=markers[1],label='He$^{+1}$,Bremsstrahlung and recombination, source: ADF11 ',alpha=0.5)

plt.ylabel('radiation energy coefficient [eVm$^3$/s]')
plt.xlabel('$T_{\mathrm{e}}$ [eV]')
plt.yscale('log')
plt.ylim(1E-20,1E-12)
plt.xlim(-1,30)
plt.legend(loc='right center',bbox_to_anchor=(1,0.5))
fig1= plt.gcf()
plt.show()
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/lt_br_rr.pdf',bbox_inches='tight')



# %% Fluxsurfaces at different angles
t=17.5 #radius of vessel [cm]

x_50=np.genfromtxt('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle50_position.txt')
y_50=np.genfromtxt('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle50_radii.txt')
x_10=np.genfromtxt('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle10_position.txt')
y_10=np.genfromtxt('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle10_radii.txt')
x_0=np.genfromtxt('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle0_position.txt')
y_0=np.genfromtxt('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle0_radii.txt')
x_30=np.genfromtxt('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.txt')
y_30=np.genfromtxt('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.txt')

size=width/4 
fig0=plt.figure(figsize=(size,size))
ax0=fig0.add_subplot(111)
for i in [0,1,2,3,4,5,6,7,8,9]:
    plt.plot(np.append(x_0[i],x_0[i][0]),np.append(y_0[i],y_0[i][0]),color=colors[0])
vessel=patches.Arc((60,0),2*t,2*t,theta1=195,theta2=165,fill=False,color='grey',linewidth=3,alpha=0.5)
plt.plot([40,42.5],[-4.75,-4.75],[40,42.5],[4.75,4.75],color='grey',linewidth=3,alpha=0.5)
ax0.add_patch(vessel)
plt.xlim(40,80)
plt.ylim(-20,20)
#plt.ylabel('Z [cm]')
plt.xlabel('R [cm]')
fig0= plt.gcf()
plt.show()
#fig0.savefig('/home/gediz/LaTex/Thesis/Figures/inner_port.pdf',bbox_inches='tight')

fig1=plt.figure(figsize=(size,size))
ax1=fig1.add_subplot(111)
for i in [0,1,2,3,4,5,6,7,8,9]:
    plt.plot(np.append(x_10[i],x_10[i][0]),np.append(y_10[i],y_10[i][0]),color=colors[1])
vessel=patches.Arc((60,0),2*t,2*t,theta1=135,theta2=45,fill=False,color='grey',linewidth=3,alpha=0.5)
plt.plot([60-12.5,60-12.5],[12.5,20],[60+12.5,60+12.5],[12.5,20],color='grey',linewidth=3,alpha=0.5)
ax1.add_patch(vessel)
plt.xlim(40,80)
plt.ylim(-20,20)
#plt.ylabel('Z [cm]')
plt.xlabel('R [cm]')
fig1= plt.gcf()
plt.show()
#fig1.savefig('/home/gediz/LaTex/Thesis/Figures/upper_port.pdf',bbox_inches='tight')

fig2=plt.figure(figsize=(size,size))
ax2=fig2.add_subplot(111)
for i in [0,1,2,3,4,5,6,7,8,9]:
    plt.plot(np.append(x_30[i],x_30[i][0]),np.append(y_30[i],y_30[i][0]),color=colors[2])
vessel=patches.Arc((60,0),2*t,2*t,theta1=45,theta2=315,fill=False,color='grey',linewidth=3,alpha=0.5)
plt.plot([72.5,80],[-12.5,-12.5],[72.5,80],[12.5,12.5],color='grey',linewidth=3,alpha=0.5)
ax2.add_patch(vessel)
plt.xlim(40,80)
plt.ylim(-20,20)
plt.ylabel('Z [cm]')
plt.xlabel('R [cm]')
fig2= plt.gcf()
plt.show()
#fig2.savefig('/home/gediz/LaTex/Thesis/Figures/outer_port.pdf',bbox_inches='tight')

fig3=plt.figure(figsize=(size,size))
ax3=fig3.add_subplot(111)
for i in [0,1,2,3,4,5,6,7,8,9]:
    plt.plot(np.append(x_50[i],x_50[i][0]),np.append(y_50[i],y_50[i][0]),color=colors[3])
vessel=patches.Arc((60,0),2*t,2*t,theta1=315,theta2=225,fill=False,color='grey',linewidth=3,alpha=0.5)
plt.plot([60-12.5,60-12.5],[-12.5,-20],[60+12.5,60+12.5],[-12.5,-20],color='grey',linewidth=3,alpha=0.5)
ax3.add_patch(vessel)
plt.xlim(40,80)
plt.ylim(-20,20)
#plt.ylabel('Z [cm]')
plt.xlabel('R [cm]')
fig3= plt.gcf()
plt.show()
#fig3.savefig('/home/gediz/LaTex/Thesis/Figures/bottom_port.pdf',bbox_inches='tight')
# %% I-V curve
fig,ax=plt.subplots(figsize=(width,height))
fig.patch.set_visible(False)
ax.axis('off')
lw=2
ls='-.'
extra='#1282A3'
x=np.arange(-10,10,0.01)
def characteristic(x,Isat,e):
    return Isat*(1-np.exp(-(e-x)/5))
Isat=-10
e=-2
plt.plot(x,characteristic(x,Isat,e),linewidth=lw,color=colors[1])
x2=np.arange(10,15,0.01)
def probe1(x,a):
    return a+np.sqrt((x-10)*100)
def probe2(x,a):
    return a+np.sqrt((x-10)*0.9)
def probe3(x,Isat,e):
    return Isat*(1-np.exp(-(e-x)/5))
a=characteristic(x,Isat,e)[-1]
plt.plot(x2,probe2(x2,a),linewidth=lw,linestyle=ls,color=colors[3],label='planar')
plt.plot(x2,probe1(x2,a),linewidth=lw,linestyle=ls,color=colors[8],label='cylindrical')
plt.plot(x2,probe3(x2,Isat,e),linewidth=lw,linestyle=ls,color=colors[4],label='spherical')
plt.ylim(-20,150)
plt.xlim(-10,15)
plt.arrow(-10,0,24,0,head_width=5,head_length=1,fill=False)
plt.arrow(0,-20,0,160,head_width=0.5,head_length=10,fill=False)
ax.axvspan(-10,-2,alpha=0.3,color=colors[0])
ax.axvspan(-1.9,10,alpha=0.2,color=colors[0])
ax.axvspan(10.1,15,alpha=0.1,color=colors[0])
plt.annotate('$\phi$$_{\mathrm{fl}}$',[-2.4,3],fontsize=11)
plt.annotate('$\phi$$_{\mathrm{p}}$',[9.5,3],fontsize=11)
plt.annotate('I$_{\mathrm{i, sat}}$',[0.1,-10],fontsize=11)
plt.annotate('I$_{\mathrm{e, sat}}$',[0.1,105],fontsize=11)
plt.annotate('-I [A]',[0.5,140],fontsize=11)
plt.annotate('U [V]',[13,6],fontsize=11)
plt.hlines(103,0,15,linestyle='dotted')
plt.hlines(-10,-10,0,linestyle='dotted')
plt.vlines(-2,-20,3,linestyle='dotted')
plt.vlines(10,-20,3,linestyle='dotted')
plt.annotate('ion \n acceleration \n region',[-6,50],fontsize=11,color=extra,ha='center')
plt.annotate('transition \n region',[4,50],fontsize=11,color=extra,ha='center')
plt.annotate('electron \n acceleration \n region',[12.5,50],fontsize=11,color=extra,ha='center')

plt.legend(loc='upper right',bbox_to_anchor=(1.31,1),title='probe shape')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/i_v_curve.pdf',bbox_inches='tight')

# %% I-V fit
fig=plt.figure(figsize=(width/2,height))
ax=fig.add_subplot(111)
x,y,z=np.genfromtxt('/data2/shot6421/kennlinien/000001.dat',unpack=True)
l=5120
def fit(x,a,b,c):
   return a*(1-np.exp(-(c-x)/b))
for i in np.arange(0,10):
    plt.plot(x[0+i*l:l+i*l]*24,-y[0+i*l:l+i*l]*10,marker='x',color=colors[i],alpha=0.05,linestyle='None')#,markersize=6-0.5*i)
s=3800
e=4600
popt,pcov=curve_fit(fit,x[s:e]*24,-y[s:e]*10)
plt.plot(x[s:e]*24,fit(x[s:e]*24,*popt)-0.2,color='red',lw=3)
plt.plot(x[s:e]*24,fit(x[s:e]*24,*popt)*0.8-0.2,color='red',lw=1)
plt.plot(x[s:e]*24,fit(x[s:e]*24,*popt)*1.2-0.2,color='red',lw=1)

#plt.xlim(-10,50)
#plt.ylim(-1,10)
plt.xlim(6.9,14.1)
plt.ylim(-0.15,4.95)
plt.xlabel('$U$ [V]')
plt.ylabel('$-I$ [mA]')
plt.hlines(0,-100,100,lw=1,alpha=0.5)
plt.vlines(0,-2,20,lw=1,alpha=0.5)
zoom=patches.Rectangle((-5,-0.9),25,8,edgecolor=colors[0],facecolor='none',linewidth=3,alpha=0.7,zorder=20)
ax.add_patch(zoom)
zoom2=patches.Rectangle((7,-0.1),7,5,edgecolor=colors[2],facecolor='none',linewidth=3,alpha=0.7,zorder=20)
ax.add_patch(zoom2)

fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/i_v_curve_fit_3.pdf',bbox_inches='tight')


# %% Density Profile procedure
s=13252
Position, char_U, char_I, I_isat,Bolo_sum, Interferometer=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=s),unpack=True)
p_t,T=pc.TemperatureProfile(s,'Values','Power')[0],pc.TemperatureProfile(s,'Values','Power')[1]
location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=s)
time, inter_original=pc.LoadData(location)['Zeit [ms]'] / 1000,pc.LoadData(location)['Interferometer digital']
inter=savgol_filter(inter_original,100,3)
fig=plt.figure(figsize=(width/2,height))
ax=fig.add_subplot(111)
fig.patch.set_facecolor('white')
ax2=ax.twinx()
ax2.plot(Position*100+z_0, Interferometer, marker='o',color=colors[1])
ax.plot(Position*100+z_0, I_isat, marker='o',color=colors[0])
ax.set_xlabel('R [cm]')
ax.set_ylabel('$I_{\mathrm{i, sat}}$ [mA]',color=colors[0])
ax2.set_ylabel('$U_{\mathrm{inter}}$ [V]',color=colors[1])
ax.tick_params(axis='y', labelcolor=colors[0])
ax2.tick_params(axis='y', labelcolor=colors[1])
ax2.set_ylim(0)
fig= plt.gcf()
plt.show()
# fig.savefig('/home/gediz/LaTex/Thesis/Figures/Isat_Iinter.pdf',bbox_inches='tight')

#fig1=plt.figure(figsize=(width/2,height))
# plt.axvspan(25,55,facecolor=colors2[6], alpha=0.5)
# plt.axvspan(56,83,facecolor=colors2[12], alpha=0.5)
# plt.plot(time,inter_original,color=colors[1],alpha=0.5,label='original')
# plt.plot(time,inter,color=colors[1],label='Savitzky-Golay filter ')
# plt.annotate('',xy=(70,0.95),xytext=(70,0.25), arrowprops=dict(arrowstyle='<->',color=colors[1],linewidth=2))
# plt.annotate('$\Delta U_{\mathrm{inter}}$',xy=(72,0.6),color=colors[1])

# plt.xlabel('time [s]')
# plt.ylabel('$U_{\mathrm{inter}}$ [V]')
# plt.ylim(0)
# plt.legend(loc='lower center')
# fig1= plt.gcf()
# plt.show()
#fig1.savefig('/home/gediz/LaTex/Thesis/Figures/Interferometer.pdf',bbox_inches='tight')

# fig2=plt.figure(figsize=(width,height))
# d=(pc.CorrectedDensityProfile(s)[1]*3.88E17)/2
# Position,I_isat=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=s),unpack=True,usecols=(0,3))
# norm_origin=integrate.trapezoid(I_isat,Position)/abs(Position[-1]-Position[0])
# Density_origin=[u*d/norm_origin for u in I_isat]
# norm_T=integrate.trapezoid(pc.CorrectedDensityProfile(s)[0],Position)/abs(Position[-1]-Position[0])
# Density_T=[u*d/norm_T for u in pc.CorrectedDensityProfile(s)[0]]
# Temperature=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}Te.dat'.format(s=s),usecols=1,unpack=True)
# norm=integrate.trapezoid([a*np.sqrt(b) for a,b in zip(pc.CorrectedDensityProfile(s)[0],Temperature)],Position)/abs(Position[-1]-Position[0])
# Density=[u*d/norm for u in [a*np.sqrt(b) for a,b in zip(pc.CorrectedDensityProfile(s)[0],Temperature)]]
# I_isat_fit=np.genfromtxt('/data6/shot{s}/kennlinien/auswert/shot{s}ne.dat'.format(s=s),usecols=1,unpack=True)
# norm_fit=integrate.trapezoid(I_isat_fit,Position)/abs(Position[-1]-Position[0])
# Density_fit=[u*d/norm_fit for u in I_isat_fit]
# plt.plot(Position*100+60,Density_origin,marker=markers[0],color=colors2[0],label='from $I_{\mathrm{i, sat}}$, uncorrected ')
# plt.plot(Position*100+60,Density,marker=markers[1],color=colors2[1],label='from $I_{\mathrm{i, sat}}$, corrected \n for interferometer dip ')
# plt.plot(Position*100+60, Density_T,marker=markers[2],color=colors2[2],label='from $I_{\mathrm{i, sat}}$, corrected \n for interferometer dip \n and temperature ')
# plt.plot(Position*100+60, Density_fit,marker=markers[3],color=colors2[4],label='from I-V curve fit,\n uncorrected ')        
# plt.xlabel('R [cm]')
# plt.ylabel('$n_\mathrm{e}(R)$ [m$^{-3}$]')
# plt.legend(loc='upper right')    
# fig2= plt.gcf()
# plt.show()
# fig2.savefig('/home/gediz/LaTex/Thesis/Figures/Density_procedure.pdf',bbox_inches='tight')

# %% Gold Absorption

h1=4.135E-15
c=299792458
fig,ax=plt.subplots(figsize=(width,height))
l,R=np.genfromtxt('/home/gediz/Results/Goldfoil_Absorption/gold_abs_Anne.txt',unpack=True)
E_P,n_P,k_P=np.genfromtxt('/home/gediz/Results/Goldfoil_Absorption/Gold_Palik.txt',unpack=True,usecols=(0,1,2),delimiter=',')
E_F,n_F,k_F,R_F=np.genfromtxt('/home/gediz/Results/Goldfoil_Absorption/Gold_Foiles.txt',unpack=True,usecols=(0,1,2,3),delimiter=',')
E_O,n_O,k_O=np.genfromtxt('/home/gediz/Results/Goldfoil_Absorption/Gold_Ordal.txt',unpack=True,usecols=(0,1,2),delimiter=',')
E_H,R_H=np.genfromtxt('/home/gediz/Results/Goldfoil_Absorption/Gold_Hagemann.txt',unpack=True)
#ax.semilogx(np.flip(E_P),[1-(((a-1)**2+(b-1)**2)/((1+a)**2+(1+b)**2)) for a,b in zip(n_P,k_P)],'gx')
# ax.semilogx(E_P,n_P,'ro',alpha=0.1)
# ax.semilogx(E_P,k_P,'bo',alpha=0.1)
# ax.semilogx(E_F,n_F,'rx',alpha=0.7)
# ax.semilogx(E_F,k_F,'bx',alpha=0.7)
# ax.semilogx([h*c/(x*10**(-6)) for x in E_O],n_O,'rs')
# ax.semilogx([h*c/(x*10**(-6)) for x in E_O],k_O,'bs')
i=0

energy=np.arange(10E-4,10E5,0.1)
all_energy= list(E_H)+list(E_F)#+[(h1*c)/(x*10**(-9)) for x in l]
all_abs=list(100-R_H)+list(100-R_F*100)#+list(R*100)
all_energy,all_abs=(list(t) for t in zip(*sorted(zip(all_energy,all_abs))))
fitted=pchip_interpolate(all_energy,all_abs,energy)
ax.semilogx(E_F,100-R_F*100,marker='o',color=colors2[i],alpha=0.7,ls='None',label='Foiles (1985)')
ax.semilogx(E_H,100-R_H,marker='s',color=colors2[i+2],alpha=0.7,ls='None',label='Hagemann (1974)')
ax.semilogx([(h1*c)/(x*10**(-9)) for x in l],R*100,marker='d',color=colors2[i+1],alpha=0.7,ls='None',label='Palik (1985)')
ax.semilogx(energy,fitted, color=colors2[i+5],label='fit to all data')
ax.set_xlim(0.0011,1400)
ax.minorticks_off()
ax2 = ax.twiny()

newlabel = ['$10^6$','$10^5$','$10^4$','$10^3$','$10^2$','$10^1$','$10^0$','$10^{-1}$','$10^{-2}$']
ax2.set_xscale('log')
ax2.minorticks_off()
newpos=[0.00123964181383,0.0123964181383,0.123964181383,1.23964181383,12.3964181383,123.964181383,1239.64181383,12396.4181383,123964.181383]
#e2l=lambda x: ((h1*c)/(x))/10**(-9)
#newlabel=[e2l(x) for x in newpos]
ax2.set_xticks(newpos)
ax2.set_xticklabels(newlabel)
ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
ax2.spines['bottom'].set_position(('outward', 36))
ax2.set_xlabel('wavelength [nm]')
ax2.set_xlim(ax.get_xlim())
ax.axvspan(1.6528557517733333,3.099104534575,facecolor=colors2[6], alpha=0.5)
ax.annotate('visible\nlight',xy=(1.5,70),xytext=(0.05,80),color=colors2[5],arrowprops=dict(facecolor=colors2[6],edgecolor='none',width=1,headwidth=5))
ax.set_xlabel('photon energy [eV]')
ax.set_ylabel('absorption [\%]')
ax.legend(loc='lower right')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/Gold_Absorption.pdf',bbox_inches='tight')

# plt.figure()
# #plt.xscale('log')
# plt.plot(energy,fitted)
# plt.show()
# plt.figure()
# #plt.xscale('log')
# l=lambda x: ((h1*c)/x)*10**(-9)
# plt.plot([l(a) for a in energy],fitted)
# plt.show()


# %%  Ohmic Calibrations Signal
time,U_sq,U_b_n,U_b= np.genfromtxt('/home/gediz/Measurements/Calibration/Ohmic_Calibration/Ohmic_Calibration_Vacuum_November/10_11_2022/NewFile20.csv',delimiter=',',unpack=True, usecols=(0,1,2,3),skip_header=2)
fig,ax1=plt.subplots(figsize=(height,height))
fig.patch.set_facecolor('white')
ax2=ax1.twinx()
def I_func(t,I_0, Delta_I, tau):
    return I_0+Delta_I*(1-np.exp(-t/tau))
I_b=U_b_n/100
ref1=20
ref2=60
start= np.argmax(np.gradient(U_sq, time))+2
stop= np.argmin(np.gradient(U_sq, time))-2
lns1=ax1.plot(time[start-ref1:stop+ref2], U_sq[start-ref1:stop+ref2],color=colors2[12],label='square pulse')
lns2=ax2.plot(time[start-ref1:stop+ref2],U_b_n[start-ref1:stop+ref2]*10,color=colors2[5],label='bolometer response')
ax1.set_ylabel('$U_{\mathrm{square}}$ [V]',color=colors2[12])
ax2.set_ylabel('$I_{\mathrm{meas.}}$ [mA]',color=colors2[5])
ax1.set(xlabel='time [s]')
ax1.tick_params(axis='y', labelcolor=colors2[12])
ax2.tick_params(axis='y', labelcolor=colors2[5])

time_cut=time[start:stop]-time[start]
I_b_cut= I_b[start:stop]*1000
popt, pcov = curve_fit(I_func, time_cut,I_b_cut)
lns3=ax2.plot(time_cut, I_func(time_cut, *popt),lw=3,color=colors2[10], label='exponential fit')
plt.hlines(I_b_cut[0],time_cut[0],time_cut[90],color=colors2[10],ls='dashed')
plt.annotate('$I (0)$',(time_cut[100],I_b_cut[0]-0.02),color=colors2[10])
plt.hlines(I_b_cut[-5],time_cut[-1],time_cut[-1]+0.07,color=colors2[10],ls='dashed')
plt.annotate('$I (\infty)$',(time_cut[-1]+0.07,I_b_cut[-5]-0.02),color=colors2[10])

leg = lns1 + lns2 +lns3
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc='lower center',bbox_to_anchor=(0.45,0))

fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/ohmic_calibration.pdf',bbox_inches='tight')

# %%  Ohmic Calibrations Tau and Kappa
x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10 = ([] for i in range(30))
for i,j,k,n in zip([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10],[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10],[k1,k2,k3,k4,k5,k6,k7,k8,k9,k10],[0,1,2,3,4,5,6,7,8,9]):
    i.append(np.genfromtxt('/home/gediz/Results/Calibration/Ohmic_Calibration/Ohmic_Calibration_Vacuum_November/ohmic_calibration_vacuum_tjk_tau_and_kappa_reduced_noise_measurement_0{}.txt'.format(n), unpack=True, usecols=(0)))
    j.append(np.genfromtxt('/home/gediz/Results/Calibration/Ohmic_Calibration/Ohmic_Calibration_Vacuum_November/ohmic_calibration_vacuum_tjk_tau_and_kappa_reduced_noise_measurement_0{}.txt'.format(n), unpack=True, usecols=(1)))
    k.append(np.genfromtxt('/home/gediz/Results/Calibration/Ohmic_Calibration/Ohmic_Calibration_Vacuum_November/ohmic_calibration_vacuum_tjk_tau_and_kappa_reduced_noise_measurement_0{}.txt'.format(n), unpack=True, usecols=(2))) 
mean_t,sd_t,sem_t=[],[],[]
fig1=plt.figure(figsize=(width/2,height*0.7))
for i,j,n in zip([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10],[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10],[0,1,2,3,4,5,6,7,8,9]):
    plt.plot(i,j,label='Measurement {} TJ-K'.format(n),marker='o',color=colors[3],alpha=0.3)
for m in [0,1,2,3,4,5,6,7]:
    val=[t1[0][m],t2[0][m],t3[0][m],t4[0][m],t5[0][m],t6[0][m],t7[0][m],t8[0][m],t9[0][m],t10[0][m]]
    mean_t.append(np.mean(val))
    sd_t.append(np.std(val,ddof=1))
    sem_t.append(np.std(val,ddof=1)/np.sqrt(len(val)))
    plt.errorbar(m+1,mean_t[m],yerr=sem_t[m],marker='o',linestyle='None', capsize=5,color=colors[4])
plt.xlabel('sensor number')
plt.ylabel(r'$\tau$ [s]')
plt.xticks([1,2,3,4,5,6,7,8])
fig1=plt.gcf()
plt.show()
print(mean_t,sem_t)
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/ohmic_calibration_tau.pdf',bbox_inches='tight')

fig2=plt.figure(figsize=(width/2,height*0.7))
mean_k,sd_k,sem_k=[],[],[]
for i,k,n in zip([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10],[k1,k2,k3,k4,k5,k6,k7,k8,k9,k10],[0,1,2,3,4,5,6,7,8,9]):
    plt.plot(i,k,label='Measurement {} TJ-K'.format(n),marker='o',color=colors[0],alpha=0.3)
for m in [0,1,2,3,4,5,6,7]:
    val=[k1[0][m],k2[0][m],k3[0][m],k4[0][m],k5[0][m],k6[0][m],k7[0][m],k8[0][m],k9[0][m],k10[0][m]]
    mean_k.append(np.mean(val))
    sd_k.append(np.std(val,ddof=1))
    sem_k.append(np.std(val,ddof=1)/np.sqrt(len(val)))
    plt.errorbar(m+1,mean_k[m],yerr=sem_k[m],marker='o',linestyle='None', capsize=5,color=colors[1])
plt.xlabel('sensor number')
plt.ylabel('$\kappa$ [W]')
plt.xticks([1,2,3,4,5,6,7,8])
fig2=plt.gcf()
plt.show()
print(mean_k,sem_k)
fig2.savefig('/home/gediz/LaTex/Thesis/Figures/ohmic_calibration_kappa.pdf',bbox_inches='tight')


# %% Laser Scan
plt.figure(figsize=(width*0.7,height*0.8))
location='/home/gediz/Measurements/Lines_of_sight/shot_data/shot60038.dat'
cut=1000
time = br.LoadData(location)['Zeit [ms]'][cut:-1] / 1000
for i in [1,2,3,4,5,6,7,8]:
    y= br.LoadData(location)["Bolo{}".format(i)][cut:-1]
    background=np.mean(y[-500:-1])
    y_1=y-background
    plt.plot(time/3.73,-y_1,color=colors2[i+1], label='sensor {}'.format(i))
plt.xlabel('y [mm]')
plt.ylabel('sensor signal [V]')
plt.annotate('1',(22/3.73,0.01),xytext=(15/3.73,0.35),arrowprops=dict(facecolor='#008e0c',edgecolor='none',alpha=0.5,width=1,headwidth=5), bbox={"boxstyle" : "circle","facecolor":'None','edgecolor':'#008e0c'},color='#008e0c')
plt.annotate('2',(35/3.73,0.01),xytext=(32/3.73,0.35),arrowprops=dict(facecolor='#008e0c',edgecolor='none',alpha=0.5,width=1,headwidth=5), bbox={"boxstyle" : "circle","facecolor":'None','edgecolor':'#008e0c'},color='#008e0c')

plt.legend(fontsize=9,loc='center right',bbox_to_anchor=(1.32,0.5))
plt.xlim(0,33)
plt.ylim(-0.05,0.65)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/Laser_scan.pdf',bbox_inches='tight')

# %% UV scan
plt.figure(figsize=(width/2,height*0.8))
location='/home/gediz/Measurements/Lines_of_sight/shot_data/shot60078_cropped.dat'
cut=0
c=0
con=0.77
time = (br.LoadData(location)['Zeit [ms]'][cut:-1] / 1000)*con
def lin (x,a,b):
    return a*x + b
for i in [1,2,3,4]:
    y0= br.LoadData(location)["Bolo{}".format(i)][cut:-1]
    background=np.mean(y0[-500:-1])
    y=savgol_filter(y0-background,1000,3)
    steps=[]
    for j in np.arange(0, len(y)-1000):
        step= (y[j]-y[j+1000])
        steps.append(abs(step))
    start=(np.argwhere(np.array([steps])>0.005)[0][1])
    stop=(np.argwhere(np.array([steps])>0.005)[-1][1])
    background_x = np.concatenate((time[0:start],time[stop:-1]))
    background_y=np.concatenate((y[0:start],y[stop:-1]))
    popt,pcov=curve_fit(lin,background_x,background_y)
    amp_origin=list((y[j]-lin(time[j],*popt))*(-1) for j in np.arange(0,len(y)))
    maximum=max(amp_origin)
    amp=list(amp_origin[j]/maximum for j in np.arange(0,len(y)))
    plt.plot(time,  amp ,color=colors2[i+c], label='sensor {}'.format(i),alpha=0.7)
    signal_edge=np.argwhere(amp>max(amp)/np.e)
    fwhm1, fwhm2=time[cut+int(signal_edge[0])],time[cut+int(signal_edge[-1])]
    plt.plot(fwhm1,amp[int(signal_edge[0])],'o',color=colors2[i+c])
    plt.plot(fwhm2,amp[int(signal_edge[-1])],'o',color=colors2[i+c])
    fwhm=float(fwhm2-fwhm1)
    plt.plot([fwhm1,fwhm2],[amp[int(signal_edge[0])],amp[int(signal_edge[-1])]], color=colors2[i+c])

plt.xlabel('x [mm]')
plt.ylabel('sensor signal [V]')
#plt.legend(loc='upper right',fontsize=9)
plt.xlim(30,180)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/UV_scan_horizontal.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,height*0.8))
location='/home/gediz/Measurements/Lines_of_sight/shot_data/shot60070_cropped.dat'
cut=0
c=0
con=0.37
time = (br.LoadData(location)['Zeit [ms]'][cut:-1] / 1000)*con
def lin (x,a,b):
    return a*x + b
for i in [1,2,3,4,5,6,7,8]:
    y0= br.LoadData(location)["Bolo{}".format(i)][cut:-1]
    background=np.mean(y0[-500:-1])
    y=savgol_filter(y0-background,1000,3)
    steps=[]
    for j in np.arange(0, len(y)-1000):
        step= (y[j]-y[j+1000])
        steps.append(abs(step))
    start=(np.argwhere(np.array([steps])>0.005)[0][1])
    stop=(np.argwhere(np.array([steps])>0.005)[-1][1])
    background_x = np.concatenate((time[0:start],time[stop:-1]))
    background_y=np.concatenate((y[0:start],y[stop:-1]))
    popt,pcov=curve_fit(lin,background_x,background_y)
    amp_origin=list((y[j]-lin(time[j],*popt))*(-1) for j in np.arange(0,len(y)))
    maximum=max(amp_origin)
    amp=list(amp_origin[j]/maximum for j in np.arange(0,len(y)))
    plt.plot(time,  amp ,color=colors2[i+c],alpha=0.7)
    signal_edge=np.argwhere(amp>max(amp)/np.e)
    fwhm1, fwhm2=time[cut+int(signal_edge[0])],time[cut+int(signal_edge[-1])]
    plt.plot(fwhm1,amp[int(signal_edge[0])],'o',color=colors2[i+c] ,label='sensor {}'.format(i))
    plt.plot(fwhm2,amp[int(signal_edge[-1])],'o',color=colors2[i+c])
    fwhm=float(fwhm2-fwhm1)
    plt.plot([fwhm1,fwhm2],[amp[int(signal_edge[0])],amp[int(signal_edge[-1])]], color=colors2[i+c])

plt.xlabel('y [mm]')
plt.ylabel('sensor signal [V]')
plt.legend(loc='center right',fontsize=10,bbox_to_anchor=(1.5,0.5))
plt.xlim(0,250)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/UV_scan_vertical.pdf',bbox_inches='tight')


# %% Spectra working gases
plt.figure(figsize=(height,height))
d=5e+17
t=10
al=0.8
k=1.13
l2e=lambda x:(h1*c)/(x*10**(-9))
hdata=adas.h_adf15(T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in hdata[0]],[a/max(hdata[1]) for a in hdata[1]],k,color=colors2[8],alpha=al,label='H$^0$')
hedata=adas.he_adf15(data='pec96#he_pju#he0',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in hedata[0]],[a/max(hedata[1]) for a in hedata[1]],k,color=colors2[1],alpha=al,label='He$^0$')
nedata=adas.ne_adf15(data='pec96#ne_pju#ne0',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in nedata[0]],[a/max(nedata[1]) for a in nedata[1]],k,color=colors2[5],alpha=al,label='Ne$^0$')
ardata=adas.ar_adf15(data='pec40#ar_ls#ar0',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in ardata[0]],[a/max(ardata[1]) for a in ardata[1]],k,color=colors2[11],alpha=al,label='Ar$^0$')
plt.plot(energy,fitted/100,color='red',ls='dotted',alpha=0.5)
plt.xlim(-1,75)
plt.ylabel('normalized PEC [m$^3$/s]')
plt.xlabel('photon energy [eV]')
plt.legend(loc='lower right', title='ADAS data \n for spectral \n lines at  \n $T_{\mathrm{e}}=$10 eV, \n $n_{\mathrm{e}}$=5$\cdot 10^{17}$ m$^{-3}$ \n due to excitation \n of neutrals ')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/spectra_neutrals.pdf',bbox_inches='tight')

k=1.5
plt.figure(figsize=(height,height))
hedata=adas.he_adf15(data='pec96#he_pju#he1',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in hedata[0]],[a/max(hedata[1]) for a in hedata[1]],k,color=colors2[2],alpha=al,label='He$^{+1}$')
nedata=adas.ne_adf15(data='pec96#ne_pju#ne1',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in nedata[0]],[a/max(nedata[1]) for a in nedata[1]],k,color=colors2[6],alpha=al,label='Ne$^{+1}$')
ardata=adas.ar_adf15(data='pec40#ar_ic#ar1',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in ardata[0]],[a/max(ardata[1]) for a in ardata[1]],k,color=colors2[12],alpha=al,label='Ar$^{+1}$')
plt.plot(energy,fitted/100,color='red',ls='dotted',alpha=0.5)

plt.xlim(-1,100)
plt.ylabel('normalized PEC [m$^3$/s]')
plt.xlabel('photon energy [eV]')
plt.legend(loc='lower right', title='ADAS data \n for spectral \n lines at  \n $T_e=$10 eV, \n $n_e$=5$\cdot 10^{17}$ m$^{-3}$ \n due to excitation \n of ions')

fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/spectra_ions.pdf',bbox_inches='tight')
# %% Reduced Spectrum
d=5e+17
t=10
fig,ax=plt.subplots(figsize=(width/2,width/2))
ax2=ax.twinx()
l2e=lambda x:(h1*c)/(x*10**(-9))
hdata=adas.h_adf15(T_max=t,density=d,Spectrum=True)
energy=[round(l2e(a),3) for a in hdata[0]]
pec=[a/max(hdata[1]) for a in hdata[1]]
gold_energy=np.arange(0,round(max(energy),2)+5,0.001)
gold=pchip_interpolate(all_energy,all_abs,gold_energy)
reduced_pec=[]
for i,j in zip(energy,pec):
    indice= int(round(i,3)*1000)
    reduced_pec.append(j*gold[indice]/100)
ax.bar(energy,pec,0.5,color=colors2[2],label='ADAS data for spectral lines at \n$T_{\mathrm{e}}=$10 eV,$n_{\mathrm{e}}$=5$\cdot 10^{17}$ m$^{-3}$ \n due to excitation of H$^0$ ')   
ax.bar(energy,reduced_pec,0.5,color=colors2[1],label='reduced spectrum:\n {}\% absorbed by gold foil'.format(float(f'{np.sum(reduced_pec)/np.sum(pec)*100:.2f}')))
ax2.plot(gold_energy,gold,color=colors2[5],ls='dotted',label='gold absorption \n characteristic \n fit to data')
ax.set_ylabel('normalized PEC [m$^3$/s]')
ax.set_xlabel('photon energy [eV]')
ax2.set_ylabel('absorption gold [\%]',color=colors2[5])
ax2.tick_params(axis='y', labelcolor=colors2[5])
ax.legend(loc='lower center',bbox_to_anchor=(0.5,-0.7))
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/reduced_spectrum_H.pdf',bbox_inches='tight')


fig,ax=plt.subplots(figsize=(width/2,width/2))
ax2=ax.twinx()
l2e=lambda x:(h1*c)/(x*10**(-9))
wl,counts=np.genfromtxt('/home/gediz/Results/Spectrometer/Spectra_of_laser_and_white_light_22_09_2022/spectrometer_data_of_lightsource_Weißlichtquelle_Wellenlängenmessung.txt',unpack=True)
energy=[round(l2e(a),3) for a in wl]
gold_energy=np.arange(0,round(energy[0],2),0.001)
gold=pchip_interpolate(all_energy,all_abs,gold_energy)
reduced_counts=[]
for i,j in zip(energy,counts):
    indice= int(round(i,3)*1000)
    reduced_counts.append(j*gold[indice]/100)
lns1=ax.plot(energy,counts,color=colors2[2],label='specrometer data for a \n white light source ')   
lns2=ax.plot(energy,reduced_counts,color=colors2[1],label='reduced spectrum:\n {}\% absorbed by gold foil'.format(float(f'{integrate.trapezoid(reduced_counts,energy)/integrate.trapezoid(counts,energy)*100:.2f}')))
ax2.plot(gold_energy,gold,color=colors2[5],ls='dotted')
ax.set_ylabel('counts')
ax.set_xlabel('photon energy [eV]')
ax2.set_ylabel('absorption gold [\%]',color=colors2[5])
ax2.tick_params(axis='y', labelcolor=colors2[5])
leg = lns1 + lns2 
labs = [l.get_label() for l in leg]
ax.legend(leg, labs, loc='lower center',bbox_to_anchor=(0.5,-0.62))
ax.set_xlim(0,4)
ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/reduced_spectrum_white.pdf',bbox_inches='tight')


# %% electrical signals
fig1,ax1=plt.subplots(figsize=(height,height))
ax1.axhline(y=0,color='black',lw=1)
t = np.linspace(0, 2, 1000, endpoint=True)
frequ=1.7777
squ=signal.square(2 * np.pi * frequ * t+np.pi)
sine=7*np.sin(t*frequ*2*np.pi)
antisine=np.sin(t*frequ*2*np.pi+np.pi)
ax1.plot(t,sine,color=colors2[2],lw=2,label='1.77 kHz offset signal \nfrom bolometer')
ax1.plot(t, squ*3,color=colors2[6],lw=2,label='1.77 kHz compensation signal')
ax1.plot(t, squ*1,color=colors2[6],lw=2,alpha=0.5)
ax1.plot(t, squ*5,color=colors2[6],lw=2,alpha=0.5)

ax1.set_xlabel('time [ms]')
ax1.set_ylabel('amplitude [mV]')
#ax1.legend(loc='lower center',bbox_to_anchor=(0.5,-0.7))
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/e_sig_with_square.pdf',bbox_inches='tight')


fig2,ax2=plt.subplots(figsize=(height,height))
ax2.axhline(y=0,color='black',lw=1)
ax2.plot(t,squ*3+sine,color=colors2[12],lw=2,label='resulting signal \nafter summarizer')
ax2.plot(t,squ*1+sine,color=colors2[12],lw=2,alpha=0.5)
ax2.plot(t,squ*5+sine,color=colors2[12],lw=2,alpha=0.5)

ax2.set_xlabel('time [ms]')
ax2.set_ylabel('amplitude [mV]')
ax2.set_ylim(ax1.get_ylim())
#ax2.legend(loc='lower center',bbox_to_anchor=(0.5,-0.55))
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/e_sig_with_square_result.pdf',bbox_inches='tight')


fig1,ax1=plt.subplots(figsize=(height,height))
ax1.axhline(y=0,color='black',lw=1)
t = np.linspace(0, 2, 1000, endpoint=True)
frequ=1.7777
antisine=np.sin(t*frequ*2*np.pi+np.pi)
sine=7*np.sin(t*frequ*2*np.pi+0.005*np.pi)
antisine=np.sin(t*frequ*2*np.pi+np.pi)
ax1.plot(t,sine,color=colors2[2],lw=2,label='1.77 kHz offset signal \nfrom bolometer')
ax1.plot(t, antisine*7,color=colors2[6],lw=2,label='1.77 kHz compensation signal')
ax1.plot(t, antisine*3,color=colors2[6],lw=2,alpha=0.5)
ax1.plot(t, antisine*5,color=colors2[6],lw=2,alpha=0.5)

ax1.set_xlabel('time [ms]')
ax1.set_ylabel('amplitude [mV]')
ax1.legend(loc='lower center',bbox_to_anchor=(0.5,-0.7))
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/e_sig_with_sine.pdf',bbox_inches='tight')


fig2,ax2=plt.subplots(figsize=(height,height))
ax2.axhline(y=0,color='black',lw=1)
ax2.plot(t,antisine*7+sine,color=colors2[12],lw=2,label='resulting signal \nafter summarizer')
ax2.plot(t,antisine*5+sine,color=colors2[12],lw=2,alpha=0.5)
ax2.plot(t,antisine*3+sine,color=colors2[12],lw=2,alpha=0.5)

ax2.set_xlabel('time [ms]')
ax2.set_ylabel('amplitude [mV]')
ax2.set_ylim(ax1.get_ylim())
ax2.legend(loc='lower center',bbox_to_anchor=(0.5,-0.55))

fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/e_sig_with_sine_result.pdf',bbox_inches='tight')

# %% Sensor resistances
path=['/home/gediz/Results/Calibration/Channel_resistances_September_2022/all_resistor_values_bolometer_sensors_calculated.txt','/home/gediz/Results/Calibration/Channel_resistances_September_2022/all_resistor_values_bolometer_sensors_calculated_second_set.txt','/home/gediz/Results/Calibration/Channel_resistances_September_2022/all_resistor_values_bolometer_sensors_calculated_third_set.txt']
x=[1,2,3,4,5,6,7,8]
i=0
plt.figure(figsize=(width/2,height*0.7))
mean,sem=[],[]
for s in x:
    M1=[]
    for p in path:
        RM1=np.genfromtxt(p,unpack=True, delimiter=',',usecols=(1))
        M1.append(RM1[s-1])
        plt.plot(s,RM1[s-1],color=colors[i+0],marker=markers[i+0],ls='None',alpha=0.5)
    mean.append(np.mean(M1))
    sem.append(np.std(M1,ddof=1)/np.sqrt(len(M1)))
plt.errorbar(x,mean,yerr=sem,ls='None',color=colors[i+0],marker=markers[i+0],capsize=5,label='$R_\mathrm{M1}$')
print(mean,sem)
plt.xticks(x)
plt.xlabel('sensor number')
plt.ylabel('resistivity [$\Omega$]')
plt.legend(loc='lower left')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/resistance_m1.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,height*0.7))
mean,sem=[],[]
for s in x:
    M2=[]
    for p in path:
        RM2=np.genfromtxt(p,unpack=True, delimiter=',',usecols=(2))
        M2.append(RM2[s-1])
        plt.plot(s,RM2[s-1],color=colors[i+1],marker=markers[i+1],ls='None',alpha=0.5)
    mean.append(np.mean(M2))
    sem.append(np.std(M2,ddof=1)/np.sqrt(len(M2)))
plt.errorbar(x,mean,yerr=sem,color=colors[i+1],marker=markers[i+1],capsize=5,ls='None',label='$R_\mathrm{M2}$')
print(mean)
print(sem)
plt.xticks(x)
plt.xlabel('sensor number')
plt.ylabel('resistivity [$\Omega$]')
plt.legend(loc='lower left')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/resistance_m2.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,height*0.7))
mean,sem=[],[]
for s in x:
    R1=[]
    for p in path:
        RR1=np.genfromtxt(p,unpack=True, delimiter=',',usecols=(3))
        R1.append(RR1[s-1])
        plt.plot(s,RR1[s-1],color=colors[i+2],marker=markers[i+2],ls='None',alpha=0.5)
    mean.append(np.mean(R1))
    sem.append(np.std(R1,ddof=1)/np.sqrt(len(R1)))
plt.errorbar(x,mean,yerr=sem,color=colors[i+2],marker=markers[i+2],capsize=5,ls='None',label='$R_\mathrm{R1}$')
print(mean)
print(sem)
plt.xticks(x)
plt.xlabel('sensor number')
plt.ylabel('resistivity [$\Omega$]')
plt.legend(loc='lower right')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/resistance_r1.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,height*0.7))
mean,sem=[],[]
for s in x:
    R2=[]
    for p in path:
        RR2=np.genfromtxt(p,unpack=True, delimiter=',',usecols=(4))
        R2.append(RR2[s-1])
        plt.plot(s,RR2[s-1],color=colors[i+3],marker=markers[i+3],ls='None',alpha=0.5)
    mean.append(np.mean(R2))
    sem.append(np.std(R2,ddof=1)/np.sqrt(len(R2)))
plt.errorbar(x,mean,yerr=sem,color=colors[i+3],marker=markers[i+3],capsize=5,ls='None',label='$R_\mathrm{R2}$')
print(mean)
print(sem)
plt.xticks(x)
plt.xlabel('sensor number')
plt.ylabel('resistivity [$\Omega$]')
plt.legend(loc='lower right')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/resistance_r2.pdf',bbox_inches='tight')

# %%Lamp spectra
plt.figure(figsize=(height,height))
e2l=lambda x:((h1*c)/x)/10**(-9)

wl_UV254,UV254=np.genfromtxt('/home/gediz/Results/Spectrometer/Spectra_of_lamps_17_08_2022/spectrometer_data_of_lightsource_254.txt',unpack=True)
wl_UV350,UV350=np.genfromtxt('/home/gediz/Results/Spectrometer/Spectra_of_lamps_17_08_2022/spectrometer_data_of_lightsource_350.txt', unpack=True)
wl_white,white=np.genfromtxt('/home/gediz/Results/Spectrometer/Spectra_of_lamps_17_08_2022/spectrometer_data_of_lightsource_breite_Weißlichtquelle_scaled.txt',unpack=True)
wl_red,red=np.genfromtxt('/home/gediz/Results/Spectrometer/Spectra_of_laser_and_white_light_22_09_2022/spectrometer_data_of_lightsource_Weißlichtquelle_Wellenlängenmessung_rote_folie.txt',unpack=True)
wl_green,green=np.genfromtxt('/home/gediz/Results/Spectrometer/Spectra_of_laser_and_white_light_22_09_2022/spectrometer_data_of_lightsource_Weißlichtquelle_Wellenlängenmessung_grüne_folie.txt',unpack=True)
wl_laser,laser=np.genfromtxt('/home/gediz/Results/Spectrometer/Spectra_of_laser_and_white_light_22_09_2022/spectrometer_data_of_lightsource_grüner_laser_rand.txt',unpack=True)
i=0
plt.plot(wl_UV254[0:600],[a/max(UV254) for a in UV254][0:600],color=colors2[9],label='UV lamp 254 nm')
plt.plot(wl_UV350,[a/max(UV350) for a in UV350],color=colors2[10],label='UV lamp 350 nm')
plt.plot(wl_white,[a/max(white) for a in white],color=colors2[2],label='white light')
plt.plot(wl_green,[a/max(green) for a in green],color=colors2[13],label='white light with green foil')
plt.plot(wl_red,[a/max(red) for a in red],color=colors2[5],label='white light with red foil')
plt.plot(wl_laser,[a/max(laser) for a in laser],color=colors2[11],label='green laser',lw=3)
plt.plot([e2l(x) for x in energy],fitted/100,color=colors2[5],ls='dotted',label='gold absorption characteristic')
plt.xlim(200,800)
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-1.15))
plt.xlabel('wavelength [nm]')
plt.ylabel('normalized counts')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/spectra_lamps.pdf',bbox_inches='tight')

# %% Optical calibration
s,laser1=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot70001/shot70001_bolometerprofile_from_radiation_powers.txt',unpack=True, usecols=(0,1))
s,laser2=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot70003/shot70003_bolometerprofile_from_radiation_powers.txt',unpack=True, usecols=(0,1))
s,uv=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot70025/shot70025_bolometerprofile_from_radiation_powers.txt',unpack=True, usecols=(0,1))
s,uv350=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot70034/shot70034_bolometerprofile_from_radiation_powers.txt',unpack=True, usecols=(0,1))
s,uv254=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot70035/shot70035_bolometerprofile_from_radiation_powers.txt',unpack=True, usecols=(0,1))
plt.plot(s,[a/np.mean(laser1) for a in laser1])
plt.plot(s,[a/np.mean(laser2) for a in laser2])
plt.plot(s,[a/np.mean(uv) for a in uv])
plt.plot(s,[a/np.mean(uv350) for a in uv350])
plt.plot(s,[a/np.mean(uv254) for a in uv254])
plt.show()
# %% Voltage Time trace
shotnumber=13257
plt.figure(figsize=(width*0.4,height))
location=  '/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
time = np.array(br.LoadData(location)['Zeit [ms]'] / 1000)[:,None]
for i,c in zip(np.arange(1,9),colors):
    bolo_raw_data = np.array(br.LoadData(location)["Bolo{}".format(i)])[:,None]
    m=min(bolo_raw_data)
    bolo_raw_data=[(k-m)+i*0.05 for k in bolo_raw_data]
    plt.plot(time,  bolo_raw_data, label="sensor {}".format(i),color=c )
plt.xlabel('time [s]')
plt.ylabel('U$_{\mathrm{out}}$ [V]')
plt.legend(loc='center right',bbox_to_anchor=(2,0.5),title='shot n$^\circ$13257,\nHe,\nMW= 2.45 GHz,\n P$_{\mathrm{MW}}$= 2.97 kW,\np= 21 mPa')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_voltage_time_traces.pdf',bbox_inches='tight')



# %% Power time trace with height
shotnumber=13257
i=1
# location=  '/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
# time,U_Li =br.LoadData(location)['Zeit [ms]'] / 1000, br.LoadData(location)["Bolo{}".format(i)]
# def power(g,k,U_ac, t, U_Li):
#     return (np.pi/g) * (2*k/U_ac) * (t* np.gradient(U_Li,time*1000 )+U_Li)
# def error(g,k,U_ac, t, U_Li,delta_t,delta_k):
#     return ((np.pi/g) * (2/U_ac) * (t* np.gradient(U_Li,time*1000 )+U_Li))*delta_k+(np.pi/g) * (2*k/U_ac) * np.gradient(U_Li,time*1000 )*delta_t
# tau,tau_sem,kappa,kappa_sem=np.genfromtxt('/home/gediz/Results/Calibration/Ohmic_Calibration/Ohmic_Calibration_Vacuum_November/ohmic_calibration_vacuum_tjk_tau_and_kappa_mean_and_sem.txt',unpack=True,usecols=(1,2,3,4))
# g1,g2,g3= 30,1,100
# g=g1*g2*g3
# U_ac,k,t,delta_t,delta_k=8,kappa[i-1],tau[i-1],tau_sem[i-1],kappa_sem[i-1]
# power=[a/10**(-6) for a in power(g,k,U_ac, t, U_Li)]
# steps=[]
# for i in np.arange(0, len(U_Li)-10):
#     step= (U_Li[i]-U_Li[i+10])
#     steps.append(abs(step))
# start,stop=np.argwhere(np.array([steps])>0.005)[0][1],np.argwhere(np.array([steps])>0.005)[-1][1]
# U_Li=U_Li*1000
# x1,y1,x2,y2 = time[start:stop],U_Li[start:stop],np.concatenate((time[0:start],time[stop:-1])),np.concatenate((U_Li[0:start],U_Li[stop:-1]))
# def lin (x,a,b):
#     return a*x + b
# popt1, pcov1 = curve_fit(lin,x1,y1)
# popt2, pcov2 = curve_fit(lin,x2,y2)
# sd=np.std([div1,div2],ddof=1)
# sem=sd/np.sqrt(2)
# plt.figure(figsize=(width/2,height/2))
# plt.plot(time[start+15],U_Li[start+15],marker='x',color=colors[1])
# plt.plot(time[stop],U_Li[stop],marker='x',color=colors[1])
# plt.plot(time,U_Li,color=colors[0],label='sensor 1')
# plt.plot(np.arange(0,240),lin(np.arange(0,240),*popt1),color=colors[3],label='fit to $U_{\mathrm{out}} (t_{\mathrm{plasma \ on}})$')
# plt.plot(np.arange(0,240),lin(np.arange(0,240),*popt2),color=colors[4],label='fit to $U_{\mathrm{out}} (t_{\mathrm{plasma \ off}})$')
# plt.ylabel('$U_{\mathrm{out}}$  [mV]')
# plt.xticks([])
# plt.ticklabel_format(axis='y', style='sci')
# #plt.legend(loc='lower right',fontsize=10)
# fig= plt.gcf()
# plt.show()
# fig.savefig('/home/gediz/LaTex/Thesis/Figures_PPP/voltage_time_trace.pdf',bbox_inches='tight')

x1,y1,x2,y2 = time[start:stop],power[start:stop],np.concatenate((time[0:start],time[stop:-1])),np.concatenate((power[0:start],power[stop:-1]))
def lin (x,a,b):
    return a*x + b
popt1, pcov1 = curve_fit(lin,x1,y1)
popt2, pcov2 = curve_fit(lin,x2,y2)
div1 = lin(time[start], *popt2)-lin(time[start], *popt1)
div2 = lin(time[stop], *popt2)-lin(time[stop], *popt1)
div_avrg = abs(float(f'{(div1+div2)/2:.4f}'))

plt.figure(figsize=(width/2,height/2))
plt.plot(time,power-lin(time,*popt2),color=colors[0],label='sensor 1')
x1,y1,x2,y2 = time[start:stop],power[start:stop]-lin(time[start:stop],*popt2),np.concatenate((time[0:start],time[stop:-1])),np.concatenate((power[0:start]-lin(time[0:start],*popt2),power[stop:-1]-lin(time[stop:-1],*popt2)))
popt1, pcov1 = curve_fit(lin,x1,y1)
popt2, pcov2 = curve_fit(lin,x2,y2)
plt.plot(np.arange(0,240),lin(np.arange(0,240),*popt1),color=colors[3],label='fit to $U_{\mathrm{out}} (t_{\mathrm{plasma \ on}})$')
plt.plot(np.arange(0,240),lin(np.arange(0,240),*popt2),color=colors[4],label='fit to $U_{\mathrm{out}} (t_{\mathrm{plasma \ off}})$')
plt.annotate('',xy=(50,lin(50,*popt2)),xytext=(50,lin(50,*popt1)), arrowprops=dict(arrowstyle='<->',color=colors[1],alpha=0.7,linewidth=2))
plt.annotate('$\Delta P_{\mathrm{rad}} \cdot a_{\mathrm{abs}}$='+str(f'{div_avrg:.2f}')+'$\mu$W',xy=(55,1),color=colors[1])
plt.xlabel('time [s]')
plt.ylabel('P$_{\mathrm{rad}}$  [$\mu$W]')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures_PPP/power_time_trace.pdf',bbox_inches='tight')

# %% Bolometer profile
plt.figure(figsize=(width/2,height*0.75))
x=[13265,13263,13261,13259,13257]
b=[1,2,3,4,5,6,7,8]
gas='He'
#x=[13254, 13253, 13252, 13251, 13255, 13250, 13249, 13248, 13247, 13246, 13245, 13244, 13243, 13242]#H pressure
#x=[13278, 13277, 13276, 13275, 13274, 13279, 13273, 13272, 13271, 13270, 13269, 13268]#He pressure
#x=[13310, 13309, 13308, 13307, 13306, 13311, 13305, 13304, 13303, 13302, 13301, 13300, 13299]#Ar pressure
#x=[13347, 13346, 13345, 13344, 13343, 13342, 13341, 13340]#Ne pressure
#x=[13289, 13290, 13288, 13287, 13285, 13286, 13284, 13291, 13283, 13282, 13281, 13280]#Ar power
#x=[13221, 13220, 13223, 13222, 13224, 13218, 13225, 13226, 13217, 13216, 13219, 13227, 13215]#H power
#x=[13265, 13264, 13263, 13262, 13261, 13260, 13259, 13258, 13257]#He power
#x=[13316 ,13317, 13318 ,13319 ,13320]#He 8Ghz
for shotnumber,i in zip(x,np.arange(0,len(x))):
    P_profile,error_P_exp=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot{s}/shot{s}_bolometerprofile_from_radiation_powers.txt'.format(s=shotnumber),unpack=True,usecols=(1,2))
    plt.errorbar(b,P_profile,yerr=error_P_exp,marker=markers[i],color=colors[i],capsize=5,alpha=0.15)

    b,c,ecmi,ecma,p,e=np.genfromtxt('/home/gediz/Results/Modeled_Data/Bolometerprofiles/shot{s}/shot{s}_modeled_powerprofile_{g}.txt'.format(s=shotnumber,g=gas),unpack=True)
    plt.errorbar(b,p,yerr=e,marker=markers[i],color=colors[i],capsize=5,label='shot n$^\circ$'+str(shotnumber) +', $P_{\mathrm{MW}}$='+str(f'{pc.GetMicrowavePower(shotnumber)[0]/1000:.1f}') +'kW')

plt.ylim(0)
plt.xticks(b)
plt.xlabel('sensor number',fontsize=10)
plt.ylabel('$\Delta P_{\mathrm{rad}}$ [$\mu$W]',fontsize=10)
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-1.0),title='He, MW=2.45 GHz, p=21 mPa')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/radiation_profiles.pdf'.format(g=gas),bbox_inches='tight')

#%% Bolometerprofile no Density and Temperature
plt.figure(figsize=(width/2,height/1.5))
gas='He'
ScanType='Pressure'
#x=[13192,13191,13190,13189,13188,13187]#He 8GHz power 
x=[13186,13181,13187,13177,13184,13185,13180]#He 8GHz pressure
#x=[13194,13195,13196,13197,13198,13200]#H 8Ghz  power
#x=[13205,13201,13202,13203,13204]#Ar 8GHz   pressure
#x=[13206,13207,13208,13209,13210,13211]#Ar 8Ghz power
for s,i in zip(x,np.arange(0,len(x))):
    if ScanType=='Pressure':
        label=r'n$^\circ$'+str(s)+r', p= '+str(f'{pc.Pressure(s,gas):.1f}')+' mPa'
        title= str(gas)+r', MW= '+str(pc.GetMicrowavePower(s)[1])+r', $P_{\mathrm{MW}}$ $\approx$ '+str(f'{pc.GetMicrowavePower(s)[0]*10**(-3):.2f}')+' kW'
    if ScanType=='Power':
        label=r'n$^\circ$'+str(s)+r', $P_{\mathrm{MW}}$ = '+str( f'{pc.GetMicrowavePower(s)[0]*10**(-3):.2f}')+' kW'
        title= str(gas)+', MW= '+str(pc.GetMicrowavePower(s)[1])+r', p $\approx$ '+str(f'{pc.Pressure(s,gas):.1f}')+' mPa'
    if ScanType=='None':
        label=r'n$^\circ$'+str(s)+r', $P_{\mathrm{MW}}$ = '+str(f'{pc.GetMicrowavePower(s)[0]*10**(-3):.2f}')+' kW,\n p= '+str(f'{pc.Pressure(s,gas):.1f}')+' mPa'
        title= str(gas)+', MW= '+str(pc.GetMicrowavePower(s)[1])
    p,e=poca.Boloprofile_correction(s,gas,savedata=True)[1],poca.Boloprofile_correction(s,gas)[2]
    plt.errorbar([1,2,3,4,5,6,7,8],p,yerr=e,marker=markers[i],color=colors2[i],capsize=5,label=label)
plt.ylim(0)
plt.xticks([1,2,3,4,5,6,7,8])
plt.xlabel('sensor number',fontsize=10)
plt.ylabel('$\Delta P_{\mathrm{rad}}$ [$\mu$W]',fontsize=10)
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-1.2),title=title)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/{g}_8GHz_{s}_radiation.pdf'.format(g=gas,s=ScanType),bbox_inches='tight')

# %% Total Power
plt.figure(figsize=(width/2,height*0.75))
shotnumbers=[[13265,13263,13261,13259,13257]]
gases=[['He'for i in range(5)]]
mw,p,emi,ema=poca.Totalpower_from_exp(shotnumbers,gases,'Power')
plt.plot(mw,p,ls='dashed',label='He, MW=2.45 GHz, p=21 mPa')
for i,j,k,m,a in zip(mw,p,emi,ema,[0,1,2,3,4]):
    plt.plot(i,j,marker=markers[0+a],color=colors[0+a])
    plt.errorbar(i,j,yerr=[[k],[m]],capsize=5,linestyle='None',color=colors[0+a])
    plt.annotate(str(f'{(j/i)*100:.1f}')+'\%',xy=(i,j),xytext=(i-300,j+60),color=colors[0+a])
plt.xlabel('$P_{\mathrm{MW}}$ [W]')
plt.ylabel('$P_{\mathrm{rad,\ net}}$ [W]')
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.45))
plt.xlim(0,3100)
plt.ylim(0,460)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/net_power_loss.pdf',bbox_inches='tight')

# %% Modelling Code

fig=plt.figure(figsize=(width,width))
ax=fig.add_subplot(111)
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position_extended.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii_extended.csv',sep=',',engine='python')
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
    x=np.array(x_.iloc[i])
    y=np.array(y_.iloc[i])
    plt.plot(np.append(x,x[0]),np.append(y,y[0]),color=colors2[i],linewidth=2,marker=markers[0],markersize=4,alpha=0.5)
ax.add_patch(Rectangle((60.5,3),2,2,edgecolor=colors2[2],facecolor='None',lw=4))
ax.plot(61.55,4.05,color=colors2[5],marker='x',ms=10,mew=3)
ax.plot(z_0,0,color=colors2[5],marker='x',ms=10,mew=3)

popt,pcov=curve_fit(lin,[z_0,61.55],[0,4.05])
ax.plot(np.arange(50,75,0.1),lin(np.arange(50,75,0.1),*popt),color=colors2[5],lw=2)

plt.ylabel('Z [cm]')
plt.xlabel('R [cm]')
plt.xlim(47,78)
plt.ylim(-15.5,15.5)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/flux_surfaces_extended.pdf',bbox_inches='tight')

fig=plt.figure(figsize=(w,w))
ax=fig.add_subplot(111)
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position_extended.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii_extended.csv',sep=',',engine='python')
def lin (x,a,b):
    return a*x + b
for k,m in zip(np.arange(60.5,62.5,0.1),np.arange(3,5,0.1)):
    ax.hlines(m,60.5,62.5,color='gray',alpha=0.2)
    ax.vlines(k,3,5,color='gray',alpha=0.2)
for i in [2,3]:
    x=np.array(x_.iloc[i])
    y=np.array(y_.iloc[i])
    ax.plot(np.append(x,x[0]),np.append(y,y[0]),color=colors2[i],ls='None',marker=markers[0],markersize=10)
    for j in np.arange(0,len(x)-1):
        popt,pcov=curve_fit(lin,[x[j],x[j+1]],[y[j],y[j+1]])
        alpha=np.arctan(abs(y[j+1]-y[j])/abs(x[j+1]-x[j]))
        extra=abs(np.cos(alpha))*0.15
        range=np.arange(np.sort([x[j],x[j+1]])[0]-extra,np.sort([x[j],x[j+1]])[1]+extra,0.0001)
        if j==20:
            ax.plot(range,lin(range,*popt),color=colors2[i])
        else:
            ax.plot(range,lin(range,*popt),color=colors2[i],alpha=0.5,ls='dashed')
ax.plot(61.55,4.05,color=colors2[5],marker='x',ms=10,mew=3)
popt,pcov=curve_fit(lin,[z_0,61.55],[0,4.05])
ax.plot(np.arange(60.5,62.5,0.1),lin(np.arange(60.5,62.5,0.1),*popt),color=colors2[5],lw=2)

ax.add_patch(Rectangle((61.5,4),0.1,0.1,color=colors2[5],alpha=0.5))
ax.add_patch(Rectangle((60.52,3.02),1.96,1.96,edgecolor=colors2[2],facecolor='None',lw=6))

    #poptm,pcovm=curve_fit(lin,[z_0,m],[0,n])
    #diff=abs(lin(range,*popt)-lin(range,*poptm))

plt.ylabel('Z [cm]')
plt.xlabel('R [cm]')
plt.xlim(60.5,62.5)
plt.ylim(3,5)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/flux_surfaces_extended_zoom.pdf',bbox_inches='tight')




# %% ADF11
temperatures=adas.h_adf11()[2]
densities=[a*10**6 for a in adas.h_adf11()[3]]

plt.figure(figsize=(width*0.4,height*0.8))
for i,c in zip([3,5,9,11,15,17,23,29],[0,1,2,3,4,5,6,7]):
    t_name='rad_t_'+str(i)
    d_dep_h=adas.h_adf11(T_max=temperatures[-1],wish=t_name)[5]
    plt.plot(densities,d_dep_h[1],color=colors2[c],marker='o',ms=5,label='$T_e$ = {} eV'.format('%.1f' %d_dep_h[0]))
plt.xscale('log')
plt.xlabel('density [m$^{-3}$]')
plt.ylabel(r'$\left\langle  \sigma v \right\rangle _{\textrm{rad}}\left\langle E_{\textrm{rad}}\right\rangle$ [eVm$^3$/s]')
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-1.2))
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/adf11_H_pec_from_d.pdf',bbox_inches='tight')

plt.figure(figsize=(width*0.4,height*0.8))
for i,c in zip([2,5,8,11,14,17,20,23],[0,1,2,3,4,5,6,7]):
    d_name='rad_d_'+str(i)
    t_dep_h=adas.h_adf11(T_max=temperatures[-1],wish=d_name)[5]
    plt.plot(temperatures,t_dep_h,color=colors2[c],marker='o',ms=5,label='$n_e$ = '+str('%.0E' % densities[i-1] )+' m$^{-3}$')
plt.xscale('log')
plt.xlabel('temperature [eV]')
plt.ylabel(r'$\left\langle  \sigma v \right\rangle _{\textrm{rad}}\left\langle E_{\textrm{rad}}\right\rangle$ [eVm$^3$/s]')
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-1.2))
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/adf11_H_pec_from_t.pdf',bbox_inches='tight')

plt.figure(figsize=(width*0.4,height*0.8))
for i,c in zip([7,8,10,11,13,14,16],[1,2,3,4,5,6,7]):
    d_name='rad_d_'+str(i)
    t_dep_h=adas.h_adf11(T_max=temperatures[-1],wish=d_name)[5]
    plt.plot(temperatures,t_dep_h,color=colors2[c],label='$n_e$ = '+str('%.0E' % densities[i-1] )+' m$^{-3}$',marker='o',ms=5,alpha=0.5)
plt.plot(temperatures, adas.h_adf11(T_max=temperatures[-1])[1], label='averaged density',marker='o',ms=10,color=colors2[2])
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.ylim(1E-13,6E-13)
plt.xlim(0,100)
plt.xlabel('temperature [eV]')
plt.ylabel(r'$\left\langle  \sigma v \right\rangle _{\textrm{rad}}\left\langle E_{\textrm{rad}}\right\rangle$ [eVm$^3$/s]')
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-1.2))
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/adf11_H_pec_from_t_log.pdf',bbox_inches='tight')


# %% ADF15

plt.figure(figsize=(width*0.4,height*0.8))
d=5e+17
t=10
e=1.602E-19
m=1E-6
al=0.8
k=2.5
l2e=lambda x:(h1*c)/(x*10**(-9))
for t,c1,c2 in zip([200,100,10],[0,1,2],[3,5,6]):
    k-=c1*0.4
    hedata0=adas.he_adf15(data='pec96#he_pju#he0',T_max=t,density=d,Spectrum=True)
    plt.bar([l2e(a) for a in hedata0[0]],[a*m for a in hedata0[1]],k,color=colors2[c1],label='He$^0$, $T_e$=' +str(hedata0[3]) +'eV')
k=2.5
for t,c1,c2 in zip([200,100,10],[0,1,2],[3,5,6]):
    k-=c1*0.4
    hedata1=adas.he_adf15(data='pec96#he_pju#he1',T_max=t,density=d,Spectrum=True)
    plt.bar([l2e(a) for a in hedata1[0]],[a*m for a in hedata1[1]],k,color=colors2[c2],label='He$^{+1}$, $T_e$=' +str(hedata0[3]) +'eV')
plt.xlim(-1,60)
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ScalarFormatter()) 
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
plt.minorticks_off()
plt.ticklabel_format(axis='y', style='sci')
plt.ylabel('PEC [m$^3$/s]')
plt.xlabel('photon energy [eV]')
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-1))
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/adf15_he_spectra.pdf',bbox_inches='tight')

plt.figure(figsize=(width*0.4,height*0.8))
h=6.626E-34
c=299792458
# hedata0=adas.h_adf15()
# plt.plot(hedata0[0],hedata0[1],color=colors2[2],label='H$^0$',marker='o')
hedata0=adas.he_adf15(data='pec96#he_pju#he0')
plt.plot(hedata0[0],hedata0[1],color=colors2[1],label='He$^0$',marker='o')
hedata1=adas.he_adf15(data='pec96#he_pju#he1')
plt.plot(hedata1[0],hedata1[1],color=colors2[5],label='He$^{+1}$',marker='o')
plt.yscale('log')
plt.ylim(1E-14,6E-13)
plt.xlim(0,100)
plt.gca().yaxis.set_major_formatter(ScalarFormatter()) 
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
plt.minorticks_off()
plt.ticklabel_format(axis='y', style='sci')
plt.yticks([1E-14,5E-14,1E-13,5E-13])
plt.xlabel('temperature [eV]')
plt.ylabel(r'$\left\langle  \sigma v \right\rangle _{\textrm{rad}}\left\langle E_{\textrm{rad}}\right\rangle$ [eVm$^3$/s]')
plt.legend(loc='lower right')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/adf15_he_from_t.pdf',bbox_inches='tight')

plt.figure(figsize=(width*0.4,height*0.8))
h=6.626E-34
c=299792458
hdata=adas.h_adf15()
hedata=adas.he_adf15(data='pec96#he_pju#he0')
nedata=adas.ne_adf15(data='pec96#ne_pju#ne0')
plt.plot(hdata[0],hdata[1],marker='o',color=colors2[2],label='H$^0$, ADF15',lw=3.5,markersize=8)
plt.plot(hedata[0],hedata[1],marker='o',color=colors2[5],label='He$^0$, ADF15',lw=3.5,markersize=8)
plt.plot(nedata[0],nedata[1],marker='o',color=colors2[12],label='Ne$^0$, ADF15',lw=3.5,markersize=8)
hdata=adas.h_adf11()
hedata=adas.he_adf11(data='plt96_he')
nedata=adas.ne_adf11(data='plt96_ne')
plt.plot(hdata[0],hdata[1],marker='s',color=colors2[1],label='H$^0$, ADF11',lw=2,markersize=5)
plt.plot(hedata[0],hedata[1],marker='s',color=colors2[4],label='He$^0$, ADF11',lw=2,markersize=5)
plt.plot(nedata[0],nedata[1],marker='s',color=colors2[10],label='Ne$^0$, ADF11',lw=2,markersize=5)
plt.xlim(0,100)
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ScalarFormatter()) 
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
plt.minorticks_off()
plt.ticklabel_format(axis='y', style='sci')
plt.ylim(1E-14,6E-13)
plt.xlabel('temperature [eV]')
plt.ylabel(r'$\left\langle  \sigma v \right\rangle _{\textrm{rad}}\left\langle E_{\textrm{rad}}\right\rangle$ [eVm$^3$/s]')
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-1))
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/adf15_compared_to_adf11.pdf',bbox_inches='tight')
# %% All PLT
plt.figure(figsize=(width,height))
h=6.626E-34
c=299792458
hdata=adas.h_adf11()
hedata=adas.he_adf11(data='plt96_he')
nedata=adas.ne_adf11(data='plt96_ne')
ardata=adas.ar_adf11()
plt.plot(hdata[0],hdata[1],marker='o',color=colors2[2],label='H$^0$')
plt.plot(hedata[0],hedata[1],marker='o',color=colors2[5],label='He$^0$')
plt.plot(nedata[0],nedata[1],marker='o',color=colors2[9],label='Ne$^0$')
plt.plot(ardata[0],ardata[1],marker='o',color=colors2[10],label='Ar$^0$')
plt.plot(hedata[0],hedata[2],marker='s',color=colors2[4],label='He$^{+1}$')
plt.plot(nedata[0],nedata[2],marker='s',color=colors2[8],label='Ne$^{+1}$')
plt.plot(ardata[0],ardata[2],marker='s',color=colors2[11],label='Ar$^{+1}$')
plt.xlim(0,60)
plt.yscale('log')
plt.ylim(1E-15,1E-12)
plt.xlabel('temperature [eV]')
plt.ylabel(r'$\left\langle  \sigma v \right\rangle _{\textrm{rad}}\left\langle E_{\textrm{rad}}\right\rangle$ [eVm$^3$/s]')
plt.legend(loc='lower right')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_plt.pdf',bbox_inches='tight')

# %% Total Power weighing method
plt.figure(figsize=(width,height))
shotnumbers=[[13265, 13264, 13263, 13262, 13261, 13260, 13259, 13258, 13257]]
gases=[['He'for i in range(10)]]
arg,c,ecma,ecmi=poca.Totalpower_calc(shotnumbers[0],gases[0],'Power')
p_tot_crossec=poca.Total_cross_section_calc(shotnumbers,gases)
plt.plot(arg,p_tot_crossec,'o--',color=colors2[6],label='from modelled mean \n power density $\overline{p}_{\mathrm{rad, mod}}$')
plt.plot(arg,c,'o--',color=colors2[5], label='from $P_{\mathrm{rad, mod}}$ \n with weighing method')
plt.xlabel('$P_{\mathrm{MW}}$ [W]')
plt.ylabel('$P_{\mathrm{rad,net}}$ [W]')
plt.legend(loc='lower right')

fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/net_power_loss_weighing_method.pdf',bbox_inches='tight')

# %% Density Profile with Errorbars
fig2=plt.figure(figsize=(width/2,height))
ax=fig2.add_subplot(111)
ax_f=ax.twinx()
ax_f.set_yticks([])
ax.set_xlim(58,79)
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position_extended.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii_extended.csv',sep=',',engine='python')
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
    x=[u-60 for u in np.array(x_.iloc[i])]
    y=np.array(y_.iloc[i])
    ax_f.plot(np.append(x,x[0])+60,np.append(y,y[0]),color='grey',linewidth=2,alpha=0.3)
ax_f.hlines(0,64,78,color=colors2[5])
for i in np.arange(0,29):
    ax_f.vlines(64+i*0.5,-0.5,0.5,color=colors2[5],lw=1)

s=13252
i=2
df=['d']
p,d,e=pc.DensityProfile(s,df,'Values')[0],pc.DensityProfile(s,df,'Values')[1],pc.DensityProfile(s,df,'Values')[2]
ax.errorbar(p*100+60,d,e,capsize=5, marker='o', color=colors2[i],label='corrected\n density\n profile')
ax.errorbar(p*100+46,np.flip(d),np.flip(e),capsize=5, marker='o', color=colors2[i],alpha=0.3)
ax.set_xlabel('R [cm]')
ax.set_ylabel('$n_\mathrm{e}(R)$ [m$^{-3}$]')
#ax.legend(loc='lower left')#,bbox_to_anchor=(0.5,-0.5))#,title='He, shot n$^\circ$13252, MW: 2.45 GHz, \n $P_{\mathrm{MW}}$ = 2.8 kW, p = 25.5 mPa') 
ax.set_zorder(ax.get_zorder()+1)
ax.set_frame_on(False)   
fig2= plt.gcf()
plt.show()
fig2.savefig('/home/gediz/LaTex/Thesis/Figures/Density_with_errorbars.pdf',bbox_inches='tight')



# %% Temperature Profile with Errorbars
fig2=plt.figure(figsize=(width/2,height))
ax=fig2.add_subplot(111)
ax_f=ax.twinx()
ax_f.set_yticks([])
ax.set_xlim(58,79)
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position_extended.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii_extended.csv',sep=',',engine='python')
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
    x=[u-60 for u in np.array(x_.iloc[i])]
    y=np.array(y_.iloc[i])
    ax_f.plot(np.append(x,x[0])+60,np.append(y,y[0]),color='grey',linewidth=2,alpha=0.3)
ax_f.hlines(0,64,78,color=colors2[5])
for i in np.arange(0,29):
    ax_f.vlines(64+i*0.5,-0.5,0.5,color=colors2[5],lw=1)

s=13252
i=6
p,t,e=pc.TemperatureProfile(s,'Values')[0],pc.TemperatureProfile(s,'Values')[1],pc.TemperatureProfile(s,'Values')[3]
ax.errorbar(p*100+60,t,e,capsize=5, marker='o', color=colors2[i],label='temperature profile \n with error bars ')
ax.errorbar(p*100+46,np.flip(t),np.flip(e),capsize=5, marker='o', color=colors2[i],alpha=0.3)
ax.set_xlabel('R [cm]')
ax.set_ylabel('$T_\mathrm{e}(R)$ [eV]')
#ax.legend(loc='lower center',bbox_to_anchor=(0.5,-0.5),title='He, shot n$^\circ$13252, MW: 2.45 GHz, \n $P_{\mathrm{MW}}$ = 2.8 kW, p = 25.5 mPa')    
ax.set_zorder(ax.get_zorder()+1)
ax.set_frame_on(False)   
fig2= plt.gcf()
plt.show()
fig2.savefig('/home/gediz/LaTex/Thesis/Figures/Temperature_with_errorbars.pdf',bbox_inches='tight')

# %% P total table
shotnumber,gas,mw,p,t,n,P,Pmin,Pmax=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
def colorchooser(j): 
    if (math.isnan(t[j])):
        m='*'
    else:
        m='o'
    if gas[j]=='H':
        c=colors2[1]
        a_span_p=max(p_h)
        a_span_mw=max(mw_h)
    if gas[j]=='He':
        c=colors2[5]
        a_span_p=max(p_he)
        a_span_mw=max(mw_he)
    if gas[j]=='Ar':
        c=colors2[11]
        a_span_p=max(p_ar)
        a_span_mw=max(mw_ar)
    if gas[j]=='Ne':
        c=colors2[8]
        a_span_p=max(p_ne)
        a_span_mw=max(mw_ne)
    return m,c,a_span_p,a_span_mw
p_h,p_he,p_ar,p_ne,mw_h,mw_he,mw_ar,mw_ne=[],[],[],[],[],[],[],[]
for j in np.arange(0,154):
    if gas[j]=='H':
        mw_h.append(mw[j])
        p_h.append(p[j])
    if gas[j]=='He':
        mw_he.append(mw[j])
        p_he.append(p[j])
    if gas[j]=='Ar':
        mw_ar.append(mw[j])
        p_ar.append(p[j])
    if gas[j]=='Ne':
        mw_ne.append(mw[j])
        p_ne.append(p[j])
plt.figure(figsize=(width/2,width/2))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$P_{\mathrm{MW}}$ [W]')
for j in np.arange(0,154):
    a=p[j]/colorchooser(j)[2]
    if 2.5*a>=1:
        al=1
    else:
        al=2.5*a
    plt.plot(mw[j],(P[j]/mw[j])*100,marker=colorchooser(j)[0],color=colorchooser(j)[1],alpha=al)
plt.plot(mw[20],(P[20]/mw[20])*100,marker=colorchooser(20)[0],color=colorchooser(20)[1],ls='None',label='H')
plt.plot(mw[32],(P[32]/mw[32])*100,marker=colorchooser(32)[0],color=colorchooser(32)[1],ls='None',label='He')
plt.plot(mw[67],(P[67]/mw[67])*100,marker=colorchooser(67)[0],color=colorchooser(67)[1],ls='None',label='Ne')
plt.plot(mw[53],(P[53]/mw[53])*100,marker=colorchooser(53)[0],color=colorchooser(53)[1],ls='None',label='Ar')
plt.legend(loc='upper right')
fig= plt.gcf()
plt.show()
#fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_power.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,width/2))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$p$ [mPa]')
for j in np.arange(0,154):
    a=mw[j]/colorchooser(j)[3]
    plt.plot(p[j],(P[j]/mw[j])*100,marker=colorchooser(j)[0],color=colorchooser(j)[1],alpha=a)
plt.plot(p[20],(P[20]/mw[20])*100,marker=colorchooser(20)[0],color=colorchooser(20)[1],ls='None',label='H')
plt.plot(p[92],(P[92]/mw[92])*100,marker=colorchooser(92)[0],color=colorchooser(92)[1],ls='None',label='He')
plt.plot(p[117],(P[117]/mw[117])*100,marker=colorchooser(117)[0],color=colorchooser(117)[1],ls='None',label='Ar')
plt.plot(p[61],(P[61]/mw[61])*100,marker=colorchooser(61)[0],color=colorchooser(61)[1],ls='None',label='Ne')
#plt.legend(loc='upper right')
fig= plt.gcf()
plt.show()
#fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_pressure.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,width/2))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$\overline{T}_{\mathrm{e}}$ [eV]')
for j in np.arange(0,117):
    plt.plot(t[j],(P[j]/mw[j])*100,marker=colorchooser(j)[0],color=colorchooser(j)[1])
plt.plot(t[20],(P[20]/mw[20])*100,marker=colorchooser(20)[0],color=colorchooser(20)[1],ls='None',label='H')
plt.plot(t[92],(P[92]/mw[92])*100,marker=colorchooser(92)[0],color=colorchooser(92)[1],ls='None',label='He')
plt.plot(t[117],(P[117]/mw[117])*100,marker=colorchooser(117)[0],color=colorchooser(117)[1],ls='None',label='Ar')
plt.plot(t[61],(P[61]/mw[61])*100,marker=colorchooser(61)[0],color=colorchooser(61)[1],ls='None',label='Ne')
plt.legend(loc='upper right')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_temperature.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,width/2))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$\overline{n}_{\mathrm{e}}$ [m$^{-3}$]')
for j in np.arange(0,117):
    plt.plot(n[j],(P[j]/mw[j])*100,marker=colorchooser(j)[0],color=colorchooser(j)[1])
plt.plot(n[20],(P[20]/mw[20])*100,marker=colorchooser(20)[0],color=colorchooser(20)[1],ls='None',label='H')
plt.plot(n[92],(P[92]/mw[92])*100,marker=colorchooser(92)[0],color=colorchooser(92)[1],ls='None',label='He')
plt.plot(n[117],(P[117]/mw[117])*100,marker=colorchooser(117)[0],color=colorchooser(117)[1],ls='None',label='Ar')
plt.plot(n[61],(P[61]/mw[61])*100,marker=colorchooser(61)[0],color=colorchooser(61)[1],ls='None',label='Ne')
#plt.legend(loc='upper right')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_density.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,width/2))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$\overline{n}_{\mathrm{e}} \cdot \overline{T}_{\mathrm{e}}$ [eVm$^{-3}$]')
for j in np.arange(0,117):
    plt.plot(n[j]*t[j],(P[j]/mw[j])*100,marker=colorchooser(j)[0],color=colorchooser(j)[1])
plt.plot(n[20]*t[j],(P[20]/mw[20])*100,marker=colorchooser(20)[0],color=colorchooser(20)[1],ls='None',label='H')
plt.plot(n[92]*t[j],(P[92]/mw[92])*100,marker=colorchooser(92)[0],color=colorchooser(92)[1],ls='None',label='He')
plt.plot(n[117]*t[j],(P[117]/mw[117])*100,marker=colorchooser(117)[0],color=colorchooser(117)[1],ls='None',label='Ar')
plt.plot(n[61]*t[j],(P[61]/mw[61])*100,marker=colorchooser(61)[0],color=colorchooser(61)[1],ls='None',label='Ne')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_energyproduct.pdf',bbox_inches='tight')

# %% P Energieinhalt
shotnumber,gas,mw,p,t,n,P,Pmin,Pmax=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_table_245.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
def colorchooser(j): 
    if (math.isnan(t[j])):
        m='*'
    else:
        m='o'
    if gas[j]=='H':
        c=colors2[1]
        a_span_p=max(p_h)
        a_span_mw=max(mw_h)
    if gas[j]=='He':
        c=colors2[5]
        a_span_p=max(p_he)
        a_span_mw=max(mw_he)
    if gas[j]=='Ar':
        c=colors2[11]
        a_span_p=max(p_ar)
        a_span_mw=max(mw_ar)
    if gas[j]=='Ne':
        c=colors2[8]
        a_span_p=max(p_ne)
        a_span_mw=max(mw_ne)
    return m,c,a_span_p,a_span_mw
p_h,p_he,p_ar,p_ne,mw_h,mw_he,mw_ar,mw_ne=[],[],[],[],[],[],[],[]
for j in np.arange(0,117):
    if gas[j]=='H':
        mw_h.append(mw[j])
        p_h.append(p[j])
    if gas[j]=='He':
        mw_he.append(mw[j])
        p_he.append(p[j])
    if gas[j]=='Ar':
        mw_ar.append(mw[j])
        p_ar.append(p[j])
    if gas[j]=='Ne':
        mw_ne.append(mw[j])
        p_ne.append(p[j])

def func(x,a,b,c):
    return a+b*(x**c)
x=np.sort(n*t)
y=[(P[i]/mw[i])*100 for i in np.argsort(n*t)]
popt, pcov = curve_fit(func,x,y)

plt.figure(figsize=(width/2,width/2))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$\overline{n}_{\mathrm{e}} \cdot \overline{T}_{\mathrm{e}}$ [eVm$^{-3}$]')
for j in np.arange(0,117):
    plt.plot(n[j]*t[j],(P[j]/mw[j])*100,marker=colorchooser(j)[0],color=colorchooser(j)[1])
plt.plot(n[20]*t[j],(P[20]/mw[20])*100,marker=colorchooser(20)[0],color=colorchooser(20)[1],ls='None',label='H')
plt.plot(n[92]*t[j],(P[92]/mw[92])*100,marker=colorchooser(92)[0],color=colorchooser(92)[1],ls='None',label='He')
plt.plot(n[117]*t[j],(P[117]/mw[117])*100,marker=colorchooser(117)[0],color=colorchooser(117)[1],ls='None',label='Ar')
plt.plot(n[61]*t[j],(P[61]/mw[61])*100,marker=colorchooser(61)[0],color=colorchooser(61)[1],ls='None',label='Ne')
plt.plot(x,func(x,*popt))
plt.legend(loc='upper right')
print(*popt)
fig= plt.gcf()
plt.show()
#fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_energyproduct.pdf',bbox_inches='tight')

# %% P total table modeled Gases pressure
shotnumber,gas,mw,p,t,n,P,Pmin,Pmax=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
shotnumberm,gasm,mwm,pm,tm,nm,Pm,Pminm,Pmaxm=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_modeled_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
def colorchooserm(j): 
    if gasm[j]=='H':
        c=colors2[1]
    if gasm[j]=='He':
        c=colors2[5]
    if gasm[j]=='Ar':
        c=colors2[11]
    if gasm[j]=='Ne':
        c=colors2[8]
    return c


plt.figure(figsize=(width/2,height*0.7))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$p$ [mPa]')
for j in np.arange(0,116):
    if gasm[j]=='H':
        if shotnumberm[j] in [13242 ,13243, 13244, 13245 ,13246, 13247, 13248,13250 ,13251, 13252 ,13253,13254 ,13255] or shotnumberm[j] in np.arange(13089,13095):
            index=[np.argwhere(shotnumber==shotnumberm[j])][0]
            plt.errorbar(pm[j],(Pm[j]/mwm[j])*100,yerr=([(Pminm[j]/mwm[j])*100],[(Pmaxm[j]/mwm[j])*100]),marker='v',color=colorchooserm(j),capsize=5)
            plt.errorbar(p[index][0],(P[index][0]/mw[index][0])*100,yerr=([(Pmin[index][0]/mw[index][0])*100],[(Pmax[index][0]/mw[index][0])*100]),marker='o',color=colorchooserm(j),alpha=0.5,capsize=5)
           # plt.vlines(pm[j],(P[index]/mw[index])*100,(Pm[j]/mwm[j])*100,color=colorchooserm(j),lw=1)
indexm=np.argwhere(shotnumberm==13242)
index=np.argwhere(shotnumber==13242)
plt.plot(pm[indexm],(Pm[indexm]/mwm[indexm])*100,marker='v',color=colorchooserm(indexm),ls='None',label='mod')
plt.plot(p[index],(P[index]/mw[index])*100,marker='o',color=colorchooserm(indexm),ls='None',label='exp',alpha=0.5)
plt.legend(loc='upper left',title='H')
plt.ylim(0)
plt.xlim(0,50)
plt.hlines(100,0,200,lw=1,ls='dotted')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_pressure_modelled_H.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,height*0.7))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$p$ [mPa]')
for j in np.arange(0,116):
    if gasm[j]=='He':
        if shotnumberm[j] in np.arange(13268,13280):
            index=[np.argwhere(shotnumber==shotnumberm[j])][0]
            plt.errorbar(pm[j],(Pm[j]/mwm[j])*100,yerr=([(Pminm[j]/mwm[j])*100],[(Pmaxm[j]/mwm[j])*100]),marker='v',color=colorchooserm(j),capsize=5)
            plt.errorbar(p[index][0],(P[index][0]/mw[index][0])*100,yerr=([(Pmin[index][0]/mw[index][0])*100],[(Pmax[index][0]/mw[index][0])*100]),marker='o',color=colorchooserm(j),alpha=0.5,capsize=5)
            #plt.vlines(pm[j],(P[index]/mw[index])*100,(Pm[j]/mwm[j])*100,color=colorchooserm(j),lw=1)
indexm=np.argwhere(shotnumberm==13268)
index=np.argwhere(shotnumber==13268)
plt.plot(pm[indexm],(Pm[indexm]/mwm[indexm])*100,marker='v',color=colorchooserm(indexm),ls='None',label='mod')
plt.plot(p[index],(P[index]/mw[index])*100,marker='o',color=colorchooserm(indexm),ls='None',label='exp',alpha=0.5)
plt.legend(loc='upper left',title='He')
plt.ylim(0,80)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_pressure_modelled_He.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,height*0.7))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$p$ [mPa]')
for j in np.arange(0,116):
    if gasm[j]=='Ne':
        index=[np.argwhere(shotnumber==shotnumberm[j])][0]
        plt.errorbar(pm[j],(Pm[j]/mwm[j])*100,yerr=([(Pminm[j]/mwm[j])*100],[(Pmaxm[j]/mwm[j])*100]),marker='v',color=colorchooserm(j),capsize=5)
        plt.errorbar(p[index][0],(P[index][0]/mw[index][0])*100,yerr=([(Pmin[index][0]/mw[index][0])*100],[(Pmax[index][0]/mw[index][0])*100]),marker='o',color=colorchooserm(j),alpha=0.5,capsize=5)
        #plt.vlines(pm[j],(P[index]/mw[index])*100,(Pm[j]/mwm[j])*100,color=colorchooserm(j),lw=1)
indexm=np.argwhere(shotnumberm==13081)
index=np.argwhere(shotnumber==13081)
plt.plot(pm[indexm],(Pm[indexm]/mwm[indexm])*100,marker='v',color=colorchooserm(indexm),ls='None',label='mod')
plt.plot(p[index],(P[index]/mw[index])*100,marker='o',color=colorchooserm(indexm),ls='None',label='exp',alpha=0.5)
plt.legend(loc='upper left',title='Ne')
plt.xlim(0,80)
plt.hlines(100,0,200,lw=1,ls='dotted')
plt.ylim(0)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_pressure_modelled_Ne.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,height*0.7))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$p$ [mPa]')
for j in np.arange(0,116):
    if gasm[j]=='Ar':
        if shotnumberm[j] in [13299, 13300, 13301, 13302, 13303, 13304, 13305, 13306, 13307, 13308, 13309, 13310,13311,13099, 13100, 13101, 13102, 13104, 13105, 13106]:
            index=[np.argwhere(shotnumber==shotnumberm[j])][0]
            plt.errorbar(pm[j],(Pm[j]/mwm[j])*100,yerr=([(Pminm[j]/mwm[j])*100],[(Pmaxm[j]/mwm[j])*100]),marker='v',color=colorchooserm(j),capsize=5)
            plt.errorbar(p[index][0],(P[index][0]/mw[index][0])*100,yerr=([(Pmin[index][0]/mw[index][0])*100],[(Pmax[index][0]/mw[index][0])*100]),marker='o',color=colorchooserm(j),alpha=0.5,capsize=5)
            #plt.vlines(pm[j],(P[index]/mw[index])*100,(Pm[j]/mwm[j])*100,color=colorchooserm(j),lw=1)
indexm=np.argwhere(shotnumberm==13299)
index=np.argwhere(shotnumber==13299)
plt.plot(pm[indexm],(Pm[indexm]/mwm[indexm])*100,marker='v',color=colorchooserm(indexm),ls='None',label='mod')
plt.plot(p[index],(P[index]/mw[index])*100,marker='o',color=colorchooserm(indexm),ls='None',label='exp',alpha=0.5)
plt.legend(loc='upper left',title='Ar')
plt.ylim(0,80)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_pressure_modelled_Ar.pdf',bbox_inches='tight')
# %% P total table modeled Gases power
shotnumber,gas,mw,p,t,n,P,Pmin,Pmax=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
shotnumberm,gasm,mwm,pm,tm,nm,Pm,Pminm,Pmaxm=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_modeled_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
def colorchooserm(j): 
    if gasm[j]=='H':
        c=colors2[1]
    if gasm[j]=='He':
        c=colors2[5]
    if gasm[j]=='Ar':
        c=colors2[11]
    if gasm[j]=='Ne':
        c=colors2[8]
    return c


plt.figure(figsize=(width/2,width/2))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$P_{\mathrm{MW}}$ [W]')
for j in np.arange(0,116):
    if gasm[j]=='H':
        index=[np.argwhere(shotnumber==shotnumberm[j])][0]
        plt.plot(mwm[j],(Pm[j]/mwm[j])*100,marker='v',color=colorchooserm(j))
        plt.plot(mw[index],(P[index]/mw[index])*100,marker='o',color=colorchooserm(j),alpha=0.5)
        plt.vlines(mwm[j],(P[index]/mw[index])*100,(Pm[j]/mwm[j])*100,color=colorchooserm(j),lw=1)
plt.plot(mwm[0],(Pm[0]/mwm[0])*100,marker='v',color=colorchooserm(0),ls='None',label='mod')
plt.plot(mw[20],(P[20]/mw[20])*100,marker='o',color=colorchooserm(0),ls='None',label='exp',alpha=0.5)
plt.legend(loc='upper right',title='H')
plt.ylim(0)
plt.xlim(0,2900)
plt.hlines(100,0,3000,lw=1,ls='dotted')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_power_modelled_H.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,width/2))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$P_{\mathrm{MW}}$ [W]')
for j in np.arange(0,116):
    if gasm[j]=='He':
        index=[np.argwhere(shotnumber==shotnumberm[j])][0]
        plt.plot(mwm[j],(Pm[j]/mwm[j])*100,marker='v',color=colorchooserm(j))
        plt.plot(mw[index],(P[index]/mw[index])*100,marker='o',color=colorchooserm(j),alpha=0.5)
        plt.vlines(mwm[j],(P[index]/mw[index])*100,(Pm[j]/mwm[j])*100,color=colorchooserm(j),lw=1)
plt.plot(mwm[26],(Pm[26]/mwm[26])*100,marker='v',color=colorchooserm(26),ls='None',label='mod')
plt.plot(mw[32],(P[32]/mw[32])*100,marker='o',color=colorchooserm(26),ls='None',label='exp',alpha=0.5)
plt.legend(loc='upper left',title='He')
plt.ylim(0)
plt.xlim(0,2900)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_power_modelled_He.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,width/2))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$P_{\mathrm{MW}}$ [W]')
for j in np.arange(0,116):
    if gasm[j]=='Ne':
        index=[np.argwhere(shotnumber==shotnumberm[j])][0]
        plt.plot(mwm[j],(Pm[j]/mwm[j])*100,marker='v',color=colorchooserm(j))
        plt.plot(mw[index],(P[index]/mw[index])*100,marker='o',color=colorchooserm(j),alpha=0.5)
        plt.vlines(mwm[j],(P[index]/mw[index])*100,(Pm[j]/mwm[j])*100,color=colorchooserm(j),lw=1)
plt.plot(mwm[107],(Pm[107]/mwm[107])*100,marker='v',color=colorchooserm(107),ls='None',label='mod')
plt.plot(mw[67],(P[67]/mw[67])*100,marker='o',color=colorchooserm(107),ls='None',label='exp',alpha=0.5)
plt.legend(loc='upper left',title='Ne')
plt.hlines(100,0,3000,lw=1,ls='dotted')
plt.xlim(0,2900)
plt.ylim(0)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_power_modelled_Ne.pdf',bbox_inches='tight')

plt.figure(figsize=(width/2,width/2))
plt.ylabel('$P_{\mathrm{rad,net}}/P_{\mathrm{MW}}$ [\%]')
plt.xlabel('$P_{\mathrm{MW}}$ [W]')
for j in np.arange(0,116):
    if gasm[j]=='Ar':
        index=[np.argwhere(shotnumber==shotnumberm[j])][0]
        plt.plot(mwm[j],(Pm[j]/mwm[j])*100,marker='v',color=colorchooserm(j))
        plt.plot(mw[index],(P[index]/mw[index])*100,marker='o',color=colorchooserm(j),alpha=0.5)
        plt.vlines(mwm[j],(P[index]/mw[index])*100,(Pm[j]/mwm[j])*100,color=colorchooserm(j),lw=1)
plt.plot(mwm[49],(Pm[49]/mwm[49])*100,marker='v',color=colorchooserm(49),ls='None',label='mod')
plt.plot(mw[53],(P[53]/mw[53])*100,marker='o',color=colorchooserm(49),ls='None',label='exp',alpha=0.5)
plt.legend(loc='upper right',title='Ar')
plt.xlim(0,2900)
plt.ylim(0)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_power_modelled_Ar.pdf',bbox_inches='tight')

#%% Faktor modeled experimental

shotnumber,gas,mw,p,t,n,P,Pmin,Pmax=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
shotnumberm,gasm,mwm,pm,tm,nm,Pm,Pminm,Pmaxm=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_modeled_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
def colorchooserm(j): 
    if gasm[j]=='H':
        c=colors2[1]
    if gasm[j]=='He':
        c=colors2[5]
    if gasm[j]=='Ar':
        c=colors2[11]
    if gasm[j]=='Ne':
        c=colors2[8]
    return c

fig, ax= plt.subplots(figsize=(width/2,height*0.7))
ax2=ax.twinx()
ax.set_ylabel('$ \delta_{\mathrm{mod}}$')
ax2.set_ylabel('$ \delta_{\mathrm{mod, Ne}}$',color=colors2[8])
ax.set_xlabel('$n_{\mathrm{e}}$ [m$^{-3}$]')
for j in np.arange(0,116):
    if gasm[j] in ['H','He','Ar']:
        ax.plot(nm[j],(Pm[j]/P[np.argwhere(shotnumber==shotnumberm[j])][0]),marker='d',color=colorchooserm(j))
    if gasm[j] =='Ne':
        ax2.plot(nm[j],(Pm[j]/P[np.argwhere(shotnumber==shotnumberm[j])][0]),marker='d',color=colorchooserm(j))
ax2.tick_params(axis='y', labelcolor=colors2[8])
fig.patch.set_facecolor('white')

for j in [0,20,107,98]:
    plt.plot(nm[j],(Pm[j]/P[np.argwhere(shotnumber==shotnumberm[j])][0]),marker='d',color=colorchooserm(j),ls='None',label=gasm[j])
plt.legend(loc='upper left')
plt.xlim(0)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_factor_mod_exp_density_wo_neon.pdf',bbox_inches='tight')

fig, ax= plt.subplots(figsize=(width/2,height*0.7))
ax2=ax.twinx()
ax.set_ylabel('$ \delta_{\mathrm{mod}}$')
ax2.set_ylabel('$ \delta_{\mathrm{mod, Ne}}$',color=colors2[8])
ax.set_xlabel('$T_{\mathrm{e}}$ [eV]')
for j in np.arange(0,116):
    if gasm[j] in ['H','He','Ar']:
        ax.plot(tm[j],(Pm[j]/P[np.argwhere(shotnumber==shotnumberm[j])][0]),marker='d',color=colorchooserm(j))
    if gasm[j] =='Ne':
        ax2.plot(tm[j],(Pm[j]/P[np.argwhere(shotnumber==shotnumberm[j])][0]),marker='d',color=colorchooserm(j))
ax2.tick_params(axis='y', labelcolor=colors2[8])
fig.patch.set_facecolor('white')

fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_factor_mod_exp_temperature_wo_neon.pdf',bbox_inches='tight')

#%% Faktor Hist
shotnumber,gas,mw,p,t,n,P,Pmin,Pmax=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
shotnumberm,gasm,mwm,pm,tm,nm,Pm,Pminm,Pmaxm=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_modeled_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)

plt.figure(figsize=(width*1.2,height*0.5))
plt.xlabel('$ \delta_{\mathrm{mod}}$')
delt_h,delt_he,delt_ar,delt_ne=[],[],[],[]
for j in np.arange(0,116):
    if gasm[j]=='H':
        delt_h.append((Pm[j]/P[np.argwhere(shotnumber==shotnumberm[j])][0])[0])
    if gasm[j]=='He':
        delt_he.append((Pm[j]/P[np.argwhere(shotnumber==shotnumberm[j])][0])[0])
    if gasm[j]=='Ar':
        delt_ar.append((Pm[j]/P[np.argwhere(shotnumber==shotnumberm[j])][0])[0])
    if gasm[j]=='Ne':
        delt_ne.append((Pm[j]/P[np.argwhere(shotnumber==shotnumberm[j])][0])[0])
plt.hist(delt_h,bins=np.arange(0,28,0.5),color=colors2[1],alpha=0.7,label='H, $\overline{\delta}_{\mathrm{mod}}$ ='+str('%.1f' %np.mean(delt_h)))
plt.hist(delt_he,bins=np.arange(0.1,28,0.5),color=colors2[5],alpha=0.7,label='He, $\overline{\delta}_{\mathrm{mod}}$ ='+str('%.1f' %np.mean(delt_he)))
plt.hist(delt_ne,bins=np.arange(0.2,28,0.5),color=colors2[8],alpha=0.7,label='Ne, $\overline{\delta}_{\mathrm{mod}}$ ='+str('%.1f' %np.mean(delt_ne)))
plt.hist(delt_ar,bins=np.arange(0.3,28,0.5),color=colors2[11],alpha=0.7,label='Ar, $\overline{\delta}_{\mathrm{mod}}$ ='+str('%.1f' %np.mean(delt_ar)))
plt.axvline(np.mean(delt_h),color=colors2[1],ls='dotted')
plt.axvline(np.mean(delt_he),color=colors2[5],ls='dotted')
plt.axvline(np.mean(delt_ne),color=colors2[8],ls='dotted')
plt.axvline(np.mean(delt_ar),color=colors2[11],ls='dotted')
plt.legend(loc='upper right')
plt.ylabel('occurrence')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_factor_mod_exp_hist.pdf',bbox_inches='tight')


# %% Model accuracy hollowness
shotnumber,gas,mw,p,t,n,P,Pmin,Pmax=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
def colorchooser(j): 
    if gas[j]=='H':
        c=colors2[1]
    if gas[j]=='He':
        c=colors2[5]
    if gas[j]=='Ar':
        c=colors2[11]
    if gas[j]=='Ne':
        c=colors2[8]
    return c
def neu(pres):
    T=290
    k=1.38E-23
    return (pres*10**(-3))/(k*T)
arg=p
# plt.figure(figsize=(width/2,height*0.7))
# for j in np.arange(0,117):
#     plt.plot(arg[j],poca.Model_accuracy(shotnumber[j],gas[j])[4],marker='v',color=colorchooser(j))
# for j in [20,32,67,53]:
#     plt.plot(arg[j],poca.Model_accuracy(shotnumber[j],gas[j])[4],marker='v',color=colorchooser(j),ls='None',label=gas[j])
# plt.legend(loc='upper right',title='modelled')
# plt.ylabel('$h_{\mathrm{mod}}$')
# plt.xlabel('$p$ [mPa]')
# plt.ylim(0.3,1.8)
# fig= plt.gcf()
# plt.show()
# fig.savefig('/home/gediz/LaTex/Thesis/Figures/model_hollowness_pressure.pdf',bbox_inches='tight')

# plt.figure(figsize=(width/2,height*0.7))
# for j in np.arange(0,117):
#     plt.plot(arg[j],poca.Model_accuracy(shotnumber[j],gas[j])[5],marker='o',color=colorchooser(j))
# for j in [20,32,67,53]:
#     plt.plot(arg[j],poca.Model_accuracy(shotnumber[j],gas[j])[5],marker='o',color=colorchooser(j),ls='None',label=gas[j])
# plt.legend(loc='upper right',title='experimental')
# plt.ylabel('$h_{\mathrm{exp}}$')
# plt.xlabel('$p$ [mPa]')
# plt.ylim(0.3,1.8)
# fig= plt.gcf()
# plt.show()
# fig.savefig('/home/gediz/LaTex/Thesis/Figures/experiment_hollowness_pressure.pdf',bbox_inches='tight')

fig=plt.figure(figsize=(width/2,height*0.7))
for j in np.arange(0,117):
    plt.plot(poca.Model_accuracy(shotnumber[j],gas[j])[5],poca.Model_accuracy(shotnumber[j],gas[j])[5]-poca.Model_accuracy(shotnumber[j],gas[j])[4],marker='d',color=colorchooser(j))
for j in [20,32,67,53]:
    plt.plot(poca.Model_accuracy(shotnumber[j],gas[j])[5],poca.Model_accuracy(shotnumber[j],gas[j])[5]-poca.Model_accuracy(shotnumber[j],gas[j])[4],marker='d',color=colorchooser(j),ls='None',label=gas[j])
plt.legend(loc='lower right')

plt.xlabel('$h_{\mathrm{exp}}$')
plt.ylabel('$h_{\mathrm{exp}} - h_{\mathrm{mod}}$',color=colors2[8])
plt.tick_params(axis='y', labelcolor=colors2[8])
fig.patch.set_facecolor('white')
plt.ylim(-0.5,0.5)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/delta_hollowness.pdf',bbox_inches='tight')

fig=plt.figure(figsize=(width/2,height*0.7))
h,P=[],[]
for j in np.arange(0,117):
    h.append(poca.Model_accuracy(shotnumber[j],gas[j])[5]-poca.Model_accuracy(shotnumber[j],gas[j])[4])
print(np.mean(h))
plt.hist(h,bins=15, color=colors2[9],alpha=0.7)
plt.xlabel('$h_{\mathrm{exp}} - h_{\mathrm{mod}}$',color=colors2[8])
plt.tick_params(axis='x', labelcolor=colors2[8])
fig.patch.set_facecolor('white')
plt.ylabel('occurrence')
plt.xlim(-0.5,0.5)

fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/delta_hollowness_hist.pdf',bbox_inches='tight')



# %% Model accuracy diff
shotnumber,gas,mw,p,t,n,P,Pmin,Pmax=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
def colorchooser(j): 
    if gas[j]=='H':
        c=colors2[1]
    if gas[j]=='He':
        c=colors2[5]
    if gas[j]=='Ar':
        c=colors2[11]
    if gas[j]=='Ne':
        c=colors2[8]
    return c
def neu(pres):
    T=290
    k=1.38E-23
    return (pres*10**(-3))/(k*T)
arg=mw
def lin(x,a,b):
    return x*a+b
plt.figure(figsize=(width/2,height*0.7))
P,h,h_good=[],[],[]
for j in np.arange(0,117):
    P.append(poca.Model_accuracy(shotnumber[j],gas[j])[6])
    h.append(abs(poca.Model_accuracy(shotnumber[j],gas[j])[5]-poca.Model_accuracy(shotnumber[j],gas[j])[4]))
    
    if shotnumber[j] not in [13253,13260,13102,13079]:
        plt.plot(poca.Model_accuracy(shotnumber[j],gas[j])[6],abs(poca.Model_accuracy(shotnumber[j],gas[j])[5]-poca.Model_accuracy(shotnumber[j],gas[j])[4]),marker='d',color=colorchooser(j))
for j in [20,32,67,53]:
    plt.plot(poca.Model_accuracy(shotnumber[j],gas[j])[6],abs(poca.Model_accuracy(shotnumber[j],gas[j])[5]-poca.Model_accuracy(shotnumber[j],gas[j])[4]),marker='d',color=colorchooser(j),ls='None',label=gas[j])
for j in np.arange(0,117):
    if shotnumber[j] in [13253,13260,13102,13079]:
        plt.plot(poca.Model_accuracy(shotnumber[j],gas[j])[6],abs(poca.Model_accuracy(shotnumber[j],gas[j])[5]-poca.Model_accuracy(shotnumber[j],gas[j])[4]),marker='*',markeredgecolor='white',markersize=20, fillstyle='none',markeredgewidth=2.5)

        plt.plot(poca.Model_accuracy(shotnumber[j],gas[j])[6],abs(poca.Model_accuracy(shotnumber[j],gas[j])[5]-poca.Model_accuracy(shotnumber[j],gas[j])[4]),marker='*',markeredgecolor=colorchooser(j),markersize=20, fillstyle='none',markeredgewidth=2)


h_1=[h[i] for i in np.argsort(P)]
P_1=np.sort(P)
popt,pcov=curve_fit(lin,P_1,h_1)
P_2=np.arange(-1,5)
plt.plot(P_2,lin(P_2,*popt),color=colors2[6],ls='dotted')
plt.plot(P_2,lin(P_2,-0.23,0.5),color=colors2[6],ls='dotted')
h_max=0.2
P_max=1.4
for j in np.arange(0,117):
    if h[j]<=h_max:
        if P[j]<=P_max:
            h_good.append(shotnumber[j])
print(len(h_good) )
plt.axvspan(-0.05,P_max,ymin=-0.02,ymax=h_max*2+0.02, color=colors2[6],alpha=0.4)
plt.legend(loc='upper right')
plt.xlabel('$P_{\mathrm{rad, diff}}$',color=colors2[5])
plt.ylabel('$|\Delta h |$',color=colors2[8])
plt.ylim(-0.02,0.5)
plt.xlim(-0.05,3.7)
plt.tick_params(axis='x', labelcolor=colors2[5])
plt.tick_params(axis='y', labelcolor=colors2[8])
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/model_hollowness_P_rad_diff.pdf',bbox_inches='tight')



# plt.figure(figsize=(width/2,height*0.7))
# P=[]
# for j in np.arange(0,117):
#     P.append(poca.Model_accuracy(shotnumber[j],gas[j])[6])
# print(np.mean(P))
# plt.hist(P, bins=15, color=colors2[6],alpha=0.7)
# plt.xlabel('$P_{\mathrm{rad, diff}}$',color=colors2[5])
# plt.tick_params(axis='x', labelcolor=colors2[5])
# fig.patch.set_facecolor('white')
# plt.xlim(-0.05,2.5)
# plt.ylabel('occurrence')
# fig= plt.gcf()
# plt.show()
# fig.savefig('/home/gediz/LaTex/Thesis/Figures/model_P_rad_diff_hist.pdf',bbox_inches='tight')


# %% Simplified Model in Parts
shotnumberm,gasm,mwm,pm,tm,nm,Pm,Pminm,Pmaxm=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_modeled_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)

shotnumber,gas,mw,p,te,n,P,Pmin,Pmax=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)
def colorchooser(j): 
    if gas[j]=='H':
        c=colors2[1]
    if gas[j]=='He':
        c=colors2[5]
    if gas[j]=='Ar':
        c=colors2[11]
    if gas[j]=='Ne':
        c=colors2[8]
    return c
def colorchooserm(j): 
    if gasm[j]=='H':
        c=colors2[1]
    if gasm[j]=='He':
        c=colors2[5]
    if gasm[j]=='Ar':
        c=colors2[11]
    if gasm[j]=='Ne':
        c=colors2[8]
    return c
def neu(pres,T):
    k=1.38E-23
    return (pres*10**(-3))/(k*T)

flux=np.genfromtxt('/home/gediz/Results/Modeled_Data/Fluxsurfaces_and_Lines_of_sight/flux_0_to_11_total.txt',unpack=True,usecols=(1))
flux_pos=[2.174,5.081,5.844,6.637,7.461,8.324,9.233,10.19,11.207,12.321,13.321,14.321,15.321,16.321]
pec_h,den_h,full_h,pec_den_h,p_h,n_h,P_h,mw_h,t_h=[],[],[],[],[],[],[],[],[]
pec_he,den_he,full_he,pec_den_he,p_he,n_he,P_he,mw_he,t_he=[],[],[],[],[],[],[],[],[]
pec_ne,den_ne,full_ne,pec_den_ne,p_ne,n_ne,P_ne,mw_ne,t_ne=[],[],[],[],[],[],[],[],[]
pec_ar,den_ar,full_ar,pec_den_ar,p_ar,n_ar,P_ar,mw_ar,t_ar=[],[],[],[],[],[],[],[],[]
V_T_2=0.237
V=sum(flux)
for j in np.arange(0,115):
    flux_t,flux_rc,pex_full,pex_pec_den,pex_den,pex_pec=[],[],[],[],[],[]
    p_t,t,mean_t,error=pc.TemperatureProfile(shotnumber[j],'Values',save=False)
    if gas[j]=='H':
        temp,pec= adas.h_adf11(T_max=201)[0],adas.h_adf11(T_max=201)[1]
        for i in np.arange(0,len(flux_pos)-1):
            interpol_t=pchip_interpolate(p_t*100,t,np.arange(flux_pos[i],flux_pos[i+1],0.01))
            flux_t.append(np.mean(interpol_t))
            interpol_rc=pchip_interpolate(temp,pec,flux_t[i])
            flux_rc.append(interpol_rc)
        for i in np.arange(0,len(flux_pos)-1):
            pex_full.append((flux_rc[i]*flux[i]*n[j]*(neu(p[j],270)-n[j]))*1.602E-19)
            pex_pec_den.append((flux_rc[i]*n[j]*(neu(p[j],270)-n[j])))
            pex_pec.append(flux_rc[i]*flux[i])
            pex_den.append(n[j]*(neu(p[j],270)-n[j]))
        den_h.append(np.mean(pex_den))
        pec_h.append(np.mean(pex_pec)/V)
        pec_den_h.append(np.mean(pex_pec_den))
        full_h.append((np.sum(pex_full)/V)*V_T_2)
        p_h.append(p[j])
        n_h.append(n[j])
        P_h.append(P[j])
        mw_h.append(mw[j])
        t_h.append(te[j])
    if gas[j]=='He':
        temp,pec,pec_2= adas.he_adf11(data='plt96_he')[0],adas.he_adf11(data='plt96_he')[1],adas.he_adf11(data='plt96_he')[2]
        for i in np.arange(0,len(flux_pos)-1):
            interpol_t=pchip_interpolate(p_t*100,t,np.arange(flux_pos[i],flux_pos[i+1],0.01))
            flux_t.append(np.mean(interpol_t))
            interpol_rc=pchip_interpolate(temp,pec,flux_t[i])
            flux_rc.append(interpol_rc)
        for i in np.arange(0,len(flux_pos)-1):
            pex_full.append((flux_rc[i]*flux[i]*n[j]*(neu(p[j],270)-n[j]))*1.602E-19)
            pex_pec_den.append((flux_rc[i]*n[j]*(neu(p[j],270)-n[j])))
            pex_pec.append(flux_rc[i]*flux[i])
            pex_den.append(n[j]*(neu(p[j],270)-n[j]))
        den_he.append(np.mean(pex_den))
        pec_he.append(np.mean(pex_pec)/V)
        pec_den_he.append(np.mean(pex_pec_den))
        full_he.append((np.sum(pex_full)/V)*V_T_2)
        p_he.append(p[j])
        n_he.append(n[j])
        P_he.append(P[j])
        mw_he.append(mw[j])
        t_he.append(te[j])
    if gas[j]=='Ne':
        temp,pec,pec_2= adas.ne_adf11(data='plt96_ne')[0],adas.ne_adf11(data='plt96_ne')[1],adas.he_adf11(data='plt96_he')[2]
        for i in np.arange(0,len(flux_pos)-1):
            interpol_t=pchip_interpolate(p_t*100,t,np.arange(flux_pos[i],flux_pos[i+1],0.01))
            flux_t.append(np.mean(interpol_t))
            interpol_rc=pchip_interpolate(temp,pec,flux_t[i])
            flux_rc.append(interpol_rc)
        for i in np.arange(0,len(flux_pos)-1):
            pex_full.append((flux_rc[i]*flux[i]*n[j]*(neu(p[j],270)-n[j]))*1.602E-19)
            pex_pec_den.append((flux_rc[i]*n[j]*(neu(p[j],270)-n[j])))
            pex_pec.append(flux_rc[i]*flux[i])
            pex_den.append(n[j]*(neu(p[j],270)-n[j]))
        den_ne.append(np.mean(pex_den))
        pec_ne.append(np.mean(pex_pec)/V)
        pec_den_ne.append(np.mean(pex_pec_den))
        full_ne.append((np.sum(pex_full)/V)*V_T_2)
        p_ne.append(p[j])
        n_ne.append(n[j])
        P_ne.append(P[j])
        mw_ne.append(mw[j])
        t_ne.append(te[j])
    if gas[j]=='Ar':
        temp,pec,pec_2= adas.ar_adf11()[0],adas.ar_adf11()[1],adas.ar_adf11()[2]
        for i in np.arange(0,len(flux_pos)-1):
            interpol_t=pchip_interpolate(p_t*100,t,np.arange(flux_pos[i],flux_pos[i+1],0.01))
            flux_t.append(np.mean(interpol_t))
            interpol_rc=pchip_interpolate(temp,pec,flux_t[i])
            flux_rc.append(interpol_rc)
        for i in np.arange(0,len(flux_pos)-1):
            pex_full.append((flux_rc[i]*flux[i]*n[j]*(neu(p[j],270)-n[j]))*1.602E-19)
            pex_pec_den.append((flux_rc[i]*n[j]*(neu(p[j],270)-n[j])))
            pex_pec.append(flux_rc[i]*flux[i])
            pex_den.append(n[j]*(neu(p[j],270)-n[j]))
        den_ar.append(np.mean(pex_den))
        pec_ar.append(np.mean(pex_pec)/V)
        pec_den_ar.append(np.mean(pex_pec_den))
        full_ar.append((np.sum(pex_full)/V)*V_T_2)
        p_ar.append(p[j])
        n_ar.append(n[j])
        P_ar.append(P[j])
        mw_ar.append(mw[j])
        t_ar.append(te[j])

# fig,ax=plt.subplots(figsize=(width/2,height*0.7))
# ax.set_yscale('log')
# ax.set_ylabel(r'$n_{\mathrm{e}} \cdot n_{\mathrm{0}} \cdot \left\langle  \sigma v \right\rangle _{\textrm{rad}}\left\langle E_{\textrm{rad}}\right\rangle$ [eV/m$^{3}$s]')
# ax.set_xlabel('$p$ [mPa]')
# ax.plot(p_h,pec_den_h,color=colors2[1],marker='d',ls='None',label='H')
# ax.plot(p_he,pec_den_he,color=colors2[5],marker='d',ls='None',label='He')
# ax.plot(p_ne,pec_den_ne,color=colors2[8],marker='d',ls='None',label='Ne')
# ax.plot(p_ar,pec_den_ar,color=colors2[11],marker='d',ls='None',label='Ar')
# plt.legend(loc='lower right')
# fig= plt.gcf()
# plt.show()
# fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_mean_densityproduct_and_pec_pressure.pdf',bbox_inches='tight')

fig,ax=plt.subplots(figsize=(width/3,height*0.7))
ax.set_yscale('log')
plt.gca().yaxis.set_major_formatter(ScalarFormatter()) 
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
ax.minorticks_off()
ax.ticklabel_format(axis='y', style='sci')
ax.set_ylabel('$n_{\mathrm{e}} \cdot n_{\mathrm{0}}$ [m$^{-6}$]',color=colors2[8])
ax.tick_params(axis='y', labelcolor=colors2[8])
fig.patch.set_facecolor('white')
ax.set_xlabel('$p$ [mPa]')
ax.plot(p_h,den_h,color=colors2[1],marker='d',ls='None',label='H')
ax.plot(p_he,den_he,color=colors2[5],marker='d',ls='None',label='He')
ax.plot(p_ne,den_ne,color=colors2[8],marker='d',ls='None',label='Ne')
ax.plot(p_ar,den_ar,color=colors2[11],marker='d',ls='None',label='Ar')
plt.legend(loc='lower right')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_mean_densityproduct_pressure.pdf',bbox_inches='tight')

fig,ax=plt.subplots(figsize=(width/3,height*0.7))
ax.set_yscale('log')
plt.gca().yaxis.set_major_formatter(ScalarFormatter()) 
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
ax.minorticks_off()
ax.ticklabel_format(axis='y', style='sci')
ax.set_ylabel('$\overline{\sigma}_{\mathrm{rad,}f}$ [eVm$^3$/s]',color=colors2[5])
ax.tick_params(axis='y', labelcolor=colors2[5])
fig.patch.set_facecolor('white')
ax.set_xlabel('$T_{\mathrm{e}}$ [eV]')
ax.plot(t_h,pec_h,color=colors2[1],marker='d',ls='None',label='H')
ax.plot(t_he,pec_he,color=colors2[5],marker='d',ls='None',label='He')
ax.plot(t_ne,pec_ne,color=colors2[8],marker='d',ls='None',label='Ne')
ax.plot(t_ar,pec_ar,color=colors2[11],marker='d',ls='None',label='Ar')
#plt.legend(loc='lower right')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_mean_pec_temperature.pdf',bbox_inches='tight')

fig,ax=plt.subplots(figsize=(width/3,height*0.7))
ax.set_yscale('log')
ax.set_ylabel('$P_{\mathrm{rad,net,mod}}$ [W]')
ax.set_xlabel('$p$ [mPa]')
ax.plot(p_h,full_h,color=colors2[1],marker='d',ls='None',label='H')
ax.plot(p_he,full_he,color=colors2[5],marker='d',ls='None',label='He')
ax.plot(p_ne,full_ne,color=colors2[8],marker='d',ls='None',label='Ne')
ax.plot(p_ar,full_ar,color=colors2[11],marker='d',ls='None',label='Ar')
#plt.legend(loc='lower right')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/all_studies_p_net_mod_pressure.pdf',bbox_inches='tight')

# fig,ax=plt.subplots(figsize=(width/2,height*0.7))
# ax2=ax.twinx()
# ax.plot(p_h,pec_h,color=colors2[1],marker='v',ls='None')
# ax2.plot(p_h,P_h,color=colors2[1],marker='o',ls='None')
# plt.show()
# fig,ax=plt.subplots(figsize=(width/2,height*0.7))
# ax2=ax.twinx()
# ax.plot(p_he,pec_he,color=colors2[5],marker='v',ls='None')
# ax2.plot(p_he,P_he,color=colors2[5],marker='o',ls='None')
# plt.show()
# fig,ax=plt.subplots(figsize=(width/2,height*0.7))
# ax2=ax.twinx()
# ax.plot(p_ne,pec_ne,color=colors2[8],marker='v',ls='None')
# ax2.plot(p_ne,P_ne,color=colors2[8],marker='o',ls='None')
# plt.show()
# fig,ax=plt.subplots(figsize=(width/2,height*0.7))
# ax2=ax.twinx()
# ax.plot(p_ar,pec_ar,color=colors2[11],marker='v',ls='None')
# ax2.plot(p_ar,P_ar,color=colors2[11],marker='o',ls='None')
# plt.show()

# fig,ax=plt.subplots(figsize=(width/2,height*0.7))
# ax2=ax.twinx()
# ax.plot(mw_h,pec_h,color=colors2[1],marker='v',ls='None')
# ax2.plot(mw_h,P_h,color=colors2[1],marker='o',ls='None')
# plt.show()
# fig,ax=plt.subplots(figsize=(width/2,height*0.7))
# ax2=ax.twinx()
# ax.plot(mw_he,pec_he,color=colors2[5],marker='v',ls='None')
# ax2.plot(mw_he,P_he,color=colors2[5],marker='o',ls='None')
# plt.show()
# fig,ax=plt.subplots(figsize=(width/2,height*0.7))
# ax2=ax.twinx()
# ax.plot(mw_ne,pec_ne,color=colors2[8],marker='v',ls='None')
# ax2.plot(mw_ne,P_ne,color=colors2[8],marker='o',ls='None')
# plt.show()
# fig,ax=plt.subplots(figsize=(width/2,height*0.7))
# ax2=ax.twinx()
# ax.plot(mw_ar,pec_ar,color=colors2[11],marker='v',ls='None')
# ax2.plot(mw_ar,P_ar,color=colors2[11],marker='o',ls='None')
# plt.show()
# %% Degree of ioniz
shotnumber,gas,mw,p,t,n,P,Pmin,Pmax=np.genfromtxt('/home/gediz/Results/Modeled_Data/Tota_P_rad/P_total_table.txt',unpack=True,dtype=[int,'<U19',float,float,float,float,float,float,float],delimiter=',',encoding=None)

def colorchooser(j): 
    if gas[j]=='H':
        c=colors2[1]
    if gas[j]=='He':
        c=colors2[5]
    if gas[j]=='Ar':
        c=colors2[11]
    if gas[j]=='Ne':
        c=colors2[8]
    return c
def neu(pres):
    T=290
    k=1.38E-23
    return (pres*10**(-3))/(k*T)

for j in np.arange(0,117):
    plt.plot(t[j],n[j]/neu(p[j]),'o',color=colorchooser(j))
plt.show()

for j in np.arange(0,117):
    plt.plot(t[j],n[j],'o',color=colorchooser(j))
plt.show()

for j in np.arange(0,117):
    plt.plot(t[j],neu(p[j]),'o',color=colorchooser(j))
plt.show()
# %%
