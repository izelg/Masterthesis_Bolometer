#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.patches as patches
from scipy.signal import savgol_filter
from scipy.interpolate import pchip_interpolate
from scipy import integrate
from scipy import signal

import plasma_characteristics as pc
import bolo_radiation as br
import adas_data as adas
import power_calculator as poca
#%% Parameter
Poster=False
Latex=True
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
    w=10
    h=7
elif Latex==True:
    w=412/72.27
    h=w*(5**.5-1)/2
    n=1.4
    plt.rcParams['text.usetex']=True
    plt.rcParams['font.family']='serif'
    plt.rcParams['axes.labelsize']=11*n
    plt.rcParams['font.size']=11*n
    plt.rcParams['legend.fontsize']=11*n
    plt.rcParams['xtick.labelsize']=11*n
    plt.rcParams['ytick.labelsize']=11*n
    plt.rcParams['lines.markersize']=6

else:
    w=10
    h=7
    plt.rc('font',size=14)
    plt.rc('figure', titlesize=15)
colors=['#1bbbe9','#023047','#ffb703','#fb8500','#c1121f','#780000','#6969B3','#D81159','#1bbbe9','#023047','#ffb703','#fb8500','#c1121f']
markers=['o','v','s','P','p','D','*','x']
colors2=['#03045E','#0077B6','#00B4D8','#370617','#9D0208','#DC2F02','#F48C06','#FFBA08','#3C096C','#7B2CBF','#C77DFF','#2D6A4F','#40916C','#52B788','#03045E','#0077B6','#00B4D8']

#%% Plasmatypes and regimes

#n=np.arange(10E5, 10E35)
#T=np.arange(10E-2,10E6)

plt.figure(figsize=(w,h))
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


fig=plt.figure(figsize=(w,w))
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
plt.annotate('slit',(a-b-0.1,-0.5),xymath=(a-b-20,-25),arrowprops=dict(facecolor=c,edgecolor='none',alpha=0.5,width=2),color=c)
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
plt.annotate('bolometer\n   heads',(a,-2),xymath=(a-15,-27),arrowprops=dict(facecolor=c,edgecolor='none',alpha=0.5,width=2),color=c)
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

fig=plt.figure(figsize=(w,w))
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

fig=plt.figure(figsize=(w,w))
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
fig=plt.figure(figsize=(w,w))
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
fig=plt.figure(figsize=(w,h))

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

plt.figure(figsize=(w,h))
plt.plot(T_LR,He_0_LR,color=colors[0],marker=markers[0],label='He$^0$, excitation, source: ADF11 ')
plt.plot(T_LR,He_1_LR,color=colors[0],marker=markers[0],label='He${^+1}$, excitation, source: ADF11 ',alpha=0.5)

plt.plot(T_RR_0,He_0_RR,color=colors[3],marker=markers[3],label=r'He$^{+1} \rightarrow$ He$^0$, recombination, source: ADF15')
plt.plot(T_RR_1,He_1_RR,color=colors[3],marker=markers[3],label=r'He$^{+2} \rightarrow$ He$^{+1}$, recombination, source: ADF15',alpha=0.5)

#plt.plot(T_BR_RR,He_0_BR_RR,color=colors[1],marker=markers[1],label='He$^0$,Bremsstrahlung and recombination, source: ADF11 ')
#plt.plot(T_BR_RR,He_1_BR_RR,color=colors[1],marker=markers[1],label='He$^{+1}$,Bremsstrahlung and recombination, source: ADF11 ',alpha=0.5)

plt.ylabel('collisional radiative coefficients [eVm$^3$/s]')
plt.xlabel('temperature [eV]')
plt.yscale('log')
plt.ylim(1E-20,1E-12)
plt.xlim(-1,30)
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.6))
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

size=w/4 
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
fig0.savefig('/home/gediz/LaTex/Thesis/Figures/inner_port.pdf',bbox_inches='tight')

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
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/upper_port.pdf',bbox_inches='tight')

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
fig2.savefig('/home/gediz/LaTex/Thesis/Figures/outer_port.pdf',bbox_inches='tight')

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
fig3.savefig('/home/gediz/LaTex/Thesis/Figures/bottom_port.pdf',bbox_inches='tight')
# %% I-V curve
fig,ax=plt.subplots(figsize=(w,h))
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
fig=plt.figure(figsize=(w/2,h))
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
plt.plot(Position, I_isat,color=colors[0])
plt.plot(Position,I_isat/np.sqrt(T))
plt.plot(Position, I_isat/np.sqrt(T)*Interferometer)
plt.show()
plt.plot(Position, Interferometer/max(Interferometer),color=colors[1])
plt.show()
plt.plot(p_t,T/max(T),color=colors[2])
plt.show()

# %% Gold Absorption

h1=4.135E-15
c=299792458
fig,ax=plt.subplots(figsize=(w,h))
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
ax.set_xlim(0.0011,140000)
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
#fig.savefig('/home/gediz/LaTex/Thesis/Figures/Gold_Absorption.pdf',bbox_inches='tight')

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
fig,ax1=plt.subplots(figsize=(h,h))
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
fig1=plt.figure(figsize=(h,h))
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

fig2=plt.figure(figsize=(h,h))
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
plt.figure(figsize=(w*0.75,h))
location='/home/gediz/Measurements/Lines_of_sight/shot_data/shot60038.dat'
cut=1000
time = br.LoadData(location)['Zeit [ms]'][cut:-1] / 1000
for i in [1,2,3,4,5,6,7,8]:
    y= br.LoadData(location)["Bolo{}".format(i)][cut:-1]
    background=np.mean(y[-500:-1])
    y_1=y-background
    plt.plot(time,-y_1,color=colors2[i+1], label='sensor {}'.format(i))
plt.xlabel('time [s]')
plt.ylabel('sensor signal [V]')
plt.annotate('1',(22,0.01),xymath=(15,0.15),arrowprops=dict(facecolor='#008e0c',edgecolor='none',alpha=0.5,width=1,headwidth=5), bbox={"boxstyle" : "circle","facecolor":'None','edgecolor':'#008e0c'},color='#008e0c')
plt.annotate('2',(35,0.01),xymath=(32,0.15),arrowprops=dict(facecolor='#008e0c',edgecolor='none',alpha=0.5,width=1,headwidth=5), bbox={"boxstyle" : "circle","facecolor":'None','edgecolor':'#008e0c'},color='#008e0c')

plt.legend(fontsize=9)
plt.xlim(-10,120)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/Laser_scan.pdf',bbox_inches='tight')

# %% UV scan
plt.figure(figsize=(w,h))
location='/home/gediz/Measurements/Lines_of_sight/shot_data/shot60078_cropped.dat'
cut=0
c=0
time = br.LoadData(location)['Zeit [ms]'][cut:-1] / 1000
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

plt.xlabel('time [s]')
plt.ylabel('sensor signal [V]')
plt.legend(loc='upper right',fontsize=9)
plt.xlim(0,250)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/UV_scan_horizontal.pdf',bbox_inches='tight')

# %% Spectra working gases
plt.figure(figsize=(h,h))
d=5e+17
t=10
al=0.8
k=1.13
l2e=lambda x:(h1*c)/(x*10**(-9))
hdata=adas.h_adf15(T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in hdata[0]],[a/max(hdata[1]) for a in hdata[1]],k,color=colors2[9],alpha=al,label='H$^0$')
hedata=adas.he_adf15(data='pec96#he_pju#he0',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in hedata[0]],[a/max(hedata[1]) for a in hedata[1]],k,color=colors2[1],alpha=al,label='He$^0$')
nedata=adas.ne_adf15(data='pec96#ne_pju#ne0',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in nedata[0]],[a/max(nedata[1]) for a in nedata[1]],k,color=colors2[5],alpha=al,label='Ne$^0$')
ardata=adas.ar_adf15(data='pec40#ar_ls#ar0',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in ardata[0]],[a/max(ardata[1]) for a in ardata[1]],k,color=colors2[12],alpha=al,label='Ar$^0$')
plt.plot(energy,fitted/100,color='red',ls='dotted',alpha=0.5)
plt.xlim(-1,75)
plt.ylabel('normalized pec [m$^3$/s]')
plt.xlabel('photon energy [eV]')
plt.legend(loc='lower right', title='ADAS data \n for spectral \n lines at  \n $T_e=$10 eV, \n $n_e$=5$\cdot 10^{17}$ m$^{-3}$ \n due to excitation \n of neutrals ')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/spectra_neutrals.pdf',bbox_inches='tight')

k=1.5
plt.figure(figsize=(h,h))
hedata=adas.he_adf15(data='pec96#he_pju#he1',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in hedata[0]],[a/max(hedata[1]) for a in hedata[1]],k,color=colors2[2],alpha=al,label='He$^1$')
nedata=adas.ne_adf15(data='pec96#ne_pju#ne1',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in nedata[0]],[a/max(nedata[1]) for a in nedata[1]],k,color=colors2[6],alpha=al,label='Ne$^1$')
ardata=adas.ar_adf15(data='pec40#ar_ic#ar1',T_max=t,density=d,Spectrum=True)
plt.bar([l2e(a) for a in ardata[0]],[a/max(ardata[1]) for a in ardata[1]],k,color=colors2[13],alpha=al,label='Ar$^1$')
plt.plot(energy,fitted/100,color='red',ls='dotted',alpha=0.5)

plt.xlim(-1,100)
plt.ylabel('normalized pec [m$^3$/s]')
plt.xlabel('photon energy [eV]')
plt.legend(loc='lower right', title='ADAS data \n for spectral \n lines at  \n $T_e=$10 eV, \n $n_e$=5$\cdot 10^{17}$ m$^{-3}$ \n due to excitation \n of ions')

fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/spectra_ions.pdf',bbox_inches='tight')
# %% Reduced Spectrum
d=5e+17
t=10
fig,ax=plt.subplots(figsize=(h,h))
ax2=ax.twinx()
l2e=lambda x:(h1*c)/(x*10**(-9))
hdata=adas.he_adf15(data='pec96#he_pju#he0',T_max=t,density=d,Spectrum=True)
energy=[round(l2e(a),3) for a in hdata[0]]
pec=[a/max(hdata[1]) for a in hdata[1]]
gold_energy=np.arange(0,round(max(energy),2)+5,0.001)
gold=pchip_interpolate(all_energy,all_abs,gold_energy)
reduced_pec=[]
for i,j in zip(energy,pec):
    indice= int(round(i,3)*1000)
    print(i,int(round(i,3)*1000))
    reduced_pec.append(j*gold[indice]/100)
ax.bar(energy,pec,0.5,color=colors2[2],label='ADAS data for spectral lines at \n$T_e=$10 eV,$n_e$=5$\cdot 10^{17}$ m$^{-3}$ \n due to excitation of H$^0$ ')   
ax.bar(energy,reduced_pec,0.5,color=colors2[1],label='reduced spectrum:\n {}\% absorbed by gold foil'.format(float(f'{np.sum(reduced_pec)/np.sum(pec)*100:.2f}')))
ax2.plot(gold_energy,gold,color=colors2[5],ls='dotted',label='gold absorption \n characteristic \n fit to data')
ax.set_ylabel('normalized pec [m$^3$/s]')
ax.set_xlabel('photon energy [eV]')
ax2.set_ylabel('absorption gold [\%]',color=colors2[5])
ax2.tick_params(axis='y', labelcolor=colors2[5])
ax.legend(loc='lower center',bbox_to_anchor=(0.5,-0.9))
fig= plt.gcf()
plt.show()
#fig.savefig('/home/gediz/LaTex/Thesis/Figures/reduced_spectrum_H.pdf',bbox_inches='tight')


# fig,ax=plt.subplots(figsize=(h,h))
# ax2=ax.twinx()
# l2e=lambda x:(h1*c)/(x*10**(-9))
# wl,counts=np.genfromtxt('/home/gediz/Results/Spectrometer/Spectra_of_laser_and_white_light_22_09_2022/spectrometer_data_of_lightsource_Weißlichtquelle_Wellenlängenmessung.txt',unpack=True)
# energy=[round(l2e(a),3) for a in wl]
# gold_energy=np.arange(0,round(energy[0],2),0.001)
# gold=pchip_interpolate(all_energy,all_abs,gold_energy)
# reduced_counts=[]
# for i,j in zip(energy,counts):
#     indice= int(round(i,3)*1000)
#     reduced_counts.append(j*gold[indice]/100)
# lns1=ax.plot(energy,counts,color=colors2[2],label='specrometer data for a \n white light source ')   
# lns2=ax.plot(energy,reduced_counts,color=colors2[1],label='reduced spectrum:\n {}\% absorbed by gold foil'.format(float(f'{integrate.trapezoid(reduced_counts,energy)/integrate.trapezoid(counts,energy)*100:.2f}')))
# ax2.plot(gold_energy,gold,color=colors2[5],ls='dotted')
# ax.set_ylabel('counts')
# ax.set_xlabel('photon energy [eV]')
# ax2.set_ylabel('absorption gold [\%]',color=colors2[5])
# ax2.tick_params(axis='y', labelcolor=colors2[5])
# leg = lns1 + lns2 
# labs = [l.get_label() for l in leg]
# ax.legend(leg, labs, loc='lower center',bbox_to_anchor=(0.5,-0.8))
# ax.set_xlim(0,4)
# fig= plt.gcf()
# plt.show()
# fig.savefig('/home/gediz/LaTex/Thesis/Figures/reduced_spectrum_white.pdf',bbox_inches='tight')


# %% electrical signals
fig1,ax1=plt.subplots(figsize=(h,h))
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


fig2,ax2=plt.subplots(figsize=(h,h))
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


fig1,ax1=plt.subplots(figsize=(h,h))
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


fig2,ax2=plt.subplots(figsize=(h,h))
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
plt.figure(figsize=(h,h))
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
plt.ylim(1170,1230)
plt.legend(loc='lower left')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/resistance_m1.pdf',bbox_inches='tight')

plt.figure(figsize=(h,h))
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
plt.ylim(1170,1230)
plt.legend(loc='lower left')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/resistance_m2.pdf',bbox_inches='tight')

plt.figure(figsize=(h,h))
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
plt.ylim(1170,1230)
plt.legend(loc='lower left')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/resistance_r1.pdf',bbox_inches='tight')

plt.figure(figsize=(h,h))
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
plt.ylim(1170,1230)
plt.legend(loc='lower left')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/resistance_r2.pdf',bbox_inches='tight')

# %%Lamp spectra
plt.figure(figsize=(h,h))
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
plt.figure(figsize=(h,h))
location=  '/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
time = np.array(br.LoadData(location)['Zeit [ms]'] / 1000)[:,None]
for i,c in zip(np.arange(1,9),colors):
    bolo_raw_data = np.array(br.LoadData(location)["Bolo{}".format(i)])[:,None]
    m=min(bolo_raw_data)
    bolo_raw_data=[(k-m)+i*0.05 for k in bolo_raw_data]
    plt.plot(time,  bolo_raw_data, label="sensor {}".format(i),color=c )
plt.xlabel('time [s]')
plt.ylabel('$U_{\mathrm{out}}$ [V]')
plt.legend(loc='center right',bbox_to_anchor=(1.5,0.5),title='shot n$^\circ$13257,\nHe,\nMW= 2.45 GHz,\n$P_{\mathrm{MW}}$= 2.97 kW,\np= 21 mPa')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/voltage_time_trace.pdf',bbox_inches='tight')



# %% Power time trace with height
shotnumber=13257
plt.figure(figsize=(h,h))
i=1
location=  '/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
time,U_Li =br.LoadData(location)['Zeit [ms]'] / 1000, br.LoadData(location)["Bolo{}".format(i)]
def power(g,k,U_ac, t, U_Li):
    return (np.pi/g) * (2*k/U_ac) * (t* np.gradient(U_Li,time*1000 )+U_Li)
def error(g,k,U_ac, t, U_Li,delta_t,delta_k):
    return ((np.pi/g) * (2/U_ac) * (t* np.gradient(U_Li,time*1000 )+U_Li))*delta_k+(np.pi/g) * (2*k/U_ac) * np.gradient(U_Li,time*1000 )*delta_t
tau,tau_sem,kappa,kappa_sem=np.genfromtxt('/home/gediz/Results/Calibration/Ohmic_Calibration/Ohmic_Calibration_Vacuum_November/ohmic_calibration_vacuum_tjk_tau_and_kappa_mean_and_sem.txt',unpack=True,usecols=(1,2,3,4))
g1,g2,g3= 30,1,100
g=g1*g2*g3
U_ac,k,t,delta_t,delta_k=8,kappa[i-1],tau[i-1],tau_sem[i-1],kappa_sem[i-1]
power=[a/10**(-6) for a in power(g,k,U_ac, t, U_Li)]
steps=[]
for i in np.arange(0, len(power)-10):
    step= (power[i]-power[i+10])
    steps.append(abs(step))
start,stop=np.argwhere(np.array([steps])>0.5)[0][1],np.argwhere(np.array([steps])>0.5)[-1][1]
x1,y1,x2,y2 = time[start:stop],power[start:stop],np.concatenate((time[0:start],time[stop:-1])),np.concatenate((power[0:start],power[stop:-1]))
def lin (x,a,b):
    return a*x + b
popt1, pcov1 = curve_fit(lin,x1,y1)
popt2, pcov2 = curve_fit(lin,x2,y2)
div1 = lin(time[start], *popt2)-lin(time[start], *popt1)
div2 = lin(time[stop], *popt2)-lin(time[stop], *popt1)
div_avrg = abs(float(f'{(div1+div2)/2:.4f}'))
sd=np.std([div1,div2],ddof=1)
sem=sd/np.sqrt(2)
plt.figure(figsize=(h,h))
plt.plot(time[start+15],power[start+15],marker='x',color=colors[1])
plt.plot(time[stop],power[stop],marker='x',color=colors[1])
plt.plot(time,power,color=colors[0],label='sensor 1')
plt.plot(np.arange(0,240),lin(np.arange(0,240),*popt1),color=colors[3],label='fit to $P_{\mathrm{rad}} (t_{\mathrm{plasma \ on}})$')
plt.plot(np.arange(0,240),lin(np.arange(0,240),*popt2),color=colors[4],label='fit to $P_{\mathrm{rad}} (t_{\mathrm{plasma \ off}})$')
plt.annotate('',xy=(50,lin(50,*popt1)),xytext=(50,lin(50,*popt2)), arrowprops=dict(arrowstyle='<->',color=colors[1],alpha=0.7,linewidth=2))
plt.annotate('$\Delta P_{\mathrm{rad}} \cdot a_{\mathrm{abs}}$='+str(f'{div_avrg:.2f}')+'$\mu$W',xy=(55,2.5),color=colors[1])
plt.xlabel('time [s]')
plt.ylabel('$P_{\mathrm{rad}}$  [$\mu$W]')
plt.legend(loc='lower right')
plt.ylim(-0.3)
plt.xlim(-10,249)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/power_time_trace.pdf',bbox_inches='tight')


# %% Bolometer profile
plt.figure(figsize=(h,h))
for shotnumber,i in zip([13265,13263,13261,13259,13257],[0,1,2,3,4]):
    s,p,e=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot{s}/shot{s}_bolometerprofile_from_radiation_powers.txt'.format(s=shotnumber),unpack=True)
    plt.errorbar(s,p,yerr=e,marker=markers[0+i],color=colors[0+i],capsize=5,alpha=0.1)
for shotnumber,i in zip([13265,13263,13261,13259,13257],[0,1,2,3,4]):
    s,c,p,e=np.genfromtxt('/home/gediz/Results/Modeled_Data/Bolometerprofiles/shot{s}/shot{s}_modeled_powerprofile_He.txt'.format(s=shotnumber),unpack=True)
    plt.errorbar(s,p,yerr=e,marker=markers[0+i],color=colors[0+i],capsize=5,label='shot n$^\circ$'+str(shotnumber) +', $P_{\mathrm{MW}}$='+str(f'{pc.GetMicrowavePower(shotnumber)[0]/1000:.1f}') +'kW')
plt.ylim(0)
plt.xticks(s)
plt.xlabel('sensor number')
plt.ylabel('$\Delta P_{\mathrm{rad}}$ [$\mu$W]')
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-1.0),title='He, MW=2.45 GHz, p=21 mPa')
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/radiation_profiles.pdf',bbox_inches='tight')

# %% Total Power
plt.figure(figsize=(h,h))
shotnumbers=[[13265,13263,13261,13259,13257]]
gases=[['He'for i in range(5)]]
mw,p,emi,ema,c,ecma,ecmi=poca.Totalpower_from_exp(shotnumbers,gases,'Power')
plt.plot(mw,p,ls='dashed',label='He, MW=2.45 GHz, p=21 mPa')
for i,j,k,m,a in zip(mw,p,emi,ema,[0,1,2,3,4]):
    plt.plot(i,j,marker=markers[0+a],color=colors[0+a])
    plt.errorbar(i,j,yerr=[[k],[m]],capsize=5,linestyle='None',color=colors[0+a])
    plt.annotate(str(f'{(j/i)*100:.1f}')+'\%',xy=(i,j),xytext=(i-300,j+60),color=colors[0+a])
plt.xlabel('$P_{\mathrm{MW}}$ [W]')
plt.ylabel('$P_{\mathrm{rad,\ net}}$ [W]')
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.45))
plt.xlim(0,3100)
plt.ylim(0,650)
fig= plt.gcf()
plt.show()
fig.savefig('/home/gediz/LaTex/Thesis/Figures/net_power_loss.pdf',bbox_inches='tight')

# %%
