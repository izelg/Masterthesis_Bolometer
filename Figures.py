#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.patches as patches

import plasma_characteristics as pc
import bolo_radiation as br
import adas_data as adas
#%% Parameter
Poster=False
Latex=True


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
    plt.rcParams['text.usetex']=True
    plt.rcParams['font.family']='serif'
    plt.rcParams['axes.labelsize']=10
    plt.rcParams['font.size']=10
    plt.rcParams['legend.fontsize']=10
    plt.rcParams['xtick.labelsize']=10
    plt.rcParams['ytick.labelsize']=10
    plt.rcParams['lines.markersize']=6

else:
    w=10
    h=7
    plt.rc('font',size=14)
    plt.rc('figure', titlesize=15)
colors=['#1bbbe9','#023047','#ffb703','#fb8500','#c1121f','#780000','#6969B3','#D81159','#1bbbe9','#023047','#ffb703','#fb8500','#c1121f']
markers=['o','v','s','P','p','D','*','x']

#%% Plasmatypes and regimes--------------------------------------------------------------------------------------------------------------------------------------------------

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

# %% Lines of sight setup-----------------------------------------------------------------------------------------------------------------------------------------------------

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

fig=plt.figure(figsize=(w,h))
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
ax=fig.add_subplot(111)
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')

x=np.arange(40,a,0.1)


plt.xlabel('R [cm]',fontsize=30)
plt.ylabel('z [cm]',fontsize=30)
f1=0.14 #Distance first channel to edge [cm]
f2=0.40 #Distance between channels [cm]
h=[-2+f1,-2+f1+c_h,-2+f1+c_h+f2,-2+f1+c_h*2+f2,-2+f1+c_h*2+f2*2,-2+f1+c_h*3+f2*2,-2+f1+c_h*3+f2*3,-2+f1+c_h*4+f2*3,f1,f1+c_h,f1+c_h+f2,f1+c_h*2+f2,f1+c_h*2+f2*2,f1+c_h*3+f2*2,f1+c_h*3+f2*3,f1+c_h*4+f2*3,f1*2+c_h*4+f2*3]

x_b=[]
y_b=[]
for i in h:
    #x_b.append(-abs(np.cos((90-alpha)*np.pi/180)*i)+a+c_d)
    #y_b.append(-np.sin((90-alpha)*np.pi/180)*i)
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

# pos_9,pos_10,pos_11,pos_12,rad_9,rad_10,rad_11,rad_12=[],[],[],[],[],[],[],[]
# for d,pos,rad in zip((1,2,3,4),(pos_9,pos_10,pos_11,pos_12),(rad_9,rad_10,rad_11,rad_12)):
#     for j in np.arange(0,len(x)):
#         r=np.sqrt((x[j]-z_0)**2+y[j]**2)
#         beta=np.arctan(y[j]/(x[j]-z_0))
#         if (x[j]-z_0)<=0:
#             x_neu=(r+d)*-np.cos(beta)+z_0
#             y_neu=(r+d)*-np.sin(beta)
#         else:
#             x_neu=(r+d)*np.cos(beta)+z_0
#             y_neu=(r+d)*np.sin(beta)
#         pos.append(x_neu)
#         rad.append(y_neu)
#     plt.plot(np.append(pos,pos[0]),np.append(rad,rad[0]),linewidth=2,color='grey',alpha=0.5)
#     print(pos)
#     print(rad)

#lines of sight
for i,j,k in zip(lines,colors,channels):
    plt.plot([x_b[i],x_b[i+1]],[y_b[i],y_b[i+1]],color='red')
    popt1,pcov1=curve_fit(lin,[x_b[i],a-b],[y_b[i],-s_h/2])
    popt2,pcov2=curve_fit(lin,[x_b[i+1],a-b],[y_b[i+1],s_h/2])
    #plt.plot(np.arange(40,x_b[i],0.1),lin(np.arange(40,x_b[i],0.1),*popt1),color=j,linestyle='dashed')
    #plt.plot(np.arange(40,x_b[i+1],0.1),lin(np.arange(40,x_b[i+1],0.1),*popt2),color=j,linestyle='dashed')
    popt3,pcov3=curve_fit(lin,[a-b,a-b-12.4,a-b-19.5,a-b-22.9],[-s_h/2,ex_1[i],ex_2[i],ex_3[i]])
    popt4,pcov4=curve_fit(lin,[a-b,a-b-12.4,a-b-19.5,a-b-22.9],[s_h/2,ex_1[i+1],ex_2[i+1],ex_3[i+1]])
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt3),color=j,linewidth=2)
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt4),color=j,linewidth=2)
    #plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[i],ex_2[i],ex_3[i]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=j)
    #plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[i+1],ex_2[i+1],ex_3[i+1]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=j)




f=25 #fontsize inside plot
c='black'#fontcolor inside plot
#torsatron
plt.annotate('plasma vessel',(50,25),color=c,fontsize=f) 
vessel=plt.Circle((60,0),t,fill=False,color='grey',linewidth=3,alpha=0.5)
#port
plt.plot([a-b-20,a-b-10.3],[-12.5,-12.5],[a-b-20,a-b-10.3],[12.5,12.5],[a-b-10.3,a-b-10.3],[-12.5,12.5],color='grey',linewidth=3,alpha=0.5)
plt.annotate('outer\n port',(71,23),color=c,fontsize=f)
#slit
plt.plot([a-b,a-b],[-12,-s_h/2],[a-b,a-b],[12,s_h/2],color='grey',linewidth=3,alpha=0.5,linestyle='dashed')
plt.annotate('slit',(a-b-0.1,-0.5),xytext=(a-b-15,-25),arrowprops=dict(facecolor=c,edgecolor='none',alpha=0.5,width=3),color=c,fontsize=f)
bolovessel=patches.Rectangle((60+21.8,-12),20.8,24,edgecolor='grey',facecolor='none',linewidth=3, alpha=0.5)
plt.annotate('bolometer\n  vessel',(83,23),color=c,fontsize=f)
#bolometerhead
ts=ax.transData
coords1=[-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2]
coords2=[-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0]
tr1 = matplotlib.transforms.Affine2D().rotate_deg_around(coords1[0],coords1[1], -alpha)
tr2 = matplotlib.transforms.Affine2D().rotate_deg_around(coords2[0],coords2[1],alpha)
bolohead1=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr1+ts)
bolohead2=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr2+ts)
plt.annotate('bolometer\n   head',(a,-2),xytext=(a-10,-27),arrowprops=dict(facecolor=c,edgecolor='none',alpha=0.5,width=3),color=c,fontsize=f)
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
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/lines_of_sight_in_TJ-K_with_fluxsurfaces.pdf')

# %% Lines of sight
a=60+32.11+3.45 #Position of Bolometerheadmiddle [cm]
b=3.45 #Distance of Bolometerhead Middle to  Slit [cm]
s_w=1.4 #Width of the slit [cm]
s_h=0.5 #Height of the slit [cm]
alpha=13 #Angle of the Bolometerhead to plane [°]
c_w=0.38 #Channelwidth of Goldsensor [cm]
c_h=0.176 #HChannelheight of Goldsensor [cm]
c_d=0.225 #depth of Goldsensor [cm]
h=2 #height of Bolometerhead [cm]
z_0=63.9    #middle of flux surfaces
t=17.5 #radius of vessel [cm]

fig=plt.figure(figsize=(w,h))
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
ax=fig.add_subplot(111)
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')

x=np.arange(40,a,0.1)


plt.xlabel('R [cm]',fontsize=18)
plt.ylabel('r [cm]',fontsize=18)
f1=0.14 #Distance first channel to edge [cm]
f2=0.40 #Distance between channels [cm]
h=[-2+f1,-2+f1+c_h,-2+f1+c_h+f2,-2+f1+c_h*2+f2,-2+f1+c_h*2+f2*2,-2+f1+c_h*3+f2*2,-2+f1+c_h*3+f2*3,-2+f1+c_h*4+f2*3,f1,f1+c_h,f1+c_h+f2,f1+c_h*2+f2,f1+c_h*2+f2*2,f1+c_h*3+f2*2,f1+c_h*3+f2*3,f1+c_h*4+f2*3,f1*2+c_h*4+f2*3]
#h=[-2+f/2,-2+f/2+c_h,-2+f/2+c_h+f,-2+f*1.5+c_h*2,-2+f*2.5+c_h*2,-2+f*2.5+c_h*3,-2+f*3.5+c_h*3,-2+f*3.5+c_h*4,f*0.5,f*0.5+c_h,f*1.5+c_h,f*1.5+c_h*+2,f*2.5+c_h*2,f*2.5+c_h*3,f*3.5+c_h*3,f*3.5+c_h*4]
#h=[-1.6-c_h/2,-1.6+c_h/2,-1.2-c_h/2,-1.2+c_h/2,-0.8-c_h/2,-0.8+c_h/2,-0.4-c_h/2,-0.4+c_h/2,0.4-c_h/2,0.4+c_h/2,0.8-c_h/2,0.8+c_h/2,1.2-c_h/2,1.2+c_h/2,1.6-c_h/2,1.6+c_h/2]
x_b=[]
y_b=[]
for i in h:
    x_b.append(-abs(np.sin((alpha)*np.pi/180)*i)+a+c_d)
    y_b.append(-np.cos((alpha)*np.pi/180)*i)

def lin(x,d,e):
    return d*x+e
# ex_1=[(x-363)*0.037 for x in [171.04	,	287.52	,	212.44	,	329.01	,	253.07	,	363.5	,	283.67	,	400.38	,	326.46	,	434.64	,	361.32	,	469.15	,	395.38	,	508.12	,	433.76	,	542.54]]
# ex_2=[(x-331)*0.037 for x in [56.774	,	178.13	,	117.46	,	242.22	,	197.04	,	297.69	,	237.72	,	359.76	,	304.27	,	412.69	,	355.45	,	472.17	,	415.96	,	536.58	,	482.62	,	603.61]]
# ex_3=[(x-293)*0.077 for x in [111.22	,	180.10	,	155.24	,	230.23	,	197.97	,	230.42	,	235.84	,	308.38	,	277.59	,	349.32	,	313.66	,	386.82	,	357.42	,	426.86	,	405.32	,	466.08]]
ex_1=[-7.23, -3.0600000000000005, -5.760000000000001, -1.5900000000000007, -4.290000000000001, -0.120000000000001, -2.820000000000002, 1.3499999999999979, -1.3500000000000014, 2.8200000000000003, 0.120000000000001, 4.289999999999999, 1.5899999999999963, 5.7599999999999945, 3.059999999999995, 7.229999999999997]
ex_2=[-10.115, -5.705, -7.855, -3.445, -5.595, -1.1849999999999996, -3.334999999999999, 1.075000000000001, -1.0749999999999993, 3.335000000000001, 1.1850000000000005, 5.594999999999999, 3.4450000000000003, 7.855000000000004, 5.705000000000004, 10.115]
ex_3=[-13.65, -8.23, -10.52, -5.1000000000000005, -7.390000000000001, -1.9700000000000024, -4.2600000000000025, 1.1599999999999993, -1.1300000000000008, 4.290000000000001, 2.0000000000000018, 7.419999999999998, 5.129999999999997, 10.549999999999999, 8.259999999999998, 13.68]
lines=[0,2,4,6,8,10,12,14]
colors=['red','blue','green','gold','magenta','darkcyan','blueviolet','darkorange']
channels=[0,1,2,3,4,5,6,7]

#lines of sight
for i,j,k in zip(lines,colors,channels):
    plt.plot([x_b[i],x_b[i+1]],[y_b[i],y_b[i+1]],color='red')
    # popt1,pcov1=curve_fit(lin,[x_b[i],a-b],[y_b[i],-s_h/2])
    # popt2,pcov2=curve_fit(lin,[x_b[i+1],a-b],[y_b[i+1],s_h/2])
    # plt.plot(np.arange(40,x_b[i],0.1),lin(np.arange(40,x_b[i],0.1),*popt1),color=j,linestyle='dashed')
    # plt.plot(np.arange(40,x_b[i+1],0.1),lin(np.arange(40,x_b[i+1],0.1),*popt2),color=j,linestyle='dashed')
    popt3,pcov3=curve_fit(lin,[a-b,a-b-12.4,a-b-19.5,a-b-22.9],[-s_h/2,ex_1[i],ex_2[i],ex_3[i]])
    popt4,pcov4=curve_fit(lin,[a-b,a-b-12.4,a-b-19.5,a-b-22.9],[s_h/2,ex_1[i+1],ex_2[i+1],ex_3[i+1]])
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt3),color=j)
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt4),color=j)
    plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[i],ex_2[i],ex_3[i]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=j)
    plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[i+1],ex_2[i+1],ex_3[i+1]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=j)

#measurements
plt.vlines([a-b-22.9,-30,a-b-19.5,a-b-12.4],-30,30,color='grey',linestyle='dotted',alpha=0.5,linewidth=3)



#slit
plt.plot([a-b,a-b],[-12,-s_h/2],[a-b,a-b],[12,s_h/2],color='grey',linewidth=3,alpha=0.5,linestyle='dashed')
plt.annotate('slit',(a-b,0),xytext=(a-b-10,-25),arrowprops=dict(facecolor='grey',edgecolor='none',alpha=0.5,width=3),color='grey',fontsize=15)
bolovessel=patches.Rectangle((60+21.8,-12),20.8,24,edgecolor='grey',facecolor='none',linewidth=3, alpha=0.5)
plt.annotate('bolometer vessel',(85,25),color='grey',fontsize=15)
#bolometerhead
ts=ax.transData
coords1=[-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2]
coords2=[-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0]
tr1 = matplotlib.transforms.Affine2D().rotate_deg_around(coords1[0],coords1[1], -alpha)
tr2 = matplotlib.transforms.Affine2D().rotate_deg_around(coords2[0],coords2[1],alpha)
bolohead1=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr1+ts)
bolohead2=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr2+ts)
plt.annotate('bolometer\n   head',(a,-2),xytext=(a-5,-27),arrowprops=dict(facecolor='grey',edgecolor='none',alpha=0.5,width=3),color='grey',fontsize=15)

ax.add_patch(bolovessel)
ax.add_patch(bolohead1)
ax.add_patch(bolohead2)

plt.xlim(60,100)
plt.ylim(-20,20)
#plt.xlim(a-3,a+3)
#plt.ylim(-3,3)
#plt.grid(True)
fig1= plt.gcf()
plt.show()
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/lines_of_sight_measurement.pdf')
# %%
#Fluxsurfaces and Temperature, Density
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

# %%
#LR, RR, BR
T=60
T_LR,He_0_LR,He_1_LR=adas.he_adf11(data='plt96_he',T_max=T)[0],adas.he_adf11(data='plt96_he',T_max=T)[1],adas.he_adf11(data='plt96_he',T_max=T)[2]
T_BR_RR,He_0_BR_RR,He_1_BR_RR=adas.he_adf11(data='prb96_he',T_max=T)[0],adas.he_adf11(data='prb96_he',T_max=T)[1],adas.he_adf11(data='prb96_he',T_max=T)[2]
T_RR_0,He_0_RR=adas.he_adf15(data='pec96#he_pju#he0',T_max=T)[0],adas.he_adf15(data='pec96#he_pju#he0',T_max=T)[1]
T_RR_1,He_1_RR=adas.he_adf15(data='pec96#he_pju#he1',T_max=T)[0],adas.he_adf15(data='pec96#he_pju#he1',T_max=T)[1]

plt.figure(figsize=(w,h))
plt.plot(T_LR,He_0_LR,color=colors[0],marker=markers[0],label='He$^0$, excitation, source: ADF11 ')
plt.plot(T_LR,He_1_LR,color=colors[0],marker=markers[0],label='He${^+1}$, excitation, source: ADF11 ',alpha=0.5)

plt.plot(T_RR_0,He_0_RR,color=colors[2],marker=markers[2],label='He$^0$,recombination, source: ADF15')
plt.plot(T_RR_1,He_1_RR,color=colors[2],marker=markers[2],label='He$^{+1}$,recombination, source: ADF15',alpha=0.5)

plt.plot(T_BR_RR,He_0_BR_RR,color=colors[1],marker=markers[1],label='He$^0$,Bremsstrahlung and recombination, source: ADF11 ')
plt.plot(T_BR_RR,He_1_BR_RR,color=colors[1],marker=markers[1],label='He$^{+1}$,Bremsstrahlung and recombination, source: ADF11 ',alpha=0.5)

plt.ylabel('collisional radiative coefficients [eVm$^3$/s]')
plt.xlabel('temperature [eV]')
plt.yscale('log')
plt.ylim(1E-20,1E-12)
plt.xlim(-1,30)
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.7))
fig1= plt.gcf()
plt.show()
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/lt_br_rr.pdf',bbox_inches='tight')



# %%
