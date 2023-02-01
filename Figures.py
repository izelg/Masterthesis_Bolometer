#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.patches as patches
import matplotlib
#%% Plasmatypes and regimes--------------------------------------------------------------------------------------------------------------------------------------------------

#n=np.arange(10E5, 10E35)
#T=np.arange(10E-2,10E6)

plt.figure(figsize=(10,7))
plt.hlines(6,5,35,linestyle='dashed',alpha=0.7,color='red')
plt.plot([25,35],[-2,5.5],linestyle='dashed',alpha=0.7,color='blue')
plt.plot([22.5,35],[-2,2.1],linestyle='dashed',alpha=0.7,color='green')
plt.xticks([5,10,15,20,25,30,35],[r'10$^5$',r'10$^{10}$',r'10$^{15}$',r'10$^{20}$',r'10$^{25}$',r'10$^{30}$',r'10$^{35}$'],fontsize=12)
plt.yticks([-2,0,2,4,6,8],[r'10$^{-2}$',r'10$^0$',r'10$^2$',r'10$^4$',r'10$^6$',''],fontsize=12)
plt.xlabel(r'density [m$^{-3}$]',fontsize=16)
plt.ylabel(r'temperature [eV]',fontsize=16)
plt.annotate('interstellar\n plasmas',(7,-1),fontsize=12)
plt.annotate('interplanetar\n   plasmas ',(9,1),fontsize=12)
plt.annotate('flames',(14,-1.5),fontsize=12)
plt.annotate(' solar\ncenter',(27,3),fontsize=12)
plt.annotate('magnetic\n  fusion',(22,4.5),fontsize=12)
plt.annotate('lightning',(24,0.5),fontsize=12)
plt.annotate(' solar\ncorona',(14,1.7),fontsize=12)
plt.annotate('e$^-$ gas\nin metal',(29,-1.1),fontsize=12)
plt.annotate(' white\ndwarfs',(32.8,1),fontsize=12)
plt.annotate('supernovae',(28,5.5),fontsize=12)
plt.annotate('TJ-K',(18,1),color='red',fontsize=16)
plt.annotate('fluorescence\n     light ',(15,-0.5),fontsize=12)

plt.annotate('relativistic plasmas',(16,6.1),color='red', alpha=0.7,fontsize=12)
plt.annotate('degenerated plasmas', (30.7,2.1), rotation=58,color='blue',alpha=0.7,fontsize=12)
plt.annotate('non-ideal plasmas',(27,-0.7),rotation=33,color='green',alpha=0.7,fontsize=12)
plt.grid(True,alpha=0.2)
plt.xlim(5,35)
plt.ylim(-2,7.5)
fig1= plt.gcf()
plt.show()
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/plasmas_in_nature_and_laboratory.pdf')

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

fig=plt.figure(figsize=(10,10))
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
colors=['red','blue','green','gold','magenta','darkcyan','blueviolet','darkorange']
channels=[0,1,2,3,4,5,6,7]

#lines of sight
for i,j,k in zip(lines,colors,channels):
    plt.plot([x_b[i],x_b[i+1]],[y_b[i],y_b[i+1]],color='red')
    popt1,pcov1=curve_fit(lin,[x_b[i],a-b],[y_b[i],-s_h/2])
    popt2,pcov2=curve_fit(lin,[x_b[i+1],a-b],[y_b[i+1],s_h/2])
    #plt.plot(np.arange(40,x_b[i],0.1),lin(np.arange(40,x_b[i],0.1),*popt1),color=j,linestyle='dashed')
    #plt.plot(np.arange(40,x_b[i+1],0.1),lin(np.arange(40,x_b[i+1],0.1),*popt2),color=j,linestyle='dashed')
    popt3,pcov3=curve_fit(lin,[a-b,a-b-12.4,a-b-19.5,a-b-22.9],[-s_h/2,ex_1[i],ex_2[i],ex_3[i]])
    popt4,pcov4=curve_fit(lin,[a-b,a-b-12.4,a-b-19.5,a-b-22.9],[s_h/2,ex_1[i+1],ex_2[i+1],ex_3[i+1]])
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt3),color=j)
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt4),color=j)
    #plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[i],ex_2[i],ex_3[i]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=j)
    #plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[i+1],ex_2[i+1],ex_3[i+1]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=j)


#fluxsurfaces
for i in [0,1,2,3,4,5,6,7]:
    x=np.array(x_.iloc[i+1])
    y=np.array(y_.iloc[i+1])
    plt.plot(x,y,marker='.',color='grey')

#torsatron
#plt.annotate('plasma vessel',(55,25),color='grey',fontsize=15) 
vessel=plt.Circle((60,0),t,fill=False,color='grey',linewidth=3,alpha=0.5)
#port
plt.plot([a-b-20,a-b-10.3],[-12.5,-12.5],[a-b-20,a-b-10.3],[12.5,12.5],[a-b-10.3,a-b-10.3],[-12.5,12.5],color='grey',linewidth=3,alpha=0.5)
#plt.annotate('outer\n  port',(73,24),color='grey',fontsize=15)
#slit
plt.plot([a-b,a-b],[-12,-s_h/2],[a-b,a-b],[12,s_h/2],color='grey',linewidth=3,alpha=0.5,linestyle='dashed')
#plt.annotate('slit',(a-b,0),xytext=(a-b-10,-25),arrowprops=dict(facecolor='grey',edgecolor='none',alpha=0.5,width=3),color='grey',fontsize=15)
bolovessel=patches.Rectangle((60+21.8,-12),20.8,24,edgecolor='grey',facecolor='none',linewidth=3, alpha=0.5)
#plt.annotate('bolometer vessel',(85,25),color='grey',fontsize=15)
#bolometerhead
ts=ax.transData
coords1=[-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2]
coords2=[-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0]
tr1 = matplotlib.transforms.Affine2D().rotate_deg_around(coords1[0],coords1[1], -alpha)
tr2 = matplotlib.transforms.Affine2D().rotate_deg_around(coords2[0],coords2[1],alpha)
bolohead1=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(-2))+a,-2),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr1+ts)
bolohead2=patches.Rectangle((-abs(np.cos((90-alpha)*np.pi/180)*(0))+a,0),2,2,edgecolor='grey',facecolor='grey',linewidth=3, alpha=0.5,transform=tr2+ts)
#plt.annotate('bolometer\n   head',(a,-2),xytext=(a-5,-27),arrowprops=dict(facecolor='grey',edgecolor='none',alpha=0.5,width=3),color='grey',fontsize=15)
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

fig=plt.figure(figsize=(10,10))
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

fig=plt.figure(figsize=(10,10))
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
ax=fig.add_subplot(111)
ax2=ax.twinx()
ax3=ax.twinx()
x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position.csv',sep=',',engine='python'),dtype=np.float64)
y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii.csv',sep=',',engine='python')
shotnumber=13105

ax.set_xlabel('R [cm]',fontsize=18)
ax.set_ylabel('density [m$^-$$^3$]',fontsize=18,color='green')
ax.tick_params(axis='y', labelcolor='green')
ax2.set_ylabel('temperature [eV]',fontsize=18,color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax3.set_yticks([])
Position1=np.genfromtxt('/data6/shot{s}/probe2D/shot{s}.dat'.format(s=shotnumber),unpack=True,usecols=0)
Density=[914693519482453.9, 1195474759430940.0, 1382528487574368.8, 1506412816111439.0, 2399344795032418.0, 2793878393832172.5, 3643906098279352.0, 4954748608531252.0, 6040984860021678.0, 7173615192571933.0, 9116318227409138.0, 1.1459678372174782e+16, 1.4179279329688002e+16, 1.6617026719077388e+16, 1.825701908154783e+16, 1.9163466946400468e+16, 1.933019898285706e+16, 1.8934100993598344e+16, 1.872500524272425e+16, 1.8658725949248148e+16, 1.8700682124910644e+16, 1.8831780959027196e+16, 1.8961669563157476e+16, 1.881558425197214e+16, 1.8613288933084424e+16, 1.8767348299445108e+16, 1.9053989938286704e+16, 1.94500612213667e+16, 1.996652924163139e+16, 2.0664326637827344e+16, 2.1283636148849756e+16, 2.1781594278794596e+16, 2.2233474349163216e+16, 2.2428691916796556e+16, 2.246131195168899e+16, 2.204625157792732e+16, 2.0818267320530564e+16, 1.8997245920151348e+16, 1.6294358473434292e+16, 1.3279216540994364e+16, 1.0460493720178184e+16, 8870470209856892.0, 7380918055498378.0, 6070002286508864.0, 5018940111381089.0, 4291152377062758.5, 3892772934550662.5, 3378430561534559.5, 2877662849789524.0, 2466093423524798.0, 2125680899761901.8, 1760000570286304.0, 1488424981127544.8]
Position2, T=np.genfromtxt('/data6/Auswertung/shot{s}/shot{s}Te.dat'.format(s=shotnumber),unpack=True)
ax2.plot(Position2*100, T,color='red',linewidth=3)
ax.plot(Position1*100,Density,color='green',linewidth=3)

#fluxsurfaces
for i in [0,1,2,3,4,5,6,7]:
    x=[u-60 for u in np.array(x_.iloc[i+1])]
    y=np.array(y_.iloc[i+1])
    ax3.plot(x,y,marker='.',color='grey',alpha=0.5)


fig1= plt.gcf()
plt.show()
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/fluxsurfaces_with_temperatureprofiles.pdf')

# %%
