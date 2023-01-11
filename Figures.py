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
plt.xlabel(r'Density [m$^{-3}$]',fontsize=16)
plt.ylabel(r'Temperature [eV]',fontsize=16)
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
f1=0.123 #Distance first channel to edge [cm]
f2=0.35 #Distance between channels [cm]
h=[-2+f1,-2+f1+c_h,-2+f1+c_h+f2,-2+f1+c_h*2+f2,-2+f1+c_h*2+f2*2,-2+f1+c_h*3+f2*2,-2+f1+c_h*3+f2*3,-2+f1+c_h*4+f2*3,f1,f1+c_h,f1+c_h+f2,f1+c_h*2+f2,f1+c_h*2+f2*2,f1+c_h*3+f2*2,f1+c_h*3+f2*3,f1+c_h*4+f2*3,f1*2+c_h*4+f2*3]
#h=[-2+f/2,-2+f/2+c_h,-2+f/2+c_h+f,-2+f*1.5+c_h*2,-2+f*2.5+c_h*2,-2+f*2.5+c_h*3,-2+f*3.5+c_h*3,-2+f*3.5+c_h*4,f*0.5,f*0.5+c_h,f*1.5+c_h,f*1.5+c_h*+2,f*2.5+c_h*2,f*2.5+c_h*3,f*3.5+c_h*3,f*3.5+c_h*4]
#h=[-1.6-c_h/2,-1.6+c_h/2,-1.2-c_h/2,-1.2+c_h/2,-0.8-c_h/2,-0.8+c_h/2,-0.4-c_h/2,-0.4+c_h/2,0.4-c_h/2,0.4+c_h/2,0.8-c_h/2,0.8+c_h/2,1.2-c_h/2,1.2+c_h/2,1.6-c_h/2,1.6+c_h/2]
x_b=[]
y_b=[]
for i in h:
    #x_b.append(-abs(np.cos((90-alpha)*np.pi/180)*i)+a+c_d)
    #y_b.append(-np.sin((90-alpha)*np.pi/180)*i)
    x_b.append(-abs(np.sin((alpha)*np.pi/180)*i)+a+c_d)
    y_b.append(-np.cos((alpha)*np.pi/180)*i)

def lin(x,d,e):
    return d*x+e
y_exp=np.flip((13.665,8.245,10.535,5.115,7.405,1.985,4.275,-1.145,1.145,-4.275,-1.985,-7.405,-5.115,-10.535,-8.245,-13.665))
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
    popt3,pcov3=curve_fit(lin,[a-b,a-b-22.9],[-s_h/2,y_exp[i]])
    popt4,pcov4=curve_fit(lin,[a-b,a-b-22.9],[s_h/2,y_exp[i+1]])
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt3),color=j)
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt4),color=j)
    
#fluxsurfaces
for i in [0,1,2,3,4,5,6,7]:
    x=np.array(x_.iloc[i+1])
    y=np.array(y_.iloc[i+1])
    plt.plot(x,y,marker='.',color='grey')

#torsatron
plt.annotate('plasma vessel',(55,25),color='grey',fontsize=15) 
vessel=plt.Circle((60,0),t,fill=False,color='grey',linewidth=3,alpha=0.5)
#port
plt.plot([a-b-20,a-b-10.3],[-12.5,-12.5],[a-b-20,a-b-10.3],[12.5,12.5],[a-b-10.3,a-b-10.3],[-12.5,12.5],color='grey',linewidth=3,alpha=0.5)
plt.annotate('outer\n  port',(73,24),color='grey',fontsize=15)
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
alpha=14 #Angle of the Bolometerhead to plane [°]
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
f1=0.123 #Distance first channel to edge [cm]
f2=0.35 #Distance between channels [cm]
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
y_exp1=np.flip((13.665,8.245,10.535,5.115,7.405,1.985,4.275,-1.145,1.145,-4.275,-1.985,-7.405,-5.115,-10.535,-8.245,-13.665))
y_exp2=np.flip((10.115,5.705,7.855,3.445,5.595,1.185,3.335,-1.075,1.075,-3.335,-1.185,-5.595,-3.445,-7.855,-5.705,-10.115))
y_exp3=np.flip((7.23,3.059,5.759,1.589,4.289,1.199,2.819,-1.35,1.35,-2.819,-1.199,-4.289,-1.589,-5.759,-3.059,-7.23))
lines=[0,4,8,12]
colors=['red','green','magenta','blueviolet']
channels=[0,2,4,6]

#lines of sight
for i,j,k in zip(lines,colors,channels):
    plt.plot([x_b[i],x_b[i+1]],[y_b[i],y_b[i+1]],color='red')
    popt1,pcov1=curve_fit(lin,[x_b[i],a-b],[y_b[i],-s_h/2])
    popt2,pcov2=curve_fit(lin,[x_b[i+1],a-b],[y_b[i+1],s_h/2])
    plt.plot(np.arange(40,x_b[i],0.1),lin(np.arange(40,x_b[i],0.1),*popt1),color=j,linestyle='dashed')
    plt.plot(np.arange(40,x_b[i+1],0.1),lin(np.arange(40,x_b[i+1],0.1),*popt2),color=j,linestyle='dashed')
    popt3,pcov3=curve_fit(lin,[a-b,a-b-22.9],[-s_h/2,y_exp1[i]])
    popt4,pcov4=curve_fit(lin,[a-b,a-b-22.9],[s_h/2,y_exp1[i+1]])
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt3),color=j)
    plt.plot(np.arange(40,a,0.1),lin(np.arange(40,a,0.1),*popt4),color=j)

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
plt.ylim(-30,30)
#plt.xlim(a-3,a+3)
#plt.ylim(-3,3)
#plt.grid(True)
fig1= plt.gcf()
plt.show()
fig1.savefig('/home/gediz/LaTex/Thesis/Figures/lines_of_sight_measurement.pdf')
# %%
