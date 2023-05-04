#%%
#Written by: Izel Gediz
#Date of Creation: 14.11.2022
#This code takes the simulated data of the cross section of the fluxsurfaces
#It also takes the modeled and exmerimentaly determined lines of sight
#Then it derives the area covered by e.g. channel 5 line of sight and fluxsurface 0 to 5 with a given resolution

import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy import integrate
import os
from matplotlib import cm
from scipy.interpolate import pchip_interpolate
from datetime import datetime

import plasma_characteristics as pc
#%% Parameter
Poster=True


if Poster==True:
    plt.rc('font',size=20)
    plt.rc('xtick',labelsize=25)
    plt.rc('ytick',labelsize=25)
    plt.rcParams['lines.markersize']=18
else:
    plt.rc('font',size=14)
    plt.rc('figure', titlesize=15)
colors=['#1bbbe9','#023047','#ffb703','#fb8500','#c1121f','#780000','#6969B3','#D81159','#1bbbe9','#023047','#ffb703','#fb8500','#c1121f']
markers=['o','v','s','P','p','D','8','*','x']

#%%

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
A_D=(c_h*c_w)/10000 #detector area in [m]
#err=0.4     #error of lines of sight in x and y direction [cm]


def LoadData(location):
    with open(location, "r") as f:
        cols = f.readlines()[3]
        cols = re.sub(r"\t+", ';', cols)[:-2].split(';')
    data = pd.read_csv(location, skiprows=4, sep="\t\t", names=cols, engine='python')
    return data

# Pixelmethod
def Pixelmethod():
    start=datetime.now()
    #-----------------------------------------------------#-
    #!!! Enter here which channels lines of sight you want to have analyzed(1 to 8), what pixel-resolution you need (in cm) and  which flux surfaces (0 to 7) should be modeled. 
    #note that more channels, a smaller resolution and more fluxsurfaces result in longer computation times
    bolo_channel=[4] #1 to 8
    bolo_power=[0,0,0,3.4E-6,3.4E-6,0,0,0]
    res=0.5
    fluxsurfaces=[0,1,2,3,4,5,6,7,8,9,10,11]
    err='Min'
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
        x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position_extended.csv',sep=',',engine='python'),dtype=np.float64)
        y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii_extended.csv',sep=',',engine='python')


        plt.xlabel('R [cm]',fontsize=30)
        plt.ylabel('z [cm]',fontsize=30)

        #Derive the exact positions of the bolometerchannels
        #I derive the x and y positions of the four upper channels lower and upper edge
        #I consider the 8 Bolometerchannels distributed over the 4cm with equal distance resulting in a distance of 0.33cm
        f1=0.14 #Distance first channel to edge [cm]
        f2=0.4 #Distance between channels [cm]
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
            
        x=np.arange(49,94,1)
        m_=list(p+0.01 for p in (np.arange(int(min(x_.iloc[12]))-1,int(max(x_.iloc[12]))+1,res)))
        n_=list(o+0.01 for o in (np.arange(int(min(y_.iloc[12]))-1,int(max(y_.iloc[12]))+1,res)))
        inside_line=[[],[],[],[],[],[],[],[]]
        lines=[0,2,4,6,8,10,12,14]
        #colors=['red','blue','green','gold','magenta','darkcyan','blueviolet','orange','darkblue','blue','green','gold','magenta']
        #here the desired line of sight is plotted from the experimental data. To see the calculate lines of sight activate the dashed lines plot
        #now the points of interest(the ones in a square of the rough size of the outer most fluxsurface) are tested for their position relative to the line of sight
        #If the point lies inside the two lines describing the line of sight of that channel, its coordinates are added to "inside_line"
        plt.plot([x_b[lines[bol-1]],x_b[lines[bol-1]+1]],[y_b[lines[bol-1]],y_b[lines[bol-1]+1]],color='red')
        popt3,pcov3=curve_fit(lin,[a-b+x_err,a-b-12.4+x_err,a-b-19.5+x_err,a-b-22.9+x_err],[-s_h/2,ex_1[lines[bol-1]]-y_err,ex_2[lines[bol-1]]-y_err,ex_3[lines[bol-1]]-y_err])
        popt4,pcov4=curve_fit(lin,[a-b+x_err,a-b-12.4+x_err,a-b-19.5+x_err,a-b-22.9+x_err],[s_h/2,ex_1[lines[bol-1]+1]+y_err,ex_2[lines[bol-1]+1]+y_err,ex_3[lines[bol-1]+1]+y_err])
        #popt3,pcov3=curve_fit(lin,[a-b+x_err,a-b-22.9+x_err],[-s_h/2,ex_3[lines[bol-1]]-y_err])
        #popt4,pcov4=curve_fit(lin,[a-b+x_err,a-b-22.9+x_err],[s_h/2,ex_3[lines[bol-1]+1]+y_err])
        plt.plot(x,lin(x,*popt3),color=colors[bol-1],linewidth=4)
        plt.plot(x,lin(x,*popt4),color=colors[bol-1],linewidth=4)
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
        inside=[[],[],[],[],[],[],[],[],[],[],[],[],[]]
        inside_=[]
        vol=[[],[],[],[],[],[],[],[],[],[],[],[],[]]
        v_i=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
        for f in fluxsurfaces:
        #m_=list(p+0.01 for p in (np.arange(int(min(x_.iloc[i+1])),int(max(x_.iloc[i+1]))+1,res)))
        #n_=list(b+0.01 for b in (np.arange(int(min(y_.iloc[i+1])),int(max(y_.iloc[i+1]))+1,res)))
            for (m,n) in inside_line[bol-1]:
                if (m,n) not in inside_:
                    plt.plot(m,n,marker='.',color=colors[f])
                    if n<0:
                        half=np.concatenate((np.arange(-1,0),np.arange(32,63)))
                    else:
                        half=np.arange(0,32)
                    x=np.array(x_.iloc[f])
                    y=np.array(y_.iloc[f])
                    plt.plot(np.append(x,x[0]),np.append(y,y[0]),color=colors[f])
                    ideal=intersections(half)[0]
                    x_inner=intersections([ideal])[2][intersections([ideal])[1]]
                    y_inner=lin(x_inner,*(intersections([ideal])[4]))
                    x=np.array(x_.iloc[f+1])
                    y=np.array(y_.iloc[f+1])
                    plt.plot(np.append(x,x[0]),np.append(y,y[0]),color=colors[f+1])
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
        
        for v in np.append(fluxsurfaces, fluxsurfaces[-1]+1):  
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
            print("The {n} Fluxsurface covers a space of ~ {s} cm\u00b3 in channel {c} line of sight which is v_i={d}cm.".format(n=v,s=float(f'{vol[v]:.2f}'),d=float(f'{v_i[v][0]:.2e}'),c=bol))
            
        plt.ylim(min(y_.iloc[f+1])-res,max(y_.iloc[f+1])+res)
        plt.xlim(z_0+min(y_.iloc[f+1])-res,80)
        fig1= plt.gcf()
        plt.show()
        fig1.savefig('/home/gediz/Results/Modeled_Data/Fluxsurfaces_and_Lines_of_sight/flux_{f1}_to_{f2}_channel_{b}.pdf'.format(f1=fluxsurfaces[0],f2=fluxsurfaces[-1],b=bolo_channel[0]))

        vol_ges=0
        v_i_ges=0
        for k in np.append(fluxsurfaces, fluxsurfaces[-1]+1):
            if len([vol[k]]) != 0:
                vol_ges+=vol[k]
            if len([v_i[k]]) != 0:
                v_i_ges+=v_i[k][0]
        vol_ges=vol_ges/1000000
        v_i_ges=v_i_ges/100
        P_ges=((4*np.pi*bolo_power[bol-1])/(((c_w*c_h)/10000)*v_i_ges))*0.1198
        print('Volume Observed by channel {c}: {v}m\u00b3'.format(c=bol,v=float(f'{vol_ges:.5f}')))
        print('Total Plasmaradiationpower: {p} Watt'.format(p=float(f'{P_ges:.2f}')))
    print(datetime.now()-start)


#Total Power from Channel 4

def Totalpower_from_exp(Type='',save=False,calc=False):
    v_i_ges_middle=[0.00978,0.0175945,0.020527,0.02182,0.02182,0.020527,0.0175945,0.00978]#0.01476
    #v_i_ges_min=0.012895
    #v_i_ges_max=0.016405
    #markers=['o','o','v','v','s','s','P']
    V_T=0.1198 #till outer most closed fluxsurface
    V_T_2=0.237 #till edge of calculation area with "additional fluxsurfaces"
    plt.figure(figsize=(10,7))
    for j in np.arange(len(shotnumbers)):
        P_ges_exp,P_ges_calc,pressures,mw=[],[],[],[]
        
        for s,g in zip(shotnumbers[j],gases[j]):
            pressures.append(pc.Pressure(s,g))
            mw.append(pc.GetMicrowavePower(s)[0])
        if Type=='Pressure':
            sortnumbers=[shotnumbers[j][i] for i in np.argsort(pressures)]
        if Type=='Power':
            sortnumbers=[shotnumbers[j][i] for i in np.argsort(mw)]
        for s,g in zip(sortnumbers,gases[j]):
            bolo_p=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot{s}/shot{s}_bolometerprofile_from_radiation_powers.txt'.format(s=s),usecols=1)[3]*10**(-6)
            P_ges_exp.append(((4*np.pi*bolo_p*mesh*gold(g))/(((c_w*c_h)/10000)*v_i_ges_middle[3]))*V_T_2)
            if calc==True:
                if os.path.exists('/home/gediz/Results/Modeled_Data/Bolometerprofiles/shot{s}/shot{s}_modeled_powerprofile_{g}.txt'.format(s=s,g=gas)):
                    calc_P=np.genfromtxt('/home/gediz/Results/Modeled_Data/Bolometerprofiles/shot{s}/shot{s}_modeled_powerprofile_{g}.txt'.format(s=s,g=gas),usecols=1)[3]*10**(-6)
                else:
                    calc_p=Boloprofile_calc(s)[3]*10**(-6)           
                P_ges_calc.append(((4*np.pi*calc_p)/(((c_w*c_h)/10000)*v_i_ges_middle[3]))*V_T_2)
        if Type=='Pressure':
            title=r'{g}, MW: {mw}, P$_M$$_W$ $\approx$ {m} kW'.format(g=g,mw=pc.GetMicrowavePower(s)[1],m=float(f'{np.mean(mw)*10**(-3):.1f}'))
            plt.plot(np.sort(pressures),P_ges_exp,marker=markers[j],color=colors[j],linestyle='dashed',label=title)
            if calc==True:
                plt.plot(np.sort(pressures),P_ges_calc,marker=markers[j+1],color=colors[j+1],linestyle='dashed',label='calculated'+title)
            plt.xlabel('pressure [mPa]')
        if Type=='Power':
            title='{g}, \t MW: {mw}, \t'.format(g=g,mw=pc.GetMicrowavePower(s)[1],)+r'p$\approx$'+'{p} mPa'.format(p=float(f'{np.mean(pressures):.1f}'))
            plt.plot(np.sort(mw),P_ges_exp,marker=markers[j],color=colors[j],linestyle='dashed',label=title)
            if calc==True:
                plt.plot(np.sort(mw),P_ges_calc,marker=markers[j+1],color=colors[j+1],linestyle='dashed',label='calculated'+title)
            plt.xlabel('P$_M$$_W$ [W]')
    plt.ylabel('P$_r$$_a$$_d$ total [W]')
    plt.ylim(0)
    plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.2*len(shotnumbers)-0.1))
    fig1= plt.gcf()   
    plt.show()
    if save==True:
        fig1.savefig('/home/gediz/Results/Modeled_Data/Tota_P_rad/comparison_{T}_{g}.pdf'.format(T=Type, g=gases), bbox_inches='tight')


def Totalpower_from_mean_T(s):
    p_t,t,p_d,d=pc.TemperatureProfile(s,'Values',save=False)[0]*100,pc.TemperatureProfile(s,'Values',save=False)[1],pc.DensityProfile(s,'Values',save=False)[0]*100,pc.DensityProfile(s,'Values',save=False)[1]
    plt12_h=[[0.2,0.3,0.5,0.7,1.0,1.5,2.0,3.0,5.0,7.0,10.0,15.0,20.0,30.0,50.0,70.0,100.0,150.0,200.0,300.0],[4.407688845190763e-35,8.516871836003884e-28,5.447289062950326e-22,1.7238686729080675e-19,1.3128247856849096e-17,3.809902586446144e-16,2.0557069822381066e-15,1.1219816909239107e-14,4.4647818960078494e-14,8.194415483209005e-14,1.3041383247019845e-13,1.8878908716623144e-13,2.2720958471076943e-13,2.7265288021124073e-13,3.1980213280329866e-13,3.4037318269587456e-13,3.5247900153249376e-13,3.554178707527905e-13,3.514790184085834e-13,3.3768040567583216e-13]]
    interpol_rc=pchip_interpolate(plt12_h[0],plt12_h[1],np.mean(t[:-5]))
    n_e_2=np.mean(d)
    n_e=pc.Densities(s)[1]
    n_0=pc.Densities(s)[2]
    R=z_0/100
    r=((0.763-z_0/100)+0.15)/2
    V_T=0.237#2*np.pi**2*r**2*R
    P=((interpol_rc*n_e*n_0*V_T)*1.602E-19)
    return (interpol_rc,P,(P/br.GetMicrowavePower(s)[0])*100)


#The power absorbed by the channels calculated with ADAS coefficients
def Boloprofile_calc(s,save=False,plot=False):
    
    flux_4=[0.00105,0.00106,0.00143,0.00185,0.0019,0.00175,0.00183,0.00188,0.00209,0.00166,0.00182,0.00163,0.00187]#spaces of the fluxsurfaces 0 to 8 occupied by ch.4 line of sight in cm^-3
    flux_3=[0,0.000207,0.00063,0.00103,0.00148,0.00195,0.00252,0.00281,0.00242,0.00205,0.00184,0.00181,0.00178]
    flux_2=[0,0,0,0,0.0000795,0.000785,0.00141,0.00195,0.00249,0.00269,0.00318,0.00276,0.00225]
    flux_1=[0,0,0,0,0,0,0,0,0.000663,0.00137,0.00203,0.00267,0.00305]
    flux_pos=[2.174,5.081,5.844,6.637,7.461,8.324,9.233,10.19,11.207,12.321,13.321,14.321,15.321,16.321]#position of the flux surface edged in cm
    p_t,t,p_d,d=pc.TemperatureProfile(s,'Values',save=False)[0]*100,pc.TemperatureProfile(s,'Values',save=False)[1],pc.DensityProfile(s,'Values',save=False)[0]*100,pc.DensityProfile(s,'Values',save=False)[1]
    plt12_h=[[0.2,0.3,0.5,0.7,1.0,1.5,2.0,3.0,5.0,7.0,10.0,15.0,20.0,30.0,50.0,70.0,100.0,150.0,200.0,300.0],[4.407688845190763e-35,8.516871836003884e-28,5.447289062950326e-22,1.7238686729080675e-19,1.3128247856849096e-17,3.809902586446144e-16,2.0557069822381066e-15,1.1219816909239107e-14,4.4647818960078494e-14,8.194415483209005e-14,1.3041383247019845e-13,1.8878908716623144e-13,2.2720958471076943e-13,2.7265288021124073e-13,3.1980213280329866e-13,3.4037318269587456e-13,3.5247900153249376e-13,3.554178707527905e-13,3.514790184085834e-13,3.3768040567583216e-13]]
    flux_t,flux_d,flux_rc=[],[],[]
    n=pc.Densities(s,gas)[0]
    n_e=pc.Densities(s,gas)[1]
    n_0=pc.Densities(s,gas)[2]
    for i in np.arange(0,len(flux_pos)-1):
        interpol_t=pchip_interpolate(p_t,t,np.arange(flux_pos[i],flux_pos[i+1],0.01))
        interpol_d=pchip_interpolate(p_d,d,np.arange(flux_pos[i],flux_pos[i+1],0.01))
        flux_t.append(np.mean(interpol_t))
        flux_d.append(np.mean(interpol_d))
        interpol_rc=pchip_interpolate(plt12_h[0],plt12_h[1],flux_t[i])
        flux_rc.append(interpol_rc)
    plt.plot(p_d,d)
    plt.plot(flux_pos[0:-1],flux_d,'bo')
    plt.hlines(n_e,p_d[0],p_d[-1])
    n=integrate.trapezoid(d,p_d)/(p_d[-1]-p_d[0])
    plt.hlines(n,p_d[0],p_d[-1],color='red')
    plt.show()
    P_profile_calc=[]
    for b in [flux_1,flux_2,flux_3,flux_4,flux_4,flux_3,flux_2,flux_1]:
        P=0
        for j in np.arange(0,len(flux_d)):
            #R=(a-40-flux_pos[j])/100
            #P+=((flux_rc[j]*n_e*n_0*b[j])*1.602E-19)*(A_D/(4*np.pi))
            P+=((flux_rc[j]*flux_d[j]*n_0*b[j])*1.602E-19)*(A_D/(4*np.pi))
        P_profile_calc.append(P/10**(-6))
    c1='#18a8d1'
    c2='#fb8500'
    if plot==True:
        fig=plt.figure(figsize=(10,7))
        ax=fig.add_subplot(111)
        ax2=ax.twinx()
        P_profile=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot{s}/shot{s}_bolometerprofile_from_radiation_powers.txt'.format(s=s),usecols=1)
        lns2=ax2.plot((1,2,3,4,5,6,7,8),P_profile_calc,'v--',color=c2,label='calculated power profile')
        lns1=ax.plot((1,2,3,4,5,6,7,8),P_profile*mesh*gold(gas),'o--',color=c1,label='measured power profile')
        ax.set_xticks((1,2,3,4,5,6,7,8))
        ax.set_ylim(0)
        ax2.set_ylim(0)
        ax.set_xlabel('bolometer channel',fontsize=35)
        ax.set_ylabel('P$_r$$_a$$_d$ [\u03bcW]',color=c1,fontsize=35)
        ax.tick_params(axis='y', labelcolor=c1)
        ax2.set_ylabel('P$_r$$_a$$_d$ [\u03bcW]',color=c2,fontsize=35)
        ax2.tick_params(axis='y', labelcolor=c2)
        fig.patch.set_facecolor('white')
        leg = lns1 + lns2 
        labs = [l.get_label() for l in leg]
        title= '{g}, shot n°{s}, MW: {mw} \n P$_M$$_W$= {m} W, p={p} mPa'.format(g=gas,s=s,mw=pc.GetMicrowavePower(s)[1],m=float(f'{pc.GetMicrowavePower(s)[0]:.1f}'),p=float(f'{pc.Pressure(s,gas):.1f}'))
        ax.legend(leg, labs, loc='lower center', title=title)
        fig1= plt.gcf()   
        plt.show()
    if save==True:
        data = np.column_stack([np.array([1,2,3,4,5,6,7,8]), np.array(P_profile_calc)])#, np.array(z), np.array(abs(y-z))])
        np.savetxt(str(outfile)+"shot{n}/shot{n}_modeled_powerprofile_{g}.txt".format(n=shotnumber, g=gas) , data, delimiter='\t \t', fmt=['%d', '%10.3f'], header='Modeled power data \n {t} \n bolometerchannel \t power [\u03bcW] '.format(t=title))
        fig1.savefig(str(outfile)+"shot{n}/shot{n}_modeled_powerprofile_{g}.pdf".format(n=shotnumber, g=gas), bbox_inches='tight')
    return P_profile_calc


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

shotnumber=13258
shotnumbers=[np.arange(13242,13256),np.arange(13268,13280),np.arange(13299,13312),np.arange(13340,13347)]#[np.arange(13256,13267),np.arange(13316,13321),np.arange(13321,13331),np.arange(13331,13340)]#[np.arange(13215,13228),np.arange(13256,13268),np.arange(13280,13292)]#
gases=[['H'for i in range(14)],['He'for i in range(12)],['Ar'for i in range(13)],['Ne'for i in range(7)]]#[['He'for i in range(11)],['He'for i in range(5)],['He'for i in range(10)],['He'for i in range(9)]]#[['H'for i in range(13)],['He'for i in range(12)],['Ar'for i in range(12)]]#
gas='He'
infile='/data6/shot{s}/kennlinien/auswert'.format(s=shotnumber)

location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
mesh=1/0.75     #multiply with this factor to account for 25% absorbance of mesh
def gold(g):
    if g=='H':
        return 1/0.81      
    if g=='He':      
        return 1/0.71      
    if g=='Ar':
        return 1/0.88
    if g=='Ne':
        return 1
# for shotnumber in np.arange(13256,13268):
#     outfile='/home/gediz/Results/Modeled_Data/Bolometerprofiles/'
#     if not os.path.exists(str(outfile)+'shot{}'.format(shotnumber)):
#         os.makedirs(str(outfile)+'shot{}'.format(shotnumber))

#     Boloprofile_calc(shotnumber,save=True,plot=True)
Totalpower_from_exp('Pressure')
#Pixelmethod()
#for s in np.arange(13242,13256):
 #   Boloprofile_calc(s,save=True,plot=True)
#Boloprofile_calc(shotnumber,plot=True)
#print(Totalpower_from_mean_T(shotnumber)[0],Totalpower_from_mean_T(shotnumber)[1])
# %%
#[np.arange(13242,13256)]#