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
from sympy import  nsolve
from sympy.abc import x

import plasma_characteristics as pc
import adas_data as adas
#%% Parameter
Latex=True
Poster=False


if Poster==True:
    plt.rc('font',size=20)
    plt.rc('xtick',labelsize=25)
    plt.rc('ytick',labelsize=25)
    plt.rcParams['lines.markersize']=18
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
    plt.rc('font',size=14)
    plt.rc('figure', titlesize=15)
#colors=['#1bbbe9','#023047','#ffb703','#fb8500','#c1121f','#780000','#6969B3','#D81159','#1bbbe9','#023047','#ffb703','#fb8500','#c1121f']
colors=['#1bbbe9','#023047','#ffb703','#fb8500','#c1121f','#780000','#6969B3','#D81159','#1bbbe9','#023047','#ffb703','#fb8500','#c1121f']
colors2=['#03045E','#0077B6','#00B4D8','#370617','#9D0208','#DC2F02','#F48C06','#FFBA08','#3C096C','#7B2CBF','#C77DFF','#2D6A4F','#40916C','#52B788','#03045E','#0077B6','#00B4D8']

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

def Method():
    popt_flux_pos=[[],[],[],[],[],[],[],[],[],[],[],[]]
    popt_flux_neg=[[],[],[],[],[],[],[],[],[],[],[],[]]
    fluxsurfaces=[0,1,2,3,4,5,6,7,8,9,10,11]
    start=datetime.now()
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111)
    x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position_extended.csv',sep=',',engine='python'),dtype=np.float64)
    y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii_extended.csv',sep=',',engine='python')


    plt.xlabel('R [cm]',fontsize=30)
    plt.ylabel('z [cm]',fontsize=30)

    def lin(x,d,e):
        return d*x+e
    for f in fluxsurfaces:
        x=np.array(x_.iloc[f])
        y=np.array(y_.iloc[f])
        for i in np.concatenate((np.arange(-1,0),np.arange(32,63))):
            popt,pcov=curve_fit(lin,[x[i],x[i+1]],[y[i],y[i+1]])
            popt_flux_neg[f].append(popt)
        for i in np.arange(0,32):
            popt,pcov=curve_fit(lin,[x[i],x[i+1]],[y[i],y[i+1]])
            popt_flux_pos[f].append(popt)
    def intersections(g):
        diff_=[]
        for i in g:
            popt,pcov=curve_fit(lin,[x[i],x[i+1]],[y[i],y[i+1]])
            range=np.arange(np.sort([x[i],x[i+1]])[0],np.sort([x[i],x[i+1]])[1],0.0001)
            poptm,pcovm=curve_fit(lin,[z_0,m],[0,n])
            diff=abs(lin(range,*popt)-lin(range,*poptm))
            diff_.append(min(diff))
        return (g[np.argmin(diff_)],np.argmin(diff),range,popt,poptm)

    # def intersections(g):
    #     diff_=[]
    #     for i,j in zip(g,np.arange(0,32)):
    #         if n<0:
    #             popt=popt_flux_neg[f][j]
    #         else:
    #             popt=popt_flux_pos[f][j]
    #         range=np.arange(np.sort([x[i],x[i+1]])[0],np.sort([x[i],x[i+1]])[1],0.0001)
    #         #plt.plot(range,lin(range,*popt),color=colors[n])
    #         poptm,pcovm=curve_fit(lin,[z_0,m],[0,n])
    #         diff=abs(lin(range,*popt)-lin(range,*poptm))
    #         diff_.append(min(diff))
    #     return (g[np.argmin(diff_)],np.argmin(diff),range,popt,poptm)

    m=60
    for n in np.arange(-5,5):
        for f in fluxsurfaces:
            plt.plot(m,n,marker='.',color=colors[f])
            if n<0:
                half=np.concatenate((np.arange(-1,0),np.arange(32,63)))
            else:
                half=np.arange(0,32)
            x=np.array(x_.iloc[f])
            y=np.array(y_.iloc[f])
            plt.plot(np.append(x,x[0]),np.append(y,y[0]),'.',color=colors[f])
            ideal=intersections(half)[0]
            x_inner=intersections([ideal])[2][intersections([ideal])[1]]
            y_inner=lin(x_inner,*(intersections([ideal])[4]))
            x=np.array(x_.iloc[f+1])
            y=np.array(y_.iloc[f+1])
            ideal=intersections(half)[0]
            x_outer=intersections([ideal])[2][intersections([ideal])[1]]
            y_outer=lin(x_outer,*(intersections([ideal])[4]))
    plt.show()
    print('total:',datetime.now()-start)


# Pixelmethod
def Pixelmethod():
    start=datetime.now()
    #-----------------------------------------------------#-
    #!!! Enter here which channels lines of sight you want to have analyzed(1 to 8), what pixel-resolution you need (in cm) and  which flux surfaces (0 to 7) should be modeled. 
    #note that more channels, a smaller resolution and more fluxsurfaces result in longer computation times
    bolo_channel=[5] #1 to 8
    Bolo= False
    res=0.5
    fluxsurfaces=[0,1,2,3,4,5,6,7,8,9,10,11]
    err='None'
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
        fig=plt.figure(figsize=(w,w))
        ax=fig.add_subplot(111)
        x_=pd.DataFrame(pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_position_extended.csv',sep=',',engine='python'),dtype=np.float64)
        y_=pd.read_csv('/home/gediz/IDL/Fluxsurfaces/example/Fluxsurfaces_10_angle30_radii_extended.csv',sep=',',engine='python')


        plt.xlabel('R [cm]')
        plt.ylabel('Z [cm]')

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
        if Bolo==True:
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
            plt.plot(x,lin(x,*popt3),color=colors[bol-1])
            plt.plot(x,lin(x,*popt4),color=colors[bol-1])
            plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[lines[bol-1]],ex_2[lines[bol-1]],ex_3[lines[bol-1]]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=colors[bol-1])
            plt.errorbar([a-b-12.4,a-b-19.5,a-b-22.9],[ex_1[lines[bol-1]+1],ex_2[lines[bol-1]+1],ex_3[lines[bol-1]+1]],yerr=0.4,xerr=0.4,marker='o', linestyle='None',capsize=5,color=colors[bol-1])

        for m in m_:
            for n in n_:
                if Bolo==True:
                    if n< lin(m,*popt4) and n>(lin(m,*popt3)):
                        inside_line[bol-1].append((m,n))
                else:
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
            x=np.array(x_.iloc[f])
            y=np.array(y_.iloc[f])
            plt.plot(np.append(x,x[0]),np.append(y,y[0]),color=colors2[f],linewidth=2,marker=markers[0],markersize=4,alpha=0.5)

        #m_=list(p+0.01 for p in (np.arange(int(min(x_.iloc[i+1])),int(max(x_.iloc[i+1]))+1,res)))
        #n_=list(b+0.01 for b in (np.arange(int(min(y_.iloc[i+1])),int(max(y_.iloc[i+1]))+1,res)))
            for (m,n) in inside_line[bol-1]:
                if (m,n) not in inside_:
                    if n<0:
                        half=np.concatenate((np.arange(-1,0),np.arange(32,63)))
                    else:
                        half=np.arange(0,32)
                    x=np.array(x_.iloc[f])
                    y=np.array(y_.iloc[f])
                    ideal=intersections(half)[0]
                    x_inner=intersections([ideal])[2][intersections([ideal])[1]]
                    y_inner=lin(x_inner,*(intersections([ideal])[4]))
                    x=np.array(x_.iloc[f+1])
                    y=np.array(y_.iloc[f+1])
                    ideal=intersections(half)[0]
                    x_outer=intersections([ideal])[2][intersections([ideal])[1]]
                    y_outer=lin(x_outer,*(intersections([ideal])[4]))
                    if f==0:
                        if abs(y_inner)>=abs(n) and abs(x_inner-z_0)>=abs(m-z_0):
                            ax.add_patch(mpl.patches.Rectangle((m-res/2,n-res/2),res,res,color=colors2[f],alpha=0.4,linewidth=0))
                            inside[f].append((m,n))
                        if abs(y_outer)>=abs(n) and abs(x_outer-z_0)>=abs(m-z_0) and abs(y_inner)<=abs(n) and abs(x_inner-z_0)<=abs(m-z_0):
                            ax.add_patch(mpl.patches.Rectangle((m-res/2,n-res/2),res,res,color=colors2[f+1],alpha=0.4,linewidth=0))
                            inside[f+1].append((m,n))
                    else:
                        if abs(y_outer)>=abs(n) and abs(x_outer-z_0)>=abs(m-z_0) and abs(y_inner)<=abs(n) and abs(x_inner-z_0)<=abs(m-z_0):
                            ax.add_patch(mpl.patches.Rectangle((m-res/2,n-res/2),res,res,color=colors2[f+1],alpha=0.4,linewidth=0))
                            inside[f+1].append((m,n))
                inside_.extend(inside[f])
        x=np.array(x_.iloc[12])
        y=np.array(y_.iloc[12])
        plt.plot(np.append(x,x[0]),np.append(y,y[0]),color=colors2[12],linewidth=2,marker=markers[0],markersize=4,alpha=0.5)

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

            if Bolo==True:
                vol[v]=np.sum(len_)*res**2
                print("The {n} Fluxsurface covers a space of ~ {s} cm\u00b3 in channel {c} line of sight which is v_i={d}cm.".format(n=v,s=float(f'{vol[v]:.2f}'),d=float(f'{v_i[v][0]:.2e}'),c=bol))
            else:
                vol[v]=len(len_)*res**3
                print("The {n} Fluxsurface covers a space of ~ {s} cm\u00b3  which is v_i={d}cm.".format(n=v,s=float(f'{vol[v]:.2f}'),d=float(f'{v_i[v][0]:.2e}')))

        #plt.ylim(min(y_.iloc[f+1])-res,max(y_.iloc[f+1])+res)
        #plt.xlim(z_0+min(y_.iloc[f+1])-res,80)
        plt.xlim(47,78)
        plt.ylim(-15.5,15.5)
        fig1= plt.gcf()
        plt.show()
        fig1.savefig('/home/gediz/LaTex/Thesis/Figures/pixelmethod_sensor_5.pdf',bbox_inches='tight')
        #fig1.savefig('/home/gediz/Results/Modeled_Data/Fluxsurfaces_and_Lines_of_sight/flux_{f1}_to_{f2}_channel_{b}.pdf'.format(f1=fluxsurfaces[0],f2=fluxsurfaces[-1],b=bolo_channel[0]))

        vol_ges=0
        v_i_ges=0
        for k in np.append(fluxsurfaces, fluxsurfaces[-1]+1):
            if len([vol[k]]) != 0:
                vol_ges+=vol[k]
            if len([v_i[k]]) != 0:
                v_i_ges+=v_i[k][0]
        vol_ges=vol_ges/1000000
        v_i_ges=v_i_ges/100
        if Bolo==True:
            print('Volume Observed by channel {c}: {v}m\u00b3'.format(c=bol,v=float(f'{vol_ges:.5f}')))
        else:
            print('Volume of Plasma Cross-section * {c}: {v}m\u00b3, and v_i_ges={vi}'.format(c=res,v=float(f'{vol_ges:.5f}'),vi=float(f'{v_i_ges:.5f}')) )

    print('total:',datetime.now()-start)

#Total Power from Channel 4

def Totalpower_from_exp(s,g,Type='',save=False,calc=False,plot=False):
    shotnumbers=s
    gases=g
    v_i_ges_middle=[0.00984,0.01764,0.02042,0.02157,0.02157,0.02042,0.01764,0.00984]#0.01476
    v_i_ges_min=[0.00894,0.01554,0.01786,0.01785,0.01785,0.01786,0.01554,0.00894]
    v_i_ges_max=[0.01086,0.01979,0.02309,0.02451,0.02451,0.02309,0.01979,0.01086]
    V_T=0.1198 #till outer most closed fluxsurface
    V_T_2=0.237 #till edge of calculation area with "additional fluxsurfaces"
    if plot==True:
        fig1=plt.figure(figsize=(10,7))
        ax=fig1.add_subplot(111)
        ax2=ax.twinx()
        fig2=plt.figure(figsize=(10,7))
        ax0=fig2.add_subplot(111)
        ax02=ax0.twinx()

    for j in np.arange(len(shotnumbers)):
        P_ges_exp,P_ges_calc,P_ges_calc_no_weighing,P_ges_no_weighing,pressures,mw,discrepancies,factors,P_ges_exp_error_min,P_ges_exp_error_max,P_ges_calc_error_min,P_ges_calc_error_max,P_ges_exp_error_bolo=[],[],[],[],[],[],[],[],[],[],[],[],[]
        for s,g in zip(shotnumbers[j],gases[j]):
            pressures.append(pc.Pressure(s,g))
            mw.append(pc.GetMicrowavePower(s)[0])
        if Type=='Pressure':
            arg=np.sort(pressures)
            sortnumbers=[shotnumbers[j][i] for i in np.argsort(pressures)]
        if Type=='Power':
            arg=np.sort(mw)
            sortnumbers=[shotnumbers[j][i] for i in np.argsort(mw)]
        for s,g in zip(sortnumbers,gases[j]):
            weight=Totalpower_from_Profile(s)[0]
            P_ges_exp_,P_ges_calc_,P_ges_exp_error_min_,P_ges_exp_error_max_,P_ges_calc_error_min_,P_ges_calc_error_max_,P_ges_exp_error_bolo_=[],[],[],[],[],[],[]
            if not os.path.exists('/home/gediz/Results/Modeled_Data/Bolometerprofiles/shot{s}/shot{s}_modeled_powerprofile_{g}.txt'.format(s=s,g=g)):
                calc_p,bolo_p,bolo_err=Boloprofile_calc(s,g,save=True,plot=True)
            else:
                calc_p,bolo_p,bolo_err=np.genfromtxt('/home/gediz/Results/Modeled_Data/Bolometerprofiles/shot{s}/shot{s}_modeled_powerprofile_{g}.txt'.format(s=s,g=g),unpack=True, usecols=(1,2,3))
            calc_p=[p*10**(-6)for p in calc_p]
            bolo_p=[p*10**(-6) for p in bolo_p]
            bolo_err=[p*10**(-6) for p in bolo_err]
            P_ges_no_weighing.append(((4*np.pi*bolo_p[3])/(((c_w*c_h)/10000)*v_i_ges_middle[3]))*V_T_2)
            p_rad=[]
            for i in [0,1,2,3,4,5,6,7]:
                p_rad.append((4*np.pi*bolo_p[i])/(((c_w*c_h)/10000)*v_i_ges_middle[i]))
                P_ges_exp_.append(((4*np.pi*bolo_p[i])/(((c_w*c_h)/10000)*v_i_ges_middle[i]))*V_T_2*weight[i])
                P_ges_exp_error_min_.append(((4*np.pi*bolo_p[i])/(((c_w*c_h)/10000)*v_i_ges_min[i]))*V_T_2*weight[i])
                P_ges_exp_error_max_.append(((4*np.pi*bolo_p[i])/(((c_w*c_h)/10000)*v_i_ges_max[i]))*V_T_2*weight[i])
                P_ges_exp_error_bolo_.append(((4*np.pi*bolo_err[i])/(((c_w*c_h)/10000)*v_i_ges_middle[i]))*V_T_2*weight[i])
            P_ges_exp.append(np.mean(P_ges_exp_))
            P_ges_exp_error_min.append(abs(np.mean(P_ges_exp_error_min_)-np.mean(P_ges_exp_)))#+np.mean(P_ges_exp_error_bolo_))
            P_ges_exp_error_max.append(abs(np.mean(P_ges_exp_error_max_)-np.mean(P_ges_exp_)))#+np.mean(P_ges_exp_error_bolo_))
            if calc==True:
                P_ges_calc_no_weighing.append(((4*np.pi*calc_p[3])/(((c_w*c_h)/10000)*v_i_ges_middle[3]))*V_T_2)
                for i in [0,1,2,3,4,5,6,7]:
                    P_ges_calc_.append(((4*np.pi*calc_p[i])/(((c_w*c_h)/10000)*v_i_ges_middle[i]))*V_T_2*weight[i])
                    P_ges_calc_error_min_.append(((4*np.pi*calc_p[i])/(((c_w*c_h)/10000)*v_i_ges_min[i]))*V_T_2*weight[i])
                    P_ges_calc_error_max_.append(((4*np.pi*calc_p[i])/(((c_w*c_h)/10000)*v_i_ges_max[i]))*V_T_2*weight[i])
                P_ges_calc.append(np.mean(P_ges_calc_))
                P_ges_calc_error_min.append(abs(np.mean(P_ges_calc_error_min_)-np.mean(P_ges_calc_)))
                P_ges_calc_error_max.append(abs(np.mean(P_ges_calc_error_max_)-np.mean(P_ges_calc_)))
            discrepancy, factor=Model_accuracy(s,g)
            discrepancies.append(discrepancy)
            factors.append(factor)

        if plot==True:
            ax0.plot(discrepancies,'o-',color=colors[j])
            ax02.plot(factors,'o-',color=colors[j+1])
            ax0.set_ylim(0)
            ax02.set_ylim(0)
            ax0.tick_params(axis='x', rotation=80)
            ax0.set_xticks(np.arange(0,len(sortnumbers)))
            ax0.set_xticklabels([str(s) for s in sortnumbers])
            ax0.set_ylabel('qualitative discrepancy',color=colors[j])
            ax0.tick_params(axis='y', labelcolor=colors[j])
            ax02.set_ylabel('factor P$_c$$_a$$_l$$_c$/P$_e$$_X$$_p$',color=colors[j+1])
            ax02.tick_params(axis='y', labelcolor=colors[j+1])
            fig2.patch.set_facecolor('white')
            if Type=='Pressure':
                title=r'{g}, MW: {mw}, P$_M$$_W$ $\approx$ {m} kW'.format(g=g,mw=pc.GetMicrowavePower(s)[1],m=float(f'{np.mean(mw)*10**(-3):.1f}'))
                lns1=ax.plot(np.sort(pressures),P_ges_exp,marker=markers[j],color=colors[j],linestyle='dashed',label=title)
                ax.errorbar(np.sort(pressures),P_ges_exp,yerr=(P_ges_exp_error_min,P_ges_exp_error_max),capsize=5,linestyle='None',color=colors[j])
                #lns2=ax.plot(np.sort(pressures),P_ges_no_weighing,marker=markers[j],color=colors[j],linestyle='dashed',label='ch4, '+title,alpha=0.5)           
                if calc==True:
                    lns3=ax2.plot(np.sort(pressures),P_ges_calc,marker=markers[j+1],color=colors[j+1],linestyle='dashed',label='calculated, '+title)
                    ax2.errorbar(np.sort(pressures),P_ges_calc,yerr=(P_ges_calc_error_min,P_ges_calc_error_max),capsize=5)
                    lns4=ax2.plot(np.sort(pressures),P_ges_calc_no_weighing,marker=markers[j+1],color=colors[j+1],linestyle='dashed',label='calculated, ch4, '+title, alpha=0.5)
                ax.set_xlabel('pressure [mPa]')
            if Type=='Power':
                title='{g}, \t MW: {mw}, \t'.format(g=g,mw=pc.GetMicrowavePower(s)[1],)+r'p$\approx$'+'{p} mPa'.format(p=float(f'{np.mean(pressures):.1f}'))
                lns1=ax.plot(np.sort(mw),P_ges_exp,marker=markers[j],color=colors[j],linestyle='dashed',label=title)
                ax.errorbar(np.sort(mw),P_ges_exp,yerr=(P_ges_exp_error_min,P_ges_exp_error_max),capsize=5,linestyle='None',color=colors[j])
                #lns2=ax.plot(np.sort(mw),P_ges_no_weighing,marker=markers[j],color=colors[j],linestyle='dashed',label='ch4, '+title, alpha=0.5)
                if calc==True:
                    lns3=ax2.plot(np.sort(mw),P_ges_calc,marker=markers[j+1],color=colors[j+1],linestyle='dashed',label='calculated, '+title)           
                    ax2.errorbar(np.sort(mw),P_ges_calc,yerr=(P_ges_calc_error_min,P_ges_calc_error_max),capsize=5)
                    lns4=ax2.plot(np.sort(mw),P_ges_calc_no_weighing,marker=markers[j+1],color=colors[j+1],linestyle='dashed',label='calculated, ch4,  '+title,alpha=0.5)
                ax.set_xlabel('P$_M$$_W$ [W]')
            ax.set_ylim(0)
            ax.set_ylabel('P$_r$$_a$$_d$ total exp [W]')
            if calc==True:
                ax2.set_ylim(0)
                leg = lns1 + lns2 + lns3 + lns4
                ax.tick_params(axis='y', labelcolor=colors[j])
                ax2.set_ylabel('P$_r$$_a$$_d$ total calc [W]',color=colors[j+1])
                ax.set_ylabel('P$_r$$_a$$_d$ total exp [W]',color=colors[j])
                ax2.tick_params(axis='y', labelcolor=colors[j+1])
                fig1.patch.set_facecolor('white')
                labs = [l.get_label() for l in leg]
                ax.legend(leg,labs,loc='lower center',bbox_to_anchor=(0.5,-0.45*len(shotnumbers)-0.1))

            else:
                ax.legend(loc='lower center',bbox_to_anchor=(0.5,-0.15*len(shotnumbers)-0.1))
                ax2.set_yticks([])
    if plot==True:
        fig1.show()
        #fig2.show()


    if save==True:
        fig1.savefig('/home/gediz/Results/Modeled_Data/Tota_P_rad/comparison_{T}_{g}.pdf'.format(T=Type, g=gases), bbox_inches='tight')
    return arg,P_ges_exp,P_ges_exp_error_min,P_ges_exp_error_max,P_ges_calc,P_ges_calc_error_min,P_ges_calc_error_max

def Totalpower_from_Profile(s):
    vol_4=[0.00010813,0.0001071,0.00015026,0.00018826,0.00019817,0.00019164,0.00019493,0.000214,0.0002304,0.00019267,0.00019731,0.00018608,0.00019864]
    vol_3=[0,0.00002208,0.00006679,0.000102,0.00015254,0.00021412,0.00026981,0.00031004,0.00027036,0.00020052,0.00019463,0.00018738,0.00019145]
    vol_2=[0,0,0,0,0.00000581,0.00008295,0.00015392,0.00021417,0.00027148,0.00027282,0.0003086,0.00026246,0.00023091]
    vol_1=[0,0,0,0,0,0,0,0,0.00007158,0.00014295,0.00020077,0.00024161,0.0002814]
    V_T=[0.005970748724526465 ,0.0266435545159973, 0.010530662608190813, 0.012503519781924724, 0.01467555157319217, 0.017209374953170112,0.02016154954536485, 0.023482155680283764, 0.02749055796497933, 0.03311158096852443, 0.03239377869474992, 0.03492039742142869, 0.03744701614810751]
    p_t,t,flux_t,weight=pc.TemperatureProfile(s,'Values',save=False)[0]*100,pc.TemperatureProfile(s,'Values',save=False)[1],[],[]
    flux_pos=[2.174,5.081,5.844,6.637,7.461,8.324,9.233,10.19,11.207,12.321,13.321,14.321,15.321,16.321]#position of the flux surface edged in cm
    for i in np.arange(0,len(flux_pos)-1):
        interpol_t=pchip_interpolate(p_t,t,np.arange(flux_pos[i],flux_pos[i+1],0.01))
        flux_t.append(np.mean(interpol_t))

    mean_T=np.mean(flux_t)
    for ch in [vol_1,vol_2,vol_3,vol_4,vol_4,vol_3,vol_2,vol_1]:
        T_dens_ges,T_dens_ch=[],[]
        for a,b,c in zip(flux_t,V_T,ch):
            T_dens_ges.append(a*b)
            T_dens_ch.append(a*c)
        weight.append((sum(T_dens_ges)/sum(V_T))/(sum(T_dens_ch)/sum(ch)))
    print(mean_T,sum(T_dens_ges)/sum(V_T),(sum(T_dens_ch)/sum(ch)),(sum(T_dens_ch)/sum(ch))*weight[-1])
    return (weight, mean_T, sum(T_dens_ges)/sum(V_T))

#The power absorbed by the channels calculated with ADAS coefficients
def Boloprofile_calc(s,g,save=False,plot=False):
    
    flux_4,flux_4_max,flux_4_min=np.genfromtxt('/home/gediz/Results/Modeled_Data/Fluxsurfaces_and_Lines_of_sight/flux_0_to_11_channel_4.txt',unpack=True,usecols=(4,5,6))
    flux_3,flux_3_max,flux_3_min=np.genfromtxt('/home/gediz/Results/Modeled_Data/Fluxsurfaces_and_Lines_of_sight/flux_0_to_11_channel_3.txt',unpack=True,usecols=(4,5,6))
    flux_2,flux_2_max,flux_2_min=np.genfromtxt('/home/gediz/Results/Modeled_Data/Fluxsurfaces_and_Lines_of_sight/flux_0_to_11_channel_2.txt',unpack=True,usecols=(4,5,6))
    flux_1,flux_1_max,flux_1_min=np.genfromtxt('/home/gediz/Results/Modeled_Data/Fluxsurfaces_and_Lines_of_sight/flux_0_to_11_channel_1.txt',unpack=True,usecols=(4,5,6))
    flux_pos=[2.174,5.081,5.844,6.637,7.461,8.324,9.233,10.19,11.207,12.321,13.321,14.321,15.321,16.321]#position of the flux surface edged in cm
    p_t,t,p_d,d=pc.TemperatureProfile(s,'Values',save=False)[0]*100,pc.TemperatureProfile(s,'Values',save=False)[1],pc.DensityProfile(s,'Values',save=False)[0]*100,pc.DensityProfile(s,'Values',save=False)[1]
    if g=='H':
        temp,pec= adas.h_adf11(T_max=201)[0],adas.h_adf11(T_max=201)[1]
    if g=='He':
        temp,pec,pec_2= adas.he_adf11(data='plt96_he')[0],adas.he_adf11(data='plt96_he')[1],adas.he_adf11(data='plt96_he')[2]
    if g=='Ar':
        temp,pec,pec_2= adas.ar_adf11()[0],adas.ar_adf11()[1],adas.ar_adf11()[2]
    if g=='Ne':
        temp,pec,pec_2= adas.ne_adf11(data='plt96_ne')[0],adas.ne_adf11(data='plt96_ne')[1],adas.he_adf11(data='plt96_he')[2]

    flux_t,flux_d,flux_rc,flux_rc_2=[],[],[],[]
    n=pc.Densities(s,g)[0]
    n_e=pc.Densities(s,g)[1]
    n_0=pc.Densities(s,g)[2]
    for i in np.arange(0,len(flux_pos)-1):
        interpol_t=pchip_interpolate(p_t,t,np.arange(flux_pos[i],flux_pos[i+1],0.01))
        interpol_d=pchip_interpolate(p_d,d,np.arange(flux_pos[i],flux_pos[i+1],0.01))
        flux_t.append(np.mean(interpol_t))
        flux_d.append(np.mean(interpol_d))
        interpol_rc=pchip_interpolate(temp,pec,flux_t[i])
        flux_rc.append(interpol_rc)
        if g=='He' or g=='Ar' or g=='Ne':
            interpol_rc_2=pchip_interpolate(temp,pec_2,flux_t[i])
            flux_rc_2.append(interpol_rc_2)
    P_profile_calc,power_from_ions,power_from_neutrals,error_P_calc=[],[],[],[[],[]]
    for b,e_max,e_min in zip([flux_1,flux_2,flux_3,flux_4,flux_4,flux_3,flux_2,flux_1],[flux_1_max,flux_2_max,flux_3_max,flux_4_max,flux_4_max,flux_3_max,flux_2_max,flux_1_max],[flux_1_min,flux_2_min,flux_3_min,flux_4_min,flux_4_min,flux_3_min,flux_2_min,flux_1_min]):
        P_0,P_1,P_0_min,P_0_max,P_1_min,P_1_max=0,0,0,0,0,0
        if g=='H':
            for j in np.arange(0,len(flux_d)):
                P_0+=((flux_rc[j]*flux_d[j]*n_0*b[j])*1.602E-19)*(A_D/(4*np.pi))
                P_0_min+=((flux_rc[j]*flux_d[j]*n_0*e_min[j])*1.602E-19)*(A_D/(4*np.pi))
                P_0_max+=((flux_rc[j]*flux_d[j]*n_0*e_max[j])*1.602E-19)*(A_D/(4*np.pi))
                P_1,P_1_min,P_1_max=0,0,0
        if g=='He' or g=='Ar' or g=='Ne':
            for j in np.arange(0,len(flux_d)):
                P_0+=((flux_rc[j]*n_e*n_0*b[j])*1.602E-19)*(A_D/(4*np.pi))
                P_0_min+=((flux_rc[j]*n_e*n_0*e_min[j])*1.602E-19)*(A_D/(4*np.pi))
                P_0_max+=((flux_rc[j]*n_e*n_0*e_max[j])*1.602E-19)*(A_D/(4*np.pi))
                P_1+=((flux_rc_2[j]*n_e*flux_d[j]*b[j])*1.602E-19)*(A_D/(4*np.pi))
                P_1_min+=((flux_rc_2[j]*flux_d[j]*n_e*e_min[j])*1.602E-19)*(A_D/(4*np.pi))
                P_1_max+=((flux_rc_2[j]*flux_d[j]*n_e*e_max[j])*1.602E-19)*(A_D/(4*np.pi))
        P_profile_calc.append((P_0+P_1)/10**(-6))
        error_P_calc[0].append(abs(P_0+P_1-P_0_min-P_1_min)/10**(-6))
        error_P_calc[1].append(abs(P_0+P_1-P_0_max-P_1_max)/10**(-6))
        power_from_ions.append(P_1/10**(-6))
        power_from_neutrals.append(P_0/10**(-6))

    P_profile,error_P_exp=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot{s}/shot{s}_bolometerprofile_from_radiation_powers.txt'.format(s=s),unpack=True,usecols=(1,2))
    P_profile_corr=[(a/(b/d*gold(g)[0]+c/d*gold(g)[1]))*mesh for a,b,c,d in zip(P_profile,power_from_neutrals,power_from_ions,P_profile_calc)]
    c1='#18a8d1'
    c2='#fb8500'
    title= '{g}, shot n°{s}, MW: {mw} \n P$_M$$_W$= {m} W, p={p} mPa'.format(g=g,s=s,mw=pc.GetMicrowavePower(s)[1],m=float(f'{pc.GetMicrowavePower(s)[0]:.1f}'),p=float(f'{pc.Pressure(s,g):.1f}'))
    if plot==True:
        print('neutral gas density:','%.2E' %n_0)
        print('mean electron density:','%.2E' %n_e)
        print('degree of ionisation:',str(round((n_e/n)*100,2)), '%')
        print('power from ions:',str(round((np.mean([a/b for a,b in zip(power_from_ions,P_profile_calc)]))*100,2)), '%')
        print('power from neutrals:',str(round((np.mean([a/b for a,b in zip(power_from_neutrals,P_profile_calc)]))*100,2)), '%')
        fig=plt.figure(figsize=(10,7))
        ax=fig.add_subplot(111)
        ax2=ax.twinx()
        lns0=ax2.plot((1,2,3,4,5,6,7,8),P_profile_calc,'v--',color=c2,label='calculated power profile')
        ax2.errorbar((1,2,3,4,5,6,7,8),P_profile_calc,yerr=(error_P_calc[0],error_P_calc[1]), capsize=5,linestyle='None',color=c2)
        #lns1=ax.plot((1,2,3,4,5,6,7,8),P_profile,'o--',color=c1,label='measured power profile, uncorr',alpha=0.5)
        #lns2=ax.plot((1,2,3,4,5,6,7,8),P_profile*mesh/gold(g)[0],'o--',color=c1,label='measured power profile, corr for neu',alpha=0.75)
        lns3=ax.plot((1,2,3,4,5,6,7,8),P_profile_corr,'o--',color=c1,label='measured power profile, corrected')
        ax.errorbar((1,2,3,4,5,6,7,8),P_profile_corr,yerr=error_P_exp, capsize=5,linestyle='None',color=c1)

        ax.set_xticks((1,2,3,4,5,6,7,8))
        ax.set_ylim(0)
        ax2.set_ylim(0)
        ax.set_xlabel('bolometer channel',fontsize=35)
        ax.set_ylabel('P$_r$$_a$$_d$ [\u03bcW]',color=c1,fontsize=35)
        ax.tick_params(axis='y', labelcolor=c1)
        ax2.set_ylabel('P$_r$$_a$$_d$ [\u03bcW]',color=c2,fontsize=35)
        ax2.tick_params(axis='y', labelcolor=c2)
        fig.patch.set_facecolor('white')
        leg = lns0  +lns3
        labs = [l.get_label() for l in leg]
        ax.legend(leg, labs, loc='lower center', title=title)
        fig1= plt.gcf()   
        plt.show()
    if save==True:
        outfile='/home/gediz/Results/Modeled_Data/Bolometerprofiles/'
        if not os.path.exists(str(outfile)+'shot{}'.format(s)):
            os.makedirs(str(outfile)+'shot{}'.format(s))
        data = np.column_stack([np.array([1,2,3,4,5,6,7,8]), np.array(P_profile_calc), np.array(P_profile_corr), np.array(error_P_exp)])#, np.array(z), np.array(abs(y-z))])
        np.savetxt(str(outfile)+"shot{n}/shot{n}_modeled_powerprofile_{g}.txt".format(n=s, g=g) , data, delimiter='\t \t', fmt=['%d', '%10.3f','%10.3f','%10.3f'], header='Modeled power data \n {t} \n bolometerchannel \t power modeled [\u03bcW] \t power measured and corrected [\u03bcW] \t error for experimental Data [\u03bcW]'.format(t=title))
        #fig1.savefig(str(outfile)+"shot{n}/shot{n}_modeled_powerprofile_{g}.pdf".format(n=s, g=gas), bbox_inches='tight')
    return P_profile_calc,P_profile_corr,error_P_exp

def Forward_modeling(s,g):
        
    flux_4=[0.00106,0.00105,0.00145,0.00181,0.00188,0.00180,0.00179,0.00191,0.00207,0.00169,0.00178,0.00164,0.00187]
    flux_3=[0,0.000216,0.000644,0.000984,0.00145,0.00202,0.00249,0.00284,0.00243,0.00194,0.00187,0.00177,0.00177]
    flux_2=[0,0,0,0,0.0000568,0.000782,0.00144,0.00198,0.00249,0.00271,0.00310,0.00271,0.00237]
    flux_1=[0,0,0,0,0,0,0,0,0.000667,0.0014,0.00207,0.00258,0.00312]
    flux_pos=[2.174,5.081,5.844,6.637,7.461,8.324,9.233,10.19,11.207,12.321,13.321,14.321,15.321,16.321]#position of the flux surface edged in cm
    p_t,t,p_d,d=pc.TemperatureProfile(s,'Values',save=False)[0]*100,pc.TemperatureProfile(s,'Values',save=False)[1],pc.DensityProfile(s,'Values',save=False)[0]*100,pc.DensityProfile(s,'Values',save=False)[1]
    P_profile=np.genfromtxt('/home/gediz/Results/Bolometer_Profiles/shot{s}/shot{s}_bolometerprofile_from_radiation_powers.txt'.format(s=s),usecols=1)
    if g=='H':
        temp,pec= adas.h_adf11(T_max=201)[0],adas.h_adf11(T_max=201)[1]

    flux_t,flux_d,flux_rc=[],[],[]
    n=pc.Densities(s,g)[0]
    n_e=pc.Densities(s,g)[1]
    n_0=pc.Densities(s,g)[2]
    for i in np.arange(0,len(flux_pos)-1):
        interpol_t=pchip_interpolate(p_t,t,np.arange(flux_pos[i],flux_pos[i+1],0.01))
        flux_t.append(np.mean(interpol_t))
    def P_rad(x,n_e,n_0,v_i):
        return x*n_e*n_0*v_i*1.602E-19*(A_D/(4*np.pi))
    pec_modeled=[]
    for a,b in zip([0,1,2,3,4,5,6,7],[flux_1,flux_2,flux_3,flux_4,flux_4,flux_3,flux_2,flux_1]):
        for j in np.arange(0,len(flux_d)):
            n_e=flux_d[j]
            n_0=pc.Densities(s,g)[2]
            v_i=b[j]
            P=P_profile[a]*mesh/gold(g)
            print(nsolve(P_rad(x,n_e,n_0,v_i),x,P))

def Model_accuracy(s,g):
    if not os.path.exists('/home/gediz/Results/Modeled_Data/Bolometerprofiles/shot{s}/shot{s}_modeled_powerprofile_{g}.txt'.format(s=s,g=g)):
        calc_p,bolo_p=Boloprofile_calc(s,g,save=True,plot=True)
    else:
        calc_p,bolo_p=np.genfromtxt('/home/gediz/Results/Modeled_Data/Bolometerprofiles/shot{s}/shot{s}_modeled_powerprofile_{g}.txt'.format(s=s,g=g),unpack=True, usecols=(1,2))
    mean_calc=np.mean(calc_p)
    mean_exp=np.mean(bolo_p)
    norm_calc=[i/mean_calc for i in calc_p]
    norm_exp=[i/mean_exp for i in bolo_p]
    diff=[]
    for i,j in zip(norm_calc,norm_exp):
        diff.append(np.abs(i-j))
    return sum(diff),(mean_calc/mean_exp)

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
if __name__ == "__main__":
    start=datetime.now()
    print('start:', start)
    for shotnumber in [13265,13263,13261,13259,13257]:
        #shotnumber=13257
        shotnumbers=[[13261]]#[np.arange(13242,13255),np.arange(13299,13311),[13098,13099,13100,13101,13102,13103,13104,13105,13106],np.arange(13340,13348),[13079,13080,13081,13082,13083,13084],[13089,13090,13091,13092,13093,13094,13095],[13089,13090,13091,13092,13093,13094,13095],[13089,13090,13091,13092,13093,13094,13095]]
        gases=[['He'for i in range(1)]]#[['H'for i in range(13)],['Ar'for i in range(13)],['Ar'for i in range(9)],['Ne'for i in range(8)],['Ne'for i in range(7)],['H'for i in range(8)],['Ar'for i in range(8)],['Ne'for i in range(8)]]
        gas='He'
        infile='/data6/shot{s}/kennlinien/auswert'.format(s=shotnumber)

        location ='/data6/shot{name}/interferometer/shot{name}.dat'.format(name=shotnumber)
        mesh=1/0.75     #multiply with this factor to account for 25% absorbance of mesh
        def gold(g):
            if g=='H':
                return [0.76 ,0]     
            if g=='He':      
                return [0.50 ,0.92]     
            if g=='Ar':
                return [0.84,0.80]
            if g=='Ne':
                return [0.61,0.78]
        Boloprofile_calc(shotnumber,gas,save=True)
    print('total:',datetime.now()-start)
  # %%


