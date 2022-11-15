#%%

#Written by: Izel Gediz
#Date of Creation: 11.11.2022


from pdb import line_prefix
from unicodedata import name
from blinker import Signal
from click import style
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import statistics
import os
import itertools

#%%

U,I,P=np.genfromtxt('/data6/Auswertung/shot13042/kennlinien/000000.dat',unpack=True)
def Kenn(x,a,b,T):
    return a+b*(1-np.exp(-x/T))
popt,pcov=curve_fit(Kenn,U,I)
print(*popt)
plt.plot(U,I,linestyle='None',marker='.')
plt.plot(U,Kenn(U,*popt))
plt.xlim(0.4,0.8)
plt.show()

plt.plot(U,np.gradient(I),linestyle='None',marker='.')
plt.xlim(0.4,0.8)
def square(x,a,b):
    a+b*x**2
popt,pcov=curve_fit()
plt.show()
# %%
