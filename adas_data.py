#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

T=[1.00E+00,	2.00E+00,	5.00E+00,	1.00E+01,	2.00E+01,	3.00E+01,	4.00E+01,	5.00E+01,	6.50E+01,	8.00E+01,	1.00E+02,	1.50E+02,	2.00E+02]
D,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13=np.genfromtxt('/home/gediz/ADAS/adas_test_density_temperature_he.txt',unpack=True)
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
fmt = mticker.FuncFormatter(g)
for i in np.arange(11):
    x=[x1[i],x2[i],x3[i],x4[i],x5[i],x6[i],x7[i],x8[i],x9[i],x10[i],x11[i],x12[i],x13[i]]
    plt.plot(T,x,label='Density: {d} 1/cm\u00b3 '.format(d=fmt(D[i])))
plt.legend(loc=1,bbox_to_anchor=(1.5,1))
plt.xlabel('Temperature [eV]')
plt.ylabel('Photon Emissivity Coefficients [cm\u00b3/s]')
plt.show()
# %%
