Folder: myPython
Created: August 2022 
by: Izel Gediz 

This folder contains all Python scripts I used during my Master Thesis.
Most of them will concerne the data analysis of Bolometerrelated measurements and modelations.
Here is a list of this folders contents and the files purposes

bolo_radiation.py: The first Script I wrote to analyse Bolometer Measurements. It contains functions to 
plot time series, functions to derive the Power from the Voltage Values of the Bolometer, functions
to derive the Signal heights per Bolometerchannel and to compare different Bolometerprofiles.

'power_calculator.py': This skript holds the pixel method used to model power absorption by the bolometer.
See the thesis for details on this method. It uses the flux surface simulations, line of sight measurement and the ADAs data to calculate
the theoretical power emission detected by the bolometer. There are also functions to calculate the total power loss over the whole plasma either 
based on the calculated, theoretical emission or based on actual bolometer measurements.


additional_functions.py: During the development of the script above I created some functions that became
obsolete later. I will collect them here for the case that I ever need them or a function of similar structure again.

golfoil_absorption.py: I wrote functions to plot the goldfoil absorption line, functions to plot the spectra of measurements
and a function to derive the percentage of an arbitrary spectra that will be absorbed by the goldfoil.

bolo_calibration.py: Here are the functions to derive kappa and tau from the omic calibration. (possibly to be extended by relative
calibration and optical absolut calibration functions.)

powermeter.py: There is only one function here to plot and analyze the height of the powermetermeasurements. We measured the powermeter
of the white light source and the green laser with a Thorlabs powermeter we borrowed.

adas_data.py: This script contains all functions I wrote to analyse the adas data. The function is utilized by the power_calculator.py function
to gain the correct photonic emmision coefficients for the respecting gas and temperature and further to calculate the modelled emitted power based on these.

'Figures.py': Here I wrote tiny scripts that create the plots for my thesis. Some are just beatufiying data that is produced in other functions.
some create data themselves I found I need during the writing of my thesis. Each figure script is named with a small descriptive comment.
I will not further document the function of and plot created in each script. Just plot them by running a cell and see for yourself.

'plasma_characteristics.py': This file contains the functions that allow to plot density and temperature curves from Langmuir probe measurements.