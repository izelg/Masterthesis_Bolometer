Folder: myPython
Created: August 2022 
by: Izel Gediz 

This folder contains all Python scripts I used during my Master Thesis.
Most of them will concerne the data analysis of Bolometerrelated measurements and modelations.
Here is a list of this folders contents and the files purposes

bolo_radiation.py: The first Script I wrote to analyse Bolometer Measurements. It contains functions to 
plot time series, functions to derive the Power from the Voltage Values of the Bolometer, functions
to derive the Signal heights per Bolometerchannel and to compare different Bolometerprofiles.

additional_functions.py: During the development of the script above I created some functions that became
obsolete later. I will collect them here for the case that I ever need them or a function of similar structure again.

golfoil_absorption.py: I wrote functions to plot the goldfoil absorption line, functions to plot the spectra of measurements
and a function to derive the percentage of an arbitrary spectra that will be absorbed by the goldfoil.

bolo_calibration.py: Here are the functions to derive kappa and tau from the omic calibration. (possibly to be extended by relative
calibration and optical absolut calibration functions.)