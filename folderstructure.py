#%%
import os
import numpy as np
import shutil
#%% create the auswert folder for your kennlinien folders
list_dir=os.listdir('/data6/Auswertung')
for s in list_dir:
    path='/data6/{}/kennlinien/auswert'.format(s)
    if os.path.exists('/data6/{}/kennlinien'.format(s)):
        if not os.path.exists(path):
            os.makedirs(path)

#%%
# %%
