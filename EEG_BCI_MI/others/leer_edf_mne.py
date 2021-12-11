# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:45:13 2019

@author: andre
"""

import mne
import numpy as np
from leer_BCI42a import leer_bci42a
import os.path
import matplotlib.pyplot as plt

filename = 'A01T.gdf'
# %%
i_muestras, i_clases, raw = leer_bci42a(filename)

# %% 

sfreq=raw.info['sfreq']
n_m = 3
#veci = range()
tmin = i_muestras[n_m]/sfreq - 2
tmax = i_muestras[n_m]/sfreq + 5
#rawi = raw[:,int(i_muestras[n_m]-sfreq*2):int(i_muestras[n_m]+sfreq*5)]
raw.save('tempraw.fif',tmin = tmin, tmax = tmax, overwrite = True)
rawf = mne.io.read_raw_fif('tempraw.fif', preload=True) 

rawf.plot()
# %%


