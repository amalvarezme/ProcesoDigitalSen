# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:20:53 2019

@author: andre
"""

#%% leer base de datos BCI competiton iv 2a

import mne
import numpy as np

#%%
def leer_bci42a(path_filename):
    
    raw = mne.io.read_raw_edf(path_filename,preload=False)
    
    #raw.save('tempraw.fif',overwrite=True)#, tmin=3, tmax=5,overwrite = True)
    #rawo = mne.io.read_raw_fif('tempraw.fif', preload=True)  # load data  
    # depurar canales
    #rawo.plot()
    
    clases = [769,770,771,772] #codigo clases
    indx_   = raw._raw_extras[0]['events'][1]           # Indices de las actividades.
    indx2_  = raw._raw_extras[0]['events'][2]           # Marcadores de las actividades.
    remov   = np.ndarray.tolist(indx2_)                 # Quitar artefactos.
    Trials_eli = 1023                                   # Elimina los trials con artefactos.
    m       = np.array([i for i,x in enumerate(remov) if x==Trials_eli])   # Identifica en donde se encuentra los artefactos.
    m_      = m+1
    tt      = np.array(raw._raw_extras[0]['events'][0]*[1],dtype=bool)
    tt[m]   = False
    tt[m_]  = False
    event1_ = indx_[tt]
    event2_ = indx2_[tt]
    # selecciona los indices de las 4 clases que contiene la base de datos.
    tt1     = np.array(event2_.shape[0]*[0],dtype=bool)
    C1      = np.array([i for i,x in enumerate(np.ndarray.tolist(event2_)) if x==clases[0]])                
    C2      = np.array([i for i,x in enumerate(np.ndarray.tolist(event2_)) if x==clases[1]])
    C3      = np.array([i for i,x in enumerate(np.ndarray.tolist(event2_)) if x==clases[2]])
    C4      = np.array([i for i,x in enumerate(np.ndarray.tolist(event2_)) if x==clases[3]])
    tt1[C1],tt1[C2],tt1[C3],tt1[C4] = True,True,True,True
    # con los indices de las clases seleccionada
    i_muestras = event1_[tt1] # indices en muestra del inicio estimulo -> tomar 2 seg antes y 5 seg despues
    i_clases = event2_[tt1] # tipo de clase
    #tt2     = np.array(te_2.shape[0]*[0],dtype=bool)
    #C_      = np.array([i for i,x in enumerate(np.ndarray.tolist(te_2)) if x==clas])
    #tt2[C_] = True
    return i_muestras, i_clases, raw


