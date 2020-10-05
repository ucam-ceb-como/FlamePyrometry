# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:09:02 2018

@author: jd766
"""

def Get_flame_temperature(Ratio_tables, R_RB, R_RG, R_BG):
    import numpy as np
    
    Ratio_tables = np.array(Ratio_tables)    
    T_ref = Ratio_tables[0,:]
    RG_ref = Ratio_tables[1,:]
    RB_ref = Ratio_tables[2,:]
    BG_ref = Ratio_tables[3,:]    
    
    T_RG = np.zeros((len(R_RG), len(R_RG[0])))
    T_RB = np.zeros((len(R_RG), len(R_RG[0])))
    T_BG = np.zeros((len(R_RG), len(R_RG[0])))
    for i in range(0,len(R_RG)):
        for j in range(0,len(R_RG[0])):
            T_RG[i,j] = T_ref[((np.abs(RG_ref - R_RG[i,j])).argmin())]
            T_RB[i,j] = T_ref[((np.abs(RB_ref - R_RB[i,j])).argmin())]
            T_BG[i,j] = T_ref[((np.abs(BG_ref - R_BG[i,j])).argmin())]
    
    return (T_RG, T_RB, T_BG)