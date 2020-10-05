# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:55:07 2018

@author: jd766
"""

def Get_soot_volume(filter, lam, pixelmm, exposure, T_ave, R_red, R_grn, R_blu, Ratio_tables,S_thermo):
    import numpy as np
    
    # Value taken from Kuhn et al.
    K_ext = 8.6
    
    Ratio_tables = np.array(Ratio_tables) 
    
    f_ave = np.zeros((len(T_ave), len(T_ave[0])))
    ratio = np.zeros((len(T_ave), len(T_ave[0])))
    
    for i in range(0, len(T_ave)):
        for j in range(0, len(T_ave[0])):
            if T_ave[i,j] != 0:
                # Get the wavelength at max. in green emission at measured temperature from the
                # look-up table
                lambda_max = Ratio_tables[4, ((np.abs(Ratio_tables[0,:] - T_ave[i,j])).argmin())]
    
                # Thermocouple emissivity at max. signal
                
                e_thermo = 1.2018e-6 * lambda_max**2 - 1.7167e-3 * lambda_max + 0.9017;
    
                # Eq. 4 in Kuhn et al. and Eq. 8 in Ma et al.; note that the
                # exposure times of soot and thermocouple seem to be wrong in the 
                # equations and were therefore switched
                ratio[i,j] = e_thermo*R_grn[i,j] / S_thermo(T_ave[i,j]) / exposure
    
                if ratio[i,j] >= 1:
                    f_ave[i,j] = 0
                else:
                    # Calculate soot volume fraction and convert to ppm; as the
                    # temperatures of the thermocouple and soot are identical the
                    # exponential term is not needed
                    f_ave[i,j] = -lambda_max*1e-9/(K_ext*pixelmm*1e-3) * np.log(1-ratio[i,j])*1e6
    
    return (f_ave)