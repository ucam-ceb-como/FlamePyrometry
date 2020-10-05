# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:43:52 2018

@author: jd766
"""

def S_fun_filter(T,l,cam_resp,lambda_first,emissivity):
    import numpy as np
    # Planck's constant in m^2 kg / s
    h = 6.62607004e-34
    
    # Boltzmann constant in m^2 kg / s^2 / K
    k = 1.38064852e-23;
    
    # Speed of light in m / s
    c = 299792458;

    i=(np.round(l*10-lambda_first).astype(int))

    S_red =  cam_resp[i] * emissivity[i] / ((l * 10**(-9))**5) / (np.exp(h*c/((l*10**(-9))*k*T))-1)
    
    return (S_red)