# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:24:52 2018

@author: jd766
"""

def BASEX(ImRed, ImGrn, ImBlu, sigma, qx):
    import os
    from os.path import abspath, exists
    import scipy.integrate as integrate 
    from numpy.linalg import inv
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create file names and check if the BASEX matrix was already calculated
    filename_rho = abspath( '{0}{1}{2}{3}{4}{5}{6}{7}'.format('BASEX_matrices//BASEX_rho_', str(len(ImRed)), 'x', str(len(ImRed[0])), '_qx', str(qx), '_sigma', str(sigma)))
    filename_G = abspath( '{0}{1}{2}{3}{4}{5}{6}{7}'.format('BASEX_matrices//BASEX_G_', str(len(ImRed)), 'x', str(len(ImRed[0])), '_qx', str(qx), '_sigma', str(sigma)) )
    filename_rho = ('{0}{1}'.format(filename_rho.replace('.','p'), '.csv'))
    filename_G = ('{0}{1}'.format(filename_G.replace('.','p'), '.csv'))
    
    # If exist load conversion matrix
    exists = os.path.isfile(filename_rho) and os.path.isfile(filename_G)
    if exists:
        rho = pd.read_csv(filename_rho, delimiter=',', header=None)
        G =  pd.read_csv(filename_G, delimiter=',', header=None)
    
    # If file doesn't exist calculate conversion matrix
    else:
        # Procedure and equations based on Apostolopoulos et al., Optics Communications 296 (2013) 25–34
        K_max = round(len(ImRed[0])/2)+1
        radius = np.linspace(0, round(len(ImRed[0])/2), num=round(len(ImRed[0])/2)+1)
        k_s = np.linspace(0, K_max-1, K_max)
        
        # Eq. 9 in ref.
        #r, k= symbols("r k")
        #rho_fun = exp(-2*(r/sigma-k)**2)
        rho_fun = lambda r, k: np.exp(-2*(r/sigma-k)**2)
        
        rho = np.zeros((K_max, len(radius) ))
        G = np.zeros((K_max, len(radius) ))
        
        for i in range(0,K_max):
            for j in range(0, len(radius)-1):
                rho[i,j] = rho_fun(radius[j], k_s[i])
                G[i,j] = 2 * integrate.quad(lambda r: np.exp(-2*(r/sigma-k_s[i])**2)*r/np.sqrt(r**2-radius[j]**2), radius[j], radius[-1], epsabs=1e-12, maxp1=200, limit=200)[0]
                # Abel transform of rho; Eq. 7b in ref.
                rho[i,j+1] =  rho_fun(radius[j+1], k_s[i])
                G[i,j+1] = 0
            
        # As G and rho are axisymmetric, only one half needs to be saved 
        np.savetxt(filename_rho, rho ,delimiter=',')
        np.savetxt(filename_G, G ,delimiter=',')
    
    # Conduct BASEX operation
    # Calculate matrix C obtained via the Tikhonov regularization; Eq. 10 in
    # ref.    
    ImRed_half = (ImRed[:,round(len(ImRed[0])/2):len(ImRed[0])]+np.flip(ImRed[:,0:round(len(ImRed[0])/2)+1], axis=1)) / 2
    ImGrn_half = (ImGrn[:,round(len(ImGrn[0])/2):len(ImGrn[0])]+np.flip(ImGrn[:,0:round(len(ImGrn[0])/2)+1], axis=1)) / 2
    ImBlu_half = (ImBlu[:,round(len(ImBlu[0])/2):len(ImBlu[0])]+np.flip(ImBlu[:,0:round(len(ImBlu[0])/2)+1], axis=1)) / 2
    
    C_red = np.dot(ImRed_half, np.dot(G.transpose(), inv(np.dot(G, G.transpose()) + np.dot(qx**2, np.identity(len(G))) )))
    C_grn = np.dot(ImGrn_half, np.dot(G.transpose(), inv(np.dot(G, G.transpose()) + np.dot(qx**2, np.identity(len(G))) )))
    C_blu = np.dot(ImBlu_half, np.dot(G.transpose(), inv(np.dot(G, G.transpose()) + np.dot(qx**2, np.identity(len(G))) )))
        
    # Eq. 4, 5, and 8 in ref.
    R_red_half = np.dot(C_red, rho)
    R_grn_half = np.dot(C_grn, rho)
    R_blu_half = np.dot(C_blu, rho)
    
    # Eq. 8 in ref.
    P_red_half = np.dot(C_red, G)
    P_grn_half = np.dot(C_grn, G)
    P_blu_half = np.dot(C_blu, G)
    
    R_red = np.concatenate((np.flip(R_red_half, axis=1), R_red_half[:,1:len(R_red_half[0])]), axis=1)
    R_grn = np.concatenate((np.flip(R_grn_half, axis=1), R_grn_half[:,1:len(R_red_half[0])]), axis=1)
    R_blu = np.concatenate((np.flip(R_blu_half, axis=1), R_blu_half[:,1:len(R_red_half[0])]), axis=1)
    
    P_red = np.concatenate((np.flip(P_red_half, axis=1), P_red_half[:,1:len(R_red_half[0])]), axis=1) 
    P_grn = np.concatenate((np.flip(P_grn_half, axis=1), P_grn_half[:,1:len(R_red_half[0])]), axis=1)
    P_blu = np.concatenate((np.flip(P_blu_half, axis=1), P_blu_half[:,1:len(R_red_half[0])]), axis=1)
    
    p1 = int(round(len(ImGrn)/2))
    p2 = int(round(len(ImGrn)/3))
    plt.figure()
    plt.plot(P_grn[p1,:], 'r')
    plt.plot(ImGrn[p1,:], 'xg')
    plt.title('Example comparison between raw data (x) and smoothened BASEX (-)')
    plt.show()
    
    plt.figure()
    plt.plot(R_grn[p1,:], 'r')
    plt.plot(R_grn[p2,:], 'g') 
    plt.title('Example for the reconstruction of the above profiles')
    plt.show()
    
    return (R_red, R_grn, R_blu, P_red, P_grn, P_blu)

