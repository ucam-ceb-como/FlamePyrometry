# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:02:21 2018

@author: jd766
"""

def Onion_Peeling_Tikhonov(ImRed, ImGrn, ImBlu, alpha):
    import numpy as np
    from numpy.linalg import inv
    import os
    from os.path import abspath, exists
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create file names and check if the BASEX matrix was already calculated
    filename_A_OP = abspath( '{0}{1}{2}{3}{4}{5}'.format('Onion_Peeling_Tikhonov_matrices//Onion_Tik_A_OP_', str(len(ImRed)), 'x', str(len(ImRed[0])), '_alpha', str(alpha)))
    filename_A_Tik = abspath( '{0}{1}{2}{3}{4}{5}'.format('Onion_Peeling_Tikhonov_matrices//Onion_Tik_A_Tik_', str(len(ImRed)), 'x', str(len(ImRed[0])), '_alpha', str(alpha)))
    filename_A_OP = ('{0}{1}'.format(filename_A_OP.replace('.','p'), '.csv'))
    filename_A_Tik = ('{0}{1}'.format(filename_A_Tik.replace('.','p'), '.csv'))
    
    # If exist load conversion matrix
    exists = os.path.isfile(filename_A_OP) and os.path.isfile(filename_A_Tik)
    if exists:
        A_OP = pd.read_csv(filename_A_OP, delimiter=',', header=None)
        A_Tik = pd.read_csv(filename_A_Tik, delimiter=',', header=None)
        
    # If file doesn't exist calculate conversion matrix
    else:
        # Procedure and equations for the Onion Peeling are based on Dasch, Applied Optics 31 (1992) 1146-1152
        # The procedure and equations for the Tikhonov regularization are based on
        # Daun et al., Applied Optics 45 (2006) 4638-4646
        
        # Calculate matrix of Eq. 4 in Daun et al.; Identical to Eq. 11 in Dasch et
        # al.
        A_OP = np.zeros((round(len(ImRed[0])/2)+1, round(len(ImRed[0])/2)+1 ))
        for i in range(0, round(len(ImRed[0])/2)+1):
            for j in range(0, round(len(ImRed[0])/2)+1):
                if j<i:
                    A_OP[i,j] = 0
                elif j==i:
                    A_OP[i,j] = 2*np.sqrt((j+0.5)**2-i**2)
                else:
                    A_OP[i,j] = 2*(np.sqrt((j+0.5)**2-i**2)-np.sqrt((j-0.5)**2-i**2))
        
        # Generate L_0 matrix as in Daun et al., Eq. 19
        L_0 = np.zeros((round(len(ImRed[0])/2)+1, round(len(ImRed[0])/2)+1 ))
        
        for m in range(0,len(L_0)):
            for n in range(0,len(L_0)):
                if m == n:
                    L_0[m,n] = 1
                elif m+1 == n:
                    L_0[m,n] = -1
                    
        A_Tik = np.dot(A_OP.transpose(),A_OP) + alpha*np.dot(A_OP.transpose(), np.dot(L_0.transpose(), L_0))
        
        # As G and rho are axisymmetric, only one half needs to be saved 
        np.savetxt(filename_A_OP, A_OP ,delimiter=',')
        np.savetxt(filename_A_Tik, A_Tik ,delimiter=',')
        
    ImRed_half = (ImRed[:,round(len(ImRed[0])/2):len(ImRed[0])]+np.flip(ImRed[:,0:round(len(ImRed[0])/2)+1], axis=1)) / 2
    ImGrn_half = (ImGrn[:,round(len(ImGrn[0])/2):len(ImGrn[0])]+np.flip(ImGrn[:,0:round(len(ImGrn[0])/2)+1], axis=1)) / 2
    ImBlu_half = (ImBlu[:,round(len(ImBlu[0])/2):len(ImBlu[0])]+np.flip(ImBlu[:,0:round(len(ImBlu[0])/2)+1], axis=1)) / 2
    
    # Conduct Tikhonov regularization with Onion Peeling matrix as in Eq. 21 in
    # Daun et al.; note: in Daun et al. the 3. A_OP is without trans. 
    b_Tik_red = np.dot(A_OP.transpose(), ImRed_half.transpose())
    b_Tik_grn = np.dot(A_OP.transpose(), ImGrn_half.transpose())
    b_Tik_blu = np.dot(A_OP.transpose(), ImBlu_half.transpose())
    
    D_Tik  = inv(A_Tik)
    
    # Conduct deconvolution according to Eq. 20 in Daun et al. 
    R_red_half = np.flip(np.dot(D_Tik, b_Tik_red), axis=0).transpose()
    R_grn_half = np.flip(np.dot(D_Tik, b_Tik_grn), axis=0).transpose()
    R_blu_half = np.flip(np.dot(D_Tik, b_Tik_blu), axis=0).transpose()
    
    P_red_half = np.dot(np.dot(ImRed_half, A_OP), np.dot(D_Tik.transpose(), A_OP.transpose()) )
    P_grn_half = np.dot(np.dot(ImGrn_half, A_OP), np.dot(D_Tik.transpose(), A_OP.transpose()) )
    P_blu_half = np.dot(np.dot(ImBlu_half, A_OP), np.dot(D_Tik.transpose(), A_OP.transpose()) )
    
    R_red = np.concatenate((R_red_half[:,0:len(R_red_half[0])], np.flip(R_red_half[:,0:len(R_red_half[0])-1], axis=1)), axis=1)
    R_grn = np.concatenate((R_grn_half[:,0:len(R_grn_half[0])], np.flip(R_grn_half[:,0:len(R_grn_half[0])-1], axis=1)), axis=1)
    R_blu = np.concatenate((R_blu_half[:,0:len(R_blu_half[0])], np.flip(R_blu_half[:,0:len(R_blu_half[0])-1], axis=1)), axis=1)
    
    P_red = np.concatenate((np.flip(P_red_half, axis=1), P_red_half[:,1:len(R_red_half[0])]), axis=1) 
    P_grn = np.concatenate((np.flip(P_grn_half, axis=1), P_grn_half[:,1:len(R_red_half[0])]), axis=1)
    P_blu = np.concatenate((np.flip(P_blu_half, axis=1), P_blu_half[:,1:len(R_red_half[0])]), axis=1)
    
    p1 = int(round(len(ImGrn)/2))
    p2 = int(round(len(ImGrn)/3))
    plt.figure()
    plt.plot(P_grn[p1,:], 'r')
    plt.plot(ImGrn[p1,:], 'xg')
    plt.title('Example comparison between raw data (x) and smoothened onion peeling (-)')
    plt.show()
    
    plt.figure()
    plt.plot(R_grn[p1,:], 'r')
    plt.plot(R_grn[p2,:], 'g') 
    plt.title('Example for the reconstruction of the above profiles')
    plt.show()
  
    return (R_red, R_grn, R_blu, P_red, P_grn, P_blu)