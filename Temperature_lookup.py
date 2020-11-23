# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:58:06 2018

@author: jd766
"""

def Temperature_lookup(filter, lam, T_calc, fit, Thermo_calib, soot_coef):
    from scipy.interpolate import griddata
    import scipy.integrate as integrate
    from scipy.optimize import curve_fit
    import numpy as np
    import pandas as pd
    import os
    from os.path import abspath, exists
    import matplotlib.pyplot as plt
    
    if fit == True:
        filename_lookup = abspath( '{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}{12}{13}' \
                          .format('Temperature_tables\T_lookup_', soot_coef, '_', filter, '_lambda_', str(int(lam[0])), \
                          '_', str(int(lam[-1])), '_T_', str(int(T_calc[0])), '_', str(int(T_calc[-1])), '_fitted', '.csv'))
    elif fit == False:
        filename_lookup = abspath( '{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{12}{13}' \
                          .format('Temperature_tables\T_lookup_', soot_coef, '_', filter, '_lambda_', str(int(lam[0])), \
                          '_', str(int(lam[-1])), '_T_', str(int(T_calc[0])), '_', str(int(T_calc[-1])), '.csv'))
    
    # Check if file already exists
    exists = os.path.isfile(filename_lookup)
    if exists:
        Ratio_tables = pd.read_csv(filename_lookup, delimiter=',', header=None).values
        
    else:
        from S_fun_filter import S_fun_filter
                    
        # Planck's constant in m^2 kg / s
        h = 6.62607004e-34
        
        # Boltzmann constant in m^2 kg / s^2 / K
        k = 1.38064852e-23
        
        # Speed of light in m / s
        c = 299792458
        
        lambda_first = (lam[0]*10)
        
        # Absorption coefficient for soot is selected
        if soot_coef=='Chang':
            alpha_soot = 1.423
        elif soot_coef=='Kuhn':
            alpha_soot = 1.38
        else:
            sys.exit("Chosen soot emissivity not found. Please set soot_coef to available values.")
            
        # Absorption coefficient for the calibration source is set. Note that Ma and Long report a polynomial for the wavelength dependent emissivity
        # of an S-type thermocouple:
        # 1.2018e-6 * lam**2 - 1.7167e-3 * lam + 0.9017
        # Here, the raw data reported by Ma and Long was fitted with a power law function to obtain the absorption coefficient rather than a polynomial 
        # function for the wavelength dependent emissivity. This simplifies the subsequent calculations and makes it consistent with the parameters used
        # for soot.
        alpha_thermo = 0.641
        
        # Get raw camera response
        blackfly_red = pd.read_csv('Filters/Blackfly_response_red.csv', delimiter=',', header=None).values
        blackfly_grn = pd.read_csv('Filters/Blackfly_response_green.csv', delimiter=',', header=None).values
        blackfly_blu = pd.read_csv('Filters/Blackfly_response_blue.csv', delimiter=',', header=None).values
        
        red = griddata(blackfly_red[:,0], blackfly_red[:,1], (lam), method='cubic')
        grn = griddata(blackfly_grn[:,0], blackfly_grn[:,1], (lam), method='cubic')
        blu = griddata(blackfly_blu[:,0], blackfly_blu[:,1], (lam), method='cubic')
        
        red_nofilter = red / 100
        grn_nofilter = grn / 100
        blu_nofilter = blu / 100
        
        # Calculate camera response with filter and lens
        if filter == 'None':
                lens_raw = pd.read_csv('Filters/MVL25M23.csv', delimiter=',', header=None).values
                camera_lens = griddata(lens_raw[:,0], lens_raw[:,1], (lam), method='cubic') / 100
                red_filter = red_nofilter * camera_lens
                grn_filter = grn_nofilter * camera_lens
                blu_filter = blu_nofilter * camera_lens
                
                
        elif filter == 'FGB7':
                FGB7_raw = pd.read_csv('Filters/FGB7.csv', delimiter=',', header=None).values
                lens_raw = pd.read_csv('Filters/MVL25M23.csv', delimiter=',', header=None).values
                camera_filter = griddata(FGB7_raw[:,0], FGB7_raw[:,1], (lam), method='cubic') / 100
                camera_lens = griddata(lens_raw[:,0], lens_raw[:,1], (lam), method='cubic') / 100
                red_filter = red_nofilter * camera_filter * camera_lens
                grn_filter = grn_nofilter * camera_filter * camera_lens
                blu_filter = blu_nofilter * camera_filter * camera_lens   
                
        elif filter == 'FGB39':
                FGB39_raw = pd.read_csv('Filters/FGB39.csv', delimiter=',', header=None).values
                lens_raw = pd.read_csv('Filters/MVL25M23.csv', delimiter=',', header=None).values
                camera_filter = griddata(FGB39_raw[:,0], FGB39_raw[:,1], (lam), method='cubic') / 100
                camera_lens = griddata(lens_raw[:,0], lens_raw[:,1], (lam), method='cubic') / 100
                red_filter = red_nofilter * camera_filter * camera_lens
                grn_filter = grn_nofilter * camera_filter * camera_lens
                blu_filter = blu_nofilter * camera_filter * camera_lens   
                
        else:
            sys.exit("Selected camera filter does not exist. Programme stopped")
        
        # Measured R-type/S-type thermocouple response
        T_measure = Thermo_calib[:,0]
        RG_measure = Thermo_calib[:,1]
        RB_measure = Thermo_calib[:,2]
        BG_measure = Thermo_calib[:,3]
        
        fun_R_thermo = np.zeros((len(T_measure)))
        fun_G_thermo = np.zeros((len(T_measure)))
        fun_B_thermo = np.zeros((len(T_measure)))
        
        cor_R_thermo = np.zeros((len(T_calc)))
        cor_G_thermo = np.zeros((len(T_calc)))
        cor_B_thermo = np.zeros((len(T_calc)))
        
        fun_R_soot = np.zeros((len(T_calc)))
        fun_G_soot = np.zeros((len(T_calc)))
        fun_B_soot = np.zeros((len(T_calc)))
        
        lambda_max = np.zeros((len(T_calc)))
        
        # Calculate theo. colour ratios as function of temperature for thermocouple. The applied equations correspond to Eq. 6 in:
        # Kempema and Long, Optics Letters 43 (2018) 1103-1106
        # Note that the same equation is reported using the emissivity instead of the absorption coefficient. The absorption coefficient is used here
        # because emissivity is defined for surfaces, not an aerosol with soot nanoparticles. However, both version will give the same result.
        for T in range(0, len(T_measure)):
            fun_R_thermo[T] = integrate.quad(lambda l: S_fun_filter(T_measure[T],l,red_filter,lambda_first,alpha_thermo), lam[0], lam[-1], maxp1=200, limit=200)[0]
            fun_G_thermo[T] = integrate.quad(lambda l: S_fun_filter(T_measure[T],l,grn_filter,lambda_first,alpha_thermo), lam[0], lam[-1], maxp1=200, limit=200)[0]
            fun_B_thermo[T] = integrate.quad(lambda l: S_fun_filter(T_measure[T],l,blu_filter,lambda_first,alpha_thermo), lam[0], lam[-1], maxp1=200, limit=200)[0]
        
        RG_thermo = fun_R_thermo / fun_G_thermo
        RB_thermo = fun_R_thermo / fun_B_thermo
        BG_thermo = fun_B_thermo / fun_G_thermo
    
        # Fit theorectical camera response to measured thermocouple response?
        if fit == True:
            # Starting values and boundaries
            # Emission source is thermocouple
            def lin_fit(x,a):
                return a*x
            
            # Fit theo. colour ratios to measured colour ratios
            popt1,pcov1 = curve_fit(lin_fit,RG_measure,RG_thermo)
            popt2,pcov2 = curve_fit(lin_fit,RB_measure,RB_thermo)
            popt3,pcov3 = curve_fit(lin_fit,BG_measure,BG_thermo)
            
            # Calculate wavelength dependent camera response using the obtained fitting parameters. The green response is maintained
            red_filter_Mod = red_filter / popt1[0]
            grn_filter_Mod = grn_filter
            blu_filter_Mod = blu_filter / popt3[0]
            
            # Calculate theo. response of camera for soot and thermocouple using the corrected camera response
            for T in range(0, len(T_calc)):
                cor_R_thermo[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,red_filter_Mod,lambda_first,alpha_thermo), lam[0], lam[-1], maxp1=200, limit=200)[0]
                cor_G_thermo[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,grn_filter_Mod,lambda_first,alpha_thermo), lam[0], lam[-1], maxp1=200, limit=200)[0]
                cor_B_thermo[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,blu_filter_Mod,lambda_first,alpha_thermo), lam[0], lam[-1], maxp1=200, limit=200)[0]
                
                fun_R_soot[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,red_filter_Mod,lambda_first,alpha_soot), lam[0], lam[-1], maxp1=200, limit=200)[0]
                fun_G_soot[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,grn_filter_Mod,lambda_first,alpha_soot), lam[0], lam[-1], maxp1=200, limit=200)[0]
                fun_B_soot[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,blu_filter_Mod,lambda_first,alpha_soot), lam[0], lam[-1], maxp1=200, limit=200)[0]
                
            RG_thermo_cor = (cor_R_thermo / cor_G_thermo)
            RB_thermo_cor = (cor_R_thermo / cor_B_thermo)
            BG_thermo_cor = (cor_B_thermo / cor_G_thermo)
    
            RG_soot = (fun_R_soot / fun_G_soot)
            RB_soot = (fun_R_soot / fun_B_soot)
            BG_soot = (fun_B_soot / fun_G_soot)        
    
            # Calculate wavelength where the product of green camera response and soot emissivity has its maximum depending on the soot temperature
            for T in range(0, len(T_calc)):
                grn_resp = S_fun_filter(T_calc[T],lam,grn_filter_Mod,lambda_first,alpha_soot)
                I_max = np.argmax(grn_resp)
                lambda_max[T] = lam[I_max]
               
        elif fit == False:
            # If no data fitting is requested, calculate colour ratios with data provided by the camera and filter manufacturer.
            
            # Calculate theo. response of camera for soot
            for T in range(0, len(T_calc)):
                cor_R_thermo[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,red_filter,lambda_first,alpha_thermo), lam[0], lam[-1], maxp1=200, limit=200)[0]
                cor_G_thermo[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,grn_filter,lambda_first,alpha_thermo), lam[0], lam[-1], maxp1=200, limit=200)[0]
                cor_B_thermo[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,blu_filter,lambda_first,alpha_thermo), lam[0], lam[-1], maxp1=200, limit=200)[0]
                
                fun_R_soot[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,red_filter,lambda_first,alpha_soot), lam[0], lam[-1], maxp1=200, limit=200)[0]
                fun_G_soot[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,grn_filter,lambda_first,alpha_soot), lam[0], lam[-1], maxp1=200, limit=200)[0]
                fun_B_soot[T] = integrate.quad(lambda l: S_fun_filter(T_calc[T],l,blu_filter,lambda_first,alpha_soot), lam[0], lam[-1], maxp1=200, limit=200)[0]
            
            RG_thermo_cor = (cor_R_thermo / cor_G_thermo)
            RB_thermo_cor = (cor_R_thermo / cor_B_thermo)
            BG_thermo_cor = (cor_B_thermo / cor_G_thermo)
            
            RG_soot = (fun_R_soot / fun_G_soot) 
            RB_soot = (fun_R_soot / fun_B_soot) 
            BG_soot = (fun_B_soot / fun_G_soot) 
                
            for T in range(0, len(T_calc)):
                grn_resp = S_fun_filter(T_calc[T],lam,grn_filter,lambda_first,alpha_soot)
                I_max = np.argmax(grn_resp)
                lambda_max[T] = lam[I_max]
        
        plt.figure()
        plt.plot(T_measure,RG_measure, 'or')
        plt.plot(T_measure,RB_measure, 'og')
        plt.plot(T_measure,BG_measure, 'ob')
        plt.plot(T_calc,RG_thermo_cor, '--r')
        plt.plot(T_calc,RB_thermo_cor, '--g')
        plt.plot(T_calc,BG_thermo_cor, '--b')
        plt.xlim((T_measure[0]-200,T_measure[-1]+200))
        plt.ylim((0,RB_measure[0]+0.5))
        plt.title('Comparision of measured and calculated colour ratios')
        plt.show()
    
        # Save temperature/colour ratio lookup table
        Ratio_tables = np.concatenate((T_calc, RG_soot, RB_soot, BG_soot, lambda_max), axis=0).reshape((-1, 5), order='F').T
        np.savetxt(filename_lookup, Ratio_tables,delimiter=',') 
        
        # Search theoretical thermocouple temperature corresponding to the experimentally observed colour ratio and compare it to 
        # the experimentally measured temperature. 
        T_diff = np.zeros((len(Thermo_calib),3))     
        for k in range(0,len(Thermo_calib)):
            T_diff[k,0] = T_measure[k] - T_calc[((np.abs(RG_measure[k] - RG_thermo_cor)).argmin())]
            T_diff[k,1] = T_measure[k] - T_calc[((np.abs(RB_measure[k] - RB_thermo_cor)).argmin())]
            T_diff[k,2] = T_measure[k] - T_calc[((np.abs(BG_measure[k] - BG_thermo_cor)).argmin())]  
            
        plt.figure()
        plt.plot(T_diff[:,0], 'or')
        plt.plot(T_diff[:,1], 'og')
        plt.plot(T_diff[:,2], 'ob')
        plt.title('Difference between theoretical and measured thermocouple temperatures.')
        plt.show()
     
    return (Ratio_tables)