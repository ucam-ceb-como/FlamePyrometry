# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:03:15 2019

@author: AM
"""

def FLiPPID(ImRed, ImGrn, ImBlu, z_range, Nx, Nc, delta_x, delta_c, fit_fun): 
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import minimize
    from os.path import abspath, exists
    import os 
    import pandas as pd
    import scipy.integrate as integrate
    
    filename_FLiPPID = abspath( '{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}'.format('FLiPPID_matrices//FLiPPID_Nx=', str(Nx), '_dx', str(delta_x), '_Nc=', str(Nc[0]), 'to', str(Nc[1]), '_dc', str(delta_x), '_', fit_fun))
    filename_FLiPPID = ('{0}{1}'.format(filename_FLiPPID.replace('.','p'), '.csv'))
        
    exists = os.path.isfile(filename_FLiPPID)
    
    # Check if integral lookup table already exists
    if exists:
        Int_file = pd.read_csv(filename_FLiPPID, delimiter=',', header=None).values
    
    # If the lookup table does not exists, calculate and save it
    else:
        print('FLiPPID table not found. It will now be calculated and saved, which might take several hours.')
        Int_file = np.zeros( (Nx+2, Nc[1]-Nc[0]+2) )
        Nx_line = np.linspace(0,Nx,Nx+1)*delta_x
        Nc_line = np.linspace(Nc[0],Nc[1],Nc[1]-Nc[0]+1)*delta_c
        
        Int_file[1:,0] = Nx_line/delta_x
        Int_file[0,1:] = Nc_line/delta_c
        
        prog_print = np.round(np.linspace(1,len(Nx_line)*len(Nc_line),10)) /(len(Nx_line)*len(Nc_line))
        count_print = 0
        count = 1
        prog = np.zeros(len(Nx_line)*len(Nc_line)+1)
        for m in range(0,len(Nx_line)):
            for n in range(0,len(Nc_line)):
                if fit_fun=='fun1':
                    Int_file[m+1,n+1] = np.log( integrate.quad(lambda sigma: np.exp(Nc_line[n]*(sigma**2 + Nx_line[m]**2)-(sigma**2+Nx_line[m]**2)**3 ), 0, np.inf, epsabs=1.49e-012, epsrel=1.49e-012)[0])
                elif fit_fun=='fun2':
                    Int_file[m+1,n+1] = np.log( integrate.quad(lambda sigma: np.exp(Nc_line[n]*(sigma**2 + Nx_line[m]**2)-(sigma**2+Nx_line[m]**2)**4 ), 0, np.inf, epsabs=1.49e-012, epsrel=1.49e-012)[0])
                elif fit_fun=='fun3':
                    Int_file[m+1,n+1] = np.log( integrate.quad(lambda sigma: np.exp(Nc_line[n]*(sigma**2 + Nx_line[m]**2)-(sigma**2+Nx_line[m]**2)**5 ), 0, np.inf, epsabs=1.49e-012, epsrel=1.49e-012)[0])
                elif fit_fun=='fun4':
                    Int_file[m+1,n+1] = np.log( integrate.quad(lambda sigma: np.exp(Nc_line[n]*(sigma**2 + Nx_line[m]**2)-(sigma**2+Nx_line[m]**2)**6 ), 0, np.inf, limit=500, epsabs=1.49e-014, epsrel=1.49e-014)[0])
                    #test[m+1,n+1] = integrate.quad(lambda sigma: np.exp(Nc_line[n]*(sigma**2 + Nx_line[m]**2)-(sigma**2+Nx_line[m]**2)**6 ), 0, np.inf, limit=500, epsabs=1.49e-014, epsrel=1.49e-014)[0]
                else:
                    sys.exit("Selected fitting function is not defined. Please choose one of the available functions for fit_fun.")
                
                prog[count] = count/(len(Nx_line)*len(Nc_line))
                
                if count/(len(Nx_line)*len(Nc_line)) == prog_print[count_print]:
                    print('{0}{1}'.format(str(count_print*10), '% of FLiPPID table were calculated.'))
                    count_print += 1
                    
                count += 1
                              
        np.savetxt(filename_FLiPPID, Int_file,delimiter=',')
                 
    x_bound = int(Int_file[-1,0])
    x_data = np.linspace(-int(np.round(len(ImRed[0])/2)),int(np.round(len(ImRed[0])/2)),num=int(len(ImRed[0])))
    
    log_Int = Int_file[1:len(Int_file),1:len(Int_file[0])]
    
    # Define function to read integral table
    def fn_logInt(i,j):
        return log_Int[i,j+int(abs(Int_file[0,1]))]
    
    # Define function to do interpolation between grid points on integral table. 
    def I_int(x,b,c):
        u = x/b
        if x < 0:
            u = -x / b
        
        i = int(np.floor(u /delta_x))
        
        j = int(np.floor(c / delta_c))
        
        xi = u/delta_x - i
    
        chi = c/delta_c - j
    
        if i > x_bound or i < -x_bound: 
            return -100
        
        if j < Int_file[0,1]:
            return -100
    
        if j > Int_file[0,-2]:
            return -100
    
        if u-int(u)==0.0 and c-int(c)==0.0:
            return fn_logInt(i,j)
        
        elif i==x_bound:
            return (1-xi)*(1-chi)*fn_logInt(i,j) + (1-xi)*(chi)*fn_logInt(i,j+1)
    
        elif j==Int_file[0,-1]:
            return (1-xi)*(1-chi)*fn_logInt(i,j) + (xi)*(1-chi)*fn_logInt(i+1,j)
        
        else:
            return (1-xi)*(1-chi)*fn_logInt(i,j) + (1-xi)*(chi)*fn_logInt(i,j+1) + (xi)*(1-chi)*fn_logInt(i+1,j) + xi*chi*fn_logInt(i+1,j+1) 
    
    # Define function to calculate P
    def P_eval(x,a,b,c):
        return (2*a / (np.pi**0.5)) * np.exp(I_int(x,b,c)) 
    # Define function to calculate R
    def R_eval(x,a,b,c):
        if fit_fun=='fun1':
            return ( (a / (b * np.pi**0.5)) * np.exp(c * (x/b)**2 - (x/b)**6 ) )
        elif fit_fun=='fun2':
            return ( (a / (b * np.pi**0.5)) * np.exp(c * (x/b)**2 - (x/b)**8 ) )
        elif fit_fun=='fun3':
            return ( (a / (b * np.pi**0.5)) * np.exp(c * (x/b)**2 - (x/b)**10 ) )
        elif fit_fun=='fun4':
            return ( (a / (b * np.pi**0.5)) * np.exp(c * (x/b)**2 - (x/b)**12 ) )
        else:
            sys.exit("Selected fitting function is not defined. Please choose one of the available functions for fit_fun.")
    
    # Define objective  function for the minimisation
    def RMSE(a,b):
        error = 0
        for i in range(len(a)):
            error += (a[i] - b[i])**2
        error = error / len(a)
        return error
    
    P_red = np.zeros( (len(ImGrn), len(x_data)) )
    R_red = np.zeros( (len(ImGrn), len(x_data)) )
    P_grn = np.zeros( (len(ImGrn), len(x_data)) )
    R_grn = np.zeros( (len(ImGrn), len(x_data)) )
    P_blu = np.zeros( (len(ImGrn), len(x_data)) )
    R_blu = np.zeros( (len(ImGrn), len(x_data)) )
    
    P_tmp = np.zeros( ( (len(ImGrn), len(x_data), 3) ) )
    R_tmp = np.zeros( ( (len(ImGrn), len(x_data), 3) ) )
    
    #fit_out = np.zeros( ( ( z_range[1]-z_range[0]+1,4,3 ) ) )
    fit_out = np.zeros( ( ( len(ImGrn),4,3 ) ) )
    
    colour = ['red', 'green', 'blue']
    
    # Find z where there is the maximum light intensity at the flame centre line. This profile is usually bell-shaped and easy to fit
    z_mid_all = z_range[0]+[ n for n,p in enumerate(ImGrn[z_range[0]:z_range[1]+1,x_bound]) if p==max(ImGrn[z_range[0]:z_range[1]+1,x_bound]) ][0]
    
    #Loop over the three colour channels downwards starting at the z with max counts
    for c in range(0,3):
        # This proofed to be a good initial guess for the fitting parameters. Other flame profiles might require different initial guesses
        if c==0:
            fit_data=ImRed
            a_trial = a_prev = np.amax(fit_data[z_mid_all,:])/1.047
            b_trial = b_prev = (x_bound - [ n for n,p in enumerate(fit_data[z_mid_all,:]) if p>np.amax(fit_data[z_mid_all,:])/2 ][0])/0.8
            c_trial = c_prev = 0
        
        # Once the red colour channel was fitted, fit the other colour channels using the optimised parameters as initial guess. Note that
        # the profiles of the three colour channels are very similar and just differ in intensity (i.e., parameter "a")
        elif c==1:
            fit_data=ImGrn
            a_trial = a_prev = fit_out[z_mid_all,0,0]/np.amax(ImRed)*np.amax(ImGrn)
            b_trial = b_prev = fit_out[z_mid_all,1,0]
            c_trial = c_prev = fit_out[z_mid_all,2,0]
    
        elif c==2:
            fit_data=ImBlu
            a_trial = a_prev = fit_out[z_mid_all,0,0]/np.amax(ImRed)*np.amax(ImBlu)
            b_trial = b_prev = fit_out[z_mid_all,1,0]
            c_trial = c_prev = fit_out[z_mid_all,2,0]
            
        z_count = 0        
        #Loop over the pixel lines starting at z with the maximum intensity going downwards
        for i in range(z_mid_all,z_range[1]): 
            z_pos = z_mid_all+z_count
            print('Bottom loop', str(z_pos))
    
            P_data = fit_data[z_pos]
            
            # If something went wrong during the previous fit, go back to the initial guess
            if a_prev == 0:
                x0 = [a_trial,b_trial,c_trial]
            else:
                x0 = [a_prev,b_prev,c_prev]
            
            # We will skip fitting for the cases where the signal is not much higher than the background noise
            if np.amax(P_data) > 400:
                
                # Define the function to be optimised
                def RMSE_fit(array):
                    a = array[0]
                    b = array[1]
                    c = array[2]
                    x_set = np.arange(-200,201,1)   
                    P_fit = [P_eval(x,a,b,c) for x in x_set]
                    return RMSE(P_data,P_fit)
                
                # Search for best fit, export the result and calculate R and P. The optimised parameters are used as initial guess for the subsequent fit
                res = minimize(RMSE_fit,x0,method='Nelder-Mead',options={'maxiter': 800, 'xatol': 0.000001, 'fatol': 0.000001})
                fit_out[z_pos,0,c] = res.x[0]
                fit_out[z_pos,1,c] = res.x[1]
                fit_out[z_pos,2,c] = res.x[2]
                fit_out[z_pos,3,c] = res.fun**0.5
                a_prev = res.x[0]
                b_prev = res.x[1]
                c_prev = res.x[2]
                P_tmp[z_pos,:,c] = np.asarray([P_eval(x,res.x[0],res.x[1],res.x[2]) for x in x_data]).transpose()
                R_tmp[z_pos,:,c] = np.asarray([R_eval(x,res.x[0],res.x[1],res.x[2]) for x in x_data]).transpose()
                
                # Show warning if the fir was not successful
                if res.success == False:
                    print('{0}{1}{2}{3}'.format('WARNING: minimisation unsuccessful for ', colour[c], ' colour channel and z=', str(z_pos)))
                       
            z_count += 1
        if c==0:
            print('Bottom part of red colour channel finished')
            
        elif c==1:
            print('Bottom part of green colour channel finished')
            
        elif c==2:
            print('Bottom part of blue colour channel finished')
            
    # Loop over the three colour channels upwards starting at the z with max counts. The other steps are identical to the above.
    for c in range(0,3):
        if c==0:
            fit_data=ImRed
            #If z_mid_all is the bottom z line, the above loop was not executed. Therefore, fitting parameters from the above loop cannot be used as an 
            #initial guess. Also, no fitting would be conducted for z=z_mid_all (see range of above loop). Adjusting z_count solves this issue.
            if z_mid_all==z_range[1]:
                z_count = 0
                a_trial = a_prev = np.amax(fit_data[z_mid_all,:])/1.047
                b_trial = b_prev = (x_bound - [ n for n,p in enumerate(fit_data[z_mid_all,:]) if p>np.amax(fit_data[z_mid_all,:])/2 ][0])/0.8
                c_trial = c_prev = 0
                
            else:
                z_count = 1
                a_trial = a_prev = fit_out[z_mid_all,0,0]
                b_trial = b_prev = fit_out[z_mid_all,1,0]
                c_trial = c_prev = fit_out[z_mid_all,2,0]
            
        elif c==1:
            fit_data=ImGrn
            if z_mid_all==z_range[1]:
                z_count = 0
                a_trial = a_prev = fit_out[z_mid_all,0,0]/np.amax(ImRed)*np.amax(ImGrn)
                b_trial = b_prev = fit_out[z_mid_all,1,0]
                c_trial = c_prev = fit_out[z_mid_all,2,0]
            else:
                z_count = 1
                a_trial = a_prev = fit_out[z_mid_all,0,1]
                b_trial = b_prev = fit_out[z_mid_all,1,1]
                c_trial = c_prev = fit_out[z_mid_all,2,1]
    
        elif c==2:
            fit_data=ImBlu
            if z_mid_all==z_range[1]:
                z_count = 0
                a_trial = a_prev = fit_out[z_mid_all,0,0]/np.amax(ImRed)*np.amax(ImBlu)
                b_trial = b_prev = fit_out[z_mid_all,1,0]
                c_trial = c_prev = fit_out[z_mid_all,2,0]
            else:
                z_count = 1
                a_trial = a_prev = fit_out[z_mid_all,0,2]
                b_trial = b_prev = fit_out[z_mid_all,1,2]
                c_trial = c_prev = fit_out[z_mid_all,2,2]
            
                   
        for i in range(z_mid_all-1,z_range[0]-1,-1): 
            z_pos = z_mid_all-z_count
            print('Top loop', str(z_pos))
    
            P_data = fit_data[z_pos]
             
            if a_prev == 0:
                x0 = [a_trial,b_trial,c_trial]
            else:
                x0 = [a_prev,b_prev,c_prev]
    
            if np.amax(P_data) > 400:
                def RMSE_fit(array):
                    a = array[0]
                    b = array[1]
                    c = array[2]
                    x_set = np.arange(-200,201,1)   
                    P_fit = [P_eval(x,a,b,c) for x in x_set]
                    return RMSE(P_data,P_fit)
           
                res = minimize(RMSE_fit,x0,method='Nelder-Mead',options={'maxiter': 800})
                fit_out[z_pos,0,c] = res.x[0]
                fit_out[z_pos,1,c] = res.x[1]
                fit_out[z_pos,2,c] = res.x[2]
                fit_out[z_pos,3,c] = res.fun**0.5
                a_prev = res.x[0]
                b_prev = res.x[1]
                c_prev = res.x[2]
                P_tmp[z_pos,:,c] = np.asarray([P_eval(x,res.x[0],res.x[1],res.x[2]) for x in x_data]).transpose()
                R_tmp[z_pos,:,c] = np.asarray([R_eval(x,res.x[0],res.x[1],res.x[2]) for x in x_data]).transpose()
                if res.success == False:
                    print('{0}{1}{2}{3}'.format('WARNING: minimisation unsuccessful for ', colour[c], ' colour channel and z=', str(z_pos)))
                       
            z_count += 1
            
        if c==0:
    
            print('Top part of red colour channel finished')
            
        elif c==1:
    
            print('Top part of green colour channel finished')
            
        elif c==2:
    
            print('Top part of blue colour channel finished')
            
    P_red = P_tmp[:,:,0]
    R_red = R_tmp[:,:,0]
    
    P_grn = P_tmp[:,:,1]
    R_grn = R_tmp[:,:,1]
    
    P_blu = P_tmp[:,:,2]
    R_blu = R_tmp[:,:,2]
    
    p1 = int(round(z_range[0]+(z_range[1]-z_range[0])/2))
    p2 = int(round(z_range[0]+(z_range[1]-z_range[0])/3))
    plt.figure()
    plt.plot(P_grn[p1,:], 'r')
    plt.plot(ImGrn[p1,:], 'xr')
    plt.plot(P_grn[p2,:], 'g')
    plt.plot(ImGrn[p2,:], 'xg')
    plt.title('Example comparison between raw data (x) and FLiPPID (-)')
    plt.show()
        
    plt.figure()
    plt.plot(R_grn[p1,:], 'r')
    plt.plot(R_grn[p2,:], 'g') 
    plt.title('Example for the reconstruction of the above profiles')
    plt.show()
    
    return (R_red, R_grn, R_blu, P_red, P_grn, P_blu, fit_out)
