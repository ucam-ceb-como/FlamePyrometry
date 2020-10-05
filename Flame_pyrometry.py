# -*- coding: utf-8 -*-
"""
This python code was developed to calculate soot temperatures and volume fractions of co-flow diffusion flames using colour ratio pyrometry. 
Details of the code can be found in the CoMo c4e preprint 217 (https://como.cheng.cam.ac.uk/index.php?Page=Preprints&No=217) and in the submitted 
manuscript Dreyer et al., Applied Optics, 2019. Please cite these sources when using this code.

The code uses a newly developed Abel inversion method for reconstructing flame cross-section intensities from their 2D projections recorded
by a camera. This method, called fitting the line-of-sight projection of a predefined intensity distribution (FLiPPID), was developed by Radomir Slavchov
and was implemented into Python by Angiras Menon. The rest of the code was written by Jochen Dreyer.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time

# General parameters
# Process new image or load old one? Available are 'Load' and '2-colour'.
#If new image is processed check parameters in corresponding case structure
load_image = 'Load'

# Which method should be used for the inverse Abel transform? Available are 
# BASEX, Onion_Peeling_Tikhonov, and FLiPPID. Check parameters in corresponding case structure. 
# Note that FLiPPID will take around 35-60 min for the provided example image.
Abel = 'BASEX'

# Get flame temperatures by comparing colour ratios to lookup table? Note that a temperature lookup table is 
# required. If this table is not found, the program calculates the table using a calibration curve and camera response data.
get_temperature = True

# Get soot volume fraction (only possible if get_temperature = True)?
get_volume_fraction= True

# Save the soot temperature, volume fraction, and, if FLiPPID was used, the fitting parameters?
save = False

# Filter used during the experiment and camera calibration. Options are 'nofilter', 'FGB7', and 'FGB39'. Others can be added in 
# the function Temperature_lookup.
filter = 'FGB7'
    
#------------------------------------------------------------------------------
# Select flame image files and parameters for image processing
#------------------------------------------------------------------------------


if load_image == 'Load':
    ImPathRed = 'Photos//Processed//100H_BG7_exp1563_Red_Ave5.csv'
    ImPathGrn = ImPathRed.replace('Red','Grn')
    ImPathBlu = ImPathRed.replace('Red','Blu')
    
    ImRed = np.genfromtxt(ImPathRed, delimiter=',')
    ImGrn = np.genfromtxt(ImPathGrn, delimiter=',')
    ImBlu = np.genfromtxt(ImPathBlu, delimiter=',')

            
elif load_image == '2-colour':
    #Set filename, how many frames to be processed (the example image contains 5 frames), and if the cropped raw image is plotted.
    #filename = '100H_BG7_exp1563'
    filename = '100H_BG7_exp1563'
    average = 5
    plot_raw = True
    
    # Parameters defining position of flame and cropped image size. Please use an uneven integer for flame_width.
    flame_height = 1500
    flame_width = 401
    HAB0 = 1614 #'100H_BG7_exp1563'
    
    """
    This function loads a tif image in raw format and crops it to the desired size. The file has to be located in the folder "Photos" 
    and its name has to specified in 'filename'. HAB0 stands for height above burner equals zero and is the vertical pixel position of the 
    burner exit. The cropping width and height are 'flame_height' and 'flame_width'. The flame centre line is detected automatically. 
    It is also possible to provide a stacked tif file with multiple frames. In this case, the code searches for the flame centre lines and flame 
    tips in each frame and averages the non-tilted ones with similar flame height.
    
    Note that some parameters in Get_flame might need adjustment if flames with different height or max. intensities are processed. 
    """    
    from Get_flame import Get_flame
    [ImRed, ImGrn, ImBlu, ImDevRed, ImDevGrn, ImDevBlu] = Get_flame(filename, average, flame_height, HAB0, flame_width, plot_raw)
else: 
    sys.exit("Selected image_load option does not exist. Programme stopped")
    
#------------------------------------------------------------------------------
# Inverse Abel transform of image
#------------------------------------------------------------------------------
"""
These functions perform the inverse Abel transform of the recorded image. Equations and descriptions of the BASEX, onion peeling, and
Tikhonov regularisation methods can be found in the following publications:
    
Apostolopoulos et al., Optics Communications 296 (2013) 25–34
Dribinski et al., Review of Scientific Instruments 73 (2002) 2634-2642
Dasch, Applied Optics 31 (1992) 1146-1152
Daun et al., Applied Optics 45 (2006) 4638-4646

The FLiPPID method is described in the CoMo c4e preprint 217 (https://como.cheng.cam.ac.uk/index.php?Page=Preprints&No=217) and 
in the submitted manuscript Dreyer et al., Applied Optics, 2019.
"""
 
# Set image background to 0 if below 200 counts.
ImRed[ImRed < 200] = 0
ImGrn[ImGrn < 200] = 0
ImBlu[ImBlu < 200] = 0

# Set bottom 20 pixel rows to 0 to remove reflections from burner exit
ImRed[len(ImRed)-20:len(ImRed),:] = 0
ImGrn[len(ImGrn)-20:len(ImGrn),:] = 0
ImBlu[len(ImBlu)-20:len(ImBlu),:] = 0

start = time.time()
if Abel == 'BASEX':
    sigma = 8
    qx = 2.2
    from BASEX import BASEX
    [R_red, R_grn, R_blu, P_red, P_grn, P_blu] = BASEX(ImRed, ImGrn, ImBlu, sigma, qx)
    
elif Abel == 'Onion_Peeling_Tikhonov':
    from Onion_Peeling_Tikhonov import Onion_Peeling_Tikhonov
    alpha = 64
    [R_red, R_grn, R_blu, P_red, P_grn, P_blu] = Onion_Peeling_Tikhonov(ImRed, ImGrn, ImBlu, alpha)
    
elif Abel == 'FLiPPID':
    from FLiPPID import FLiPPID
    # For which z values should FLiPPID be executed? Allowed is a range (z_min, z_max) or string 'all'
    z_range = 'all'
    
    # Select function for R to fit to the recorded data. Available are:
    # fun1: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^6)
    # fun2: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^8)
    # fun3: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^10)
    # fun4: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^12)
    fit_fun = 'fun1'
    
    # Define range of integral lookup table. Nx is the range for x/b*delta_c, Nc is the range for c*delta_c.
    if fit_fun == 'fun1':
        Nx = 200
        Nc = (-500, 2500)
        delta_x = 0.01
        delta_c = 0.01
    elif fit_fun == 'fun2':
        Nx = 200
        Nc = (-500, 2500)
        delta_x = 0.01
        delta_c = 0.01
    elif fit_fun == 'fun3':
        Nx = 170
        Nc = (-300, 2500)
        delta_x = 0.01
        delta_c = 0.01
    elif fit_fun == 'fun4':
        Nx = 170
        Nc = (-300, 2200)
        delta_x = 0.01
        delta_c = 0.01
        
        
    if z_range == 'all':
        del z_range
        z_range=(0,len(ImRed))
        
    [R_red, R_grn, R_blu, P_red, P_grn, P_blu, fit_out] = FLiPPID(ImRed, ImGrn, ImBlu, z_range, Nx, Nc, delta_x, delta_c, fit_fun) 
    
    # Find maximum root mean square error.
    rmse_red_max = [ n for n,p in enumerate(fit_out[:,3,0]) if p==max(fit_out[:,3,0]) ][0]
    rmse_grn_max = [ n for n,p in enumerate(fit_out[:,3,1]) if p==max(fit_out[:,3,1]) ][0]
    rmse_blu_max = [ n for n,p in enumerate(fit_out[:,3,2]) if p==max(fit_out[:,3,2]) ][0]
    
    rmse_max_red = float("{0:.2f}".format(fit_out[rmse_red_max,3,0])) 
    rmse_max_grn = float("{0:.2f}".format(fit_out[rmse_grn_max,3,1])) 
    rmse_max_blu = float("{0:.2f}".format(fit_out[rmse_blu_max,3,2])) 
    
    print('{0}{1}{2}{3}{4}{5}'.format('Max. standard deviation of the residuals, red=', str(rmse_max_red), ', green=', str(rmse_max_grn), ', blue=', str(rmse_max_blu)))

    plt.figure()
    plt.plot(ImRed[z_range[0]+rmse_red_max,:], 'r')
    plt.plot(ImGrn[z_range[0]+rmse_grn_max,:], 'g')
    plt.plot(ImBlu[z_range[0]+rmse_blu_max,:], 'b')
    plt.plot(P_red[z_range[0]+rmse_red_max,:], '-r')
    plt.plot(P_grn[z_range[0]+rmse_grn_max,:], '-g')
    plt.plot(P_blu[z_range[0]+rmse_blu_max,:], '-b')
    plt.title('The worst FLiPPID fits for the red, green, and blue channel')
    plt.show() 

else: 
    sys.exit("Selected inverse Abel transform method does not exist. Programme stopped")

end = time.time()
    
#------------------------------------------------------------------------------
# Calculate colour ratios
#------------------------------------------------------------------------------
# Define threshold above which colour ratio will be calculated
threshold_ratio_red = 1.4
threshold_ratio_grn = 1.4
threshold_ratio_blu = 1.4

GoodRG = (R_grn > threshold_ratio_grn) & (R_red > threshold_ratio_red)
GoodRB = (R_blu > threshold_ratio_blu) & (R_red > threshold_ratio_red)
GoodBG = (R_blu > threshold_ratio_blu) & (R_grn > threshold_ratio_grn)

R_RG = np.zeros((len(R_red),len(R_red[0])))
R_RB = np.zeros((len(R_red),len(R_red[0])))
R_BG = np.zeros((len(R_red),len(R_red[0])))

R_RG[GoodRG] = np.array(R_red[GoodRG] / R_grn[GoodRG])
R_RB[GoodRB] = np.array(R_red[GoodRB] / R_blu[GoodRB])
R_BG[GoodBG] = np.array(R_blu[GoodBG] / R_grn[GoodBG])

plt.figure()
plt.subplot(131)
plt.imshow(R_RG, vmin=np.mean(R_RG[GoodRG])-0.05, vmax=np.mean(R_RG[GoodRG])+0.05)
plt.title('Red/green colour ratio')
plt.subplot(132)
plt.imshow(R_RB, vmin=np.mean(R_RB[GoodBG])-0.05, vmax=np.mean(R_RB[GoodBG])+0.15)
plt.title('Red/blue colour ratio')
plt.subplot(133)
plt.imshow(R_BG, vmin=np.mean(R_BG[GoodBG])-0.1, vmax=np.mean(R_BG[GoodBG])+0.05)
plt.title('Blue/green colour ratio')
plt.show()

#------------------------------------------------------------------------------
# Get the temperature lookup table and calculate the soot temperature
#------------------------------------------------------------------------------
if get_temperature == True:
    # Define wavelength range in nm
    wavelength = [300, 700]
    lam = np.linspace(wavelength[0], wavelength[1], (wavelength[1]-wavelength[0])*10+1)
    # Define temperature range in K
    temperature = [1000, 3000]
    T_calc = np.linspace(temperature[0], temperature[1], (temperature[1]-temperature[0])+1)
    # Fit camera response to measured values of colour ratios?
    fit = True    
    # What soot dispersion exponent (alpha) is used for soot is used? Possible are 'Chang' with lambda^-1.423 as derived from the refractive index 
    # measurements by:
    # Chang and Charalampopoulos, Proc. R. Soc. Lond. A 430 (1990) 577-591
    # or 'Kuhn' for lambda^-1.38 as reported by Kuhn et al. for the same raw data:
    # Kuhn et al., Proceedings of the Combustion Institute 33 (2011) 743-750
    soot_coef = 'Chang'
    
    # Provide location of the colour ratio calibration file. Column 0 has to contain the measure temperature in K and
    # column 1, 2, 3 the corresponding RG, RB, and BG colour ratios. The other columns in the example file are not
    # required but show the exposure times and counts of the red, green, and blue channel. Dividing column 6 (green counts)
    # by column 4 (exposure time) and plotting the result over column 0 (temperature) leads to the power law function
    # as required for the soot volume fractions (see below).
    if filter == 'FGB7':
        Thermo_calib = pd.read_csv('Temperature_tables/Measured_R_Thermo_FGB7.csv', delimiter=',', header=None).values
    elif filter == 'FGB39':
        Thermo_calib = pd.read_csv('Temperature_tables/Measured_S_Thermo_FGB39.csv', delimiter=',', header=None).values
    else:
        sys.exit("No calibration data for the selected camera filter found. Program terminated.")
        
    """
    This function calculates the temperature lookup table, i.e., a table of the colour ratios expected to be recorded by the camera 
    as a function of the soot temperature. The function uses the theoretical camera, lens, and filter response to calculate colour 
    ratios of a hot thermocouple and compares it to experimentally measured values. The blue and red colour channels are scaled to 
    match the theoretical and observed colour ratios after which the ratios of hot soot are calculated. Further details can be found
    in the following publications:
    Ma and Long, Proceedings of the Combustion Institute 34 (2013) 3531-3539
    Kuhn et al., Proceedings of the Combustion Institute 33 (2011) 743-750
    """     
    from Temperature_lookup import Temperature_lookup
    Ratio_tables = Temperature_lookup(filter, lam, T_calc, fit, Thermo_calib, soot_coef)
    
    """
    This function uses the temperature lookup table to calculate the soot temperature profiles of the flame.  
    """     
    from Get_flame_temperature import Get_flame_temperature
    [T_RG, T_RB, T_BG] = Get_flame_temperature(Ratio_tables, R_RB, R_RG, R_BG)
    
    # If one of the three light intensities is below pre-defined threshold, the temperature is set to zero.
    Good_T = (R_grn > threshold_ratio_grn) & (R_red > threshold_ratio_red) & (R_blu > threshold_ratio_blu)
    
    # The average temperature obtained from the three colour ratios is calculated. 
    T_ave = (T_RG + T_RB + T_BG) / 3 * Good_T 
    T_RG = T_RG * GoodRG
    T_RB = T_RB * GoodRB
    T_BG = T_BG * GoodBG
    
    plt.figure()
    plt.subplot(141)
    plt.imshow(T_RG, vmin=1500, vmax=2000)
    plt.title('Temperature from red/green colour ratio')
    plt.subplot(142)
    plt.imshow(T_RB, vmin=1500, vmax=2000)
    plt.title('Temperature from red/blue colour ratio')
    plt.subplot(143)
    plt.imshow(T_BG, vmin=1500, vmax=2000)
    plt.title('Temperature from blue/green colour ratio')
    plt.subplot(144)
    plt.imshow(T_ave, vmin=1500, vmax=2000)
    plt.title('Average temperature')
    plt.show()

#------------------------------------------------------------------------------
# Get soot volume fraction
#------------------------------------------------------------------------------
"""
This function calculates the soot volume fraction using the previously calculated soot temperatures, the recorded light intensity, 
the exposure time used while imaging the flame, and the camera calibration. Further details can be found in the following 
publications:
Ma and Long, Proceedings of the Combustion Institute 34 (2013) 3531-3539
Kuhn et al., Proceedings of the Combustion Institute 33 (2011) 743-750  
"""    
if get_volume_fraction == True and get_temperature == True:
    
    # mm for each pixel and exposure time while taking the flame images
    pixelmm = 1/34

    exposure = 1563 #'100H_BG7_exp1563'
    
    # Measured green singal emitted from a hot thermocouple devided by the expsoure time as a function of 
    # temperature. The equation is obtained by plotting the green counts / exposure time over temperature and 
    # fitting a power law function to it.
    if filter=='FGB7':
        def S_thermo(T):
            return 1.41928E-47*T**(1.50268E+01)
    elif filter=='FGB39':
        def S_thermo(T):
            return 3.53219E-54*T**(1.72570E+01)
    else:
        sys.exit("No calibration data for the selected camera filter found. Program terminated.")

    from Get_soot_volume import Get_soot_volume
    f_ave = Get_soot_volume(filter, lam, pixelmm, exposure, T_ave, R_red, R_grn, R_blu, Ratio_tables, S_thermo)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(T_ave, vmin=1500, vmax=2000)
    plt.title('Average soot temperature [K]')
    plt.subplot(122)
    plt.imshow(f_ave, vmin=0.05, vmax=2.5)
    plt.title('soot volume fraction [ppm]')
    plt.show()
    
elif get_volume_fraction == True & get_temperature == False:
    sys.exit("Soot volume fraction can only be calculated after the soot temperature is known. Set get_temperature to True and re-run the programme.")

if save==True:
    ImPathT = ImPathRed.replace('Red','T-ave')
    ImPathfv = ImPathRed.replace('Red','fv')
    
    np.savetxt(ImPathT, T_ave ,delimiter=',')
    np.savetxt(ImPathfv, f_ave ,delimiter=',')
    
    if 'fit_out' in locals():
        ImPathfit = ImPathRed.replace('Red','fit_out')
        fit_save = np.concatenate((fit_out[:,:,0], fit_out[:,:,1], fit_out[:,:,2]), axis=1)
        np.savetxt(ImPathfit, fit_save ,delimiter=',')

    
Abel_time = float("{0:.2f}".format(end-start))    
print('{0}{1}{2}'.format('Abel inversion required', str(Abel_time), 's'))