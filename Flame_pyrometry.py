# -*- coding: utf-8 -*-
"""
This python code was developed to calculate soot temperatures and volume fractions of co-flow diffusion flames using colour ratio pyrometry. 
Details of the code can be found in the CoMo c4e preprint 217 (https://como.ceb.cam.ac.uk/preprints/217/) and in the 
manuscript Dreyer et al., Applied Optics 58 (2019) 2662-2670 (https://doi.org/10.1364/AO.58.002662). Please cite these sources when using this code.

The code uses a newly developed Abel inversion method for reconstructing flame cross-section intensities from their 2D projections recorded
by a camera. This method, called fitting the line-of-sight projection of a predefined intensity distribution (FLiPPID), was developed by Radomir Slavchov
and was implemented into Python by Angiras Menon. The rest of the code was written by Jochen Dreyer.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time

# Most of the required Python packages should be installed by default. Two additional ones used by this code are PyAbel and OpenCV-Python:
# https://pyabel.readthedocs.io/en/latest/index.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/index.html
# Details and installation instructions can be found on the above websites. PyAbel us used to perfor the inverse Abel transform if methods other than 
# FLiPPID are used. OpenCV is a package for Computer Vision and Machine Learning and is used for the demosaicing of the Bayer raw images.

import abel
import cv2

# General parameters
# Process new image or load old one? Available are 'Load' and 'Pre-process'.
#If new image is processed check parameters in corresponding case structure
load_image = 'Pre-process'

# Which method should be used for the inverse Abel transform? Some standard Abel inverse methods are available as implemented in the PyAbel package:
# https://pyabel.readthedocs.io/en/latest/index.html
# These currently are 'BASEX', '3-point', 'Dasch-Onion', and 'Hansen-Law'. It should be noted that in its current version, a regularization of the raw
# data is only possible with BASEX. 

# Also the FLiPPID method as reported in Dreyer et al., Applied Optics, 2019 is available. 
# Note that 'FLiPPID' will take around 35-60 min for the provided example image.
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
    # Load a previously pre-processed flame image. Only the file path for the csv file containg the red colour channel has to be defined.
    ImPathRed = 'Photos//Processed//100H_BG7_exp1563_Red_Ave5.csv'
    ImPathGrn = ImPathRed.replace('Red','Grn')
    ImPathBlu = ImPathRed.replace('Red','Blu')
    
    ImRed = np.genfromtxt(ImPathRed, delimiter=',')
    ImGrn = np.genfromtxt(ImPathGrn, delimiter=',')
    ImBlu = np.genfromtxt(ImPathBlu, delimiter=',')
    
    # Define locations where the soot temperature and volume fractions are saved. Also a path for the FLiPPID optimised fitting parameters is defined.
    ImPathT = ImPathRed.replace('Red','T-ave')
    ImPathfv = ImPathRed.replace('Red','fv')
    ImPathfit = ImPathRed.replace('Red','fit_out')
            
elif load_image == 'Pre-process':
    # Set filename, how many frames to be processed (the example image contains 5 frames), and if the cropped raw image is plotted.
    filename = '100H_BG7_exp1563'
    average = 5
    plot_raw = True
    
    # If save_single=False, only the averaged values of the pre-processed images are saved. If True, also each individual frame is saved.
    save_single = False
    
    # Parameters defining position of flame and cropped image size. Please use an uneven integer for flame_width.
    flame_height = 1500
    flame_width = 401
    HAB0 = 1614 #'100H_BG7_exp1563'
    
    # Threshold for selecting non-tilted flames with similar height. Flames with a tip larger than +-thresh_tip are removed. Flames with a standard 
    # deviation of the centreline larger than thresh_std are removed. Note that an error will occur if non of the frames fulfills these conditions.
    thresh_tip = 4 
    thresh_std = 0.8
    
    # Define locations where the soot temperature and volume fractions are saved. Also a path for the FLiPPID optimised fitting parameters is defined.
    ImPathT = ('{0}{1}{2}'.format('Photos//Processed//', filename, 'T-ave.csv'))
    ImPathfv = ('{0}{1}{2}'.format('Photos//Processed//', filename, 'fv.csv'))
    ImPathfit = ('{0}{1}{2}'.format('Photos//Processed//', filename, 'fit_out.csv'))
    
    """
    This function loads a tif image in raw format and crops it to the desired size. The file has to be located in the folder "Photos" 
    and its name has to specified in 'filename'. HAB0 stands for height above burner equals zero and is the vertical pixel position of the 
    burner exit. The cropping width and height are 'flame_height' and 'flame_width'. The flame centre line is detected automatically. 
    It is also possible to provide a stacked tif file with multiple frames. In this case, the code searches for the flame centre lines and flame 
    tips in each frame and averages the non-tilted ones with similar flame height.
    
    Note that some parameters in Get_flame might need adjustment if flames with different height or max. intensities are processed. 
    """    
    
    from Get_flame import Get_flame
    [ImRed, ImGrn, ImBlu, ImDevRed, ImDevGrn, ImDevBlu] = Get_flame(filename, average, flame_height, HAB0, flame_width, plot_raw, thresh_tip, thresh_std, save_single)
else: 
    sys.exit("Selected image_load option does not exist. Programme stopped")
    
#------------------------------------------------------------------------------
# Inverse Abel transform of image
#------------------------------------------------------------------------------
"""
These functions perform the inverse Abel transform of the recorded image. Equations and descriptions of the BASEX, onion peeling, 3-point, and
Fast Hangel (Hansen-Law) methods can be found in the following publications:
    
Apostolopoulos et al., Optics Communications 296 (2013) 25–34
Dribinski et al., Review of Scientific Instruments 73 (2002) 2634-2642
Dasch, Applied Optics 31 (1992) 1146-1152
Daun et al., Applied Optics 45 (2006) 4638-4646
Hansen and Law, Journal of the Optical Society of America A 2 (1985) 510-520

The FLiPPID method is described in the CoMo c4e preprint 217 (https://como.cheng.cam.ac.uk/index.php?Page=Preprints&No=217) and 
was accepted for publication: Dreyer et al., Applied Optics, 2019.
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

# The first few options use the PyAbel package for the inverse Abel transform. For details see:
# https://pyabel.readthedocs.io/en/latest/index.html
# Just some of the available transforms are used but others can easly be added.

ImRed_half = ((ImRed[:,round(len(ImRed[0])/2):len(ImRed[0])]+np.flip(ImRed[:,0:round(len(ImRed[0])/2)+1], axis=1)) / 2)
ImGrn_half = ((ImGrn[:,round(len(ImGrn[0])/2):len(ImGrn[0])]+np.flip(ImGrn[:,0:round(len(ImGrn[0])/2)+1], axis=1)) / 2)
ImBlu_half = ((ImBlu[:,round(len(ImBlu[0])/2):len(ImBlu[0])]+np.flip(ImBlu[:,0:round(len(ImBlu[0])/2)+1], axis=1)) / 2)

if Abel == 'BASEX':
    # These are the BASEX regularisation parameters. 
    sig = 8
    qx = 2.2

    R_red_half = abel.basex.basex_transform(ImRed_half, sigma=sig, reg=qx, correction=True, basis_dir=u'./BASEX_matrices/', dr=1.0, verbose=True, direction=u'inverse')
    R_grn_half = abel.basex.basex_transform(ImGrn_half, sigma=sig, reg=qx, correction=True, basis_dir=u'./BASEX_matrices/', dr=1.0, verbose=True, direction=u'inverse')
    R_blu_half = abel.basex.basex_transform(ImBlu_half, sigma=sig, reg=qx, correction=True, basis_dir=u'./BASEX_matrices/', dr=1.0, verbose=True, direction=u'inverse')
    
if Abel == '3-point':
    R_red_half = abel.dasch.three_point_transform(ImRed_half, basis_dir=u'./3-point_matrices', dr=1, direction=u'inverse', verbose=False)
    R_grn_half = abel.dasch.three_point_transform(ImGrn_half, basis_dir=u'./3-point_matrices', dr=1, direction=u'inverse', verbose=False)
    R_blu_half = abel.dasch.three_point_transform(ImBlu_half, basis_dir=u'./3-point_matrices', dr=1, direction=u'inverse', verbose=False)
    
elif Abel == 'Dasch-Onion':
    R_red_half = abel.dasch.onion_peeling_transform(ImRed_half, basis_dir=u'./Dasch-Onion_matrices', dr=1, direction=u'inverse', verbose=False)
    R_grn_half = abel.dasch.onion_peeling_transform(ImGrn_half, basis_dir=u'./Dasch-Onion_matrices', dr=1, direction=u'inverse', verbose=False)
    R_blu_half = abel.dasch.onion_peeling_transform(ImBlu_half, basis_dir=u'./Dasch-Onion_matrices', dr=1, direction=u'inverse', verbose=False)
    
elif Abel == 'Hansen-Law':
    R_red_half = abel.hansenlaw.hansenlaw_transform(ImRed_half, dr=1, direction=u'inverse', hold_order=1, sub_pixel_shift=0)
    R_grn_half = abel.hansenlaw.hansenlaw_transform(ImGrn_half, dr=1, direction=u'inverse', hold_order=1, sub_pixel_shift=0)
    R_blu_half = abel.hansenlaw.hansenlaw_transform(ImBlu_half, dr=1, direction=u'inverse', hold_order=1, sub_pixel_shift=0)
    
elif Abel == 'FLiPPID':
    from FLiPPID import FLiPPID
    # For which z values should FLiPPID be executed? Allowed is a range (z_min, z_max) or string 'all'
    z_range = (500,510)# 'all'
    
    # Select function for R to fit to the recorded data. Available are:
    # fun1: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^6)
    # fun2: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^8)
    # fun3: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^10)
    # fun4: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^12)
    fit_fun = 'fun1'
    
    # Define range of integral lookup table. Nx is the range for x/b*delta_c, Nc is the range for c*delta_c. 
    delta_x = 0.01
    delta_c = 0.01
    if fit_fun == 'fun1':
        Nx = 200
        Nc = (-500, 2500)
        
    elif fit_fun == 'fun2':
        Nx = 200
        Nc = (-500, 2500)

    elif fit_fun == 'fun3':
        Nx = 170
        Nc = (-300, 2500)

    elif fit_fun == 'fun4':
        Nx = 170
        Nc = (-300, 2200)
        
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
    plt.plot(ImRed[rmse_red_max,:], 'r')
    plt.plot(ImGrn[rmse_grn_max,:], 'g')
    plt.plot(ImBlu[rmse_blu_max,:], 'b')
    plt.plot(P_red[rmse_red_max,:], '-r')
    plt.plot(P_grn[rmse_grn_max,:], '-g')
    plt.plot(P_blu[rmse_blu_max,:], '-b')
    plt.title('The worst FLiPPID fits for the red, green, and blue channel')
    plt.show() 

# The transforms in PyAbel only process one half of the image. Because the code below uses a full image, two halfs of the image are concentrated.    
if Abel != 'FLiPPID':
    R_red = np.concatenate((np.flip(R_red_half[:,1:len(R_red_half[0])], axis=1), R_red_half), axis=1)
    R_grn = np.concatenate((np.flip(R_grn_half[:,1:len(R_grn_half[0])], axis=1), R_grn_half), axis=1)
    R_blu = np.concatenate((np.flip(R_blu_half[:,1:len(R_blu_half[0])], axis=1), R_blu_half), axis=1)

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

    from Get_soot_volume import Get_soot_volume
    f_ave = Get_soot_volume(filter, lam, pixelmm, exposure, T_ave, R_red, R_grn, R_blu, Ratio_tables, S_thermo)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(T_ave, vmin=1500, vmax=2000)
    plt.title('Average soot temperature [K]')
    plt.subplot(122)
    plt.imshow(f_ave, vmin=0.05, vmax=1.5)
    plt.title('Soot volume fraction [ppm]')
    plt.show()

# If True save the soot temperature and volume fraction. If FLiPPID was used the optimised fitting parameters are also saved.
if save==True:    
    np.savetxt(ImPathT, T_ave ,delimiter=',')
    np.savetxt(ImPathfv, f_ave ,delimiter=',')
    
    if Abel == 'FLiPPID':
        fit_save = np.concatenate((fit_out[:,:,0], fit_out[:,:,1], fit_out[:,:,2]), axis=1)
        np.savetxt(ImPathfit, fit_save ,delimiter=',')

    
Abel_time = float("{0:.2f}".format(end-start))    
print('{0}{1}{2}'.format('Abel inversion required', str(Abel_time), 's'))