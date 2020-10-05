# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:24:52 2018

@author: jd766
"""

def Get_flame(filename, average, flame_height, HAB0, flame_width, plot_raw):

    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import io
    from os.path import abspath, exists
    
    from colour_demosaicing import (
        EXAMPLES_RESOURCES_DIRECTORY,
        demosaicing_CFA_Bayer_bilinear,
        demosaicing_CFA_Bayer_Malvar2004,
        demosaicing_CFA_Bayer_Menon2007,
        mosaicing_CFA_Bayer)
    
    """
    If the automatic flame centre and edge detection fails check the below values, i.e., the thresholds for edge and tip detection, 
    the selection criteria for straight flames with similar height, and the number and positions where flame centres are searched
    """
    
    # Number of lines where the centre is detected
    centres=63
    
    # First z to check for centres
    z1 = 614
    
    # Spacing in z where centre is searched
    z_check = 10
    
    # Threshold for the edge and tip detection
    thresh_red = thresh_grn = thresh_blu = 2000
    
    # Threshold for selecting non-tilted flames with similar height
    thresh_tip = 4 
    thresh_std = 0.8
    
    
    scale = 65535
    
    # Read raw image in Bayer formate from
    fname = abspath( '{0}{1}{2}'.format('Photos//', filename, '.tif') )
    ImDatBayer_all=io.imread(fname)
    
    ImDatBayer = np.zeros((average, len(ImDatBayer_all[0]), len(ImDatBayer_all[0][0])))
    ImDat = np.zeros(((len(ImDatBayer[0]), len(ImDatBayer[0][0]), average, 3) ) ) 
            
    ImDatBayer = ImDatBayer_all[0:average,:,:]
    del ImDatBayer_all
    
    print('Finished loading image')
    
    for i in range(0, average):
        ImDat[:,:,i,:] = (demosaicing_CFA_Bayer_bilinear(ImDatBayer[i,:,:], 'GRBG'))
    
    ImDatRed = ImDat[:,:,:,2]
    ImDatGrn = ImDat[:,:,:,1] 
    ImDatBlu = ImDat[:,:,:,0]  
    del ImDat      
    print('Finished demosaicing image')
    
    row = np.zeros(centres, dtype=int)
    middle_std = np.zeros(len(ImDatGrn[0][0]))
    left = np.zeros([len(ImDatGrn[0][0]),centres], dtype=int)
    right = np.zeros([len(ImDatGrn[0][0]),centres], dtype=int)
    middle = np.zeros([len(ImDatGrn[0][0]),centres], dtype=int)
    
    # Find left and right flame edge to calculate the flame centre at different pixel rows
    for j in range(0,len(ImDatGrn[0][0])):
        for i in range(0,centres):
            row[i] = HAB0 - z1 - (i+1)*z_check
            left[j,i] = [ n for n,p in enumerate(ImDatGrn[row[i],:,j]) if p>thresh_grn ][0]
            right[j,i] = len(ImDatGrn[0]) - [ n for n,i in enumerate(np.flip(ImDatGrn[row[i],:,j], 0)) if i>thresh_grn ][0]
            middle[j,i] = left[j,i]+(right[j,i]-left[j,i])/2
            middle_std[j] = np.std(middle[j,:])
    
    tip = np.zeros(len(ImDatGrn[0][0]), dtype=int)
    
    # Find the flame tip at the flame centre
    for j in range(0,len(ImDatGrn[0][0])):
        tip[j] = [ n for n,p in enumerate(ImDatGrn[:,int(round(np.mean(middle[j,:]))),j]) if p>thresh_grn ][0]     
    
    print('Flame edge and tip detection finished')
    
    ave_tip = np.mean(tip)
    G_filter = 0
    ImDatRed_filter = np.array(np.zeros((len(ImDatRed),len(ImDatRed[0]),1)))
    ImDatGrn_filter = np.zeros((len(ImDatRed),len(ImDatRed[0]),1))
    ImDatBlu_filter = np.zeros((len(ImDatRed),len(ImDatRed[0]),1))
    middle_filter = np.zeros((1,len(middle[0])))
    j_good = np.zeros((len(middle[0])))
    
    # The non-tilted flames with similar height are selected.
    for j in range(0,len(ImDatGrn[0][0])):
        if (middle_std[j] < thresh_std) and (tip[j]+thresh_tip > ave_tip) and (tip[j]-thresh_tip < ave_tip):
            if G_filter == 0:
                ImDatRed_filter[:,:,G_filter] = ImDatRed[:,:,j]
                ImDatGrn_filter[:,:,G_filter] = ImDatGrn[:,:,j]
                ImDatBlu_filter[:,:,G_filter] = ImDatBlu[:,:,j]
                middle_filter[G_filter,:] = middle[j,:]
                
            else:    
                ImDatRed_filter = np.insert(ImDatRed_filter, 1, ImDatRed[:,:,j], axis=2)
                ImDatGrn_filter = np.insert(ImDatGrn_filter, 1, ImDatGrn[:,:,j], axis=2)
                ImDatBlu_filter = np.insert(ImDatBlu_filter, 1, ImDatBlu[:,:,j], axis=2)
                middle_filter = np.insert(middle_filter, 1, middle[j,:], axis=0)
                
            G_filter = G_filter+1
    
    # Calculate single and averaged flame images.
    ImRed_single = ImDatRed_filter[:,:,0]
    ImGrn_single = ImDatGrn_filter[:,:,0]
    ImBlu_single = ImDatBlu_filter[:,:,0]
    
    ImRed_filter = np.mean(ImDatRed_filter, axis=2)
    ImGrn_filter = np.mean(ImDatGrn_filter, axis=2)
    ImBlu_filter = np.mean(ImDatBlu_filter, axis=2)
    
    ImDevRed_filter = np.std(ImDatRed_filter, axis=2)
    ImDevGrn_filter = np.std(ImDatGrn_filter, axis=2)
    ImDevBlu_filter = np.std(ImDatBlu_filter, axis=2)
     
    if len(ImDatGrn[0][0]) > 1:
        plt.figure()
        plt.title('Standard deviation of detected centres in each frame and the deviation threshold')
        plt.plot(middle_std, 'r')
        plt.plot(np.ones(len(ImDatGrn[0][0]))*thresh_std, 'b')
        plt.show()
        
        plt.figure()
        plt.title('Flame tip in each frame and range of tip threshold')
        plt.plot(tip, 'xr')
        plt.plot(np.ones(len(ImDatGrn[0][0]))*(ave_tip+thresh_tip),'g--')
        plt.plot(np.ones(len(ImDatGrn[0][0]))*(ave_tip-thresh_tip),'g--')
        plt.show()
    
    middle_crop = int(round(np.mean(np.mean(middle_filter))))
    
    # Crop the flame images to the desired size.
    ImRed_single_crop = ImRed_single[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    ImGrn_single_crop = ImGrn_single[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    ImBlu_single_crop = ImBlu_single[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    
    Cropped_ImRed = ImRed_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    Cropped_ImGrn = ImGrn_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    Cropped_ImBlu = ImBlu_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    
    Cropped_ImDevRed = ImDevRed_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    Cropped_ImDevGrn = ImDevGrn_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    Cropped_ImDevBlu = ImDevBlu_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
     
    ImRed = Cropped_ImRed
    ImGrn = Cropped_ImGrn
    ImBlu = Cropped_ImBlu
    
    ImDevRed = Cropped_ImDevRed
    ImDevGrn = Cropped_ImDevGrn
    ImDevBlu = Cropped_ImDevBlu
    
    middle_line = np.zeros(2)
    row_plot = row-(HAB0-flame_height)
    left_plot = np.mean(left-(middle_crop-np.floor(flame_width/2)), axis=0)
    right_plot = np.mean(right-(middle_crop-np.floor(flame_width/2)), axis=0);
    middle_plot = np.mean(middle_filter-(middle_crop-np.floor(flame_width/2)), axis=0);
    middle_line[0] = middle_crop-(middle_crop-(flame_width/2))
    middle_line[1] = middle_line[0]
    
    # Save the processed image.
    fsave_Red = abspath( '{0}{1}{2}{3}{4}'.format('Photos//Processed//', filename, '_Red_Ave', str(G_filter), '.csv') )
    fsave_Grn = abspath( '{0}{1}{2}{3}{4}'.format('Photos//Processed//', filename, '_Grn_Ave', str(G_filter), '.csv') )
    fsave_Blu = abspath( '{0}{1}{2}{3}{4}'.format('Photos//Processed//', filename, '_Blu_Ave', str(G_filter), '.csv') )
    
    np.savetxt(fsave_Red, ImRed ,delimiter=',')
    np.savetxt(fsave_Grn, ImGrn ,delimiter=',')
    np.savetxt(fsave_Blu, ImBlu ,delimiter=',')
    
        
    if plot_raw==True:
        plt.figure()
        plt.subplot(131)
        plt.imshow(ImRed, vmin=0, vmax=scale)
        plt.title('Original red channel intensity')
        plt.subplot(132)
        plt.imshow(ImGrn, vmin=0, vmax=scale)
        plt.title('Original green channel intensity and detected centre line')
        plt.plot(left_plot,row_plot,'r')
        plt.plot(right_plot,row_plot,'r')
        plt.plot(middle_plot,row_plot,'xr')
        plt.plot(middle_line,[row_plot[0], row_plot[-1]],'r')
        plt.subplot(133)
        plt.imshow(ImBlu, vmin=0, vmax=scale)
        plt.title('Original blue channel intensity')
        plt.show()
        
        plt.figure()
        plt.plot(Cropped_ImGrn[300,:], 'r')
        plt.plot(Cropped_ImGrn[450,:], 'r')
        plt.plot(Cropped_ImGrn[660,:], 'r')
        plt.plot(ImGrn_single_crop[300,:], 'g')
        plt.plot(ImGrn_single_crop[450,:], 'g')
        plt.plot(ImGrn_single_crop[660,:], 'g')
        plt.title('Comparision single frame to averaged frames')
        plt.show()

    return (ImRed, ImGrn, ImBlu, ImDevRed, ImDevGrn, ImDevBlu)