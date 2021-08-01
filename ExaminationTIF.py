# -*- coding: utf-8 -*-
"""
A useful tool to examine a series of raw2tif images for the FlamePyrometry code.
For more information on the FlamePyrometry code, please see [https://doi.org/10.1364/AO.58.002662].

The inverse Abel transformation is very sensitive to the symmetry of flame and the uniformity of stacked images.
Thus, this code was developed to examine a series of raw2tif images from BlackflyS camera.
Several flame images with similar flame shape can be selected for further processing in FlamePyrometry code.

Note that several lines of this code are copied and modified from the original FlamePyrometry code.

The raw images from BlackflyS camera can be converted to Tif format images using ImageJ.
The Tif images need to be renamed as "0, 1, 2, 3..." using some kind of renamed tool, which can be found in the Internet.
In addition, these Tif images must be placed in "//Photos//Examination//Input//" folder.

Then, variable "img_number" need to be changed by the number of these images. 
Note that the positions of HAB0 and flame tip, and flame width must be given.

For each image, the standard deviation of the detected centre points (i.e., line), 
and the position of flame tip can be saved in csv format in "//Photos//Examination//".
The demosaicing image of flame, and the R, G, and B channels of this image can also be saved.

The code can also save tecplor format files for all frames that stores the Red, Green and Blue channel of the flame image. 
Note that each tecplot file (*.dat) may be very large.
If you don't want to save the tecplot file, just change the save_tecplot to 'False'.

Created on Sat Jun 26, 2021

@ zhangw106
@ zhangw106@tju.edu.cn
"""

from skimage import io, exposure
import numpy as np
from os.path import abspath
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import gc

# Number of images for examination
img_number = 3

# mm for each pixel of flame images
pixelmm = 1/38.5

# Save tecplot file. 'True' or 'False'
save_tecplot = True

# Parameters defining position of flame and cropped image size. 
# Please use an uneven integer for flame_width.
HAB0 = 3634           # 1614 for '100H_BG7_exp1563'
HAB_tip = 1800        #  114 for '100H_BG7_exp1563'
flame_width = 501     #  401 for '100H_BG7_exp1563'
flame_height = HAB0 - HAB_tip

# It is possible to slightly rotate tilded flame images
degree = 0

# Threshold for the tip detection
thresh_grn  = 2000

# Max. intensity count of the camera
scale = 65535

# Two arrows to save middle_std and tip for each frame
# Another arrow to save both the middle_std and tip
middle_std = np.zeros(img_number) 
tip = np.zeros(img_number)
tif_exam = np.zeros((img_number, 3))

for k in range(0, img_number, 1):
    filename = k
    fname = abspath( '{0}{1}{2}'.format('Photos//Examination//Input//', filename, '.tif') )

    ImDatBayer = io.imread(fname)   # Row*Column = 2048*1536

    ImDat = np.zeros(((len(ImDatBayer), len(ImDatBayer[0]), 3) ) )        # 2048*1536*3

    # Demosaic the image. COLOR_BayerGB2RGB, COLOR_BayerGB2RGB_EA, COLOR_BayerGB2RGB_VNG. 
    ImDat = ndimage.rotate((cv2.cvtColor(ImDatBayer, cv2.COLOR_BayerGB2RGB_EA)), degree, reshape=False)

    # It is possible to adjust the lightness of color image.
    Img_color = exposure.adjust_gamma(ImDat, 1)  #  <1 means brighter, >1 means darker

    del ImDatBayer

    ImDatRed = ImDat[:,:,0]   # 2048*1536
    ImDatGrn = ImDat[:,:,1]   # 2048*1536
    ImDatBlu = ImDat[:,:,2]   # 2048*1536
    
    left = np.zeros([len(ImDatGrn)], dtype=int)   # 2048
    right = np.zeros([len(ImDatGrn)], dtype=int)  # 2048
    middle = np.zeros([len(ImDatGrn)])            # 2048

    # Find left and right flame edge to calculate the flame centre at different pixel rows
    gray = cv2.cvtColor(np.uint8(ImDat/scale*255), cv2.COLOR_RGB2GRAY)         # 2048*1536 
    th_gray = cv2.adaptiveThreshold(gray,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,85,6)                                        # 2048*1536 

    for i in range(0,len(ImDatGrn)):
        left[i] = np.argmax(th_gray[i,:]<0.5)
        right[i] = len(ImDatGrn[0]) - np.argmax(np.flip(th_gray[i,:],0)<0.5)- 1

    # Remove the noisy top and bottom part of the flame    
    left[0:np.argmax(left[:]>0)+20] = 0
    left[len(left) - np.argmax(np.flip(left[:])>0)-50:-1] = 0

    middle = (left + right)/2 * (left!=0) * (right!=len(ImDatGrn[0])-1)
    middle_std[k] = np.std(middle[middle!=0]) 

    # Find the flame tip at flame centre
    tip[k] = np.argmax(ImDatGrn[:,int(round(np.mean(middle[np.nonzero(middle)])))]>thresh_grn)

    del ImDat, left, right, gray, th_gray,
    
    # Save img_number, middle_std and tip in the exam_tif arrow
    tif_exam[k,0] = k
    tif_exam[k,1] = middle_std[k]
    tif_exam[k,2] = tip[k]

    #-----------------------------------------------------
    # Crop the flame image to desired size
    #-----------------------------------------------------
    middle_ave = int(np.average(middle[middle!=0]))
    Img_color_crop = Img_color[(HAB0 - flame_height) : HAB0, int(middle_ave - flame_width/2) : int(middle_ave + flame_width/2), :]
    ImDatRed_crop = ImDatRed[(HAB0 - flame_height) : HAB0, int(middle_ave - flame_width/2) : int(middle_ave + flame_width/2)]
    ImDatGrn_crop = ImDatGrn[(HAB0 - flame_height) : HAB0, int(middle_ave - flame_width/2) : int(middle_ave + flame_width/2)]
    ImDatBlu_crop = ImDatBlu[(HAB0 - flame_height) : HAB0, int(middle_ave - flame_width/2) : int(middle_ave + flame_width/2)]
    
    del middle, Img_color, ImDatRed, ImDatGrn, ImDatBlu

    ImDatRG = ImDatRed_crop / ImDatGrn_crop
    ImDatRB = ImDatRed_crop / ImDatBlu_crop
    ImDatBG = ImDatBlu_crop / ImDatGrn_crop
    
    # Save the debayer color image and the R G B images
    plt.figure()
    plt.title('Color image')   
    
    plt.subplot(221)
    plt.axis('off')  # Hide both the x and y axises
    plt.imshow(Img_color_crop/scale, vmin=0, vmax=scale)
    plt.title('Color')
    
    plt.subplot(222)
    plt.axis('off')  
    plt.imshow(ImDatRed_crop, vmin=0, vmax=scale)
    plt.title('Red')
    
    plt.subplot(223)
    plt.axis('off')  
    plt.imshow(ImDatGrn_crop, vmin=0, vmax=scale)
    plt.title('Green')
    
    plt.subplot(224)
    plt.axis('off')  
    plt.imshow(ImDatBlu_crop, vmin=0, vmax=scale)
    plt.title('Blue')
    
    plt.subplots_adjust(hspace=None, wspace=-0.7)
    
    # Path to save the figure, and save it.
    fsave_color = abspath( '{0}{1}{2}'.format('Photos//Examination//Color_image-', str(k), '.png') )
    plt.savefig(fsave_color, bbox_inches='tight', dpi=500)

    #plt.draw()
    #plt.pause(3)      # Figure will show for 5 seconds
    #plt.cla()          # Clear axis
    #plt.clf()          # clear figure
    plt.close()   # close figure

    #-----------------------------------------------------
    # Save Red, Green and Blue channels in tecplot format
    #-----------------------------------------------------
    if save_tecplot == True:
        # Get the shape of ImDatRed_crop
        m = len(ImDatRed_crop)        # Row
        n = len(ImDatRed_crop[0])     # Column
        tecplot = np.zeros((m*n, 5))  # Creat a matrix

        for i in range(0, m):       # Searching each row
            for j in range(0, n):   # Searching each column
                tecplot[i*n+j, 0] = (j-(n-1)/2)*pixelmm/10    # Set the middle pixel as r=0 cm
                tecplot[i*n+j, 1] = (m-i)*pixelmm/10          # Flip the image, cm
                tecplot[i*n+j, 2] = ImDatRed_crop[i,j]
                tecplot[i*n+j, 3] = ImDatGrn_crop[i,j]
                tecplot[i*n+j, 4] = ImDatBlu_crop[i,j]

        header_str = ('{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(' TITLE = "Flame-', str(filename), '" \n VARIABLES = "r (cm)", "HAB (cm)", "Red", "Green", "Blue" \n ZONE T = "Flame-', str(filename), '", I = ', str(n),', J = ', str(m), ', F = POINT'))
        
        ImPathtecplot = ('{0}{1}{2}{3}'.format('Photos//Examination//', 'Color_channel_', filename, '-tecplot.dat'))
        
        np.savetxt(ImPathtecplot, tecplot, delimiter=' ', header= header_str, comments='')

        del tecplot

        print('{0}{1}{2}'.format('Frame ', str(k), ' have been examined.'))

    del Img_color_crop, ImDatRed_crop, ImDatGrn_crop, ImDatBlu_crop
    
    # Release Ram
    gc.collect()       # Release Ram

#-----------------------------------------------------
# Save data of flame centre and tip for all frames
#-----------------------------------------------------
header_str = ('NO.,STD,Tip')
path_exam = ('Photos/Examination/0_Flame_centre_and_tip.csv')
np.savetxt(path_exam, tif_exam, fmt='%3d,%1.3f,%4d', header= header_str, comments='')
print('Flame centre and tip detection finished.')

#-----------------------------------------------------
# Show figure of flame centre and tip, and save it
#-----------------------------------------------------
plt.figure()
plt.title('Centre standard deviation')
plt.xlabel("Image number")
plt.ylabel("Standard deviation")
plt.plot(middle_std, '*b')
plt.savefig('./Photos/Examination/1_Flame_centre.png', bbox_inches='tight', dpi=500)
if img_number != 1:
    plt.draw()
    plt.pause(3)  # Figure will show for 3 seconds
plt.close()       # Close figure and continue
print('Flame centre standard deviation saved.')

plt.figure()
plt.title('Flame tip')
plt.xlabel("Image number")
plt.ylabel("Flame tip")
plt.plot(tip, 'xr')
plt.savefig('./Photos/Examination/2_Flame_tip.png', bbox_inches='tight', dpi=500)
if img_number != 1:
    plt.draw()
    plt.pause(3)  # Figure will show for 3 seconds
plt.close()       # Close figure and continue
print('Flame tip saved.')