## Description ##

This Python code was developed to calculate soot temperatures and volume fractions of co-flow diffusion flames using colour ratio pyrometry.

A detailed description can be found in the paper Dreyer et al., Applied Optics 58 (2019), 2662-2670, [doi:10.1364/AO.58.002662](https://doi.org/10.1364/AO.58.002662). Please cite this paper when using the code. A preprint of this paper is available [here](https://como.ceb.cam.ac.uk/preprints/217/).

The code was developed using Python 3.6.6 for Windows 10. The packages PyAbel and OpenCv-Python should be installed.

## How to use ##

The main file is:
```sh
Flame_pyrometry.py
```
Depending on the selections made, other functions are called. More detailed descriptions can be found in the code.

Please note that calibration files for the experimental setup are required. Examples are provided (see 'useful links' below) but these will only give meaningful results for the optical system used to record the example image. 

Different functions perform the inverse Abel transform of the recorded image. Equations and descriptions of the BASEX, onion peeling, 3-point, and Fast Hangel (Hansen-Law) methods can be found in the following publications:
Apostolopoulos et al., Optics Communications 296 (2013) 25–34
Dribinski et al., Review of Scientific Instruments 73 (2002) 2634-2642
Dasch, Applied Optics 31 (1992) 1146-1152
Daun et al., Applied Optics 45 (2006) 4638-4646
Hansen and Law, Journal of the Optical Society of America A 2 (1985) 510-520

The FLiPPID method enables smooth reconstruction of flame cross sections, even close to the flame center line, where other methods struggle. 

The "PyAbel" package includes different methods, such as 'BASEX', '3-point', 'Dasch-Onion', and 'Hansen-Law'. It should be noted that in its current version, a regularization of the raw data is only possible with BASEX.

This code includes the FLiPPID method, which takes longer to process an image. 

##### General parameters

In the first part of the code, you have two options for processing images. For a new image use 'Pre-process', to load a previously pre-processed flame image use 'Load'. You should choose what kind of inverse Abel transform you want to use. To get the soot temperature, you need to supply a temperature lookup table; otherwise, the program calculates the table using a calibration curve and camera response data (see below).

You can calculate the soot volume fraction if the temperature is calculated. And you need to select the filter used during the experiment and camera calibration. Options are 'nofilter', 'FGB7', and 'FGB39'. Others can be added in the function Temperature_lookup.

##### Select flame image files and parameters for image processing

The function in this second part of the code loads a tif image in raw format and crops it to the desired size. The file has to be located in the folder "Photos". The example image '100H_BG7_exp1563' is saved in the Photos folder of the example data. This image was recorded in a raw format using a Balckfly S colour camera (FLIR Integrated Imaging Solutions) with a CMOS sensor (2048 x 1536 pixels). 
You should not use any balance colour ratio in the camera settings for taking the images. It is also possible to provide a stacked tif file with multiple frames. It can be done using ImageJ or similar software. 

If you are going to load a previously pre-processed flame image, you need to define the file path for the csv file containg the red colour channel. You can find this file in the "Photos" folder.
If you are going to use 'Pre-process' to analyze a new image, you should specify the file name in 'filename', how many frames to be processed (the example image contains 5 frames), and you can choose to plot the cropped image. 
The flame centre line is detected automatically. You need to define the position of the flame and cropped image size you need to adjust the 'flame_height' and 'flame_width' (Use an uneven integer for flame_width), HAB0 stands for height above burner equals zero and is the vertical pixel position of the burner exit. 
If you supplied a stacked tif file with multiple frames, the code searches for the flame centre lines and flame tips in each frame and averages the non-tilted ones with similar flame height. You can change the threshold for selecting non-tilted flames with similar height. Flames with a tip larger than +-thresh_tip are removed. Flames with a standard  deviation of the centre line larger than thresh_std are removed. Note that an error will occur if non of the frames fulfills these conditions.

If the automatic flame centre and edge detection fails, it is necessary to check the values in Get_flame.py, i.e., the thresholds for edge and tip detection, the selection criteria for straight flames with similar height, and the number and positions where flame centres are searched.
A code to demosaic the image is used. Different algorithms can be used: COLOR_BayerGB2RGB, COLOR_BayerGB2RGB_EA, COLOR_BayerGB2RGB_VNG. Note that COLOR_BayerRG2RGB_VNG requires a 8 bit image. In case the camera has a different Bayer pattern the 'BayerGR' has to be replaced by the appropriate pattern. Actually, the BlackflyS used for the example image has a BayerRG pattern but the raw image was rotated by 90 degrees such that the flame is vertical. 
The 'scale' value represents the maximum intensity count of the camera. In this case, it is set for a 12-bit camera, if you use a 8-bit camera you need to change the value. 

##### Inverse Abel transform of image

As the FLiPPID method takes a significant amount of time, the code has an option to check if the images/Abel transform are working fine. 
In the line 177, you can define a range of z values to execute FLiPPID, you can change the range (z_min, z_max) or string 'all' to perform the inverse Abel transform for all the flame using the FLiPPID method. 

##### Get the temperature lookup table and calculate the soot temperature

To calculate the soot temperature, you need to check the wavelength and temperature range. The soot temperature calculation includes the soot dispersion exponent (alpha parameter), different values are available in the literature. 
The code offers to option to calculate the soot temperature using 'Chang' with lambda^-1.423 as derived from the refractive index measurements by: Chang and Charalampopoulos, Proc. R. Soc. Lond. A 430 (1990) 577-591 or 'Kuhn' for lambda^-1.38 as reported by Kuhn et al., Proceedings of the Combustion Institute 33 (2011) 743-750.

You need to provide the location of the colour ratio calibration file. You can find an example in Temperature_tables/Measured_R_Thermo_FGB7. Column 0 has to contain the measure temperature in K and column 1, 2, 3 the corresponding RG, RB, and BG colour ratios. 
In this example, the other columns in the file are not required but show the exposure times and counts of the red, green, and blue channel. Dividing column 6 (green counts) by column 4 (exposure time) and plotting the result over column 0 (temperature) leads to the power law function as required for the soot volume fractions.
The code can calculate the temperature lookup table, i.e., a table of the colour ratios expected to be recorded by the camera as a function of the soot temperature. 
The function uses the theoretical camera, lens, and filter response to calculate colour ratios of a hot thermocouple and compares it to experimentally measured values. 
the blue and red colour channels are scaled to match the theoretical and observed colour ratios after which the ratios of hot soot are calculated. Further details can be found in the following publications:
Ma and Long, Proceedings of the Combustion Institute 34 (2013) 3531-3539 (https://www.sciencedirect.com/science/article/pii/S1540748912000314)
Kuhn et al., Proceedings of the Combustion Institute 33 (2011) 743-750 (https://www.sciencedirect.com/science/article/pii/S1540748910000209)
Once the code has calculated the lookup table, it is able to calculate the soot temperature profiles of the flame. 

##### Get soot volume fraction

The code includes the function to calculate the soot volume fraction using the previously calculated soot temperatures, the recorded light intensity, the exposure time used while imaging the flame, and the camera calibration. Further details can be found in the following 
publications:
Ma and Long, Proceedings of the Combustion Institute 34 (2013) 3531-3539
Kuhn et al., Proceedings of the Combustion Institute 33 (2011) 743-750  

You need to update the 'pixelmm value' and 'exposure'. 'pixelmm' is the mm for each pixel and 'exposure' corresponds to the exposure time used while taking the flame images.
To correlate the exposure time with the temperature you need a function. This function can be obtained form the measured green signal emitted from a hot thermocouple divided by the exposure time as a function of temperature. The equation is obtained by plotting the green counts / exposure time over temperature and fitting a power law function to it.
The equation is obtained by plotting the green counts / exposure time over temperature and fitting a power law function to it. 

## Useful links ##

* An example image and data are available for download [here](https://como.ceb.cam.ac.uk/media/resources/FlPyroImageAndData.zip).
* [Flame pyrometry resources page](https://como.ceb.cam.ac.uk/resources/flpyro/) on the [CoMo website](https://como.ceb.cam.ac.uk/)
* [Dreyer et al.](https://como.ceb.cam.ac.uk/publications/AO-58-2662-2670/) paper
* [Preprint 217](https://como.ceb.cam.ac.uk/preprints/217/)