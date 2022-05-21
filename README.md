# Data Set
The Chesapeake Watershed data set is derived from satellite imagery over all of the US states that are part of the Chesapeake Bay watershed system. We are using the patches part of the data set. Each patch is a 256 x 256 image with 26 channels, in which each pixel corresponds to a 1m x 1m area of space. Some of these channels are visible light channels (RGB), while others encode surface reflectivity at different frequencies. In addition, each pixel is labeled as being one of:

    0 = no class
    1 = water
    2 = tree canopy / forest
    3 = low vegetation / field
    4 = barren land
    5 = impervious (other)
    6 = impervious (road) 

Here is an example of the RGB image of one patch and the corresponding pixel labels: 

Notes:
  1. Detailed description of the data set
