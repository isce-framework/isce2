#!/usr/bin/env python3


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2017 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: David Bekaert
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import sys
from osgeo import gdal
import argparse
import os
import numpy as np
import scipy.linalg

# command line parsing of input file
def cmdLineParse():
   '''
   Command line parser.
   '''
   parser = argparse.ArgumentParser(description='Generate the gdal command needed to cropping RDR data based on a lon-lat BBox')
   parser.add_argument('-i','--input', dest='file', type=str, required=False, help='Input filename (GDAL supported)')
   parser.add_argument('-b', '--bbox', dest='bbox', type=str, required=True, help='Lat/Lon Bounding SNWE')
   parser.add_argument('-dfac', '--downfac', dest='down_sample', type=str, required=False, default='100', help='Lon/Lat downsample factor used when mapping GEO-coordiantes to RDR')
   parser.add_argument('-nd', '--nodata', dest='nodata', type=str, required=False, default='0', help='Lon/Lat no-data value')
   parser.add_argument('-lat', '--lat', dest='latfile', type=str, required=False, default='lat.rdr.full', help='Lat filename')
   parser.add_argument('-lon', '--lon', dest='lonfile', type=str, required=False, default='lon.rdr.full', help='Lon filename')
   return parser.parse_args() 

# main script
if __name__ == '__main__':
   '''
   Main driver.
   '''

   # parsing the command line inputs
   inps = cmdLineParse() 
   down_sample = int(inps.down_sample)
   lonfile = inps.lonfile
   latfile = inps.latfile
   nodata = inps.nodata
   bbox = inps.bbox
   bbox_SNWE = np.fromstring(bbox, dtype=float, sep=' ')

   # loading the longitude and latitude
   print("Load longitude and latitude")
   LonData =  gdal.Open(lonfile)
   LatData =  gdal.Open(latfile)
   LonBand = LonData.GetRasterBand(1)
   LatBand = LatData.GetRasterBand(1)
   LonArray = LonBand.ReadAsArray()
   # total number of lines and pixels
   n_lines_full, n_pixels_full = LonArray.shape
   LonArray_coarse = LonArray[1::down_sample,1::down_sample];
   # no need to keep the high res
   del LonArray
   LatArray = LatBand.ReadAsArray()
   LatArray_coarse = LatArray[1::down_sample,1::down_sample];
   # no need to keep the high res
   del LatArray

   # coarse grid size
   n_lines, n_pixels = LatArray_coarse.shape
   PIXELS,LINES = np.meshgrid(np.arange(1, n_pixels+1, 1), np.arange(1, n_lines+1, 1))
   Pixels = np.reshape(PIXELS, (-1,1))
   Lines = np.reshape(LINES, (-1,1))
    
   # flatten the lon and latitude in the same way
   Lat = np.reshape(LatArray_coarse, (-1,1))
   Lon = np.reshape(LonArray_coarse, (-1,1))


   # remove the no-data values for lon and lat
   ix_drop = np.where(Lat == 0)[0]
   Lat = np.delete(Lat,ix_drop,0)
   Lon = np.delete(Lon,ix_drop,0)
   Pixels = np.delete(Pixels,ix_drop,0)
   Lines = np.delete(Lines,ix_drop,0)
   ix_drop = np.where(Lon == 0)[0]
   Lat = np.delete(Lat,ix_drop,0)
   Lon = np.delete(Lon,ix_drop,0)
   Pixels = np.delete(Pixels,ix_drop,0)
   Lines = np.delete(Lines,ix_drop,0)

   # fit a plan to the lon and lat data in radar coordinates
   A = np.c_[Lon[:,0], Lat[:,0], np.ones(Lon.shape[0])]
   # Pixels plane as function of geo-coordinates
   CPixels,_,_,_ = scipy.linalg.lstsq(A, Pixels[:,0])   
   # Lines plane as function of geo-coordinates
   CLines,_,_,_ = scipy.linalg.lstsq(A, Lines[:,0])

   # loop over the BBOX as specified by the user     
   #  evaluate it on grid
   querry_lonlat = np.array([ [bbox_SNWE[2] ,bbox_SNWE[0] ] , [bbox_SNWE[2] ,bbox_SNWE[1]] , [bbox_SNWE[3] ,bbox_SNWE[1]] , [bbox_SNWE[3], bbox_SNWE[0]]])

   # initialize the estimate for the pixels and lines
   print('Mapping coordinates:')
   estimate_LinePixel = []
   for row in range(4):
       Pixel_est = int(down_sample*(CPixels[0]*querry_lonlat[row,0] + CPixels[1]*querry_lonlat[row,1] + CPixels[2]))
       Line_est = int(down_sample*(CLines[0]*querry_lonlat[row,0] + CLines[1]*querry_lonlat[row,1] + CLines[2]))

       # make sure the pixel falls within the bounds of the data
       # if smaller than 1 then put to 1
       extra_str = ''
       if Pixel_est<1:
           Pixel_est = 1
           extra_str = '(projected to edge)'
       if Line_est<1:
           Line_est=1
           extra_str = '(projected to edge)'
       # if larger than the dataset size then put to maximum bounds of the data
       if Pixel_est>n_pixels_full:
           Pixel_est = n_pixels_full
           extra_str = '(projected to edge)'
       if Line_est>n_lines_full:
           Line_est=n_lines_full
           extra_str = '(projected to edge)'

       # store the information 
       estimate_LinePixel.append([Line_est , Pixel_est ])

       # output to user:
       print('(Lon,lat): (' + str(querry_lonlat[row,0]) + ';' + str(querry_lonlat[row,1]) + ') \t->\t (Line,Pixel): ' + str(Line_est) + ';' + str(Pixel_est) + ')  \t ' + extra_str )


   # Only take the extreme of the bounds, to ensure the requested area is covered
   estimate_LinePixel = np.array(estimate_LinePixel)
   # maximum and minimum for the pixels and lines
   max_LinePixel = np.max(estimate_LinePixel,axis=0)
   min_LinePixel = np.min(estimate_LinePixel,axis=0)
   print('Lines: ' + str(min_LinePixel[0]) + '\t' + str(max_LinePixel[0]))
   print('Pixels: ' + str(min_LinePixel[1]) + '\t' + str(max_LinePixel[1])) 

   print('gdalwarp -to SRC_METHOD=NO_GEOTRANSFORM -of envi -te ' + str(min_LinePixel[1]) + ' ' + str(min_LinePixel[0]) + ' ' + str(max_LinePixel[1]) + ' ' +  str(max_LinePixel[0]) + ' ')
#   print('gdalwarp -to SRC_METHOD=NO_GEOTRANSFORM -of envi -co INTERLEAVE=BIP -te ' + str(min_LinePixel[1]) + ' ' + str(min_LinePixel[0]) + ' ' + str(max_LinePixel[1]) + ' ' +  str(max_LinePixel[0]))
   print('gdal_translate -srcwin ' +  str(min_LinePixel[1]) + ' ' + str(min_LinePixel[0]) + ' ' + str(max_LinePixel[1]-min_LinePixel[1]) + ' ' + str(max_LinePixel[0]-min_LinePixel[0]) + ' -of envi -co INTERLEAVE=BIP ' )

