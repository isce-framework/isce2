#!/usr/bin/env python3
#

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



import os
import sys
import argparse
from osgeo import gdal
from isce.applications.gdal2isce_xml import gdal2isce_xml


# command line parsing of input file
def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser(description='Generate down-sample DEM from wgs84.vrt DEM file')
    parser.add_argument('-i','--input', dest='input_dem_vrt', type=str, required=True, help='Input DEM vrt filename (GDAL supported)')
    parser.add_argument('-rmeter','--res_meter', dest='res_meter', type=str, default='', required=False, help='DEM output resolution in m units')
    parser.add_argument('-rsec','--res_seconds', dest='res_seconds', type=str, default ='', required=False, help='DEM output resolution in arc seconds units')
    return parser.parse_args()


# main script
if __name__ == '__main__':
    '''
    Main driver.
    '''

    # Parse command line
    inps = cmdLineParse()

    if inps.res_meter == '' and inps.res_seconds == '':
        raise Exception('Provide either rmeter or rsec argument for DEM resolution')

    # check if the input file exist
    if not os.path.isfile(inps.input_dem_vrt):
        raise Exception('Input file is not found ....')
    # check if the provided input file is a .vrt file and also get the envi filename
    input_dem_envi, file_extension = os.path.splitext(inps.input_dem_vrt)
    if file_extension != '.vrt':
        raise Exception('Input file is not a vrt file ....')
    # get the file path
    input_path = os.path.dirname(os.path.abspath(inps.input_dem_vrt))


    # convert the output resolution from m in degrees
    # (this is approximate, could use instead exact expression)
    if inps.res_meter != '':
        gdal_opts =  gdal.WarpOptions(format='ENVI',
                                      outputType=gdal.GDT_Int16,
                                      dstSRS='EPSG:4326',
                                      xRes=float(inps.res_meter)/110/1000,
                                      yRes=float(inps.res_meter)/110/1000,
                                      targetAlignedPixels=True)
#        res_degree = float(inps.res_meter)/110/1000
    elif inps.res_seconds != '':
        gdal_opts =  gdal.WarpOptions(format='ENVI',
                                      outputType=gdal.GDT_Int16,
                                      dstSRS='EPSG:4326',
                                      xRes=float(inps.res_seconds)*1/60*1/60,
                                      yRes=float(inps.res_seconds)*1/60*1/60,
                                      targetAlignedPixels=True)
#        res_degree = float(1/60*1/60*float(inps.res_seconds))

    # The ENVI filename of the coarse DEM to be generated
    coarse_dem_envi = os.path.join(input_path, "Coarse_"  + input_dem_envi)

    # Using gdal to down-sample the WGS84 DEM
 #   cmd = "gdalwarp -t_srs EPSG:4326 -ot Int16 -of ENVI -tap -tr " + str(res_degree) + " " + str(res_degree) + " " + inps.input_dem_vrt + " " + coarse_dem_envi
 #   os.system(cmd)
    ds = gdal.Warp(coarse_dem_envi,inps.input_dem_vrt,options=gdal_opts)
    ds = None

    # Generating the ISCE xml and vrt of this coarse DEM
    gdal2isce_xml(coarse_dem_envi+'.vrt')

