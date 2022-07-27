#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2019 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Piyush Agram, Yang Lei
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






import pdb
import subprocess
import re
import string

class GeogridOptical():
    '''
    Class for mapping regular geographic grid on radar imagery.
    '''

    def runGeogrid(self):
        '''
        Do the actual processing.
        '''
        
        from . import geogridOptical

        ##Determine appropriate EPSG system
        self.epsgDem = self.getProjectionSystem(self.demname)
        self.epsgDat = self.getProjectionSystem(self.dat1name)

        ###Determine extent of data needed
        bbox = self.determineBbox()
        
        ##Create and set parameters
        self.setState()
        
        ##check parameters
        self.checkState()

        ##Run
        geogridOptical.geogridOptical_Py(self._geogridOptical)
        self.get_center_latlon()
    
        ##Get parameters
        self.getState()
    
        ##Clean up
        self.finalize()
    
    def get_center_latlon(self):
        '''
        Get center lat/lon of the image.
        '''
        from osgeo import gdal
        gdal.AllRegister()
        self.epsgDem = 4326
        self.epsgDat = self.getProjectionSystem(self.dat1name)
        self.determineBbox()
        if gdal.__version__[0] == '2':
            self.cen_lat = (self._ylim[0] + self._ylim[1]) / 2
            self.cen_lon = (self._xlim[0] + self._xlim[1]) / 2
        else:
            self.cen_lon = (self._ylim[0] + self._ylim[1]) / 2
            self.cen_lat = (self._xlim[0] + self._xlim[1]) / 2
        print("Scene-center lat/lon: " + str(self.cen_lat) + "  " + str(self.cen_lon))
    

    def getProjectionSystem(self, filename):
        '''
        Testing with Greenland.
        '''
        if not filename:
            raise Exception('File {0} does not exist'.format(filename))

        from osgeo import gdal, osr
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        srs.AutoIdentifyEPSG()
        ds = None
#        pdb.set_trace()

        if srs.IsGeographic():
            epsgstr = srs.GetAuthorityCode('GEOGCS')
        elif srs.IsProjected():
            epsgstr = srs.GetAuthorityCode('PROJCS')
        elif srs.IsLocal():
            raise Exception('Local coordinate system encountered')
        else:
            raise Exception('Non-standard coordinate system encountered')
        if not epsgstr:  #Empty string->use shell command gdalsrsinfo for last trial
            cmd = 'gdalsrsinfo -o epsg {0}'.format(filename)
            epsgstr = subprocess.check_output(cmd, shell=True)
#            pdb.set_trace()
            epsgstr = re.findall("EPSG:(\d+)", str(epsgstr))[0]
#            pdb.set_trace()
        if not epsgstr:  #Empty string
            raise Exception('Could not auto-identify epsg code')
#        pdb.set_trace()
        epsgcode = int(epsgstr)
#        pdb.set_trace()
        return epsgcode

    def determineBbox(self, zrange=[-200,4000]):
        '''
        Dummy.
        '''
        import numpy as np
        import datetime
        from osgeo import osr

#        import pdb
#        pdb.set_trace()


        samples = self.startingX + np.array([0, self.numberOfSamples-1]) * self.XSize
        lines = self.startingY + np.array([0, self.numberOfLines-1]) * self.YSize

        coordDat = osr.SpatialReference()
        if self.epsgDat:
            coordDat.ImportFromEPSG(self.epsgDat)
        else:
            raise Exception('EPSG code does not exist for image data')


        coordDem = osr.SpatialReference()
        if self.epsgDem:
            coordDem.ImportFromEPSG(self.epsgDem)
        else:
            raise Exception('EPSG code does not exist for DEM')


        trans = osr.CoordinateTransformation(coordDat, coordDem)



        utms = []
        xyzs = []


        ### Four corner coordinates
        for ss in samples:
            for ll in lines:
                for zz in zrange:
                    utms.append([ss,ll,zz])
                    x,y,z = trans.TransformPoint(ss, ll, zz)
                    xyzs.append([x,y,z])

        utms = np.array(utms)
        xyzs = np.array(xyzs)

        self._xlim = [np.min(xyzs[:,0]), np.max(xyzs[:,0])]
        self._ylim = [np.min(xyzs[:,1]), np.max(xyzs[:,1])]


    def getState(self):

        from . import geogridOptical
        
        self.pOff = geogridOptical.getXOff_Py(self._geogridOptical)
        self.lOff = geogridOptical.getYOff_Py(self._geogridOptical)
        self.pCount = geogridOptical.getXCount_Py(self._geogridOptical)
        self.lCount = geogridOptical.getYCount_Py(self._geogridOptical)
        self.X_res = geogridOptical.getXPixelSize_Py(self._geogridOptical)
        self.Y_res = geogridOptical.getYPixelSize_Py(self._geogridOptical)
    
    def setState(self):
        '''
        Create C object and populate.
        '''

        from . import geogridOptical

        if self._geogridOptical is not None:
            geogridOptical.destroyGeoGridOptical_Py(self._geogridOptical)
    
        self._geogridOptical = geogridOptical.createGeoGridOptical_Py()
        geogridOptical.setOpticalImageDimensions_Py( self._geogridOptical, self.numberOfSamples, self.numberOfLines)
        geogridOptical.setXParameters_Py( self._geogridOptical, self.startingX, self.XSize)
        geogridOptical.setYParameters_Py( self._geogridOptical, self.startingY, self.YSize)
        geogridOptical.setRepeatTime_Py(self._geogridOptical, self.repeatTime)
        
        geogridOptical.setDtUnity_Py( self._geogridOptical, self.srs_dt_unity)
        geogridOptical.setMaxFactor_Py( self._geogridOptical, self.srs_max_scale)
        geogridOptical.setUpperThreshold_Py( self._geogridOptical, self.srs_max_search)
        geogridOptical.setLowerThreshold_Py(self._geogridOptical, self.srs_min_search)
        
        geogridOptical.setEPSG_Py(self._geogridOptical, self.epsgDem, self.epsgDat)
        geogridOptical.setChipSizeX0_Py(self._geogridOptical, self.chipSizeX0)
        geogridOptical.setGridSpacingX_Py(self._geogridOptical, self.gridSpacingX)
        
        geogridOptical.setXLimits_Py(self._geogridOptical, self._xlim[0], self._xlim[1])
        geogridOptical.setYLimits_Py(self._geogridOptical, self._ylim[0], self._ylim[1])
        if self.demname:
            geogridOptical.setDEM_Py(self._geogridOptical, self.demname)

        if (self.dhdxname is not None) and (self.dhdyname is not None):
            geogridOptical.setSlopes_Py(self._geogridOptical, self.dhdxname, self.dhdyname)
        
        if (self.vxname is not None) and (self.vyname is not None):
            geogridOptical.setVelocities_Py(self._geogridOptical, self.vxname, self.vyname)
        
        if (self.srxname is not None) and (self.sryname is not None):
            geogridOptical.setSearchRange_Py(self._geogridOptical, self.srxname, self.sryname)

        if (self.csminxname is not None) and (self.csminyname is not None):
            geogridOptical.setChipSizeMin_Py(self._geogridOptical, self.csminxname, self.csminyname)
        
        if (self.csmaxxname is not None) and (self.csmaxyname is not None):
            geogridOptical.setChipSizeMax_Py(self._geogridOptical, self.csmaxxname, self.csmaxyname)

        if (self.ssmname is not None):
            geogridOptical.setStableSurfaceMask_Py(self._geogridOptical, self.ssmname)
        
        geogridOptical.setWindowLocationsFilename_Py( self._geogridOptical, self.winlocname)
        geogridOptical.setWindowOffsetsFilename_Py( self._geogridOptical, self.winoffname)
        geogridOptical.setWindowSearchRangeFilename_Py( self._geogridOptical, self.winsrname)
        geogridOptical.setWindowChipSizeMinFilename_Py( self._geogridOptical, self.wincsminname)
        geogridOptical.setWindowChipSizeMaxFilename_Py( self._geogridOptical, self.wincsmaxname)
        geogridOptical.setWindowStableSurfaceMaskFilename_Py( self._geogridOptical, self.winssmname)
        geogridOptical.setRO2VXFilename_Py( self._geogridOptical, self.winro2vxname)
        geogridOptical.setRO2VYFilename_Py( self._geogridOptical, self.winro2vyname)
        geogridOptical.setNodataOut_Py(self._geogridOptical, self.nodata_out)
        
    
    def checkState(self):
        '''
        Create C object and populate.
        '''
        if self.repeatTime < 0:
            raise Exception('Input image 1 must be older than input image 2')


    def finalize(self):
        '''
        Clean up all the C pointers.
        '''

        from . import geogridOptical
        
        geogridOptical.destroyGeoGridOptical_Py(self._geogridOptical)
        self._geogridOptical = None


    






    def coregister(self,in1,in2):
        import os
        import numpy as np

        from osgeo import gdal, osr
        import struct

        DS1 = gdal.Open(in1, gdal.GA_ReadOnly)
        trans1 = DS1.GetGeoTransform()
        xsize1 = DS1.RasterXSize
        ysize1 = DS1.RasterYSize
        epsg1 = self.getProjectionSystem(in1)

        DS2 = gdal.Open(in2, gdal.GA_ReadOnly)
        trans2 = DS2.GetGeoTransform()
        xsize2 = DS2.RasterXSize
        ysize2 = DS2.RasterYSize
        epsg2 = self.getProjectionSystem(in2)
        
        if epsg1 != epsg2:
            raise Exception('The current version of geo_autoRIFT assumes the two images are in the same projection, i.e. it cannot handle two different projections; the users are thus recommended to do the tranformation themselves before running geo_autoRIFT.')
        
        

        W = np.max([trans1[0],trans2[0]])
        N = np.min([trans1[3],trans2[3]])
        E = np.min([trans1[0]+(xsize1-1)*trans1[1],trans2[0]+(xsize2-1)*trans2[1]])
        S = np.max([trans1[3]+(ysize1-1)*trans1[5],trans2[3]+(ysize2-1)*trans2[5]])

        x1a = int(np.round((W-trans1[0])/trans1[1]))
        x1b = int(np.round((E-trans1[0])/trans1[1]))
        y1a = int(np.round((N-trans1[3])/trans1[5]))
        y1b = int(np.round((S-trans1[3])/trans1[5]))

        x2a = int(np.round((W-trans2[0])/trans2[1]))
        x2b = int(np.round((E-trans2[0])/trans2[1]))
        y2a = int(np.round((N-trans2[3])/trans2[5]))
        y2b = int(np.round((S-trans2[3])/trans2[5]))
        
        if (x1a > (xsize1-1))|(x1b > (xsize1-1))|(x2a > (xsize2-1))|(x2b > (xsize2-1))|(y1a > (ysize1-1))|(y1b > (ysize1-1))|(y2a > (ysize2-1))|(y2b > (ysize2-1)):
            raise Exception('Uppper bound of coregistered image index should be <= size of image1 (and image2) minus 1')

        if (x1a < 0)|(x1b < 0)|(x2a < 0)|(x2b < 0)|(y1a < 0)|(y1b < 0)|(y2a < 0)|(y2b < 0):
            raise Exception('Lower bound of coregistered image index should be >= 0')

        if ((x1b-x1a) != (x2b-x2a))|((y1b-y1a) != (y2b-y2a)):
            raise Exception('Coregistered image size mismatch between image1 and image2')

        x1a = int(x1a)
        x1b = int(x1b)
        y1a = int(y1a)
        y1b = int(y1b)
        x2a = int(x2a)
        x2b = int(x2b)
        y2a = int(y2a)
        y2b = int(y2b)

        trans = (W, trans1[1], 0.0, N, 0.0, trans1[5])

        return x1a, y1a, x1b-x1a+1, y1b-y1a+1, x2a, y2a, x2b-x2a+1, y2b-y2a+1, trans








    def __init__(self):
        super(GeogridOptical, self).__init__()

        ##Optical image related parameters
        self.startingY = None
        self.startingX = None
        self.XSize = None
        self.YSize = None
        self.numberOfSamples = None
        self.numberOfLines = None
        self.repeatTime = None
        self.chipSizeX0 = None
        self.gridSpacingX = None

        ##Input related parameters
        self.dat1name = None
        self.demname = None
        self.dhdxname = None
        self.dhdyname = None
        self.vxname = None
        self.vyname = None
        self.srxname = None
        self.sryname = None
        self.csminxname = None
        self.csminyname = None
        self.csmaxxname = None
        self.csmaxyname = None
        self.ssmname = None

        ##Output related parameters
        self.winlocname = None
        self.winoffname = None
        self.winsrname = None
        self.wincsminname = None
        self.wincsmaxname = None
        self.winssmname = None
        self.winro2vxname = None
        self.winro2vyname = None
        
        ##dt-varying search range scale (srs) rountine parameters
        self.srs_dt_unity = 182
        self.srs_max_scale = 5
        self.srs_max_search = 20000
        self.srs_min_search = 0

        ##Coordinate system
        self.epsgDem = None
        self.epsgDat = None
        self._xlim = None
        self._ylim = None
        self.nodata_out = None
        
        ##Pointer to C
        self._geogridOptical = None

        ##parameters for autoRIFT
        self.pOff = None
        self.lOff = None
        self.pCount = None
        self.lCount = None
        self.X_res = None
        self.Y_res = None
