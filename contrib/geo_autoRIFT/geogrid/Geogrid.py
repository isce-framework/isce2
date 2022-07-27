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




import isce
from iscesys.Component.Component import Component
import pdb
import subprocess
import re
import string

class Geogrid(Component):
    '''
    Class for mapping regular geographic grid on radar imagery.
    '''

    def geogrid(self):
        '''
        Do the actual processing.
        '''
        import isce
        from components.contrib.geo_autoRIFT.geogrid import geogrid

        ##Determine appropriate EPSG system
        self.epsg = self.getProjectionSystem()
        
        ###Determine extent of data needed
        bbox = self.determineBbox()

        ###Load approrpriate DEM from database
        if self.demname is None:
            self.demname, self.dhdxname, self.dhdyname, self.vxname, self.vyname, self.srxname, self.sryname, self.csminxname, self.csminyname, self.csmaxxname, self.csmaxyname, self.ssmname = self.getDEM(bbox)


        ##Create and set parameters
        self.setState()
        
        ##check parameters
        self.checkState()
        
        ##Run
        geogrid.geogrid_Py(self._geogrid)
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
        self.epsg = 4326
        self.determineBbox()
        if gdal.__version__[0] == '2':
            self.cen_lat = (self._ylim[0] + self._ylim[1]) / 2
            self.cen_lon = (self._xlim[0] + self._xlim[1]) / 2
        else:
            self.cen_lon = (self._ylim[0] + self._ylim[1]) / 2
            self.cen_lat = (self._xlim[0] + self._xlim[1]) / 2
        print("Scene-center lat/lon: " + str(self.cen_lat) + "  " + str(self.cen_lon))
    

    def getProjectionSystem(self):
        '''
        Testing with Greenland.
        '''
        if not self.demname:
            raise Exception('At least the DEM parameter must be set for geogrid')

        from osgeo import gdal, osr
        ds = gdal.Open(self.demname, gdal.GA_ReadOnly)
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
            cmd = 'gdalsrsinfo -o epsg {0}'.format(self.demname)
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
        from osgeo import osr,gdal
        
#        import pdb
#        pdb.set_trace()

#        rng = self.startingRange + np.linspace(0, self.numberOfSamples, num=21)
        rng = self.startingRange + np.linspace(0, self.numberOfSamples-1, num=21) * self.rangePixelSize
        deltat = np.linspace(0, 1., num=21)[1:-1]

        lonlat = osr.SpatialReference()
        lonlat.ImportFromEPSG(4326)

        coord = osr.SpatialReference()
        coord.ImportFromEPSG(self.epsg)
        
        trans = osr.CoordinateTransformation(lonlat, coord)

        llhs = []
        xyzs = []


        ###First range line
        for rr in rng:
            for zz in zrange:
                llh = self.orbit.rdr2geo(self.sensingStart, rr, side=self.lookSide, height=zz)
                llhs.append(llh)
                if gdal.__version__[0] == '2':
                    x,y,z = trans.TransformPoint(llh[1], llh[0], llh[2])
                else:
                    x,y,z = trans.TransformPoint(llh[0], llh[1], llh[2])
                xyzs.append([x,y,z])

        ##Last range line
        sensingStop = self.sensingStart + datetime.timedelta(seconds = (self.numberOfLines-1) / self.prf)
        for rr in rng:
            for zz in zrange:
                llh = self.orbit.rdr2geo(sensingStop, rr, side=self.lookSide, height=zz)
                llhs.append(llh)
                if gdal.__version__[0] == '2':
                    x,y,z = trans.TransformPoint(llh[1], llh[0], llh[2])
                else:
                    x,y,z = trans.TransformPoint(llh[0], llh[1], llh[2])
                xyzs.append([x,y,z])


        ##For each line in middle, consider the edges
        for frac in deltat:
            sensingTime = self.sensingStart + datetime.timedelta(seconds = frac * (self.numberOfLines-1)/self.prf)
#            print('sensing Time: %f %f %f'%(sensingTime.minute,sensingTime.second,sensingTime.microsecond))
            for rr in [rng[0], rng[-1]]:
                for zz in zrange:
                    llh = self.orbit.rdr2geo(sensingTime, rr, side=self.lookSide, height=zz)
                    llhs.append(llh)
                    if gdal.__version__[0] == '2':
                        x,y,z = trans.TransformPoint(llh[1], llh[0], llh[2])
                    else:
                        x,y,z = trans.TransformPoint(llh[0], llh[1], llh[2])
                    xyzs.append([x,y,z])


        llhs = np.array(llhs)
        xyzs = np.array(xyzs)


        self._xlim = [np.min(xyzs[:,0]), np.max(xyzs[:,0])]
        self._ylim = [np.min(xyzs[:,1]), np.max(xyzs[:,1])]

        
    def getIncidenceAngle(self, zrange=[-200,4000]):
        '''
        Dummy.
        '''
        import numpy as np
        import datetime
        from osgeo import osr,gdal
        from isceobj.Util.geo.ellipsoid import Ellipsoid
        from isceobj.Planet.Planet import Planet
        
        planet = Planet(pname='Earth')
        refElp = Ellipsoid(a=planet.ellipsoid.a, e2=planet.ellipsoid.e2, model='WGS84')
        
        deg2rad = np.pi/180.0
        
        thetas = []
        
        midrng = self.startingRange + (np.floor(self.numberOfSamples/2)-1) * self.rangePixelSize
        midsensing = self.sensingStart + datetime.timedelta(seconds = (np.floor(self.numberOfLines/2)-1) / self.prf)
        masterSV = self.orbit.interpolateOrbit(midsensing, method='hermite')
        mxyz = np.array(masterSV.getPosition())
        
        for zz in zrange:
            llh = self.orbit.rdr2geo(midsensing, midrng, side=self.lookSide, height=zz)
            targxyz = np.array(refElp.LLH(llh[0], llh[1], llh[2]).ecef().tolist())
            los = (mxyz-targxyz) / np.linalg.norm(mxyz-targxyz)
            n_vec = np.array([np.cos(llh[0]*deg2rad)*np.cos(llh[1]*deg2rad), np.cos(llh[0]*deg2rad)*np.sin(llh[1]*deg2rad), np.sin(llh[0]*deg2rad)])
            theta = np.arccos(np.dot(los, n_vec))
            thetas.append([theta])
        
        thetas = np.array(thetas)
        
        self.incidenceAngle = np.mean(thetas)
        
    def getDEM(self, bbox):
        '''
        Look up database and return values.
        '''
        
        return "", "", "", "", ""

    def getState(self):
        from components.contrib.geo_autoRIFT.geogrid import geogrid
        
        self.pOff = geogrid.getXOff_Py(self._geogrid)
        self.lOff = geogrid.getYOff_Py(self._geogrid)
        self.pCount = geogrid.getXCount_Py(self._geogrid)
        self.lCount = geogrid.getYCount_Py(self._geogrid)
        self.X_res = geogrid.getXPixelSize_Py(self._geogrid)
        self.Y_res = geogrid.getYPixelSize_Py(self._geogrid)
    
    def setState(self):
        '''
        Create C object and populate.
        '''
        from components.contrib.geo_autoRIFT.geogrid import geogrid
        from iscesys import DateTimeUtil as DTU

        if self._geogrid is not None:
            geogrid.destroyGeoGrid_Py(self._geogrid)

        self._geogrid = geogrid.createGeoGrid_Py()
        geogrid.setRadarImageDimensions_Py( self._geogrid, self.numberOfSamples, self.numberOfLines)
        geogrid.setRangeParameters_Py( self._geogrid, self.startingRange, self.rangePixelSize)
        geogrid.setAzimuthParameters_Py( self._geogrid, DTU.seconds_since_midnight(self.sensingStart), self.prf)
        geogrid.setRepeatTime_Py(self._geogrid, self.repeatTime)
        
        geogrid.setDtUnity_Py( self._geogrid, self.srs_dt_unity)
        geogrid.setMaxFactor_Py( self._geogrid, self.srs_max_scale)
        geogrid.setUpperThreshold_Py( self._geogrid, self.srs_max_search)
        geogrid.setLowerThreshold_Py(self._geogrid, self.srs_min_search)

        geogrid.setEPSG_Py(self._geogrid, self.epsg)
        geogrid.setIncidenceAngle_Py(self._geogrid, self.incidenceAngle)
        geogrid.setChipSizeX0_Py(self._geogrid, self.chipSizeX0)
        geogrid.setGridSpacingX_Py(self._geogrid, self.gridSpacingX)
        
        geogrid.setXLimits_Py(self._geogrid, self._xlim[0], self._xlim[1])
        geogrid.setYLimits_Py(self._geogrid, self._ylim[0], self._ylim[1])
        if self.demname:
            geogrid.setDEM_Py(self._geogrid, self.demname)

        if (self.dhdxname is not None) and (self.dhdyname is not None):
            geogrid.setSlopes_Py(self._geogrid, self.dhdxname, self.dhdyname)

        if (self.vxname is not None) and (self.vyname is not None):
            geogrid.setVelocities_Py(self._geogrid, self.vxname, self.vyname)
        
        if (self.srxname is not None) and (self.sryname is not None):
            geogrid.setSearchRange_Py(self._geogrid, self.srxname, self.sryname)
        
        if (self.csminxname is not None) and (self.csminyname is not None):
            geogrid.setChipSizeMin_Py(self._geogrid, self.csminxname, self.csminyname)
        
        if (self.csmaxxname is not None) and (self.csmaxyname is not None):
            geogrid.setChipSizeMax_Py(self._geogrid, self.csmaxxname, self.csmaxyname)
        
        if (self.ssmname is not None):
            geogrid.setStableSurfaceMask_Py(self._geogrid, self.ssmname)

        geogrid.setWindowLocationsFilename_Py( self._geogrid, self.winlocname)
        geogrid.setWindowOffsetsFilename_Py( self._geogrid, self.winoffname)
        geogrid.setWindowSearchRangeFilename_Py( self._geogrid, self.winsrname)
        geogrid.setWindowChipSizeMinFilename_Py( self._geogrid, self.wincsminname)
        geogrid.setWindowChipSizeMaxFilename_Py( self._geogrid, self.wincsmaxname)
        geogrid.setWindowStableSurfaceMaskFilename_Py( self._geogrid, self.winssmname)
        geogrid.setRO2VXFilename_Py( self._geogrid, self.winro2vxname)
        geogrid.setRO2VYFilename_Py( self._geogrid, self.winro2vyname)
        geogrid.setLookSide_Py(self._geogrid, self.lookSide)
        geogrid.setNodataOut_Py(self._geogrid, self.nodata_out)

        self._orbit  = self.orbit.exportToC()
        geogrid.setOrbit_Py(self._geogrid, self._orbit)

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

        from components.contrib.geo_autoRIFT.geogrid import geogrid
        from isceobj.Util import combinedlibmodule

        combinedlibmodule.freeCOrbit(self._orbit)
        self._orbit = None

        geogrid.destroyGeoGrid_Py(self._geogrid)
        self._geogrid = None

    def __init__(self):
        super(Geogrid, self).__init__()

        ##Radar image related parameters
        self.orbit = None
        self.sensingStart = None
        self.startingRange = None
        self.prf = None
        self.rangePixelSize = None
        self.numberOfSamples = None
        self.numberOfLines = None
        self.lookSide = None
        self.repeatTime = None
        self.incidenceAngle = None
        self.chipSizeX0 = None
        self.gridSpacingX = None

        ##Input related parameters
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
        self.epsg = None
        self._xlim = None
        self._ylim = None
        self.nodata_out = None

        ##Pointer to C 
        self._geogrid = None
        self._orbit = None

        ##parameters for autoRIFT
        self.pOff = None
        self.lOff = None
        self.pCount = None
        self.lCount = None
        self.X_res = None
        self.Y_res = None
