#!/usr/bin/env python3 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import isce
import xml.etree.ElementTree as ElementTree
import datetime
import isceobj
from isceobj.Util import Poly1D, Poly2D
from isceobj.Scene.Frame import Frame
from isceobj.Sensor.Sensor import Sensor
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import StateVector, Orbit
from isceobj.Orbit.OrbitExtender import OrbitExtender
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
from GRDProduct import GRDProduct
import os
import glob
import json
import numpy as np
import shelve
import re
import matplotlib.pyplot as plt

sep = "\n"
tab = "    "
lookMap = { 'RIGHT' : -1,
    'LEFT' : 1}


class Terrasar_GRD(Component):
    """
        A Class representing Terrasar data
    """
    def __init__(self):
        Component.__init__(self)        
        self.xml = None
        self.xml2 = None
        self.tiff = None
        self.bin = None
        self.orbitFile = None
        self.auxFile = None
        self.orbitDir = None
        self.auxDir = None
        #self.lutSigmaXml = None
        self.productXml = None
        self.noiseXml = None
        self.noiseCorrectionApplied = False
        self.noiseLUT = None
        #self.manifest = None
        #self.IPFversion = None
        self.slantRangeFile = None
        self._xml_root=None
        self._xml2_root=None
        self.output= None
        self.lutSigma = None
        self.product = GRDProduct()
        self.product.configure()
    
    #This is where the product.xml is read in
    def parse(self):

        try:
            with open(self.xml, 'r') as fid:
                xmlstring = fid.read()
                xmlstr = re.sub('\\sxmlns="[^"]+"', '', xmlstring, count=1)
                #print('firstxmlstr=',xmlstr)
        except:
            raise Exception('Could not read xml file {0}'.format(self.xml))
                
        with open(self.xml2, 'r') as fid:
            xmlstring2 = fid.read()
            xmlstr2 = re.sub('\\mainAnnotationFileName="[^"]+"', '', xmlstring2, count=1)
                #print('secondxmlstr=',xmlstr2)
                
                
        self._xml_root = ElementTree.fromstring(xmlstr)
        self._xml2_root = ElementTree.fromstring(xmlstr2)
        self.populateBbox()
        self.populateMetadata()

############

        ####Tru and locate an orbit file
        if self.orbitFile is None:
            if self.orbitDir is not None:
                self.orbitFile = self.findOrbitFile()
    
        
        ####Read in the orbits
        if self.orbitFile:
            orb = self.extractPreciseOrbit()
        else:
            orb = self.getOrbitFromXML()
        
        self.product.orbit.setOrbitSource('Header')
        for sv in orb:
            self.product.orbit.addStateVector(sv)
        
        ####Read in the gcps
        #if self.readGCPsFromXML:
            #gcps = self.readGCPsFromXML()
        
        
        ########
        #self.populateIPFVersion()
        self.extractlutSigma()
        #self.extractNoiseLUT()
        self.extractgr2sr()


########################



    def getxmlattr(self, path, key):
        try:
            res = self._xml_root.find(path).attrib[key]
        except:
            raise Exception('Cannot find attribute %s at %s'%(key, path))

        return res

    def getxmlvalue(self, root, path):
        try:
            res = root.find(path).text
        except:
            raise Exception('Tag= %s not found'%(path))

        if res is None:
            raise Exception('Tag = %s not found'%(path))

        return res

    def getxmlelement(self, root, path):
        try:
            res = root.find(path)
        except:
            raise Exception('Cannot find path %s'%(path))

        if res is None:
            raise Exception('Cannot find path %s'%(path))

        return res

    def convertToDateTime(self, string):
        dt = datetime.datetime.strptime(string,"%Y-%m-%dT%H:%M:%S.%fZ")
        return dt

    def convertToDateTime2(self, string):
        dt = datetime.datetime.strptime(string,"%Y-%m-%dT%H:%M:%S.%f")
        return dt

    def getFrame(self):
        return self.frame


    def populateMetadata(self):
        """
            Create metadata objects from the metadata files
        """
        ####Set each parameter one - by - one

        mission = self.getxmlvalue(self._xml_root, 'productInfo/missionInfo/mission')
        print(mission)
        swath = self.getxmlvalue(self._xml_root, 'productInfo/acquisitionInfo/elevationBeamConfiguration')
        polarization = self.getxmlvalue(self._xml_root, 'productInfo/acquisitionInfo/polarisationList/polLayer')
        print('swatch=',swath)
        orig_prf = float(self.getxmlvalue(self._xml_root, 'instrument/settings/settingRecord/PRF'))
        print('orig_prf=',orig_prf)
        #orbitnumber = int(self.getxmlvalue('adsHeader/absoluteOrbitNumber'))
        frequency = float(self.getxmlvalue(self._xml_root, 'instrument/radarParameters/centerFrequency'))
        #passDirection = self.getxmlvalue('sourceAttributes/orbitAndAttitude/orbitInformation/passDirection')
        
        groundRangePixelSize = float(self.getxmlvalue(self._xml_root, 'productInfo/imageDataInfo/imageRaster/rowSpacing'))
        azimuthPixelSize = float(self.getxmlvalue(self._xml_root, 'productInfo/imageDataInfo/imageRaster/columnSpacing'))
        mappingGridRow = float(self.getxmlvalue(self._xml_root, 'productSpecific/projectedImageInfo/mappingGridInfo/imageRaster/numberOfRows'))
        mappingGridColumn = float(self.getxmlvalue(self._xml_root, 'productSpecific/projectedImageInfo/mappingGridInfo/imageRaster/numberOfColumns'))
        mappingGrid = float(self.getxmlvalue(self._xml_root, 'productSpecific/projectedImageInfo/mappingGridInfo/imageRaster/rowSpacing'))
        calFactor = float(self.getxmlvalue(self._xml_root, 'calibration/calibrationConstant/calFactor'))
        
##################################

        MP_grid = np.fromfile(self.bin,dtype='>f').reshape(mappingGridRow,mappingGridColumn,2)
        grid = MP_grid[:,:,0]
        #grid_re = np.reshape(grid,(mappingGridRow*mappingGridColumn))
        grid_mean = np.mean(np.diff(grid[:,0]))
        print('grid_mean=',grid_mean)
        azimuthTimeInterval = (grid_mean * groundRangePixelSize)/(mappingGrid)
        print('azimuthTimeInterval=',azimuthTimeInterval)
        
##################################

        #orb = self.getOrbitFromXML()
        #self.product.orbit.setOrbitSource('Header')

        #print('orborb=',orb)
        #gcps = self.readGCPsFromXML()
        #print('gcps=',gcps)

        #azt = np.zeros((len(gcps),3), dtype=np.float)
        #nvalid = 0
        #for ind,gcp in enumerate(gcps):
            #try:
                #tt,rr = orb.geo2rdr(gcp[2:])
                #aztime = tt.hour * 3600 + tt.minute * 60 + tt.second + 1e-6 * tt.microsecond
                #azt[nvalid,:] = [gcp[0], gcp[1], aztime]  #line, pixel, time
                #nvalid += 1
                #print('nvalid=',nvalid)
                #print('aztime=',aztime)
                #print('azt=',azt)

            #except:
                #pass

###Fit linear polynomial
        #pp = np.polyfit(azt[:nvalid,0], azt[:nvalid,2],1)
        #azimuthTimeInterval = abs(pp[0])
        #print('azimuthTimeInterval=',azimuthTimeInterval)
        #print("Offset should be close to sensing start: ", datetime.timedelta(seconds=pp[0]))
        #gcp = [line, pixel, lat, lon, hgt]
####################################

        lines = int(self.getxmlvalue(self._xml_root, 'productInfo/imageDataInfo/imageRaster/numberOfRows'))
        print('lines=',lines)
        samples = int(self.getxmlvalue(self._xml_root, 'productInfo/imageDataInfo/imageRaster/numberOfColumns'))
        totalProcessedAzimuthBandwidth = self.getxmlvalue(self._xml_root, 'processing/processingParameter/totalProcessedAzimuthBandwidth')
        prf = totalProcessedAzimuthBandwidth # effective PRF can be double original, suggested by Piyush
        #slantRangeTime = float(self.getxmlvalue('imageAnnotation/imageInformation/slantRangeTime'))

        startingSlantRange = float(self.getxmlvalue(self._xml_root, 'productInfo/sceneInfo/rangeTime/firstPixel')) * (Const.c/2)
        incidenceAngle = float(self.getxmlvalue(self._xml_root, 'productInfo/sceneInfo/sceneCenterCoord/incidenceAngle'))
        slantRange = float(self.getxmlvalue(self._xml_root, 'productSpecific/complexImageInfo/projectedSpacingRange/slantRange'))
        

        #lineFlip =  self.getxmlvalue('imageAttributes/rasterAttributes/lineTimeOrdering').upper() == 'DECREASING'
        
        #print('lineFlip',lineFlip)
    
        #if lineFlip:
            #sensingStop = self.convertToDateTime(self.getxmlvalue('imageGenerationParameters/sarProcessingInformation/zeroDopplerTimeFirstLine'))
            #sensingStart = self.convertToDateTime(self.getxmlvalue('imageGenerationParameters/sarProcessingInformation/zeroDopplerTimeLastLine'))
        #else:
        sensingStart = self.convertToDateTime(self.getxmlvalue(self._xml_root, 'productInfo/sceneInfo/start/timeUTC'))
        sensingStop = self.convertToDateTime(self.getxmlvalue(self._xml_root, 'productInfo/sceneInfo/stop/timeUTC'))
        mid_utc_time = self.convertToDateTime(self.getxmlvalue(self._xml_root, 'productInfo/sceneInfo/sceneCenterCoord/azimuthTimeUTC'))
        timeUTC = self.convertToDateTime(self.getxmlvalue(self._xml_root, 'processing/doppler/dopplerCentroid/dopplerEstimate/timeUTC'))
        MidazimuthTimeUTC = self.convertToDateTime(self.getxmlvalue(self._xml_root, 'productInfo/sceneInfo/sceneCenterCoord/azimuthTimeUTC'))
        

        #lookSide = -1
        lookSide = lookMap[self.getxmlvalue(self._xml_root, 'productInfo/acquisitionInfo/lookDirection').upper()]
        print('lookSide=',lookSide)
        ###Read ascending node for phase calibration
       
        ###Noise correction
        #correctionApplied = self.getxmlvalue('imageGenerationParameters/sarProcessingInformation/radiometricSmoothingPerformed').upper() == 'TRUE'
        correctionApplied = False
        print('correctionApplied=',correctionApplied)

        #Populate Frame
        self.product.numberOfSamples = samples
        self.product.numberOfLines = lines
        self.product.startingGroundRange = 0.0
        self.product.startingSlantRange = startingSlantRange
        #self.product.trackNumber = ((orbitnumber-73)%175) + 1
        #self.product.orbitNumber = orbitnumber
        self.product.frameNumber = 1
        self.product.polarization = polarization
        self.product.prf = prf
        self.product.azimuthTimeInterval = azimuthTimeInterval
        self.product.calFactor = calFactor
        #self.product.passDirection = passDirection
        self.product.radarWavelength = Const.c / frequency
        self.product.groundRangePixelSize = groundRangePixelSize
        #self.product.ascendingNodeTime = ascTime
        #self.product.slantRangeTime = slantRangeTime
        self.product.sensingStart = sensingStart
        self.product.sensingStop = sensingStop
        self.product.SensingMid = mid_utc_time
        self.product.MidazimuthTimeUTC = MidazimuthTimeUTC
        self.product.slantRange = slantRange
        self.product.timeUTC = timeUTC
        self.noiseCorrectionApplied = correctionApplied


#This is where the GEOREF.xml is read in

    ############

    def populateBbox(self, margin=0.1):
        '''
        Populate the bounding box from metadata.
        '''

        glist = (self.getxmlelement(self._xml2_root, 'geolocationGrid'))
        lat = []
        lon = []
        for child in glist:
            for grandchild in child:
                string = ElementTree.tostring(child, encoding = 'unicode', method = 'xml')
                string = string.split("<")[1]
                string = string.split(">")[0]
                if string.startswith('gridPoint'):
                    #print('stringtwo=',string)
                    lat.append( float(child.find('lat').text))
                    lon.append( float(child.find('lon').text))
        self.product.bbox = [min(lat) - margin, max(lat) + margin, min(lon) - margin, max(lon) + margin]
        print('self.product.bbox=',self.product.bbox)
        return        

#################################

    def getOrbitFromXML(self):
        '''
        Populate orbit.
        '''
    
        orb = Orbit()
        orb.configure()
    
        for node in self._xml_root.find('platform/orbit'):
            if node.tag == 'stateVec':
                sv = StateVector()
                sv.configure()
                for z in node:
                    if z.tag == 'timeUTC':
                        timeStamp = self.convertToDateTime2(z.text)
                    elif z.tag == 'posX':
                        xPosition = float(z.text)
                    elif z.tag == 'posY':
                            yPosition = float(z.text)
                    elif z.tag == 'posZ':
                            zPosition = float(z.text)
                    elif z.tag == 'velX':
                            xVelocity = float(z.text)
                    elif z.tag == 'velY':
                            yVelocity = float(z.text)
                    elif z.tag == 'velZ':
                            zVelocity = float(z.text)
        
                sv.setTime(timeStamp)
                sv.setPosition([xPosition, yPosition, zPosition])
                sv.setVelocity([xVelocity, yVelocity, zVelocity])
                orb.addStateVector(sv)
                print('sv=',sv)
    
    
        orbExt = OrbitExtender(planet=Planet(pname='Earth'))
        orbExt.configure()
        newOrb = orbExt.extendOrbit(orb)

        return newOrb

        self.product.orbit.setOrbitSource('Header')
        for sv in newOrb:
            self.product.orbit.addStateVector(sv)


    def readGCPsFromXML(self):
        '''
        Populate GCPS
        '''
        gcps = []
    
        for node in self._xml2_root.find('geolocationGrid'):
            if not node.tag == 'gridPoint':
                continue
    
            for zz in node:
                if zz.tag == 't':
                    az_time = float(zz.text)
                elif zz.tag == 'tau':
                    rg_time = float(zz.text)
                elif zz.tag == 'lat':
                    lat = float(zz.text)
                elif zz.tag == 'lon':
                    lon = float(zz.text)
                elif zz.tag == 'height':
                    hgt = float(zz.text)

            pt = [az_time, rg_time, lat, lon, hgt]
            gcps.append(pt)

        return gcps

  #====================================


    def extractlutSigma(self):
        '''
        Extract Sigma nought look up table from calibration file.
        '''
        node2 = []
        from scipy.interpolate import RectBivariateSpline
        from scipy.interpolate import interp1d
        from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

        for node in self._xml_root.find('calibration/calibrationData'):
            if not node.tag == 'antennaPattern':
                continue

            for z in node:
                if z.tag == 'elevationPattern':
                    for zz in z:
                        if zz.tag == 'gainExt':
                            node = float(zz.text)
                            node2.append(node)

        sigmadata = np.asarray(node2)
        self.lutSigma = sigmadata

        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(data)
            plt.colorbar()
            plt.show()

        return
            

    #====================================

    def extractgr2sr(self):
        '''
        Extract Slant Range to Ground Range polynomial
        '''
        from scipy.interpolate import interp1d
        from isceobj.Constants import SPEED_OF_LIGHT

        #node = self.getxmlelement(self._xml_root,'productSpecific/projectedImageInfo')
        #num = int(node.attrib['count'])
        lines = []
        data = []
        pp = []
        
        for node in self._xml_root.find('productSpecific'):
            if not node.tag == 'projectedImageInfo':
                continue
            for z in node:
                if z.tag == 'slantToGroundRangeProjection':
                    for zz in z:
                        if zz.tag == 'coefficient':
                            p = float(zz.text)*(SPEED_OF_LIGHT)
                            pp.append(p)

                    print('t0=',self.product.MidazimuthTimeUTC)
                    lines.append( (self.product.MidazimuthTimeUTC-self.product.sensingStart).total_seconds()/self.product.azimuthTimeInterval)
                    data.append(pp[::1]) #-1
                    print('lines123=',lines)
                    #print('data0=',data)
                    ###If polynomial starts beyond first line
                    if lines[0] > 0:
                        lines.insert(0, 0.)
                        pp = data[0]
                        data.insert(0,pp)
                        #print('data1=',data)
                    ####If polynomial ends before last line
                    if lines[-1] < (self.product.numberOfLines-1):
                        lines.append(self.product.numberOfLines-1.0)
                        pp = data[-1]
                        data.append(pp)
                        #print('data2=',data)
                    lines = np.array(lines)
                    data = np.array(data)
                    print('lines=',lines)
                    #print('data=',data)
                    LUT = []

                    for ii in range(data.shape[1]):
                        col = data[:,ii]
                        #print('col=',col)
                        LUT.append(interp1d(lines, col, bounds_error=False, assume_sorted=True))

                    self.gr2srLUT = LUT
                    print('LUT=',len(self.gr2srLUT))
                    if False:
                        import matplotlib.pyplot as plt
                        plt.figure()
                        plt.imshow(data)
                        plt.colorbar()
                        plt.show()

                    return
  #====================================


    def extractImage(self, parse=False, removeNoise=False, verbose=True):
        """
            Use gdal python bindings to extract image
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise Exception('GDAL python bindings not found. Need this for RSAT2/ TandemX / Sentinel1A.')
                        
        from scipy.interpolate import interp2d
                            
        if parse:
            self.parse()
                                    
        print('Extracting normalized image ....')
                                        
        src = gdal.Open(self.tiff.strip(), gdal.GA_ReadOnly)
        band = src.GetRasterBand(1)

        if self.product.numberOfSamples != src.RasterXSize:
            raise Exception('Width in metadata and image dont match')
                                                        
        if self.product.numberOfLines != src.RasterYSize:
            raise Exception('Length in metadata and image dont match')
                                                                
                                                                
        noiseFactor = 0.0
        if (not removeNoise) and self.noiseCorrectionApplied:
            print('User asked for data without noise corrections, but product appears to be corrected. Adding back noise from LUT ....')
            noiseFactor = 1.0
        elif removeNoise and (not self.noiseCorrectionApplied):
            print('User asked for data with noise corrections, but the products appears to not be corrected. Applying noise corrections from LUT ....')
            noiseFactor = -1.0
        elif (not removeNoise) and (not self.noiseCorrectionApplied):
            print('User asked for data without noise corrections. The product contains uncorrected data ... unpacking ....')
        else:
            print('User asked for noise corrected data and the product appears to be noise corrected .... unpacking ....')
                                                                                                            
        ###Write original GRD to file
        fid = open(self.output, 'wb')
        #lineFlip = self.getxmlvalue('imageAttributes/rasterAttributes/lineTimeOrdering').upper() == 'DECREASING'
        #pixFlip = self.getxmlvalue('imageAttributes/rasterAttributes/pixelTimeOrdering').upper() == 'DECREASING'

        lines = int(self.getxmlvalue(self._xml_root, 'productInfo/imageDataInfo/imageRaster/numberOfRows'))
        samples = int(self.getxmlvalue(self._xml_root, 'productInfo/imageDataInfo/imageRaster/numberOfColumns'))

        for ii in range(self.product.numberOfLines//lines + 1):  #initially lines was 100
            ymin = int(ii*lines)
            ymax = int(np.clip(ymin+lines,0, self.product.numberOfLines))
            #print('ymin=',ymin)
            #print('ymax=',ymax)

            if ymax == ymin:
                break
                                                                                                                                        
            #lin = np.arange(ymin,ymax)
            ####Read in one line of data
            data = 1.0 * band.ReadAsArray(0, ymin, self.product.numberOfSamples, ymax-ymin)
            #print('data=',data)
            lut = self.lutSigma
            calibrationFactor_dB = (self.product.calFactor)
            outdata = calibrationFactor_dB*(data*data)
            #outdata = (data*data)/(lut*lut) + noiseFactor * noise/(lut*lut)
            #outdata = 10 * np.log10(outdata)
            
            #if lineFlip:
                #if verbose:
                    #print('Vertically Flipping data')
                #outdata = np.flipud(outdata)

            #if pixFlip:
                #if verbose:
                    #print('Horizontally Flipping data')
                #outdata = np.fliplr(outdata)
   
            outdata.astype(np.float32).tofile(fid)
        fid.close()
            
        ####Render ISCE XML
        L1cImage = isceobj.createImage()
        L1cImage.setByteOrder('l')
        L1cImage.dataType = 'FLOAT'
        L1cImage.setFilename(self.output)
        L1cImage.setAccessMode('read')
        L1cImage.setWidth(self.product.numberOfSamples)
        L1cImage.setLength(self.product.numberOfLines)
        L1cImage.renderHdr()
        L1cImage.renderVRT()
        
        
        band = None
        src = None
        
        self.extractSlantRange()
        
        return
            
    def extractSlantRange(self):
        '''
        Extract pixel-by-pixel slant range file for GRD files.
        '''
        import numpy.polynomial.polynomial as poly
        from scipy import optimize
        
        if self.slantRangeFile is None:
            return
                
        print('Extracing slant range ....')


        ###Write original L1c to file
        fid = open(self.slantRangeFile, 'wb')
        pix = np.arange(self.product.numberOfSamples) * (self.product.groundRangePixelSize)

        lin = np.arange(self.product.numberOfLines)
        polys = np.zeros((self.product.numberOfLines, len(self.gr2srLUT)))
        for ii, intp in enumerate(self.gr2srLUT):
            polys[:,ii] = intp(lin)


        def func(Y):
            return self.A - self.A0 + self.B * Y + self.C * Y ** 2 + self.D * Y ** 3
        def fprime(Y):
            return self.B + 2 * self.C * Y + 3 * self.D * Y ** 2
        
        res = []
        for iii in range(self.product.numberOfSamples):
            pp = polys[iii,:]
            pp = pp[::1]
            print('pp=',pp)
            x0 = 600              #The initial value
            self.A = pp[0]
            self.B = pp[1]
            self.C = pp[2]
            self.D = pp[3]
            self.A0 = pix[iii]

            res1 = optimize.newton(func, 600, fprime=fprime, maxiter=2000)
            res.append(res1)
        outdata = np.tile(res,(self.product.numberOfLines,1))
        print('outdata=',outdata)
        outdata.tofile(fid)
        #for ii in range(self.product.numberOfLines):
            #pp = polys[ii,:]
            #outdata = np.polyval(res, pix)
            #print('outdata=',outdata)

            #outdata.tofile(fid)

        fid.close()

        ####Render ISCE XML
        L1cImage = isceobj.createImage()
        L1cImage.setByteOrder('l')
        L1cImage.dataType = 'DOUBLE'
        L1cImage.setFilename(self.slantRangeFile)
        L1cImage.setAccessMode('read')
        L1cImage.setWidth(self.product.numberOfSamples)
        L1cImage.setLength(self.product.numberOfLines)
        L1cImage.renderHdr()
        L1cImage.renderVRT()
        self.product.slantRangeimage = L1cImage


def createParser():
    import argparse

    parser = argparse.ArgumentParser( description = 'Radarsar parser' )

    parser.add_argument('-d', '--dirname', dest='dirname', type=str,
            default=None, help='SAFE format directory. (Recommended)')

    parser.add_argument('-o', '--outdir', dest='outdir', type=str,
            required=True, help='Output L1c prefix.')

    parser.add_argument('--orbitdir', dest='orbitdir', type=str,
            default=None, help = 'Directory with all the orbits')

    parser.add_argument('--auxdir', dest='auxdir', type=str,
            default=None, help = 'Directory with all the aux products')

    parser.add_argument('--denoise', dest='denoise', action='store_true',
            default=False, help = 'Add noise back if removed')
    
    return parser

def cmdLineParse(iargs=None):
    '''
    Command Line Parser.
    '''
    import glob
    import fnmatch

    parser = createParser()
    inps = parser.parse_args(args=iargs)
    print('inpssss=',inps.dirname)
    if inps.dirname is None:
        raise Exception('File is not provided')

    ####First find vv Product path
    swathid0 = 'GEOREF.xml'
    swathid = 'TSX*.xml'
    swathid1 = 'IMAGE*'
    swathid2 = 'MAPPING_GRID.bin'
    inps.data = {}
    #Find XML file
    patt = os.path.join('DOT_*', swathid)
    match = glob.glob(patt)
    if len(match) == 0:
        raise Exception('Target file {0} not found'.format(patt))
    inps.data['xml'] = match[0]

    #Find XML2 file
    patt = os.path.join('DOT_*/ANNOTATION', swathid0)
    match = glob.glob(patt)
    if len(match) == 0:
        raise Exception('Target file {0} not found'.format(patt))
    inps.data['xml2'] = match[0]
    
    #Find TIFF file
    patt = os.path.join('DOT_*/IMAGEDATA', swathid1+'*')
    match = glob.glob(patt)
    if len(match) == 0 :
        raise Exception('Target file {0} not found'.format(patt))
    inps.data['tiff'] = match[0]

    #Find Bin file
    patt = os.path.join('DOT_*/AUXRASTER', swathid2)
    match = glob.glob(patt)
    if len(match) == 0 :
        raise Exception('Target file {0} not found'.format(patt))
    inps.data['bin'] = match[0]

    return inps

        
def main(iargs=None):
    inps = cmdLineParse(iargs)

    rangeDone = False
    #for key, val in inps.data.items():

    #print('Processing polarization: ', key)
    #print(val)
    obj = Terrasar_GRD()
    obj.configure()
    obj.xml = inps.data['xml']
    obj.xml2 = inps.data['xml2']
    obj.tiff = inps.data['tiff']
    obj.bin = inps.data['bin']
    #obj.lutSigmaXml = inps.data['calibration']
    #obj.noiseXml = inps.data['noise']
    #obj.manifest = inps.manifest

    if not os.path.isdir(inps.outdir):
        os.mkdir(inps.outdir)
    else:
        print('Output directory {0} already exists.'.format(inps.outdir))

    obj.output = os.path.join(inps.outdir, 'Sigma0_' + 'vv' + '.img')

    if not rangeDone:
        obj.slantRangeFile = os.path.join(inps.outdir, 'slantRange.img')
        rangeDone = True

    obj.parse()
    obj.extractImage(removeNoise=inps.denoise)

    dbname = os.path.join(inps.outdir, 'metadata')
    with shelve.open(dbname) as db:
        db['vv'] = obj.product


if __name__ == '__main__':
    '''
    Main driver.
    '''

    main()

















