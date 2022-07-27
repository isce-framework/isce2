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


class Radarsat2_GRD(Component):
    """
        A Class representing RadarSAT 2 data
    """
    def __init__(self):
        Component.__init__(self)        
        self.xml = None
        self.tiff = None
        self.orbitFile = None
        self.auxFile = None
        self.orbitDir = None
        self.auxDir = None
        self.lutSigmaXml = None
        self.productXml = None
        self.noiseXml = None
        self.noiseCorrectionApplied = False
        self.noiseLUT = None
        #self.manifest = None
        #self.IPFversion = None
        self.slantRangeFile = None
        self._xml_root=None
        self.output= None
        self.lutSigma = None
        self.product = GRDProduct()
        self.product.configure()
    
                                               
    def parse(self):

        try:
            with open(self.xml, 'r') as fid:
                xmlstring = fid.read()
                xmlstr = re.sub('\\sxmlns="[^"]+"', '', xmlstring, count=1)
        except:
            raise Exception('Could not read xml file {0}'.format(self.xml))

        self._xml_root = ElementTree.fromstring(xmlstr)

        self.populateMetadata()
        self.populateBbox()
        
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
        if self.readGCPsFromXML:
            gcps = self.readGCPsFromXML()
        
        
        ########
        self.extractlutSigma()
        #self.extractNoiseLUT()
        self.extractgr2sr()


    def getxmlattr(self, path, key):
        try:
            res = self._xml_root.find(path).attrib[key]
        except:
            raise Exception('Cannot find attribute %s at %s'%(key, path))

        return res

    def getxmlvalue(self, path):
        try:
            res = self._xml_root.find(path).text
        except:
            raise Exception('Tag= %s not found'%(path))

        if res is None:
            raise Exception('Tag = %s not found'%(path))

        return res

    def getxmlelement(self, path):
        try:
            res = self._xml_root.find(path)
        except:
            raise Exception('Cannot find path %s'%(path))

        if res is None:
            raise Exception('Cannot find path %s'%(path))

        return res

    def convertToDateTime(self, string):
        dt = datetime.datetime.strptime(string,"%Y-%m-%dT%H:%M:%S.%fZ")
        return dt

    def getFrame(self):
        return self.frame


    def populateMetadata(self):
        """
            Create metadata objects from the metadata files
        """
        ####Set each parameter one - by - one

        mission = self.getxmlvalue('sourceAttributes/satellite')
        swath = self.getxmlvalue('sourceAttributes/radarParameters/beams')
        polarization = self.getxmlvalue('sourceAttributes/radarParameters/polarizations')
        orig_prf = float(self.getxmlvalue('sourceAttributes/radarParameters/pulseRepetitionFrequency'))
        #orbitnumber = int(self.getxmlvalue('adsHeader/absoluteOrbitNumber'))
        frequency = float(self.getxmlvalue('sourceAttributes/radarParameters/radarCenterFrequency'))
        #passDirection = self.getxmlvalue('sourceAttributes/orbitAndAttitude/orbitInformation/passDirection')

        groundRangePixelSize = float(self.getxmlvalue('imageAttributes/rasterAttributes/sampledPixelSpacing'))
        azimuthPixelSize = float(self.getxmlvalue('imageAttributes/rasterAttributes/sampledLineSpacing'))
        rank = self.getxmlvalue('sourceAttributes/radarParameters/rank')
##################################

        orb = self.getOrbitFromXML()
        self.product.orbit.setOrbitSource('Header')

        gcps = self.readGCPsFromXML()
        #print('gcps=',gcps)

        azt = np.zeros((len(gcps),3), dtype=np.float)
        nvalid = 0
        for ind,gcp in enumerate(gcps):
            try:
                tt,rr = orb.geo2rdr(gcp[2:])
                aztime = tt.hour * 3600 + tt.minute * 60 + tt.second + 1e-6 * tt.microsecond
                azt[nvalid,:] = [gcp[0], gcp[1], aztime]  #line, pixel, time
                nvalid += 1
                #print('nvalid=',nvalid)
                #print('aztime=',aztime)
                #print('azt=',azt)

            except:
                pass

###Fit linear polynomial
        pp = np.polyfit(azt[:nvalid,0], azt[::-1,2],1)
        azimuthTimeInterval = abs(pp[0])
        print('azimuthTimeInterval=',azimuthTimeInterval)
        #print("Offset should be close to sensing start: ", datetime.timedelta(seconds=pp[0]))
        #gcp = [line, pixel, lat, lon, hgt]
####################################

        lines = int(self.getxmlvalue('imageAttributes/rasterAttributes/numberOfLines'))
        print('startlines=',lines)
        samples = int(self.getxmlvalue('imageAttributes/rasterAttributes/numberOfSamplesPerLine'))
        totalProcessedAzimuthBandwidth = self.getxmlvalue('imageGenerationParameters/sarProcessingInformation/totalProcessedAzimuthBandwidth')
        prf = totalProcessedAzimuthBandwidth # effective PRF can be double original, suggested by Piyush
        #slantRangeTime = float(self.getxmlvalue('imageAnnotation/imageInformation/slantRangeTime'))

        startingSlantRange = float(self.getxmlvalue('imageGenerationParameters/slantRangeToGroundRange/slantRangeTimeToFirstRangeSample')) * (Const.c/2)
        incidenceAngle = (float(self.getxmlvalue('imageGenerationParameters/sarProcessingInformation/incidenceAngleNearRange')) + float(self.getxmlvalue('imageGenerationParameters/sarProcessingInformation/incidenceAngleFarRange')))/2.0



        lineFlip =  self.getxmlvalue('imageAttributes/rasterAttributes/lineTimeOrdering').upper() == 'DECREASING'
        
        print('lineFlip',lineFlip)
    
        if lineFlip:
            sensingStop = self.convertToDateTime(self.getxmlvalue('imageGenerationParameters/sarProcessingInformation/zeroDopplerTimeFirstLine'))
            sensingStart = self.convertToDateTime(self.getxmlvalue('imageGenerationParameters/sarProcessingInformation/zeroDopplerTimeLastLine'))
        else:
            sensingStart = self.convertToDateTime(self.getxmlvalue('imageGenerationParameters/sarProcessingInformation/zeroDopplerTimeFirstLine'))
            sensingStop = self.convertToDateTime(self.getxmlvalue('imageGenerationParameters/sarProcessingInformation/zeroDopplerTimeLastLine'))
        
        

        ####Radarsat 2 me be right looking or left looking
        #lookSide = -1
        lookSide = lookMap[self.getxmlvalue('sourceAttributes/radarParameters/antennaPointing').upper()]
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

        #self.product.passDirection = passDirection
        self.product.radarWavelength = Const.c / frequency
        self.product.groundRangePixelSize = groundRangePixelSize
        #self.product.ascendingNodeTime = ascTime
        #self.product.slantRangeTime = slantRangeTime
        self.product.sensingStart = sensingStart
        self.product.sensingStop = sensingStop
        self.noiseCorrectionApplied = correctionApplied

    def populateBbox(self, margin=0.1):
        '''
        Populate the bounding box from metadata.
        '''

        glist = (self.getxmlelement('imageAttributes/geographicInformation/geolocationGrid'))
        lat = []
        lon = []
        for child in glist:
            for grandchild in child:
                string = ElementTree.tostring(grandchild, encoding = 'unicode', method = 'xml')
                string = string.split("<")[1]
                string = string.split(">")[0]
                if string == 'geodeticCoordinate':
                    lat.append( float(grandchild.find('latitude').text))
                    lon.append( float(grandchild.find('longitude').text))
        self.product.bbox = [min(lat) - margin, max(lat) + margin, min(lon) - margin, max(lon) + margin]
        print('coordinate=',self.product.bbox)

        return        

#################################

    def getOrbitFromXML(self):
        '''
        Populate orbit.
        '''
    
        orb = Orbit()
        orb.configure()
    
        for node in self._xml_root.find('sourceAttributes/orbitAndAttitude/orbitInformation'):
            if node.tag == 'stateVector':
                sv = StateVector()
                sv.configure()
                for z in node:
                    if z.tag == 'timeStamp':
                        timeStamp = self.convertToDateTime(z.text)
                    elif z.tag == 'xPosition':
                        xPosition = float(z.text)
                    elif z.tag == 'yPosition':
                            yPosition = float(z.text)
                    elif z.tag == 'zPosition':
                            zPosition = float(z.text)
                    elif z.tag == 'xVelocity':
                            xVelocity = float(z.text)
                    elif z.tag == 'yVelocity':
                            yVelocity = float(z.text)
                    elif z.tag == 'zVelocity':
                            zVelocity = float(z.text)
        
                sv.setTime(timeStamp)
                sv.setPosition([xPosition, yPosition, zPosition])
                sv.setVelocity([xVelocity, yVelocity, zVelocity])
                orb.addStateVector(sv)
    
    
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
    
        for node in self._xml_root.find('imageAttributes/geographicInformation/geolocationGrid'):
            if not node.tag == 'imageTiePoint':
                continue
    
            for z in node:
                if z.tag == 'imageCoordinate':
                    for zz in z:
                        if zz.tag == 'line':
                            line = float(zz.text)
                        elif zz.tag == 'pixel':
                            pixel = float(zz.text)
        
                if z.tag == 'geodeticCoordinate':
                    for zz in z:
                        if zz.tag == 'latitude':
                            lat = float(zz.text)
                        elif zz.tag == 'longitude':
                            lon = float(zz.text)
                        elif zz.tag == 'height':
                            hgt = float(zz.text)

            pt = [line, pixel, lat, lon, hgt]
            gcps.append(pt)

        return gcps
#######################################

    def extractlutSigma(self, iargs=None):
        '''
        Extract Sigma nought look up table from calibration file.
        '''

        from scipy.interpolate import RectBivariateSpline
        from scipy.interpolate import interp1d
        from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

        if self.lutSigmaXml is None:
            raise Exception('No calibration file provided')

        try:
            with open(self.lutSigmaXml, 'r') as fid:
                xmlstr = fid.read()

        except:
                raise Exception('Could not read calibration file: {0}'.format(self.lutSigmaXml))
        _xml_root = ElementTree.fromstring(xmlstr)
        #print(_xml_root)
        node = _xml_root.find('gains').text
        node = node.split(' ')
        node = [float(x) for x in node]
        #data = None


        #nump = len(node) #this is the length of gains
        #numps = list(range(nump))
        sigmadata = np.asarray(node)
        self.lutSigma = sigmadata
        #self.lutSigma = interp1d(numps, sigmadata, kind='cubic')
        #self.lutSigma = InterpolatedUnivariateSpline(numps, sigmadata)

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

        node = self._xml_root.find('imageGenerationParameters')
        #num = int(node.attrib['count'])

        lines = []
        data = []

        for child in node:
            for child in node:
                string = ElementTree.tostring(child, encoding = 'unicode', method = 'xml')
                string = string.split("<")[1]
                string = string.split(">")[0]
                if string == 'slantRangeToGroundRange':

                    t0 = self.convertToDateTime(child.find('zeroDopplerAzimuthTime').text)
                    lines.append( (t0-self.product.sensingStart).total_seconds()/self.product.azimuthTimeInterval)

                    pp =  [float(x) for x in child.find('groundToSlantRangeCoefficients').text.split()]
                    meangr = float( child.find('groundRangeOrigin').text)
                    if meangr !=0 :
                        raise Exception('Ground range image does not start at zero coordinate')

                    data.append(pp[::-1])
                    #print('lines123=',lines)
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
                    #print('lines=',lines)
                    #print('data=',data)
                    LUT = []

                    for ii in range(data.shape[1]):
                        col = data[:,ii]
                        #print('col=',col)
                        LUT.append(interp1d(lines, col, bounds_error=False, assume_sorted=True))
                
                    self.gr2srLUT = LUT
                    #print('lines=',len(self.gr2srLUT))
                    #print('data=',data)
                    if False:
                        import matplotlib.pyplot as plt
                        plt.figure()
                        plt.imshow(data)
                        plt.colorbar()
                        plt.show()

                    return

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
        lineFlip = self.getxmlvalue('imageAttributes/rasterAttributes/lineTimeOrdering').upper() == 'DECREASING'
        pixFlip = self.getxmlvalue('imageAttributes/rasterAttributes/pixelTimeOrdering').upper() == 'DECREASING'

        lines = int(self.getxmlvalue('imageAttributes/rasterAttributes/numberOfLines'))
        pix = np.arange(self.product.numberOfSamples)
        lin =np.arange(self.product.numberOfLines)

        for ii in range(self.product.numberOfLines//lines + 1):
            ymin = int(ii*lines)
            ymax = int(np.clip(ymin+lines,0, self.product.numberOfLines))
            print('ymin=',ymin)
            print('ymax=',ymax)
            if ymax == ymin:
                break
                                                                                                                                        
            lin = np.arange(ymin,ymax)
            ####Read in one line of data
            data = 1.0 * band.ReadAsArray(0, ymin, self.product.numberOfSamples, ymax-ymin)
            #lut = self.lutSigma(lin,pix,grid=True)
            lut = self.lutSigma
            if noiseFactor != 0.0:
                noise = self.noiseLUT(lin,pix)
                #noise = self.noiseLUT(lin,pix,grid=True)

            else:
                noise = 0.0
                                                                                                                                                                    
            #outdata = data
            outdata = data*data/(lut*lut) + noiseFactor * noise/(lut*lut)
            #outdata = 10 * np.log10(outdata)
            
            if lineFlip:
                if verbose:
                    print('Vertically Flipping data')
                outdata = np.flipud(outdata)

            if pixFlip:
                if verbose:
                    print('Horizontally Flipping data')
                outdata = np.fliplr(outdata)
   
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
            
        if self.slantRangeFile is None:
            return
                
        print('Extracing slant range ....')


        ###Write original L1c to file
        fid = open(self.slantRangeFile, 'wb')
        pix = np.arange(self.product.numberOfSamples) * self.product.groundRangePixelSize
        lin = np.arange(self.product.numberOfLines)

        polys = np.zeros((self.product.numberOfLines, len(self.gr2srLUT)))

        for ii, intp in enumerate(self.gr2srLUT):
            polys[:,ii] = intp(lin)


        for ii in range(self.product.numberOfLines):
            pp = polys[ii,:]
            outdata = np.polyval(pp, pix)
            outdata.tofile(fid)

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
    import fnmatch
    import glob
    
    parser = createParser()
    inps = parser.parse_args(args=iargs)
    print('inpssss=',inps.dirname)
    if inps.dirname is None:
        raise Exception('File is not provided')

    ####First find HH Product path
    swathid = 'product.xml'
    swathid1 = 'imagery_HH.'
    swathid2 = 'lutSigma.xml'
    inps.data = {}
    #Find XML file
    patt = os.path.join('RS2_*', swathid)
    match = glob.glob(patt)
    if len(match) == 0:
        raise Exception('Target file {0} not found'.format(patt))
    inps.data['xml'] = match[0]

    #Find TIFF file
    patt = os.path.join('RS2_*', swathid1+'*')
    match = glob.glob(patt)
    if len(match) == 0 :
        raise Exception('Target file {0} not found'.format(patt))
    inps.data['tiff'] = match[0]

    #Find Calibration file
    patt = os.path.join('RS2_*', swathid2)
    match = glob.glob(patt)
    if len(match) == 0 :
        raise Exception('Target file {0} not found'.format(patt))
    inps.data['calibration'] = match[0]

    return inps

def main(iargs=None):
    inps = cmdLineParse(iargs)

    rangeDone = False

    obj = Radarsat2_GRD()
    obj.configure()
    obj.xml = inps.data['xml']
    obj.tiff = inps.data['tiff']
    obj.lutSigmaXml = inps.data['calibration']

    if not os.path.isdir(inps.outdir):
        os.mkdir(inps.outdir)
    else:
        print('Output directory {0} already exists.'.format(inps.outdir))

    obj.output = os.path.join(inps.outdir, 'Sigma0_' + 'HH' + '.img')

    if not rangeDone:
        obj.slantRangeFile = os.path.join(inps.outdir, 'slantRange.img')
        rangeDone = True

    obj.parse()
    obj.extractImage(removeNoise=inps.denoise)

    dbname = os.path.join(inps.outdir, 'metadata')
    with shelve.open(dbname) as db:
        db['HH'] = obj.product


if __name__ == '__main__':
    '''
    Main driver.
    '''

    main()

















