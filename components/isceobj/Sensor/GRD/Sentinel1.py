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
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import StateVector, Orbit
from isceobj.Orbit.OrbitExtender import OrbitExtender
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
from .GRDProduct import GRDProduct
import os
import glob
import json
import numpy as np
import shelve


SAFE = Component.Parameter('safe',
            public_name = 'safe',
            default = None,
            type = str,
            doc = 'SAFE folder with the GRD product')

POLARIZATION = Component.Parameter('polarization',
            public_name = 'polarization',
            default = None,
            type = str,
            doc = 'Polarization to unpack')

ORBIT_DIR = Component.Parameter('orbitDir',
            public_name = 'orbit directory',
            default = None,
            type = str,
            doc = 'Directory to search for orbit data')

ORBIT_FILE = Component.Parameter('orbitFile',
            public_name = 'orbit file',
            default = None,
            type = str,
            doc = 'External orbit file with state vectors')

OUTPUT = Component.Parameter('output',
            public_name = 'output directory',
            default = None,
            type = str,
            doc = 'Directory where the data gets unpacked')

SLANT_RANGE_FILE = Component.Parameter('slantRangeFile',
            public_name = 'slant range file',
            default = None,
            type = str,
            doc = 'Slant range file at full resolution')

####List of facilities
PRODUCT = Component.Facility('product',
            public_name='product',
            module = 'isceobj.Sensor.GRD',
            factory = 'createGRDProduct',
            args = (),
            mandatory = True,
            doc = 'GRD Product populated by the reader')

class Sentinel1(Component):
    """
        A Class representing Sentinel1 data
    """

    family = 's1grd'
    logging = 'isce.sensor.grd.s1'

    parameter_list = (SAFE,
                      POLARIZATION,
                      ORBIT_DIR,
                      ORBIT_FILE,
                      OUTPUT,
                      SLANT_RANGE_FILE)

    facility_list = (PRODUCT,)

    def __init__(self):
        Component.__init__(self)        
        self.xml = None
        self.tiff = None
        self.calibrationXml = None
        self.noiseXml = None

        self.manifest = None
        self.noiseCorrectionApplied = False

        self.betaLUT = None
        self.noiseLUT = None
        self.gr2srLUT = None


        self._xml_root=None


    def validateUserInputs(self):
        '''
        Validate inputs provided by user.
        Populat tiff and xml list using SAFE folder names.
        '''

        import zipfile
        import fnmatch

        if self.safe is None:
            raise Exception('SAFE directory is not provided')

        if self.polarization in ['',None]:
            raise Exception('Polarization is not provided')

        ###Check if zip file / unpacked directory is provided.
        iszipfile = False
        if self.safe.endswith('.zip'):
            iszipfile = True
            with zipfile.ZipFile(self.safe, 'r') as zf:
                namelist = zf.namelist()

   
        


        ####First find VV annotation file
        swathid = 's1?-iw-grd-' + self.polarization.lower()

        if iszipfile:
            #Find XML file
            patt = os.path.join('*.SAFE', 'annotation', swathid+'*')
            match = fnmatch.filter(namelist, patt)
   
            if len(match) == 0 :
                raise Exception('Annotation file for {0} not found in {1}'.format(self.polarization, self.safe))
            
            self.xml = os.path.join('/vsizip', self.safe, match[0]) 


            #Find TIFF file
            patt = os.path.join('*.SAFE', 'measurement', swathid+'*')
            match = fnmatch.filter(namelist, patt)

            if len(match) == 0 :
                raise Exception('Annotation file found for {0} but no measurement in {1}'.format(self.polarization, self.safe))

            self.tiff = os.path.join('/vsizip', self.safe, match[0])

            #Find Calibration file
            patt = os.path.join('*.SAFE', 'annotation', 'calibration', 'calibration-'+swathid+'*')
            match = fnmatch.filter(namelist, patt)

            if len(match) == 0 :
                raise Exception('Annotation file found for {0} but no calibration in {1}'.format(self.polarization, self.safe))

            self.calibrationXml = os.path.join('/vsizip', self.safe, match[0]) 

            #Find Noise file
            patt = os.path.join('*.SAFE', 'annotation', 'calibration', 'noise-'+swathid+'*')
            match = fnmatch.filter(namelist, patt)

            if len(match) == 0 :
                raise Exception('Annotation file found for {0} but no noise in {1}'.format(self.polarization, self.safe))

            self.noiseXml = os.path.join('/vsizip', self.safe, match[0])


            patt = os.path.join('*.SAFE', 'manifest.safe')
            match = fnmatch.filter(namelist, patt)
            if len(match) == 0:
                raise Exception('No manifest file found in {0}'.format(self.safe))
            self.manifest = os.path.join('/vsizip', self.safe, match[0])

        else:
           
            ####Find annotation file
            patt = os.path.join( self.safe, 'annotation', swathid + '*')
            match = glob.glob(patt)

            if len(match) == 0:
                raise Exception('Annotation file for {0} not found in {1}'.format(self.polarization, self.safe))

            self.xml = match[0]

            ####Find TIFF file
            patt = os.path.join( self.safe, 'measurement', swathid+'*')
            match = glob.glob(patt)
        
            if len(match) == 0:
                raise Exception('Annotation file found for {0} but not measurement in {1}'.format(self.polarization, self.safe))

            self.tiff= match[0]


            ####Find calibration file
            patt = os.path.join( self.safe, 'annotation', 'calibration', 'calibration-' + swathid + '*')
            match = glob.glob(patt)

            if len(match) == 0 :
                raise Exception('Annotation file found for {0} but not calibration in {1}'.format(self.polarization, self.safe))

            self.calibrationXml= match[0]


            ####Find noise file
            patt = os.path.join( self.safe, 'annotation', 'calibration', 'noise-' + swathid + '*')
            match = glob.glob(patt)

            if len(match) == 0 :
                raise Exception('Annotation file found for {0} but not noise in {1}'.format(self.polarization, self.safe))

            self.noiseXml = match[0]

	    ####Find manifest file
            self.manifest = os.path.join(self.safe, 'manifest.safe')



        print('XML: ', self.xml)
        print('TIFF: ', self.tiff)
        print('CALIB: ', self.calibrationXml)
        print('NOISE: ', self.noiseXml)
        print('MANIFEST: ', self.manifest)

        return


    def parse(self):
        import zipfile

        self.validateUserInputs()

        if '.zip' in self.xml:
            try:
                parts = self.xml.split(os.path.sep)
                zipname = os.path.join('/',*(parts[:-3]))
                fname = os.path.join(*(parts[-3:]))

                with zipfile.ZipFile(zipname, 'r') as zf:
                    xmlstr = zf.read(fname)
            except:
                raise Exception('Could not read xml file {0}'.format(self.xml))
        else:
            try:
                with open(self.xml, 'r') as fid:
                    xmlstr = fid.read()
            except:
                raise Exception('Could not read xml file {0}'.format(self.xml))

        self._xml_root = ElementTree.fromstring(xmlstr)

        self.populateMetadata()
        self.populateBbox()

        ####Try and locate an orbit file
        if self.orbitFile is None:
            if self.orbitDir is not None:
                self.orbitFile = self.findOrbitFile()
                print('Found this orbitfile: %s' %self.orbitFile)

        ####Read in the orbits
        if '_POEORB_' in self.orbitFile:
            orb = self.extractPreciseOrbit()
        elif '_RESORB_' in self.orbitFile:
            orb = self.extractOrbit()
        
        self.product.orbit.setOrbitSource('Header')
        for sv in orb:
            self.product.orbit.addStateVector(sv)

        self.populateIPFVersion()
        self.extractBetaLUT()
        self.extractNoiseLUT()

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
        dt = datetime.datetime.strptime(string,"%Y-%m-%dT%H:%M:%S.%f")
        return dt

    def populateMetadata(self):
        """
            Create metadata objects from the metadata files
        """
        ####Set each parameter one - by - one
        mission = self.getxmlvalue('adsHeader/missionId')
        swath = self.getxmlvalue('adsHeader/swath')
        polarization = self.getxmlvalue('adsHeader/polarisation')
        orbitnumber = int(self.getxmlvalue('adsHeader/absoluteOrbitNumber'))
        frequency = float(self.getxmlvalue('generalAnnotation/productInformation/radarFrequency'))
        passDirection = self.getxmlvalue('generalAnnotation/productInformation/pass')

        groundRangePixelSize = float(self.getxmlvalue('imageAnnotation/imageInformation/rangePixelSpacing'))
        azimuthPixelSize = float(self.getxmlvalue('imageAnnotation/imageInformation/azimuthPixelSpacing'))
        azimuthTimeInterval = float(self.getxmlvalue('imageAnnotation/imageInformation/azimuthTimeInterval'))

        lines = int(self.getxmlvalue('imageAnnotation/imageInformation/numberOfLines'))
        samples = int(self.getxmlvalue('imageAnnotation/imageInformation/numberOfSamples'))
        
        slantRangeTime = float(self.getxmlvalue('imageAnnotation/imageInformation/slantRangeTime'))
        startingSlantRange = float(self.getxmlvalue('imageAnnotation/imageInformation/slantRangeTime'))*Const.c/2.0
        incidenceAngle = float(self.getxmlvalue('imageAnnotation/imageInformation/incidenceAngleMidSwath'))
       

        sensingStart = self.convertToDateTime( self.getxmlvalue('imageAnnotation/imageInformation/productFirstLineUtcTime'))
        sensingStop = self.convertToDateTime( self.getxmlvalue('imageAnnotation/imageInformation/productLastLineUtcTime'))

        ####Sentinel is always right looking
        lookSide = -1

        ###Read ascending node for phase calibration
        ascTime = self.convertToDateTime(self.getxmlvalue('imageAnnotation/imageInformation/ascendingNodeTime'))
       
        ###Noise correction
        correctionApplied = self.getxmlvalue('imageAnnotation/processingInformation/thermalNoiseCorrectionPerformed').upper() == 'TRUE'

        self.product.lookSide = 'RIGHT'
        self.product.numberOfSamples = samples
        self.product.numberOfLines = lines
        self.product.startingGroundRange = 0.0
        self.product.startingSlantRange = startingSlantRange
        self.product.trackNumber = ((orbitnumber-73)%175) + 1 
        self.product.orbitNumber = orbitnumber 
        self.product.frameNumber = 1 
        self.product.polarization = polarization
        self.product.passDirection = passDirection
        self.product.radarWavelength = Const.c / frequency
        self.product.groundRangePixelSize = groundRangePixelSize
        self.product.azimuthPixelSize = azimuthPixelSize
        self.product.azimuthTimeInterval = azimuthTimeInterval
        self.product.ascendingNodeTime = ascTime
        self.product.slantRangeTime = slantRangeTime
        self.product.sensingStart = sensingStart
        self.product.sensingStop = sensingStop
        self.noiseCorrectionApplied = correctionApplied


    def populateBbox(self, margin=0.1):
        '''
        Populate the bounding box from metadata.
        '''

        glist = self.getxmlelement('geolocationGrid/geolocationGridPointList')

        lat = []
        lon = []

        for child in glist.getchildren():
            lat.append( float(child.find('latitude').text))
            lon.append( float(child.find('longitude').text))

        self.product.bbox = [min(lat) - margin, max(lat) + margin, min(lon) - margin, max(lon) + margin]
        print(self.product.bbox)
        return        

    def populateIPFVersion(self):
        '''
        Get IPF version from the manifest file.
        '''

        if self.manifest is None:
            return

        nsp = "{http://www.esa.int/safe/sentinel-1.0}"

        if '.zip' in self.manifest:

            import zipfile    
            parts = self.manifest.split(os.path.sep)
            zipname = os.path.join('/',*(parts[:-2]))
            fname = os.path.join(*(parts[-2:]))

            try:
                with zipfile.ZipFile(zipname, 'r') as zf:
                    xmlstr = zf.read(fname)
            except:
                raise Exception('Could not read manifest file : {0}'.format(self.manifest))
        else:
            try:
                with open(self.manifest, 'r') as fid:
                    xmlstr = fid.read()
            except:
                raise Exception('Could not read manifest file: {0}'.format(self.manifest))

        try:
            root = ElementTree.fromstring(xmlstr)
        
            elem = root.find('.//metadataObject[@ID="processing"]')
            rdict = elem.find('.//xmlData/' + nsp + 'processing/' + nsp + 'facility/' + nsp + 'software').attrib

            self.IPFversion = rdict['version']
            print('Setting IPF version to : ', self.IPFversion) 

        except:
            print('Could not read version number successfully from manifest file: ', self.manifest)
            pass
        return

    def findOrbitFile(self):
        '''
        Find correct orbit file in the orbit directory.
        '''

        datefmt = "%Y%m%dT%H%M%S"
        types = ['POEORB', 'RESORB']
        filelist = []
        match = []
        timeStamp = self.product.sensingStart+(self.product.sensingStop - self.product.sensingStart)/2.
        
        for orbType in types:
            files = glob.glob( os.path.join(self.orbitDir, 'S1A_OPER_AUX_' + orbType + '_OPOD*'))
            filelist.extend(files)
            ###List all orbit files

        for result in filelist:
            fields = result.split('_')
            taft = datetime.datetime.strptime(fields[-1][0:15], datefmt)
            tbef = datetime.datetime.strptime(fields[-2][1:16], datefmt)
            print(taft, tbef)
                
            #####Get all files that span the acquisition
            if (tbef <= timeStamp) and (taft >= timeStamp):
                tmid = tbef + 0.5 * (taft - tbef)
                match.append((result, abs((timeStamp-tmid).total_seconds())))
            #####Return the file with the image is aligned best to the middle of the file
            if len(match) != 0:
                bestmatch = min(match, key = lambda x: x[1])
                return bestmatch[0]

       
        if len(match) == 0:
            raise Exception('No suitable orbit file found. If you want to process anyway - unset the orbitdir parameter')

    def extractOrbit(self):
        '''
        Extract orbit information from xml node.
        '''
        node = self._xml_root.find('generalAnnotation/orbitList')
        
        print('Extracting orbit from annotation XML file')
        frameOrbit = Orbit()
        frameOrbit.configure()

        for child in node.getchildren():
            timestamp = self.convertToDateTime(child.find('time').text)
            pos = []
            vel = []
            posnode = child.find('position')
            velnode = child.find('velocity')
            for tag in ['x','y','z']:
                pos.append(float(posnode.find(tag).text))

            for tag in ['x','y','z']:
                vel.append(float(velnode.find(tag).text))

            vec = StateVector()
            vec.setTime(timestamp)
            vec.setPosition(pos)
            vec.setVelocity(vel)
            frameOrbit.addStateVector(vec)

        return frameOrbit
            
    def extractPreciseOrbit(self):
        '''
        Extract precise orbit from given Orbit file.
        '''
        try:
            fp = open(self.orbitFile,'r')
        except IOError as strerr:
            print("IOError: %s" % strerr)
            return

        _xml_root = ElementTree.ElementTree(file=fp).getroot()

        node = _xml_root.find('Data_Block/List_of_OSVs')

        orb = Orbit()
        orb.configure()

        margin = datetime.timedelta(seconds=40.0)
        tstart = self.product.sensingStart - margin
        tend = self.product.sensingStop + margin

        for child in node.getchildren():
            timestamp = self.convertToDateTime(child.find('UTC').text[4:])

            if (timestamp >= tstart) and (timestamp < tend):

                pos = [] 
                vel = []

                for tag in ['VX','VY','VZ']:
                    vel.append(float(child.find(tag).text))

                for tag in ['X','Y','Z']:
                    pos.append(float(child.find(tag).text))

                vec = StateVector()
                vec.setTime(timestamp)
                vec.setPosition(pos)
                vec.setVelocity(vel)
#                print(vec)
                orb.addStateVector(vec)

        fp.close()

        return orb


    def extractBetaLUT(self):
        '''
        Extract Beta nought look up table from calibration file.
        '''

        from scipy.interpolate import RectBivariateSpline

        if self.calibrationXml is None:
            raise Exception('No calibration file provided')

        if '.zip' in self.calibrationXml:
            import zipfile
            parts = self.calibrationXml.split(os.path.sep)
            zipname = os.path.join('/',*(parts[:-4]))
            fname = os.path.join(*(parts[-4:]))
            
            try:
                with zipfile.ZipFile(zipname, 'r') as zf:
                    xmlstr = zf.read(fname)
            except:
                raise Exception('Could not read calibration file: {0}'.format(self.calibrationXml))
        else:
            try:
                with open(self.calibrationXml, 'r') as fid:
                    xmlstr = fid.read()
            except:
                raise Exception('Could not read calibration file: {0}'.format(self.calibrationXml))

        _xml_root = ElementTree.fromstring(xmlstr)

        node = _xml_root.find('calibrationVectorList')
        num = int(node.attrib['count'])

        lines = []
        pixels = []
        data = None

        for ii, child in enumerate(node.getchildren()):
            pixnode = child.find('pixel')
            nump = int(pixnode.attrib['count'])

            if ii==0:
                data = np.zeros((num,nump))
                pixels = [float(x) for x in pixnode.text.split()]


            lines.append( int(child.find('line').text))
            signode = child.find('betaNought')
            data[ii,:] = [float(x) for x in signode.text.split()]

      
        lines = np.array(lines)
        pixels = np.array(pixels)

        self.betaLUT = RectBivariateSpline(lines,pixels, data, kx=1, ky=1)

        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(data)
            plt.colorbar()
            plt.show()

        return

    def extractNoiseLUT(self):
        '''
        Extract Noise look up table from calibration file.
        '''

        if not self.noiseCorrectionApplied:
            self.noiseLUT = 0.0
            return
		

        from scipy.interpolate import RectBivariateSpline

        if self.noiseXml is None:
            raise Exception('No calibration file provided')

        if self.noiseXml.startswith('/vsizip'):
            import zipfile
            parts = self.noiseXml.split(os.path.sep)
            zipname = os.path.join(*(parts[2:-4]))
            fname = os.path.join(*(parts[-4:]))

            try:
                with zipfile.ZipFile(zipname, 'r'):
                    xmlstr = zf.read(fname)
            except:
                raise Exception('Could not read noise file: {0}'.format(self.calibrationXml))
        else:
            try:
                with open(self.noiseXml, 'r') as fid:
                    xmlstr = fid.read()
            except:
                raise Exception('Could not read noise file: {0}'.format(self.calibrationXml))

        _xml_root = ElementTree.fromstring(xmlstr)

        node = _xml_root.find('noiseVectorList')
        num = int(node.attrib['count'])

        lines = []
        pixels = []
        data = None

        for ii, child in enumerate(node.getchildren()):
            pixnode = child.find('pixel')
            nump = int(pixnode.attrib['count'])

            if ii==0:
                data = np.zeros((num,nump))
                pixels = [float(x) for x in pixnode.text.split()]


            lines.append( int(child.find('line').text))
            signode = child.find('noiseLut')
            data[ii,:] = [float(x) for x in signode.text.split()]

        fp.close()
        lines = np.array(lines)
        pixels = np.array(pixels)

        self.noiseLUT = RectBivariateSpline(lines,pixels, data, kx=1,ky=1)

        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(data)
            plt.colorbar()
            plt.show()

        return

    def extractImage(self, parse=False, removeNoise=False):
        """
           Use gdal python bindings to extract image
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise Exception('GDAL python bindings not found. Need this for RSAT2/ TandemX / Sentinel1.')

        from scipy.interpolate import interp2d

        if parse:
            self.parse()

        print('Extracting normalized image ....')

        src = gdal.Open('/vsizip//'+self.tiff.strip(), gdal.GA_ReadOnly)
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
          
        ###Write original SLC to file
        fid = open(self.output, 'wb')
        pix = np.arange(self.product.numberOfSamples)

        for ii in range(self.product.numberOfLines//100 + 1):
            ymin = int(ii*100)
            ymax = int(np.clip(ymin+100,0, self.product.numberOfLines))

            if ymax == ymin:
                break

            lin = np.arange(ymin,ymax)
            ####Read in one line of data
            data = 1.0 * band.ReadAsArray(0, ymin, self.product.numberOfSamples, ymax-ymin)

            lut = self.betaLUT(lin,pix,grid=True)
	    
            if noiseFactor != 0.0:
                noise = self.noiseLUT(lin,pix,grid=True)
            else:
                noise = 0.0


            #outdata = data
            outdata = data*data/(lut*lut) + noiseFactor * noise/(lut*lut)
            #outdata = 10 * np.log10(outdata)

            outdata.astype(np.float32).tofile(fid)

        fid.close()
               

        ####Render ISCE XML
        slcImage = isceobj.createImage()
        slcImage.setByteOrder('l')
        slcImage.dataType = 'FLOAT'
        slcImage.setFilename(self.output)
        slcImage.setAccessMode('read')
        slcImage.setWidth(self.product.numberOfSamples)
        slcImage.setLength(self.product.numberOfLines)
        slcImage.renderHdr()
        slcImage.renderVRT()
        self.product.image = slcImage 

        band = None
        src = None

        return


    def extractSlantRange(self, full=True):
        '''
        Extract pixel-by-pixel slant range file for GRD images.
        '''

        print('Extracing slant range ....')
        from scipy.interpolate import interp1d
        from isceobj.Util import Poly1D

        node = self._xml_root.find('coordinateConversion/coordinateConversionList')
        num = int(node.attrib['count'])

        lines = []
        data = []


        for ii, child in enumerate(node.getchildren()):
            t0 = self.convertToDateTime(child.find('azimuthTime').text)
           
            lines.append( (t0-self.product.sensingStart).total_seconds()/self.product.azimuthTimeInterval)

            
            pp =  [float(x) for x in child.find('grsrCoefficients').text.split()]
    
            meangr = float( child.find('gr0').text)
            if meangr !=0 :
                raise Exception('Ground range image does not start at zero coordinate')



            data.append(pp[::-1])

        ###If polynomial starts beyond first line
        if lines[0] > 0:
            lines.insert(0, 0.)
            pp = data[0]
            data.insert(0,pp) 

        ####If polynomial ends before last line
        if lines[-1] < (self.product.numberOfLines-1):
            lines.append(self.product.numberOfLines-1.0)
            pp = data[-1]
            data.append(pp)


        lines = np.array(lines)
        data = np.array(data)

        LUT = []

        for ii in range(data.shape[1]):
            col = data[:,ii]
            LUT.append(interp1d(lines, col, bounds_error=False, assume_sorted=True))

            
        self.gr2srLUT = LUT

        ###Write original SLC to file
        fid = open(self.slantRangeFile, 'wb')
        pix = np.arange(self.product.numberOfSamples) * self.product.groundRangePixelSize
        lin = np.arange(self.product.numberOfLines)

        polys = np.zeros((self.product.numberOfLines, len(self.gr2srLUT)))

        for ii, intp in enumerate(self.gr2srLUT):
            polys[:,ii] = intp(lin)


        minrng = 1e11
        maxrng = -1e11

        for ii in range(self.product.numberOfLines):
            pp = polys[ii,:]
            outdata = np.polyval(pp, pix)
            minrng = min(minrng, outdata[0])   ###Can be made more efficient
            maxrng = max(maxrng, outdata[-1])

            outdata.tofile(fid)

        fid.close()

        self.product.startingSlantRange = minrng
        self.product.endingSlantRange = maxrng


        ####Render ISCE XML
        slcImage = isceobj.createImage()
        slcImage.setByteOrder('l')
        slcImage.dataType = 'DOUBLE'
        slcImage.setFilename(self.slantRangeFile)
        slcImage.setAccessMode('read')
        slcImage.setWidth(self.product.numberOfSamples)
        slcImage.setLength(self.product.numberOfLines)
        slcImage.renderHdr()
        slcImage.renderVRT()
        self.product.slantRangeImage = slcImage 
