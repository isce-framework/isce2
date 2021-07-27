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



# APR. 02, 2015    add the ability to extract Restituted Orbit
#                  by Cunren Liang
#
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import xml.etree.ElementTree as ET
import datetime
import isceobj
from isceobj.Scene.Frame import Frame
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import StateVector, Orbit
from isceobj.Orbit.OrbitExtender import OrbitExtender
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
import os
import glob
import numpy as np

sep = "\n"
tab = "    "
lookMap = { 'RIGHT' : -1,
            'LEFT' : 1}

XML = Component.Parameter('xml',
        public_name = 'xml',
        default = None,
        type = str,
        doc = 'Input XML file')

TIFF = Component.Parameter('tiff',
        public_name = 'tiff',
        default = None,
        type = str,
        doc = 'Input Tiff file')

MANIFEST = Component.Parameter('manifest',
        public_name = 'manifest',
        default = None,
        type = str,
        doc = 'Manifest file with IPF version')

SAFE = Component.Parameter('safe',
        public_name = 'safe',
        default = None,
        type = str,
        doc = 'SAFE folder / zip file')

ORBIT_FILE = Component.Parameter('orbitFile',
        public_name = 'orbit file',
        default = None,
        type = str,
        doc = 'External orbit file with state vectors')

ORBIT_DIR = Component.Parameter('orbitDir',
        public_name = 'orbit directory',
        default = None,
        type = str,
        doc = 'Directory to search for orbit files')

POLARIZATION = Component.Parameter('polarization',
        public_name = 'polarization',
        default = 'vv',
        type = str,
        mandatory = True,
        doc = 'Polarization')

from .Sensor import Sensor
class Sentinel1(Sensor):
    """
        A Class representing Sentinel1 StripMap data
    """

    family = 's1sm'
    logging = 'isce.sensor.S1_SM'

    parameter_list = (  XML,
                        TIFF,
                        MANIFEST,
                        SAFE,
                        ORBIT_FILE,
                        ORBIT_DIR,
                        POLARIZATION,) + Sensor.parameter_list

    def __init__(self,family='',name=''):

        super(Sentinel1,self).__init__(family if family else  self.__class__.family, name=name)
        
        self.frame = Frame()
        self.frame.configure()
    
        self._xml_root=None
    

    def validateUserInputs(self):
        '''
        Validate inputs from user.
        Populate tiff and xml from SAFE folder name.
        '''

        import fnmatch
        import zipfile

        if not self.xml:
            if not self.safe:
                raise Exception('SAFE directory is not provided')


        ####First find annotation file
        ####Dont need swath number when driving with xml and tiff file
        if not self.xml:
            swathid = 's1?-s?-slc-{}'.format(self.polarization)

        dirname = self.safe
        if not self.xml:
            match = None
                
            if dirname.endswith('.zip'):
                pattern = os.path.join('*SAFE','annotation', swathid) + '*.xml'
                zf = zipfile.ZipFile(dirname, 'r')
                match = fnmatch.filter(zf.namelist(), pattern)
                zf.close()

                if (len(match) == 0):
                    raise Exception('No annotation xml file found in zip file: {0}'.format(dirname))

                ####Add /vsizip at the start to make it a zip file
                self.xml = '/vsizip/'+os.path.join(dirname, match[0]) 

            else:
                pattern = os.path.join('annotation',swathid)+'*.xml'
                match = glob.glob( os.path.join(dirname, pattern))

                if (len(match) == 0):
                    raise Exception('No annotation xml file found in {0}'.format(dirname))
            
                self.xml = match[0]

        if not self.xml:
            raise Exception('No annotation files found')

        print('Input XML file: ', self.xml)

        ####Find TIFF file
        if (not self.tiff) and (self.safe):
            match = None

            if dirname.endswith('.zip'):
                pattern = os.path.join('*SAFE','measurement', swathid) + '*.tiff'
                zf = zipfile.ZipFile(dirname, 'r')
                match = fnmatch.filter(zf.namelist(), pattern)
                zf.close()

                if (len(match) == 0):
                    raise Exception('No tiff file found in zip file: {0}'.format(dirname))

                ####Add /vsizip at the start to make it a zip file
                self.tiff = '/vsizip/' + os.path.join(dirname, match[0]) 


            else:
                pattern = os.path.join('measurement', swathid) + '*.tiff'
                match = glob.glob(os.path.join(dirname, pattern))

                if len(match) == 0 :
                    raise Exception('No tiff file found in directory: {0}'.format(dirname))

                self.tiff = match[0]

            print('Input TIFF files: ', self.tiff)


        ####Find manifest files
        if self.safe:
            if dirname.endswith('.zip'):
                pattern='*SAFE/manifest.safe'
                zf = zipfile.ZipFile(dirname, 'r')
                match = fnmatch.filter(zf.namelist(), pattern)
                zf.close()
                self.manifest = '/vsizip/' + os.path.join(dirname, match[0])
            else:
                self.manifest = os.path.join(dirname, 'manifest.safe')
    
            print('Manifest files: ', self.manifest)


        return
                                               
    def getFrame(self):
        return self.frame
    
    def parse(self):
        '''
        Actual parsing of the metadata for the product.
        '''
        from isceobj.Sensor.TOPS.Sentinel1 import s1_findOrbitFile 
        ###Check user inputs
        self.validateUserInputs()

        if self.xml.startswith('/vsizip'):
            import zipfile
            parts = self.xml.split(os.path.sep)

            if parts[2] == '':
                parts[2] = os.path.sep

            zipname = os.path.join(*(parts[2:-3]))
            fname = os.path.join(*(parts[-3:]))

            zf = zipfile.ZipFile(zipname, 'r')
            xmlstr = zf.read(fname)
            zf.close()
        else:
            with open(self.xml,'r') as fid:
                xmlstr = fid.read()

        self._xml_root = ET.fromstring(xmlstr)                    
        self.populateMetadata()
    
        if self.manifest:
            self.populateIPFVersion()
        else:
            self.frame.setProcessingFacility('ESA')
            self.frame.setProcessingSoftwareVersion('IPFx.xx')

        if not self.orbitFile:
            if self.orbitDir:
                self.orbitFile = s1_findOrbitFile(self.orbitDir,
                                                self.frame.sensingStart,
                                                self.frame.sensingStop,
                                                mission = self.frame.getInstrument().getPlatform().getMission())

        if self.orbitFile:
            orb = self.extractPreciseOrbit()
            self.frame.orbit.setOrbitSource( os.path.basename(self.orbitFile))
        else:
            orb = self.extractOrbitFromAnnotation()
            self.frame.orbit.setOrbitSource('Annotation')

        for sv in orb:
            self.frame.orbit.addStateVector(sv)


    def grab_from_xml(self, path):
        try:
            res = self._xml_root.find(path).text
        except:
            raise Exception('Tag= %s not found'%(path))

        if res is None:
            raise Exception('Tag = %s not found'%(path))

        return res

    def convertToDateTime(self, string):
        dt = datetime.datetime.strptime(string,"%Y-%m-%dT%H:%M:%S.%f")
        return dt

    
    def populateMetadata(self):
        """
            Create metadata objects from the metadata files
        """
        ####Set each parameter one - by - one
        mission = self.grab_from_xml('adsHeader/missionId')
        swath = self.grab_from_xml('adsHeader/swath')
        polarization = self.grab_from_xml('adsHeader/polarisation')

        frequency = float(self.grab_from_xml('generalAnnotation/productInformation/radarFrequency'))
        passDirection = self.grab_from_xml('generalAnnotation/productInformation/pass')

        rangePixelSize = float(self.grab_from_xml('imageAnnotation/imageInformation/rangePixelSpacing'))
        azimuthPixelSize = float(self.grab_from_xml('imageAnnotation/imageInformation/azimuthPixelSpacing'))
        rangeSamplingRate = Const.c/(2.0*rangePixelSize)
        prf = 1.0/float(self.grab_from_xml('imageAnnotation/imageInformation/azimuthTimeInterval'))

        lines = int(self.grab_from_xml('imageAnnotation/imageInformation/numberOfLines'))
        samples = int(self.grab_from_xml('imageAnnotation/imageInformation/numberOfSamples'))

        startingRange = float(self.grab_from_xml('imageAnnotation/imageInformation/slantRangeTime'))*Const.c/2.0
        incidenceAngle = float(self.grab_from_xml('imageAnnotation/imageInformation/incidenceAngleMidSwath'))
        dataStartTime = self.convertToDateTime(self.grab_from_xml('imageAnnotation/imageInformation/productFirstLineUtcTime'))
        dataStopTime = self.convertToDateTime(self.grab_from_xml('imageAnnotation/imageInformation/productLastLineUtcTime'))


        pulseLength = float(self.grab_from_xml('generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues/txPulseLength'))
        chirpSlope = float(self.grab_from_xml('generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues/txPulseRampRate'))
        pulseBandwidth = pulseLength * chirpSlope

        ####Sentinel is always right looking
        lookSide = -1

#        height = self.product.imageGenerationParameters.sarProcessingInformation._satelliteHeight

        ####Populate platform
        platform = self.frame.getInstrument().getPlatform()
        platform.setPlanet(Planet(pname="Earth"))
        platform.setMission(mission)
        platform.setPointingDirection(lookSide)
        platform.setAntennaLength(2*azimuthPixelSize)

        ####Populate instrument
        instrument = self.frame.getInstrument()
        instrument.setRadarFrequency(frequency)
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setPulseLength(pulseLength)
        instrument.setChirpSlope(pulseBandwidth/pulseLength)
        instrument.setIncidenceAngle(incidenceAngle)
        #self.frame.getInstrument().setRangeBias(0)
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setRangeSamplingRate(rangeSamplingRate)
        instrument.setBeamNumber(swath)
        instrument.setPulseLength(pulseLength)


        #Populate Frame
        #self.frame.setSatelliteHeight(height)
        self.frame.setSensingStart(dataStartTime)
        self.frame.setSensingStop(dataStopTime)
        diffTime = DTUtil.timeDeltaToSeconds(dataStopTime - dataStartTime)/2.0
        sensingMid = dataStartTime + datetime.timedelta(microseconds=int(diffTime*1e6))
        self.frame.setSensingMid(sensingMid)
        self.frame.setPassDirection(passDirection)
        self.frame.setPolarization(polarization) 
        self.frame.setStartingRange(startingRange)
        self.frame.setFarRange(startingRange + (samples-1)*rangePixelSize)
        self.frame.setNumberOfLines(lines)
        self.frame.setNumberOfSamples(samples)
        
        self.frame.setPassDirection(passDirection)

    def extractOrbitFromAnnotation(self):
        '''
        Extract orbit information from xml node.
        '''

        node = self._xml_root.find('generalAnnotation/orbitList')
        frameOrbit = Orbit()
        frameOrbit.setOrbitSource('Header')

        for child in node:
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

        planet = self.frame.instrument.platform.planet
        orbExt = OrbitExtender(planet=planet)
        orbExt.configure()
        newOrb = orbExt.extendOrbit(frameOrbit)

        return newOrb

    def extractPreciseOrbit(self):
        '''
        Extract precise orbit from given Orbit file.
        '''
        try:
            fp = open(self.orbitFile,'r')
        except IOError as strerr:
            print("IOError: %s" % strerr)
            return

        _xml_root = ET.ElementTree(file=fp).getroot()
       
        node = _xml_root.find('Data_Block/List_of_OSVs')

        orb = Orbit()
        orb.configure()

        margin = datetime.timedelta(seconds=40.0)
        tstart = self.frame.getSensingStart() - margin
        tend = self.frame.getSensingStop() + margin

        for child in node:
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
                orb.addStateVector(vec)

        fp.close()

        return orb

    def extractImage(self):
        """
           Use gdal python bindings to extract image
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise Exception('GDAL python bindings not found. Need this for RSAT2/ TandemX / Sentinel1A.')

        self.parse()
        width = self.frame.getNumberOfSamples()
        lgth = self.frame.getNumberOfLines()

        src = gdal.Open(self.tiff.strip(), gdal.GA_ReadOnly)
        band = src.GetRasterBand(1)
        fid = open(self.output, 'wb')
        for ii in range(lgth):
            data = band.ReadAsArray(0,ii,width,1)
            data.tofile(fid)

        fid.close()
        src = None
        band = None

        ####
        slcImage = isceobj.createSlcImage()
        slcImage.setByteOrder('l')
        slcImage.setFilename(self.output)
        slcImage.setAccessMode('read')
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setLength(self.frame.getNumberOfLines())
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        self.frame.setImage(slcImage)

    def extractDoppler(self):
        '''
        self.parse()
        Extract doppler information as needed by mocomp
        '''
        from isceobj.Util import Poly1D

        node = self._xml_root.find('dopplerCentroid/dcEstimateList')

        tdiff = 1.0e9
        dpoly = None

        for index, burst in enumerate(node):
            refTime = self.convertToDateTime( burst.find('azimuthTime').text)
            
            delta = abs((refTime - self.frame.sensingMid).total_seconds())
            if delta < tdiff:
                tdiff = delta
                r0 = 0.5 * Const.c * float(burst.find('t0').text)
                coeffs = [float(val) for val in burst.find('dataDcPolynomial').text.split()]

                poly = Poly1D.Poly1D()
                poly.initPoly(order = len(coeffs) - 1)
                poly.setMean(r0)
                poly.setNorm(0.5 * Const.c)
                poly.setCoeffs(coeffs)

                dpoly = poly

        if dpoly is None:
            raise Exception('Could not extract Doppler information for S1 scene')

        ###Part for insarApp
        ###Should be removed in the future
        rmid = self.frame.startingRange + 0.5 * self.frame.getNumberOfSamples() * self.frame.getInstrument().getRangePixelSize()

        quadratic = {}
        quadratic['a'] = dpoly(rmid) / self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic['b'] = 0.
        quadratic['c'] = 0.


        ###Actual Doppler Polynomial for accurate processing
        ###Will be used in roiApp
        pix = np.linspace(0, self.frame.getNumberOfSamples(), num=dpoly._order+2)
        rngs = self.frame.startingRange + pix * self.frame.getInstrument().getRangePixelSize()
        evals = dpoly(rngs)
        fit = np.polyfit(pix, evals, dpoly._order)
        self.frame._dopplerVsPixel = list(fit[::-1])
        print('Doppler Fit : ', fit[::-1])

        return quadratic



    def populateIPFVersion(self):
        '''
        Get IPF version from the manifest file.
        '''

        try:
            if self.manifest.startswith('/vsizip'):
                import zipfile
                parts = self.manifest.split(os.path.sep)
                if parts[2] == '':
                    parts[2] = os.path.sep
                zipname = os.path.join(*(parts[2:-2]))
                fname = os.path.join(*(parts[-2:]))
                print('MANS: ', zipname, fname)

                zf = zipfile.ZipFile(zipname, 'r')
                xmlstr = zf.read(fname)

            else:
                with open(self.manifest, 'r') as fid:
                    xmlstr = fid.read()

            ####Setup namespace
            nsp = "{http://www.esa.int/safe/sentinel-1.0}"

            root = ET.fromstring(xmlstr)

            elem = root.find('.//metadataObject[@ID="processing"]')

            rdict = elem.find('.//xmlData/' + nsp + 'processing/' + nsp + 'facility').attrib
            self.frame.setProcessingFacility(rdict['site'] +', '+ rdict['country'])

            rdict = elem.find('.//xmlData/' + nsp + 'processing/' + nsp + 'facility/' + nsp + 'software').attrib

            self.frame.setProcessingSoftwareVersion(rdict['name'] + ' ' + rdict['version'])

        except:   ###Not a critical error ... continuing
            print('Could not read version number successfully from manifest file: ', self.manifest)
            pass

        return
