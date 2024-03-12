#!/usr/bin/python3

# Reader for Lutan-1 SLC data
# Used Sentinel1.py and ALOS.py as templates
# Author: Bryan Marfito, EOS-RS


import os
import glob
import xml.etree.ElementTree as ET
import datetime
import isce
import isceobj
from iscesys.Component.Component import Component
from isceobj.Sensor.Sensor import Sensor
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector, Orbit
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
from isceobj.Orbit.OrbitExtender import OrbitExtender

sep = "\n"
tab = "    "
lookMap = { 'RIGHT' : -1,
            'LEFT' : 1}
antennaLength = 9.8


TIFF = Component.Parameter('tiff',
                            public_name ='tiff',
                            default = None,
                            type=str,
                            doc = 'Input image file')

ORBIT_FILE = Component.Parameter('orbitFile',
                            public_name ='orbitFile',
                            default = None,
                            type=str,
                            doc = 'Orbit file')


class Lutan1(Sensor):

    "Class for Lutan-1 SLC data"
    
    family = 'l1sm'
    logging_name = 'isce.sensor.LuTan1'

    parameter_list = (TIFF, ORBIT_FILE) + Sensor.parameter_list

    def __init__(self, name = ''):
        super(LuTan1,self).__init__(self.__class__.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        self.xml_root = None


    
    def parse(self):
        with open(self.xml, 'r') as fid:
            xmlstr = fid.read()
        
        self.xml_root = ET.fromstring(xmlstr)
        self.populateMetadata()


    def convertToDateTime(self,string):
        dt = datetime.datetime.strptime(string,"%Y-%m-%dT%H:%M:%S.%f")
        return dt


    def grab_from_xml(self, path):
        try:
            res = self._xml_root.find(path).text
        except:
            raise Exception('Tag= %s not found'%(path))

        if res is None:
            raise Exception('Tag = %s not found'%(path))
        return res
    

    def populateMetadata(self):
        mission = self.grab_from_xml('generalHeader/mission')
        polarization = self.grab_from_xml('generalHeader/acquisitionInfo/polarisationMode')
        frequency = float(self.grab_from_xml('instrument/radarParameters/centerFrequency'))
        passDirection = self.grab_from_xml('productInfo/missionInfo/orbitDirection')
        rangePixelSize = float(self.grab_from_xml('productInfo/imageDataInfo/imageRaster/columnSpacing units="m"'))
        azimuthPixelSize = float(self.grab_from_xml('productInfo/imageDataInfo/imageRaster/rowSpacing units="m"'))
        rangeSamplingRate = Const.c/(2.0*rangePixelSize)

        prf = float(self.grab_from_xml('instrument/settings/settingRecord/PRF'))
        lines = int(self.grab_from_xml('productInfo/imageDataInfo/numberOfRows'))
        samples = int(self.grab_from_xml('productInfo/imageDataInfo/numberOfColumns'))

        startingRange = float(self.grab_from_xml('sceneInfo/rangeTime/firstPixel'))*Const.c/2.0
        #slantRange = float(self.grab_from_xml('productSpecific/complexImageInfo/'))
        incidenceAngle = float(self.grab_from_xml('sceneInfo/sceneCenterCoord/incidenceAngle'))
        dataStartTime = self.convertToDateTime(self.grab_from_xml('sceneInfo/start/timeUTC'))
        dataStopTime = self.convertToDateTime(self.grab_from_xml('sceneInfo/stop/timeUTC'))


        pulseLength = float(self.grab_from_xml('processing/processingParameter/rangeCompression/chirps/pulseLength'))
        pulseBandwidth = float(self.grab_from_xml('processing/processingParameter/rangeCompression/chirps/pulseBandwidth'))
        chirpSlope = pulseBandwidth/pulseLength
        lookSide = lookMap['RIGHT']

        # Platform parameters
        platform = self.frame.getInstrument().getPlatform()
        platform.setPlanet(pname='Earth')
        platform.setMission(mission)
        platform.setPointingDirection(lookSide)
        platform.setAntennaLength(antennaLength)

        # Instrument parameters
        instrument = self.frame.getInstrument()
        instrument.setRadarFrequency(frequency)
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setPulseLength(pulseLength)
        instrument.ChirpSlope = chirpSlope
        instrument.setIncidenceAngle(incidenceAngle)
        instrument.setRangePixelSize(rangePixelSize)
        instrument.RangeSamplingRate(rangeSamplingRate)
        instrument.setPulseLength(pulseLength)

        # Frame parameters
        self.frame.setSensingStart(dataStartTime)
        self.frame.setSensingStop(dataStopTime)

        # Two-way travel time 
        diffTime = DTUtil.timeDeltaToSeconds(dataStopTime - dataStartTime) / 2.0
        sensingMid = dataStartTime + datetime.timedelta(microseconds=int(diffTime*1e6))
        self.frame.setSensingMid(sensingMid)
        self.frame.setPassDirection(passDirection)
        self.frame.setPolarization(polarization)
        self.frame.setStartingRange(startingRange)
        self.frame.setFarRange(startingRange + rangePixelSize * (samples - 1))
        self.frame.setNumberOfLines(lines)
        self.frame.setNumberOfSamples(samples)
        
        return


    def extractOrbit(self):

        '''
        Extract orbit information from the orbit file
        '''

        try:
            orbitFile = "/Volumes/jupiter/LuTan-1/orbits/LT1A_20230228143413185_V20230209T235500_20230211T000500_ABSORBIT_SCIE.xml"
            fp = open(orbitFile, 'r')
        except IOError as strerr:
            print("IOError: %s" % strerr)
        
    
        _xml_root = ET.ElementTree(file=fp).getroot()

        node = _xml_root.find('Data_Block/List_of_OSVs')

        orb = Orbit()
        orb.configure()

        margin = datetime.timedelta(seconds=40.0)
        tstart = self.frame.getSensingStart() - margin
        tend = self.frame.getSensingStop() + margin

        for child in node:
            timestamp = self.convertDateTime(child.find('UTC').text)

            if (tstart <= timestamp) and (timestamp < tend):
                pos = []
                vel = []
        
                for tag in ['VX', 'VY', 'VZ']:
                    vel.append(float(child.find(tag).text))

                for tag in ['X', 'Y', 'Z']:
                    pos.append(float(child.find(tag).text))

            vec = StateVector()
            vec.setTime(timestamp)
            vec.setPosition(pos)
            vec.setVelocity(vel)
            orb.addStateVector(vec)
            print("Velocity: ",vel)
        fp.close()

        return orb
    
    def extractOrbitFromAnnotation(self):

        '''
        Extract orbit information from xml annotation
        WARNING! Only used this method if precise orbit file is not available
        '''

        node = self.xml_root.find('platform/orbit')
        frameOrbit = Orbit()
        frameOrbit.setOrbitSource('Header')

        for child in node:
            timestamp = self.convertToDateTime(child.find('timeUTC').text)
            pos = []
            vel = []

            for tag in ['posX', 'posY', 'posZ']:
                pos.append(float(child.find(tag).text))

            for tag in ['velX', 'velY', 'velZ']:
                vel.append(float(child.find(tag).text))

            vec = StateVector()
            vec.setTime(timestamp)
            vec.setPosition(pos)
            vec.setVelocity(vel)
            frameOrbit.addStateVector(vec)
        
        planet = self.frame.instrument.platform.planet
        orbExt = OrbitExtender(planet=planet)
        orbExt.configure()
        newOrb = orbExt.setOrbit(frameOrbit)

        return newOrb
    
    def extractImage(self):
        try:
            from osgeo import gdal
        except ImportError:
            raise Exception('GDAL python bindings not found. Need this for Lutan-1.')
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





k = LuTan1()
#k.extractOrbit()
#k.extractOrbit()




  


