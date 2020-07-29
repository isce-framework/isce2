#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# copyright: 2010 to the present, california institute of technology.
# all rights reserved. united states government sponsorship acknowledged.
# any commercial use must be negotiated with the office of technology transfer
# at the california institute of technology.
# 
# this software may be subject to u.s. export control laws. by accepting this
# software, the user agrees to comply with all applicable u.s. export laws and
# regulations. user has the responsibility to obtain export licenses,  or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
# 
# installation and use of this software is restricted by a license agreement
# between the licensee and the california institute of technology. it is the
# user's responsibility to abide by the terms of the license agreement.
#
# Author: Andrés Solarte - Leonardo Euillades
# Instituto de Capacitación Especial y Desarrollo de la Ingeniería Asistida por Computadora (CEDIAC) Fac. Ing. UNCuyo
# Instituto de Altos Estudios Espaciales "Mario Gulich" CONAE-UNC
# Consejo Nacional de Investigaciones Científicas y Técnicas (CONICET)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import numpy as np
import datetime
import logging
import isceobj
from isceobj import *
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import StateVector
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Scene.Frame import Frame
from iscesys.Component.Component import Component

XEMTFILE = Component.Parameter(
    'xemtFile',
    public_name='XEMTFILE',
    default='',
    type=str,
    mandatory=True,
    intent='input',
    doc='xml file with generic metadata.'
)

XMLFILE = Component.Parameter(
    'xmlFile',
    public_name='XMLFILE',
    default='',
    type=str,
    mandatory=True,
    intent='input',
    doc='Input metadata file in xml format.'
)

IMAGEFILE = Component.Parameter(
    '_imageFileName',
    public_name='IMAGEFILE',
    default='',
    type=str,
    mandatory=True,
    intent='input',
    doc='Input image file.'
)

from .Sensor import Sensor
class SAOCOM_SLC(Sensor):

    parameter_list = (IMAGEFILE,
                      XEMTFILE,
                      XMLFILE)           + Sensor.parameter_list

    """
        A Class for parsing SAOCOM instrument and imagery files
    """

    family = 'saocom_slc'

    def __init__(self,family='',name=''):
        super(SAOCOM_SLC, self).__init__(family if family else  self.__class__.family, name=name)
        self._imageFile = None
        self._xemtFileParser = None
        self._xmlFileParser = None
        self._instrumentFileData = None
        self._imageryFileData = None
        self.dopplerRangeTime = None
        self.rangeRefTime = None
        self.azimuthRefTime = None
        self.rangeFirstTime = None
        self.rangeLastTime = None
        self.logger = logging.getLogger("isce.sensor.SAOCOM_SLC")
        self.frame = None
        self.frameList = []
        
        self.lookMap = {'RIGHT': -1,
                        'LEFT': 1}
        self.nearIncidenceAngle = {'S1DP': 20.7, 
        'S2DP': 24.9, 
        'S3DP': 29.1, 
        'S4DP': 33.7, 
        'S5DP': 38.2, 
        'S6DP': 41.3, 
        'S7DP': 44.6, 
        'S8DP': 47.2, 
        'S9DP': 48.8, 
        'S1QP': 17.6, 
        'S2QP': 19.5, 
        'S3QP': 21.4, 
        'S4QP': 23.2, 
        'S5QP': 25.3, 
        'S6QP': 27.2, 
        'S7QP': 29.6, 
        'S8QP': 31.2, 
        'S9QP': 33.0, 
        'S10QP': 34.6}
        self.farIncidenceAngle = {'S1DP': 25.0, 
        'S2DP': 29.2, 
        'S3DP': 33.8, 
        'S4DP': 38.3, 
        'S5DP': 41.3, 
        'S6DP': 44.5, 
        'S7DP': 47.1, 
        'S8DP': 48.7, 
        'S9DP': 50.2, 
        'S1QP': 19.6, 
        'S2QP': 21.5, 
        'S3QP': 23.3, 
        'S4QP': 25.4, 
        'S5QP': 27.3, 
        'S6QP': 29.6, 
        'S7QP': 31.2, 
        'S8QP': 33.0, 
        'S9QP': 34.6, 
        'S10QP': 35.5}
        
    def parse(self):
        """
            Parse both imagery and instrument files and create
            objects representing the platform, instrument and scene
        """

        self.frame = Frame()
        self.frame.configure()
        self._xemtFileParser = XEMTFile(fileName=self.xemtFile)
        self._xemtFileParser.parse()
        self._xmlFileParser = XMLFile(fileName=self.xmlFile)
        self._xmlFileParser.parse()
        self.populateMetadata()

    def populateMetadata(self):
        self._populatePlatform()
        self._populateInstrument()
        self._populateFrame()
        self._populateOrbit()
        self._populateExtras()
        
    def _populatePlatform(self):
        """Populate the platform object with metadata"""
        platform = self.frame.getInstrument().getPlatform()
        
        # Populate the Platform and Scene objects
        platform.setMission(self._xmlFileParser.sensorName)
        platform.setPointingDirection(self.lookMap[self._xmlFileParser.sideLooking])
        platform.setAntennaLength(9.968)
        platform.setPlanet(Planet(pname="Earth"))
        
    def _populateInstrument(self):
        """Populate the instrument object with metadata"""
        instrument = self.frame.getInstrument()
        
        rangePixelSize = self._xmlFileParser.PSRng
        azimuthPixelSize = self._xmlFileParser.PSAz
        radarWavelength = Const.c/float(self._xmlFileParser.fc_hz)
        instrument.setRadarWavelength(radarWavelength)
        instrument.setPulseRepetitionFrequency(self._xmlFileParser.prf)
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setAzimuthPixelSize(azimuthPixelSize)
        instrument.setPulseLength(self._xmlFileParser.pulseLength)
        instrument.setChirpSlope(float(self._xmlFileParser.pulseBandwidth)/float(self._xmlFileParser.pulseLength))

        instrument.setRangeSamplingRate(self._xmlFileParser.frg)

        incAngle = 0.5*(self.nearIncidenceAngle[self._xemtFileParser.beamID] + self.farIncidenceAngle[self._xemtFileParser.beamID])
        instrument.setIncidenceAngle(incAngle)
        
    def _populateFrame(self):
        """Populate the scene object with metadata"""
        
        rft = self._xmlFileParser.rangeStartTime
        slantRange = float(rft)*Const.c/2.0
        self.frame.setStartingRange(slantRange)

        sensingStart = self._parseNanoSecondTimeStamp(self._xmlFileParser.azimuthStartTime)
        sensingTime = self._xmlFileParser.lines/self._xmlFileParser.prf
        sensingStop = sensingStart + datetime.timedelta(seconds=sensingTime)
        sensingMid = sensingStart + datetime.timedelta(seconds=0.5*sensingTime)

        self.frame.setPassDirection(self._xmlFileParser.orbitDirection)
        self.frame.setProcessingFacility(self._xemtFileParser.facilityID)
        self.frame.setProcessingSoftwareVersion(self._xemtFileParser.softVersion)
        self.frame.setPolarization(self._xmlFileParser.polarization)
        self.frame.setNumberOfLines(self._xmlFileParser.lines)
        self.frame.setNumberOfSamples(self._xmlFileParser.samples)
        self.frame.setSensingStart(sensingStart)
        self.frame.setSensingMid(sensingMid)
        self.frame.setSensingStop(sensingStop)

        rangePixelSize = self.frame.getInstrument().getRangePixelSize()
        farRange = slantRange +  (self.frame.getNumberOfSamples()-1)*rangePixelSize
        self.frame.setFarRange(farRange)
        
    def _populateOrbit(self):
        orbit = self.frame.getOrbit()
        orbit.setReferenceFrame('ECR')
        orbit.setOrbitSource('Header')
        t0 = self._parseNanoSecondTimeStamp(self._xmlFileParser.orbitStartTime)
        t = np.arange(self._xmlFileParser.numberSV)*self._xmlFileParser.deltaTimeSV
        position = self._xmlFileParser.orbitPositionXYZ
        velocity = self._xmlFileParser.orbitVelocityXYZ

        for i in range(0,self._xmlFileParser.numberSV):
            vec = StateVector()
            dt = t0 + datetime.timedelta(seconds=t[i])
            vec.setTime(dt)
            vec.setPosition([position[i*3],position[i*3+1],position[i*3+2]])
            vec.setVelocity([velocity[i*3],velocity[i*3+1],velocity[i*3+2]])
            orbit.addStateVector(vec)
            print("valor "+str(i)+": "+str(dt))
        
    def _populateExtras(self):
        from isceobj.Doppler.Doppler import Doppler
        
        self.dopplerRangeTime = self._xmlFileParser.dopRngTime
        self.rangeRefTime = self._xmlFileParser.trg
        self.rangeFirstTime = self._xmlFileParser.rangeStartTime
        
    def extractImage(self):
        """
        Exports GeoTiff to ISCE format.
        """
        from osgeo import gdal
        
        ds = gdal.Open(self._imageFileName)
        metadata = ds.GetMetadata()
        geoTs = ds.GetGeoTransform() #GeoTransform
        prj = ds.GetProjection() #Projection
        dataType = ds.GetRasterBand(1).DataType
        gcps = ds.GetGCPs()
        
        sds = ds.ReadAsArray()
        
        # Output raster array to ISCE file
        driver = gdal.GetDriverByName('ISCE')  
        export = driver.Create(self.output, ds.RasterXSize, ds.RasterYSize, 1, dataType)
        band = export.GetRasterBand(1)
        band.WriteArray(sds)
        export.SetGeoTransform(geoTs)
        export.SetMetadata(metadata)
        export.SetProjection(prj)
        export.SetGCPs(gcps,prj)
        band.FlushCache()
        export.FlushCache()        
        
        self.parse()
        slcImage = isceobj.createSlcImage()
        slcImage.setFilename(self.output)
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setAccessMode('r')
        self.frame.setImage(slcImage)
        
    def _parseNanoSecondTimeStamp(self,timestamp):
        """
            Parse a date-time string with microsecond precision and return a datetime object
        """
        dateTime,decSeconds = timestamp.split('.')
        microsec = float("0."+decSeconds)*1e6
        dt = datetime.datetime.strptime(dateTime,'%d-%b-%Y %H:%M:%S')
        dt = dt + datetime.timedelta(microseconds=microsec)
        return dt

    def extractDoppler(self):
        """
        Return the doppler centroid.
        """
        quadratic = {}

        r0 = self.frame.getStartingRange()
        dr = self.frame.instrument.getRangePixelSize()
        width = self.frame.getNumberOfSamples()
        
        midr = r0 + (width/2.0) * dr
        midtime = 2 * midr/ Const.c - self.rangeRefTime

        fd_mid = 0.0
        tpow = midtime
        
        for kk in self.dopplerRangeTime:
            fd_mid += kk * tpow
            tpow *= midtime

        ####For insarApp
        quadratic['a'] = fd_mid/self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic['b'] = 0.
        quadratic['c'] = 0.

        ####For roiApp
        ####More accurate
        from isceobj.Util import Poly1D
        
        coeffs = self.dopplerRangeTime
        dr = self.frame.getInstrument().getRangePixelSize()
        rref = 0.5 * Const.c * self.rangeRefTime 
        r0 = self.frame.getStartingRange()
        norm = 0.5*Const.c/dr
        
        dcoeffs = []
        for ind, val in enumerate(coeffs):
            dcoeffs.append( val / (norm**ind))

        poly = Poly1D.Poly1D()
        poly.initPoly(order=len(coeffs)-1)
        poly.setMean( (rref - r0)/dr - 1.0)
        poly.setCoeffs(dcoeffs)

        pix = np.linspace(0, self.frame.getNumberOfSamples(), num=len(coeffs)+1)
        evals = poly(pix)
        fit = np.polyfit(pix,evals, len(coeffs)-1)
        self.frame._dopplerVsPixel = list(fit[::-1])
        print('Doppler Fit: ', fit[::-1])
        
        return quadratic


class XMLFile():
    """Parse a SAOCOM xml file"""
    
    def __init__(self, fileName=None):
        self.fileName = fileName

    def parse(self):
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(self.fileName)
            root = tree.getroot()
            product = root.findall('Channel')
            rasterInfo = [feat.findall("RasterInfo") for feat in product][0]
            datasetInfo = [feat.findall("DataSetInfo") for feat in product][0]
            constants = [feat.findall("SamplingConstants") for feat in product][0]
            pulse = [feat.findall("Pulse") for feat in product][0]
            burstInfo = [feat.findall("BurstInfo") for feat in product][0]
            burst = [feat.findall("Burst") for feat in burstInfo][0]
            stateVectorData = [feat.findall("StateVectorData") for feat in product][0]
            swathInfo = [feat.findall("SwathInfo") for feat in product][0]
            orbitPosition=[feat.findall("pSV_m") for feat in stateVectorData][0]
            orbitPosition2=[feat.findall("val") for feat in orbitPosition][0]
            orbitVel=[feat.findall("vSV_mOs") for feat in stateVectorData][0]
            orbitVel2=[feat.findall("val") for feat in orbitVel][0]
            dopplerCentroid = [feat.findall("DopplerCentroid") for feat in product][0]
            dopplerRate = [feat.findall("DopplerRate") for feat in product][0]
            
            self.lines = int([lines.find("Lines").text for lines in rasterInfo][0])
            self.samples = int([samp.find("Samples").text for samp in rasterInfo][0])
            if [sn.find("SensorName").text for sn in datasetInfo][0]=='SAO1A':
            	self.sensorName = 'SAOCOM1A'
            elif [sn.find("SensorName").text for sn in datasetInfo][0]=='SAO1B':
            	self.sensorName = 'SAOCOM1B'
            else:
            	self.sensorName = [sn.find("SensorName").text for sn in datasetInfo][0]
            self.fc_hz = float([fc.find("fc_hz").text for fc in datasetInfo][0])
            self.sideLooking = [sl.find("SideLooking").text for sl in datasetInfo][0]
            self.prf = float([prf.find("faz_hz").text for prf in constants][0])
            self.frg = float([frg.find("frg_hz").text for frg in constants][0])
            self.PSRng = float([psr.find("PSrg_m").text for psr in constants][0])
            self.PSAz = float([psa.find("PSaz_m").text for psa in constants][0])
            self.azBandwidth = float([baz.find("Baz_hz").text for baz in constants][0])
            self.pulseLength = float([pl.find("PulseLength").text for pl in pulse][0])
            self.pulseBandwidth = float([bw.find("Bandwidth").text for bw in pulse][0])
            self.rangeStartTime = float([rst.find("RangeStartTime").text for rst in burst][0])
            self.azimuthStartTime = [ast.find("AzimuthStartTime").text for ast in burst][0]
            self.orbitDirection = [od.find("OrbitDirection").text for od in stateVectorData][0]
            self.polarization = [pol.find("Polarization").text for pol in swathInfo][0].replace("/","")
            self.acquisitionStartTime = [st.find("AcquisitionStartTime").text for st in swathInfo][0]
            self.orbitPositionXYZ = [float(xyz.text) for xyz in orbitPosition2]
            self.orbitVelocityXYZ = [float(xyz.text) for xyz in orbitVel2]
            self.orbitStartTime = [ost.find("t_ref_Utc").text for ost in stateVectorData][0]
            self.deltaTimeSV = float([dt.find("dtSV_s").text for dt in stateVectorData][0])
            self.numberSV = int([n.find("nSV_n").text for n in stateVectorData][0])
            trg = []
            for feat in dopplerCentroid:
                for feat2 in feat.findall("trg0_s"):
                    trg.append(float(feat2.text))
            
            for feat in dopplerRate:
                for feat2 in feat.findall("trg0_s"):
                    trg.append(float(feat2.text))
            self.trg = np.mean(np.array(trg))
            
            self.dopRngTime_old = []
            self.dopRngTime = []

            for feat in dopplerCentroid:
                for feat2 in feat.findall("pol"):
                    for val in feat2.findall("val"):
                        if feat.get("Number")=='2':
                            self.dopRngTime.append(float(val.text))
                      
        except IOError as errs:
            errno,strerr = errs
            print("IOError: {} {}".format(strerr,self.fileName))
            return
            

class XEMTFile():
    """Parse a SAOCOM xemt file"""
    
    def __init__(self, fileName=None):
        self.fileName = fileName

    def parse(self):
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(self.fileName)
            root = tree.getroot()
            product = root.findall('product')
            features = [feat.findall("features") for feat in product][0]
            acquisition = [acq.findall("acquisition") for acq in features][0]
            parameters = [param.findall("parameters") for param in acquisition][0]
            prodHistory = [feat.findall("productionHistory") for feat in product][0]
            software = [feat.findall("software") for feat in prodHistory][0]
            excecEnvironment = [feat.findall("executionEnvironment") for feat in prodHistory][0]
            
            self.beamID =[beam.find("beamID").text for beam in parameters][0]
            self.softVersion = [sversion.find("version").text for sversion in software][0]
            self.countryID = [country.find("countryID").text for country in excecEnvironment][0]
            self.agencyID = [agency.find("agencyID").text for agency in excecEnvironment][0]
            self.facilityID = [facility.find("facilityID").text for facility in excecEnvironment][0]
            self.serviceID = [service.find("serviceID").text for service in excecEnvironment][0]
           
        except IOError as errs:
            errno,strerr = errs
            print("IOError: {} {}".format(strerr,self.fileName))
            return
            
