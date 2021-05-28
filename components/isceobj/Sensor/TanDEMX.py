#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Walter Szeliga, Eric Gurrola, Maxim Neumann
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import logging
import datetime
import isceobj
from xml.etree.ElementTree import ElementTree
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector
from isceobj.Planet.Planet import Planet
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Sensor import cosar
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from iscesys.Component.Component import Component

sep = "\n"
tab = "    "

XML = Component.Parameter(
    'xml',
    public_name='xml',
    default=None,
    type=str,
    mandatory=True,
    doc='Name of the xml file.'
)

OUTPUT = Component.Parameter(
    'output',
    public_name='output',
    default=None,
    type=str,
    mandatory=False,
    doc='Name of the output file.'
)

class TanDEMX(Component):
    """
    A class representing a Level1Product meta data.
    Level1Product(xml=filename) will parse the xml
    file and produce an object with attributes that
    represent the element tree of the xml file.
    """

    family='tandemx'
    logging_name = 'isce.Sensor.TanDEMX'

    parameter_list = (XML, OUTPUT)

    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self.xml = None
        self.output = None
        self.generalHeader = _GeneralHeader()
        self.productComponents = _ProductComponents()
        self.productInfo = _ProductInfo()
        self.productSpecific = _ProductSpecific()
        self.platform = _Platform()
        self.instrument = _Instrument()
        self.processing = _Processing()
#        self.logger = logging.getLogger(
        self.frame = Frame()
        self.frame.configure()
        # Some extra processing parameters unique to TSX (currently)
        self.zeroDopplerVelocity = None
        self.dopplerArray = []

        self.descriptionOfVariables = {}

        self.lookDirectionEnum = {'RIGHT': -1,
                                  'LEFT': 1}
        return

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.Sensor.TanDEMX')
        return


    def getFrame(self):
        return self.frame

    def parse(self):
        try:
            fp = open(self.xml,'r')
        except IOError as errs:
            errno, strerr = errs
            self.logger.error("IOError: %s" % strerr)
            raise IOError(strerr)

        self._xml_root = ElementTree(file=fp).getroot()
        for z in self._xml_root:
            if z.tag == 'generalHeader':
                self.generalHeader.set_from_etnode(z)
            if z.tag == 'productComponents':
                self.productComponents.set_from_etnode(z)
            if z.tag == 'productInfo':
                self.productInfo.set_from_etnode(z)
            if z.tag == 'productSpecific':
                self.productSpecific.set_from_etnode(z)
            if z.tag == 'platform':
                self.platform.set_from_etnode(z)
            if z.tag == 'instrument':
                self.instrument.set_from_etnode(z)
            if z.tag == 'processing':
                self.processing.set_from_etnode(z)
        self.populateMetadata()
        fp.close()

    def populateMetadata(self):
        """
        Populate our Metadata objects
        """

        self._populatePlatform()
        self._populateInstrument()
        self._populateFrame()
        self._populateOrbit()
        self._populateExtras()

    def _populatePlatform(self):
        platform = self.frame.getInstrument().getPlatform()
        mission = self.productInfo.missionInfo.mission
        pointingDirection = self.lookDirectionEnum[self.productInfo.acquisitionInfo.lookDirection]

        platform.setMission(mission)
        platform.setPointingDirection(pointingDirection)
        platform.setPlanet(Planet(pname="Earth"))

    def _populateInstrument(self):
        instrument = self.frame.getInstrument()
        rowSpacing = self.productInfo.imageDataInfo.imageRaster.rowSpacing
        incidenceAngle = self.productInfo.sceneInfo.sceneCenterCoord.incidenceAngle
        rangeSamplingFrequency = 1/(2*rowSpacing)
        rangePixelSize = (Const.c*rowSpacing/2)
        chirpPulseBandwidth = self.processing.processingParameter.rangeCompression.chirps.referenceChirp.pulseBandwidth
        rangePulseLength = self.processing.processingParameter.rangeCompression.chirps.referenceChirp.pulseLength
        prf = self.productSpecific.complexImageInfo.commonPRF
        frequency = self.instrument.radarParameters.centerFrequency

        instrument.setRadarFrequency(frequency)
        instrument.setIncidenceAngle(incidenceAngle)
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setRangePixelSize(rangePixelSize)
        #jng no sampling rate extracted before.
        instrument.setRangeSamplingRate(1/rowSpacing)
        instrument.setPulseLength(rangePulseLength)
        instrument.setChirpSlope(chirpPulseBandwidth/rangePulseLength)
        #instrument.setRangeBias(0)

    def _populateFrame(self):
        orbitNumber = self.productInfo.missionInfo.absOrbit
        lines = self.productInfo.imageDataInfo.imageRaster.numberOfRows
        samples = self.productInfo.imageDataInfo.imageRaster.numberOfColumns
        facility = self.productInfo.generationInfo.level1ProcessingFacility
        startingRange = self.productInfo.sceneInfo.rangeTime.firstPixel * (Const.c/2)
        #jng farRange missing in frame. Compute as in alos
        farRange = startingRange + samples*self.frame.getInstrument().getRangePixelSize()
        polarization = self.instrument.settings.polLayer
        first_utc_time = datetime.datetime.strptime(self.productInfo.sceneInfo.start.timeUTC[0:38],"%Y-%m-%dT%H:%M:%S.%fZ")
        last_utc_time = datetime.datetime.strptime(self.productInfo.sceneInfo.stop.timeUTC[0:38],"%Y-%m-%dT%H:%M:%S.%fZ")
        mid_utc_time = datetime.datetime.strptime(self.productInfo.sceneInfo.sceneCenterCoord.azimuthTimeUTC[0:38],"%Y-%m-%dT%H:%M:%S.%fZ")

        self.frame.setPolarization(polarization)
        self.frame.setOrbitNumber(orbitNumber)
        self.frame.setStartingRange(startingRange)
        self.frame.setFarRange(farRange)
        self.frame.setProcessingFacility(facility)
        self.frame.setNumberOfLines(lines)
        self.frame.setNumberOfSamples(samples)
        self.frame.setSensingStart(first_utc_time)
        self.frame.setSensingMid(mid_utc_time)
        self.frame.setSensingStop(last_utc_time)

    def _populateOrbit(self):
        orbit = self.frame.getOrbit()

        orbit.setOrbitSource('Header')
        quality = self.platform.orbit.orbitHeader.accuracy
        if (quality == 'SCIE'):
            orbit.setOrbitQuality('Science')
        elif (quality == 'RAPD'):
            orbit.setOrbitQuality('Rapid')
        elif (quality == 'PRED'):
            orbit.setOrbitQuality('Predicted')
        elif (quality == 'REFE'):
            orbit.setOrbitQuality('Reference')
        elif (quality == 'QUKL'):
            orbit.setOrbitQuality('Quick Look')
        else:
            orbit.setOrbitQuality('Unknown')

        stateVectors = self.platform.orbit.stateVec
        for i in range(len(stateVectors)):
            position = [stateVectors[i].posX,stateVectors[i].posY,stateVectors[i].posZ]
            velocity = [stateVectors[i].velX,stateVectors[i].velY,stateVectors[i].velZ]
            vec = StateVector()
            vec.setTime(stateVectors[i].timeUTC)
            vec.setPosition(position)
            vec.setVelocity(velocity)
            orbit.addStateVector(vec)

    def _populateExtras(self):
        """
        Populate some of the extra fields unique to processing TSX data.
        In the future, other sensors may need this information as well,
        and a re-organization may be necessary.
        """
        from isceobj.Doppler.Doppler import Doppler
        self.zeroDopplerVelocity = self.processing.geometry.zeroDopplerVelocity.velocity
        numberOfRecords = self.processing.doppler.dopplerCentroid.numberOfDopplerRecords
        for i in range(numberOfRecords):
            estimate = self.processing.doppler.dopplerCentroid.dopplerEstimate[i]
            fd = estimate.dopplerAtMidRange
            # These are the polynomial coefficients over slant range time, not range bin.
            #ambiguity = estimate.dopplerAmbiguity
            #centroid = estimate.combinedDoppler.coefficient[0]
            #linear = estimate.combinedDoppler.coefficient[1]
            #quadratic = estimate.combinedDoppler.coefficient[2]
            #doppler = Doppler(prf=self.productSpecific.complexImageInfo.commonPRF)
            #doppler.setDopplerCoefficients([centroid,linear,quadratic,0.0],inHz=True)
            #doppler.ambiguity = ambiguity
            time = DTU.parseIsoDateTime(estimate.timeUTC)
            #jng added the dopplerCoefficients needed by TsxDopp.py
            self.dopplerArray.append({'time': time, 'doppler': fd,'dopplerCoefficients':estimate.combinedDoppler.coefficient,'rangeTime': estimate.combinedDoppler.referencePoint})

    def extractImage_old_TSX(self): # kept for reference - delete!
        import os
        self.parse()
        basepath = os.path.dirname(self.xml)
        image = os.path.join(basepath,self.productComponents.imageData.file.location.path,self.productComponents.imageData.file.location.filename)
        cosar.cosar_Py(image,self.output)
        slcImage = isceobj.createSlcImage()
        slcImage.setFilename(self.output)
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setAccessMode('r')
        self.frame.setImage(slcImage)

    def cosarSaveImageBand(self, infile, outfile, blocksize=1000,
                           blockwise=False, verbose=True):
        """Read in cosar float16 SLC and save as float32.
        Currently uses gdal."""
        import numpy as np
        try:
            from osgeo import gdal
        except ImportError:
            raise Exception('GDAL python bindings not found. Need this for RSAT2 / TandemX / Sentinel1A.')

        f = gdal.Open(infile)
        band = f.GetRasterBand(1)
        if verbose:
            print("Number of bands: ",f.RasterCount)
            print('Driver: ', f.GetDriver().ShortName,'/',\
                  f.GetDriver().LongName)
            print('Size is ',f.RasterXSize,'x',f.RasterYSize)
            print('Band Type=',gdal.GetDataTypeName(band.DataType))
        n = [band.YSize, band.XSize]
        if blockwise:
            with open(outfile,'w') as out:
                i = 0
                while i*blocksize < n[0]:
                    yS = int(np.min([n[0]-i*blocksize,blocksize]))
                    b = band.ReadRaster(0,i*blocksize,n[1],yS)
                    b = np.frombuffer(b, dtype=np.float16)
                    out.write(np.array(b,dtype=np.float32))
                    i += 1
        else:
            b = band.ReadRaster(0,0,n[1],n[0])
            b = np.frombuffer(b, dtype=np.float16) #.reshape(yS,n[1],2)
            b.astype(np.float32).tofile(outfile)
        return b

    def extractImage(self):
        import os
        self.parse()
        basepath = os.path.dirname(self.xml)
        image = os.path.join(basepath,self.productComponents.imageData.file.location.path,self.productComponents.imageData.file.location.filename)

        self.cosarSaveImageBand(image,self.output)
##      cosar.cosar_Py(image,self.output) # <<<< this line saves input to output filenames! output is raw binary flat file; all parameters are set afterwards.

        slcImage = isceobj.createSlcImage()
        slcImage.setFilename(self.output)
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setAccessMode('r')
        self.frame.setImage(slcImage)

    def __str__(self):
        retstr  = "Level1Product:"+sep
        retlst  = ()
        retstr += "%s"+sep
        retlst += (str(self.generalHeader),)
        retstr += "%s"+sep
        retlst += (str(self.productComponents),)
        retstr += "%s"+sep
        retlst += (str(self.productInfo),)
        retstr += "%s"+sep
        retlst += (str(self.productSpecific),)
        retstr += "%s"+sep
        retlst += (str(self.platform),)
        retstr += "%s"
        retlst += (str(self.instrument),)
        retstr += "%s"
        retlst += (str(self.processing),)
        retstr += sep+":Level1Product"
        return retstr % retlst


    def extractDoppler(self):
        '''
        Return the doppler centroid as a function of range.
        TSX provides doppler estimates at various azimuth times.
        2x2 polynomial in azimuth and range suffices for a good representation.
        ISCE can currently only handle a function of range. 
        Doppler function at mid image in azimuth is a good value to use.
        '''
        import numpy as np

        tdiffs = []

        for dd in self.processing.doppler.dopplerCentroid.dopplerEstimate:
            tentry = datetime.datetime.strptime(dd.timeUTC,"%Y-%m-%dT%H:%M:%S.%fZ")

            tdiffs.append(np.abs( (tentry - self.frame.sensingMid).total_seconds()))

        ind = np.argmin(tdiffs)

        ####Corresponds to entry closest to sensingMid
        coeffs = self.processing.doppler.dopplerCentroid.dopplerEstimate[ind].combinedDoppler.coefficient
        tref = self.processing.doppler.dopplerCentroid.dopplerEstimate[ind].combinedDoppler.referencePoint

        
        quadratic = {}
        midtime = (self.frame.getStartingRange() + self.frame.getFarRange())/Const.c - tref

        fd_mid = 0.0
        x = 1.0
        for ind,val in enumerate(coeffs):
            fd_mid += val*x
            x *= midtime

        ####insarApp
        quadratic['a'] = fd_mid / self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic['b'] = 0.0
        quadratic['c'] = 0.0


        ####For RoiApp
        ####More accurate
        from isceobj.Util import Poly1D
        
        dr = self.frame.getInstrument().getRangePixelSize()
        rref = 0.5 * Const.c * tref 
        r0 = self.frame.getStartingRange()
        norm = 0.5*Const.c/dr

        tmin = 2 * self.frame.getStartingRange()/ Const.c

        tmax = 2 * self.frame.getFarRange() / Const.c
        

        poly = Poly1D.Poly1D()
        poly.initPoly(order=len(coeffs)-1)
        poly.setMean( tref)
        poly.setCoeffs(coeffs)


        tpix = np.linspace(tmin, tmax,num=len(coeffs)+1)
        pix = np.linspace(0, self.frame.getNumberOfSamples(), num=len(coeffs)+1)
        evals = poly(tpix)
        fit = np.polyfit(pix,evals, len(coeffs)-1)
        self.frame._dopplerVsPixel = list(fit[::-1])
        print('Doppler Fit: ', fit[::-1])

        return quadratic





###########################################################
# General Header                                          #
###########################################################

class _GeneralHeader(object):
    def __init__(self):
        self.fileName = None
        self.fileVersion = None
        self.status = None
        self.itemName = None
        self.mission = None
        self.source = None
        self.destination = None
        self.generationSystem = None
        self.generationTime = None
        self.referenceDocument = None
        self.revision = None
        self.revisionComment = None
        self.remark = None
        return

    def set_from_etnode(self,node):
        self.fileName = node.attrib['fileName']
        self.fileVersion = node.attrib['fileVersion']
        self.status = node.attrib['status']
        for z in node:
            if z.tag == 'itemName':
                self.itemName = z.text
            if z.tag == 'mission':
                self.mission = z.text
            if z.tag == 'source':
                self.source = z.text
            if z.tag == 'destination':
                self.destination = z.text
            if z.tag == 'generationSystem':
                self.generationSystem = z.text
            if z.tag == 'generationTime':
                self.generationTime = z.text
            if z.tag == 'referenceDocument':
                self.referenceDocument = z.text
            if z.tag == 'revision':
                self.revision = z.text
            if z.tag == 'revisionComment':
                self.revisionComment = z.text
            if z.tag == 'remark':
                self.remark = z.text
        return

    def __str__(self):
        retstr  = "GeneralHeader:"+sep+tab
        retlst  = ()
        retstr += "fileName=%s"+sep+tab
        retlst += (self.fileName,)
        retstr += "fileVersion=%s"+sep+tab
        retlst += (self.fileVersion,)
        retstr += "status=%s"+sep+tab
        retlst += (self.status,)
        retstr += "itemName=%s"+sep+tab
        retlst  += (self.itemName,)
        retstr += "mission=%s"+sep+tab
        retlst += (self.mission,)
        retstr += "source=%s"+sep+tab
        retlst += (self.source,)
        retstr += "destination=%s"+sep+tab
        retlst += (self.destination,)
        retstr += "generationSystem=%s"+sep+tab
        retlst += (self.generationSystem,)
        retstr += "generationTime=%s"+sep+tab
        retlst += (self.generationTime,)
        retstr += "referenceDocument=%s"+sep+tab
        retlst += (self.referenceDocument,)
        retstr += "revision=%s"+sep+tab
        retlst += (self.revision,)
        retstr += "revisionComment=%s"+sep+tab
        retlst += (self.revisionComment,)
        retstr += "remark=%s"
        retlst += (self.remark,)
        retstr += sep+":GeneralHeader"
        return retstr % retlst

###########################################################
# Product Components                                      #
###########################################################


class _ProductComponents(object):
    def __init__(self):
        self.annotation = []
        self.imageData = _ImageData()
        self.quicklooks = _QuickLooks()
        self.compositeQuicklook = _CompositeQuickLook()
        self.browseImage = _BrowseImage()
        self.mapPlot = _MapPlot()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'annotation':
                self.annotation.append(_Annotation())
                self.annotation[-1].set_from_etnode(z)
            if z.tag == 'imageData':
                self.imageData.set_from_etnode(z)
            if z.tag == 'quicklooks':
                self.quicklooks.set_from_etnode(z)
            if z.tag == 'compositeQuicklook':
                self.compositeQuicklook.set_from_etnode(z)
            if z.tag == 'browseImage':
                self.browseImage.set_from_etnode(z)
            if z.tag == 'mapPlot':
                self.mapPlot.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "ProductComponents:"+sep+tab
        retlst  = ()
        for a in self.annotation:
            retstr += sep+"%s"
            retlst += (str(a),)
        retstr += sep+"%s"
        retlst += (str(self.imageData),)
        retstr += sep+"%s"
        retlst += (str(self.quicklooks),)
        retstr += sep+"%s"
        retlst += (str(self.compositeQuicklook),)
        retstr += sep+"%s"
        retlst += (str(self.browseImage),)
        retstr += sep+"%s"
        retlst += (str(self.mapPlot),)
        retstr += sep+":ProductComponents"
        return retstr % retlst

class _Annotation(object):
    def __init__(self):
        self.type = None
        self.file = _File()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'type':
                self.type = z.text
            if z.tag == 'file':
                self.file.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "Annotation:"+sep+tab
        retlst  = ()
        retstr += sep+tab+"type=%s"
        retlst += (self.type,)
        retstr += sep+"%s"
        retlst += (str(self.file),)
#        retstr += sep+"%s"
        retstr += sep+":Annotation"
        return retstr % retlst

class _ImageData(object):
    def __init__(self):
        self.layerIndex = None
        self.polLayer = None
        self.file = _File()
        return

    def set_from_etnode(self,node):
        self.layerIndex = int(node.attrib['layerIndex'])
        for z in node:
            if z.tag == 'polLayer':
                self.type = z.text
            if z.tag == 'file':
                self.file.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "ImageData:"+sep+tab
        retlst  = ()
        retstr += sep+tab+"type=%d"
        retlst += (self.layerIndex,)
        retstr += sep+tab+"type=%s"
        retlst += (self.polLayer,)
        retstr += sep+"%s"
        retlst += (str(self.file),)
#        retstr += sep+"%s"
        retstr += sep+":ImageData"
        return retstr % retlst

class _QuickLooks(object):
    def __init__(self):
        self.layerIndex = None
        self.polLayer = None
        self.file = _File()
        return

    def set_from_etnode(self,node):
        self.layerIndex = int(node.attrib['layerIndex'])
        for z in node:
            if z.tag == 'polLayer':
                self.type = z.text
            if z.tag == 'file':
                self.file.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "QuickLooks:"+sep+tab
        retlst  = ()
        retstr += sep+tab+"type=%d"
        retlst += (self.layerIndex,)
        retstr += sep+tab+"type=%s"
        retlst += (self.polLayer,)
        retstr += sep+"%s"
        retlst += (str(self.file),)
#        retstr += sep+"%s"
        retstr += sep+":QuickLooks"
        return retstr % retlst

class _CompositeQuickLook(object):
    def __init__(self):
        self.file = _File()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'file':
                self.file.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "CompositeQuickLook:"+sep+tab
        retlst  = ()
        retstr += sep+"%s"
        retlst += (str(self.file),)
#        retstr += sep+"%s"
        retstr += sep+":CompositeQuickLook"
        return retstr % retlst

class _BrowseImage(object):
    def __init__(self):
        self.file = _File()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'file':
                self.file.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "BrowseImage:"+sep+tab
        retlst  = ()
        retstr += sep+"%s"
        retlst += (str(self.file),)
#        retstr += sep+"%s"
        retstr += sep+":BrowseImage"
        return retstr % retlst

class _MapPlot(object):
    def __init__(self):
        self.file = _File()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'file':
                self.file.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "MapPlot:"+sep+tab
        retlst  = ()
        retstr += sep+"%s"
        retlst += (str(self.file),)
        retstr += sep+":MapPlot"
        return retstr % retlst




###########################################################
# Product Info                                            #
###########################################################

class _ProductInfo(object):
    def __init__(self):
        self.generationInfo = _GenerationInfo()
        self.missionInfo = _MissionInfo()
        self.acquisitionInfo = _AcquisitionInfo()
        self.productVariantInfo = _ProductVariantInfo()
        self.imageDataInfo = _ImageDataInfo()
        self.sceneInfo = _SceneInfo()

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'generationInfo':
                self.generationInfo.set_from_etnode(z)
            if z.tag == 'missionInfo':
                self.missionInfo.set_from_etnode(z)
            if z.tag == 'acquisitionInfo':
                self.acquisitionInfo.set_from_etnode(z)
            if z.tag == 'productVariantInfo':
                self.productVariantInfo.set_from_etnode(z)
            if z.tag == 'imageDataInfo':
                self.imageDataInfo.set_from_etnode(z)
            if z.tag == 'sceneInfo':
                self.sceneInfo.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "ProductInfo:"
        retlst  = ()
        retstr += sep+"%s"
        retlst += (str(self.generationInfo),)
        retstr += sep+"%s"
        retlst += (str(self.missionInfo),)
        retstr += sep+"%s"
        retlst += (str(self.acquisitionInfo),)
        retstr += sep+"%s"
        retlst += (str(self.productVariantInfo),)
        retstr += sep+"%s"
        retlst += (str(self.imageDataInfo),)
        retstr += sep+"%s"
        retlst += (str(self.sceneInfo),)
        retstr += sep+":ProductInfo"
        return retstr % retlst

class _GenerationInfo(object):
    def __init__(self):
        self.logicalProductID = None
        self.receivingStation = None
        self.level0ProcessingFacility = None
        self.level1ProcessingFacility = None
        self.groundOperationsType = None
        self.deliveryInfo = None
        self.copyrightInfo = None
        self.qualityInfo = _QualityInfo()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'logicalProductID':
                self.logicalProductID = z.text
            if z.tag == 'receivingStation':
                self.receivingStation = z.text
            if z.tag == 'level0ProcessingFacility':
                self.level0ProcessingFacility = z.text
            if z.tag == 'level1ProcessingFacility':
                self.level1ProcessingFacility = z.text
            if z.tag == 'groundOperationsType':
                self.groundOperationsType = z.text
            if z.tag == 'deliveryInfo':
                self.deliveryInfo = z.text
            if z.tag == 'copyrightInfo':
                self.copyrightInfo = z.text
            if z.tag == 'qualityInfo':
                self.qualityInfo.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "GenerationInfo:"
        retlst  = ()
        retstr += sep+tab+"logicalProductID=%s"
        retlst += (self.logicalProductID,)
        retstr += sep+tab+"receivingStation=%s"
        retlst += (self.receivingStation,)
        retstr += sep+tab+"level0ProcessingFacility=%s"
        retlst += (self.level0ProcessingFacility,)
        retstr += sep+tab+"level1ProcessingFacility=%s"
        retlst += (self.level1ProcessingFacility,)
        retstr += sep+tab+"groundOperationsType=%s"
        retlst += (self.groundOperationsType,)
        retstr += sep+tab+"deliveryInfo=%s"
        retlst += (self.deliveryInfo,)
        retstr += sep+tab+"copyrightInfo=%s"
        retlst += (self.copyrightInfo,)
        retstr += sep+"%s"
        retlst += (str(self.qualityInfo),)
        retstr += sep+":GenerationInfo"
        return retstr % retlst

class _QualityInfo(object):
    def __init__(self):
        self.qualityInspection = None
        self.qualityRemark = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'qualityInspection':
                self.qualityInspection = z.text
            if z.tag == 'qualityRemark':
                self.qualityRemark = z.text
        return

    def __str__(self):
        retstr  = "QualityInfo:"
        retlst  = ()
        retstr += sep+tab+"qualityInspection=%s"
        retlst += (self.qualityInspection,)
        retstr += sep+tab+"qualityRemark=%s"
        retlst += (self.qualityRemark,)
        retstr += sep+":QualityInfo"
        return retstr % retlst


class _MissionInfo(object):
    def __init__(self):
        self.mission = None
        self.orbitPhase = None
        self.orbitCycle = None
        self.absOrbit = None
        self.relOrbit = None
        self.numOrbitsInCycle = None
        self.orbitDirection = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'mission':
                self.mission = z.text
            if z.tag == 'orbitPhase':
                self.orbitPhase = int(z.text)
            if z.tag == 'orbitCycle':
                self.orbitCycle = int(z.text)
            if z.tag == 'absOrbit':
                self.absOrbit = int(z.text)
            if z.tag == 'relOrbit':
                self.relOrbit = int(z.text)
            if z.tag == 'numOrbitsInCycle':
                self.numOrbitsInCycle = int(z.text)
            if z.tag == 'orbitDirection':
                self.orbitDirection = z.text


    def __str__(self):
        retstr  = "MissionInfo:"+sep+tab
        retstr += "mission=%s"+sep+tab
        retlst  = (self.mission,)
        retstr += "orbitPhase=%d"+sep+tab
        retlst += (self.orbitPhase,)
        retstr += "orbitCycle=%d"+sep+tab
        retlst += (self.orbitCycle,)
        retstr += "absOrbit=%d"+sep+tab
        retlst += (self.absOrbit,)
        retstr += "relOrbit=%d"+sep+tab
        retlst += (self.relOrbit,)
        retstr += "numOrbitsInCycle=%d"+sep+tab
        retlst += (self.numOrbitsInCycle,)
        retstr += "orbitDirection=%s"
        retlst += (self.orbitDirection,)
        retstr += sep+":MissionInfo"
        return retstr % retlst

class _PolarisationList(object):
    def __init__(self):
        self.polLayer = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'polLayer':
                self.polLayer = z.text

    def __str__(self):
        retstr  = "PolarisationList:"+sep+tab
        retstr += "polLayer=%s"
        retlst  = (self.polLayer,)
        retstr += sep+":PolarisationList"
        return retstr % retlst


class _ImagingModeStripMap(object):
    def __init__(self):
        self.azimuthBeamID = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'azimuthBeamID':
                self.azimuthBeamID = z.text
        return

    def __str__(self):
        retstr  = "StripMap:"+sep+tab
        retstr += "aziuthBeamID=%s"
        retlst  = (self.azimuthBeamID,)
        retstr += sep+":StripMap"
        return retstr % retlst

class _ImagingModeSpecificInfo(object):
    def __init__(self):
        self.stripMap = _ImagingModeStripMap()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'stripMap':
                self.stripMap.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "ImagingModeSpecificInfo:"+sep
        retstr += "%s"
        retlst  = (str(self.stripMap),)
        retstr += sep+":ImagingModeSpecificInfo"
        return retstr % retlst

class _AcquisitionInfo(object):
    def __init__(self):
        self.sensor = None
        self.imagingMode = None
        self.lookDirection = None
        self.antennaReceiveConfiguration = None
        self.polarisationMode = None
        self.polarisationList = _PolarisationList()
        self.elevationBeamConfiguration = None
        self.imagingModeSpecificInfo = _ImagingModeSpecificInfo()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'sensor':
                self.sensor = z.text
            if z.tag == 'imagingMode':
                self.imagingMode = z.text
            if z.tag == 'antennaReceiveConfiguration':
                self.antennaReceiveConfiguration = z.text
            if z.tag == 'polarisationMode':
                self.polarisationMode = z.text
            if z.tag == 'polarisationList':
                self.polarisationList.set_from_etnode(z)
            if z.tag == 'lookDirection':
                self.lookDirection = z.text
            if z.tag == 'elevationBeamConfiguration':
                self.elevationBeamConfiguration = z.text
            if z.tag == 'imagingModeSpecificInfo':
                self.imagingModeSpecificInfo.set_from_etnode(z)

    def __str__(self):
        retstr  = "AcquisitionInfo:"+sep+tab
        retstr += "sensor=%s"+sep+tab
        retlst  = (self.sensor,)
        retstr += "imagingMode=%s"+sep+tab
        retlst += (self.imagingMode,)
        retstr += "lookDirection=%s"+sep+tab
        retlst += (self.lookDirection,)
        retstr += "antennaReceiveConfiguration=%s"+sep+tab
        retlst += (self.antennaReceiveConfiguration,)
        retstr += "polarisationMode=%s"+sep
        retlst += (self.polarisationMode,)
        retstr += "%s"+sep+tab
        retlst += (str(self.polarisationList),)
        retstr += "elevationBeamConfiguration=%s"+sep+tab
        retlst += (self.elevationBeamConfiguration,)
        retstr += "%s"
        retlst += (str(self.imagingModeSpecificInfo),)
        retstr += sep+":AcquisitionInfo"
        return retstr % retlst

class _ProductVariantInfo(object):
    def __init__(self):
        self.productType = None
        self.productVariant = None
        self.projection = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'productType':
                self.productType = z.text
            if z.tag == 'productVariant':
                self.productVariant = z.text
            if z.tag == 'projection':
                self.projection = z.text
        return

    def __str__(self):
        retstr  = "ProductVariant:"+sep+tab
        retlst  = ()
        retstr += "productType=%s"+sep+tab
        retlst += (self.productType,)
        retstr += "productVariant=%s"+sep+tab
        retlst += (self.productVariant,)
        retstr += "projection=%s"
        retlst += (self.projection,)
        retstr += sep+":ProductVariant"
        return retstr % retlst

class _ImageRaster(object):
    def __init__(self):
        self.numberOfRows = None
        self.numberOfColumns = None
        self.rowSpacing = None
        self.columnSpacing = None
        self.groundRangeResolution = None
        self.azimuthResolution = None
        self.azimuthLooks = None
        self.rangeLooks = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'numberOfRows':
                self.numberOfRows = int(z.text)
            if z.tag == 'numberOfColumns':
                self.numberOfColumns = int(z.text)
            if z.tag == 'rowSpacing':
                self.rowSpacing = float(z.text)
            if z.tag == 'columnSpacing':
                self.columnSpacing = float(z.text)
            if z.tag == 'groundRangeResolution':
                self.groundRangeResolution = float(z.text)
            if z.tag == 'azimuthResolution':
                self.azimuthResolution = float(z.text)
            if z.tag == 'azimuthLooks':
                self.azimuthLooks = float(z.text)
            if z.tag == 'rangeLooks':
                self.rangeLooks = float(z.text)
        return

    def __str__(self):
        retstr  = "ImageRaster:"
        retlst  = ()
        retstr += sep+tab+"numberOfRows=%d"
        retlst += (self.numberOfRows,)
        retstr += sep+tab+"numberOfColumns=%d"
        retlst += (self.numberOfColumns,)
        retstr += sep+tab+"rowSpacing=%-27.20g"
        retlst += (self.rowSpacing,)
        retstr += sep+tab+"columnSpacing=%-27.20g"
        retlst += (self.columnSpacing,)
        retstr += sep+tab+"groundRangeResolution=%-27.20g"
        retlst += (self.groundRangeResolution,)
        retstr += sep+tab+"azimuthResolution=%-27.20g"
        retlst += (self.azimuthResolution,)
        retstr += sep+tab+"azimuthLooks=%-27.20g"
        retlst += (self.azimuthLooks,)
        retstr += sep+tab+"rangeLooks=%-27.20g"
        retlst += (self.rangeLooks,)
        retstr += sep+":ImageRaster"
        return retstr % retlst


class _ImageDataInfo(object):
    def __init__(self):
        self.imageRaster = _ImageRaster()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'imageRaster':
                self.imageRaster.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "ImageDataInfo:"
        retlst  = ()
        retstr += sep+"%s"
        retlst += (str(self.imageRaster),)
        retstr += sep+":ImageDataInfo"
        return retstr % retlst


class _SceneInfoTime(object):
    def __init__(self):
        self.timeUTC = None
        self.timeGPS = None
        self.timeGPSFraction = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'timeUTC':
                self.timeUTC = z.text
            if z.tag == 'timeGPS':
                self.timeGPS = float(z.text)
            if z.tag == 'timeGPSFraction':
                self.timeGPSFraction = float(z.text)

    def __str__(self):
        retstr  = "Time:"+sep+tab
        retlst  = ()
        retstr += "timeUTC=%s"
        retlst += (self.timeUTC,)
        retstr += "timeGPS=%s"
        retlst += (self.timeGPS,)
        retstr += "timeGPSFraction=%s"
        retlst += (self.timeGPSFraction,)
        retstr += sep+":Time"
        return retstr % retlst

class _SceneInfoRangeTime(object):
    def __init__(self):
        self.firstPixel = None
        self.lastPixel = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'firstPixel':
                self.firstPixel = float(z.text)
            if z.tag == 'lastPixel':
                self.lastPixel = float(z.text)

    def __str__(self):
        retstr  = "RangeTime:"+sep+tab
        retlst  = ()
        retstr += "firstPixel=%-27.20g"+sep+tab
        retlst += (self.firstPixel,)
        retstr += "lastPixel=%-27.20g"
        retlst += (self.lastPixel,)
        retstr += sep+":RangeTime"
        return retstr % retlst

class _SceneInfoSceneCornerCoord(object):
    def __init__(self):
        self.refRow = None
        self.refColumn = None
        self.lat = None
        self.lon = None
        self.azimuthTimeUTC = None
        self.rangeTime = None
        self.incidenceAngle = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'refRow':
                self.refRow = int(z.text)
            if z.tag == 'refColumn':
                self.refColumn = int(z.text)
            if z.tag == 'lat':
                self.lat = float(z.text)
            if z.tag == 'lon':
                self.lon = float(z.text)
            if z.tag == 'azimuthTimeUTC':
                self.azimuthTimeUTC = z.text
            if z.tag == 'rangeTime':
                self.rangeTime = float(z.text)
            if z.tag == 'incidenceAngle':
                self.incidenceAngle = float(z.text)

    def __str__(self):
        retstr  = "SceneCornerCoord:"+sep+tab
        retlst  = ()
        retstr += "refRow=%d"+sep+tab
        retlst += (self.refRow,)
        retstr += "refColumn=%d"+sep+tab
        retlst += (self.refColumn,)
        retstr += "lat=%-27.20g"+sep+tab
        retlst += (self.lat,)
        retstr += "lon=%-27.20g"+sep+tab
        retlst += (self.lon,)
        retstr += "azimuthTimeUTC=%s"+sep+tab
        retlst += (self.azimuthTimeUTC,)
        retstr += "rangeTime=%-27.20g"+sep+tab
        retlst += (self.rangeTime,)
        retstr += "incidenceAngle=%-27.20g"
        retlst += (self.incidenceAngle,)
        retstr += sep+":SceneCornerCoord"
        return retstr % retlst


class _SceneCenterCoord(object):
    def __init__(self):
        self.refRow = None
        self.refColumn = None
        self.lat = None
        self.lon = None
        self.azimuthTimeUTC = None
        self.rangeTime = None
        self.incidenceAngle = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'refRow':
                self.refRow = int(z.text)
            if z.tag == 'refColumn':
                self.refColumn = int(z.text)
            if z.tag == 'lat':
                self.lat = float(z.text)
            if z.tag == 'lon':
                self.lon = float(z.text)
            if z.tag == 'azimuthTimeUTC':
                self.azimuthTimeUTC = z.text
            if z.tag == 'rangeTime':
                self.rangeTime = float(z.text)
            if z.tag == 'incidenceAngle':
                self.incidenceAngle = float(z.text)
        return

    def __str__(self):
        retstr  = "SceneCenterCoord:"+sep+tab
        retlst  = ()
        retstr += "refRow=%d"+sep+tab
        retlst += (self.refRow,)
        retstr += "refColumn=%d"+sep+tab
        retlst += (self.refColumn,)
        retstr += "lat=%-27.20g"+sep+tab
        retlst += (self.lat,)
        retstr += "lon=%-27.20g"+sep+tab
        retlst += (self.lon,)
        retstr += "azimuthTimeUTC=%s"+sep+tab
        retlst += (self.azimuthTimeUTC,)
        retstr += "rangeTime=%-27.20g"
        retlst += (self.rangeTime,)
        retstr += "incidenceAngle=%-27.20g"
        retlst += (self.incidenceAngle,)
        retstr += sep+":SceneCenterCoord"
        return retstr % retlst

class _SceneInfo(object):
    def __init__(self):
        self.sceneID = None
        self.start = _SceneInfoTime()
        self.stop = _SceneInfoTime()
        self.rangeTime = _SceneInfoRangeTime()
        self.sceneCornerCoord = [_SceneInfoSceneCornerCoord(),_SceneInfoSceneCornerCoord(),_SceneInfoSceneCornerCoord(),_SceneInfoSceneCornerCoord()]
        self.sceneCenterCoord = _SceneCenterCoord()
        return

    def set_from_etnode(self,node):
        iCorner = -1
        for z in node:
            if z.tag == 'sceneID':
                self.sceneID = z.text
            if z.tag == 'start':
                self.start.set_from_etnode(z)
            if z.tag == 'stop':
                self.stop.set_from_etnode(z)
            if z.tag == 'rangeTime':
                self.rangeTime.set_from_etnode(z)
            if z.tag == 'sceneCornerCoord':
                iCorner += 1
                self.sceneCornerCoord[iCorner].set_from_etnode(z)
            if z.tag == 'sceneCenterCoord':
                self.sceneCenterCoord.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "SceneInfo:"+sep+tab
        retlst  = ()
        retstr += "sceneID=%s"+sep
        retlst += (self.sceneID,)
        retstr += "%s"+sep
        retlst += (str(self.start),)
        retstr += "%s"+sep
        retlst += (str(self.stop),)
        retstr += "%s"
        retlst += (str(self.rangeTime),)
        for i in range(4):
            retstr += sep+"%s"
            retlst += (str(self.sceneCornerCoord[i]),)
        retstr += sep+"%s"
        retlst += (str(self.sceneCenterCoord),)
        retstr += sep+":SceneInfo"
        return retstr % retlst


###########################################################
# Product Specific                                        #
###########################################################

class _ProductSpecific(object):
    def __init__(self):
        self.complexImageInfo = _ComplexImageInfo()
        return
    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'complexImageInfo':
                self.complexImageInfo.set_from_etnode(z)
        return

    def __str__(self):
        return "ProductSpecific:\n%s\n:ProductSpecific" % (str(self.complexImageInfo),)

class _ComplexImageInfo(object):
    def __init__(self):
        self.commonPRF = None
        self.commonRSF = None
        self.slantRangeResolution = None
        self.projectedSpacingAzimuth = None
        self.projectedSpacingRange = _ProjectedSpacingRange()
        self.imageCoordinateType = None
        self.imageDataStartWith = None
        self.quicklookDataStartWith = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'commonPRF':
                self.commonPRF = float(z.text)
            if z.tag == 'commonRSF':
                self.commonRSF = float(z.text)
            if z.tag == 'slantRangeResolution':
                self.slantRangeResolution = float(z.text)
            if z.tag == 'projectedSpacingAzimuth':
                self.projectedSpacingAzimuth = float(z.text)
            if z.tag == 'projectedSpacingRange':
                self.projectedSpacingRange.set_from_etnode(z)
            if z.tag == 'imageCoordinateType':
                self.imageCoordinateType = z.text
            if z.tag == 'imageDataStartWith':
                self.imageDataStartWith = z.text
            if z.tag == 'quicklookDataStartWith':
                self.quicklookDataStartWith = z.text

    def __str__(self):
        retstr  = "ComplexImageInfo:"+sep+tab
        retstr += "commonPRF=%-27.20g"+sep+tab
        retlst  = (self.commonPRF,)
        retstr += "commonRSF=%-27.20g"+sep+tab
        retlst += (self.commonRSF,)
        retstr += "slantRangeResolution=%-27.20g"+sep+tab
        retlst += (self.slantRangeResolution,)
        retstr += "projectedSpacingAzimuth=%-27.20g"+sep
        retlst += (self.projectedSpacingAzimuth,)
        retstr += "%s"+sep+tab
        retlst += (self.projectedSpacingRange,)
        retstr += "imageCoordinateType=%s"+sep+tab
        retlst += (self.imageCoordinateType,)
        retstr += "imageDataStartWith=%s"+sep+tab
        retlst += (self.imageDataStartWith,)
        retstr += "quicklookDataStartWith=%s"
        retlst += (self.quicklookDataStartWith,)
        retstr += sep+":ComplexImageInfo"
        return retstr % retlst

class _ProjectedSpacingRange(object):
    def __init__(self):
        self.groundNear = None
        self.groundFar = None
        self.slantRange = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'groundNear':
                self.groundNear = float(z.text)
            if z.tag == 'groundFar':
                self.groundFar = float(z.text)
            if z.tag == 'slantRange':
                self.slantRange = float(z.text)

    def __str__(self):
        retstr  = "ProjectedSpacingRange:"
        retlst  = ()
        retstr += sep+tab+"groundNear=%-27.20g"
        retlst += (self.groundNear,)
        retstr += sep+tab+"groundFar=%-27.20g"
        retlst += (self.groundFar,)
        retstr += sep+tab+"slantRange=%-27.20g"
        retlst += (self.slantRange,)
        retstr += sep+":ProjectedSpacingRange"
        return retstr  % retlst


###########################################################
# Platform                                                #
###########################################################

class _Platform(object):
    def __init__(self):
        self.referenceData = _PlatformReferenceData()
        self.orbit = _Orbit()
        self.attitude = _Attitude()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'referenceData':
                self.referenceData.set_from_etnode(z)
            if z.tag == 'orbit':
                self.orbit.set_from_etnode(z)
            if z.tag == 'attitude':
                self.attitude.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "Platform:"+sep+tab
        retstr += "%s"+sep
        retlst  = (str(self.referenceData),)
        retstr += "%s"+sep
        retlst += (str(self.orbit),)
        retstr += "%s"
        retlst += (str(self.attitude),)
        retstr += sep+":Platform"
        return retstr % retlst

class _SARAntennaPosition(object):
    def __init__(self):
        self.DRAoffset = None
        self.x = None
        self.y = None
        self.z = None

    def set_from_etnode(self,node):
        self.DRAoffset = node.attrib['DRAoffset']
        for w in node:
            if w.tag == 'x':
                self.x = float(w.text)
            if w.tag == 'y':
                self.y = float(w.text)
            if w.tag == 'z':
                self.z = float(w.text)

    def __str__(self):
        retstr  = "SARAntennaPosition:"+sep+tab
        retstr += "DRAoffset=%s"+sep+tab
        retlst  = (self.DRAoffset,)
        retstr += "x=%-27.20g"+sep+tab+"y=%-27.20g"+sep+tab+"z=%-27.20g"
        retlst += (self.x,self.y,self.z)
        retstr += sep+":SARAntennaPosition"
        return retstr % retlst

class _GPSAntennaPosition(object):
    def __init__(self):
        self.GPSreceiver = None
        self.unit = None
        self.x = None
        self.y = None
        self.z = None

    def set_from_etnode(self,node):
        self.GPSreceiver = node.attrib['GPSreceiver']
        self.unit = node.attrib['unit']
        for w in node:
            if w.tag == 'x':
                self.x = float(w.text)
            if w.tag == 'y':
                self.y = float(w.text)
            if w.tag == 'z':
                self.z = float(w.text)

    def __str__(self):
        retstr  = "GPSAntennaPosition:"+sep+tab
        retstr += "GPSreceiver=%s"+sep+tab
        retlst  = (self.GPSreceiver,)
        retstr += "unit=%s"+sep+tab
        retlst += (self.unit,)
        retstr += "x=%-27.20g"+sep+tab+"y=%-27.20g"+sep+tab+"z=%-27.20g"
        retlst += (self.x,self.y,self.z)
        retstr += sep+":GPSAntennaPosition"
        return retstr % retlst

class _PlatformReferenceData(object):
    def __init__(self):
        self.SARAntennaMechanicalBoresight = None
        self.SARAntennaPosition = _SARAntennaPosition()
        self.GPSAntennaPosition = (_GPSAntennaPosition(),)
        self.GPSAntennaPosition += (_GPSAntennaPosition(),)
        self.GPSAntennaPosition += (_GPSAntennaPosition(),)
        self.GPSAntennaPosition += (_GPSAntennaPosition(),)
        return

    def set_from_etnode(self,node):
        iGPSAnt = -1
        for x in node:
            if x.tag == 'SARAntennaMechanicalBoresight':
                self.SARAntennaMechanicalBoresight = float(x.text)
            if x.tag == 'SARAntennaPosition':
                self.SARAntennaPosition.set_from_etnode(x)
            if x.tag == 'GPSAntennaPosition':
                iGPSAnt += 1
                self.GPSAntennaPosition[iGPSAnt].set_from_etnode(x)

    def __str__(self):
        retstr  = "ReferenceData:"+sep+tab
        retstr += "SARAntennaMechanicalBoresight=%-27.20g"+sep
        retlst  = (self.SARAntennaMechanicalBoresight,)
        retstr += "%s"
        retlst += (self.SARAntennaPosition,)
        for i in range(4):
            retstr += sep+"%s"
            retlst += (self.GPSAntennaPosition[i],)
        retstr += sep+":ReferenceData"
        return retstr % retlst

class _FirstStateTime(object):
    def __init__(self):
        self.firstStateTimeUTC = None
        self.firstStateTimeGPS = None
        self.firstStateTimeGPSFraction = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'firstStateTimeUTC':
                self.firstStateTimeUTC = z.text
            if z.tag == 'firstStateTimeGPS':
                self.firstStateTimeGPS = float(z.text)
            if z.tag == 'firstStateTimeGPSFraction':
                self.firstStateTimeGPSFraction = float(z.text)

    def __str__(self):
        retstr  = "FirstStateTime:"+sep+tab
        retstr += "firstStateTimeUTC=%s"+sep+tab
        retlst  = (self.firstStateTimeUTC,)
        retstr += "firstStateTimeGPS=%-27.20g"+sep+tab
        retlst += (self.firstStateTimeGPS,)
        retstr += "firstStateTimeGPSFraction=%-27.20g"
        retlst += (self.firstStateTimeGPSFraction,)
        retstr += sep+":FirstStateTime"
        return retstr % retlst

class _LastStateTime(object):
    def __init__(self):
        self.lastStateTimeUTC = None
        self.lastStateTimeGPS = None
        self.lastStateTimeGPSFraction = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'lastStateTimeUTC':
                self.lastStateTimeUTC = z.text
            if z.tag == 'lastStateTimeGPS':
                self.lastStateTimeGPS = float(z.text)
            if z.tag == 'lastStateTimeGPSFraction':
                self.lastStateTimeGPSFraction = float(z.text)

    def __str__(self):
        retstr  = "LastStateTime:"+sep+tab
        retstr += "lastStateTimeUTC=%s"+sep+tab
        retlst  = (self.lastStateTimeUTC,)
        retstr += "lastStateTimeGPS=%-27.20g"+sep+tab
        retlst += (self.lastStateTimeGPS,)
        retstr += "lastStateTimeGPSFraction=%-27.20g"
        retlst += (self.lastStateTimeGPSFraction,)
        retstr += sep+":LastStateTime"
        return retstr % retlst

class _OrbitHeader(object):
    def __init__(self):
        self.generationSystem = None
        self.generationSystemVersion = None
        self.sensor = None
        self.accuracy = None
        self.stateVectorRefFrame = None
        self.stateVectorRefTime = None
        self.stateVecFormat = None
        self.numStateVectors = None
        self.firstStateTime = _FirstStateTime()
        self.lastStateTime = _LastStateTime()
        self.stateVectorTimeSpacing = None
        self.positionAccuracyMargin = None
        self.velocityAccuracyMargin = None
        self.recProcessingTechnique = None
        self.recPolDegree = None
        self.dataGapIndicator = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'generationSystem':
                self.generationSystem = z.text
                self.generationSystemVersion = z.attrib['version']
            if z.tag == 'sensor':
                self.sensor = z.text
            if z.tag == 'accuracy':
                self.accuracy = z.text
            if z.tag == 'stateVectorRefFrame':
                self.stateVectorRefFrame = z.text
            if z.tag == 'stateVectorRefTime':
                self.stateVectorRefTime = z.text
            if z.tag == 'stateVecFormat':
                self.stateVecFormat = z.text
            if z.tag == 'numStateVectors':
                self.numStateVectors = int(z.text)
            if z.tag == 'firstStateTime':
                self.firstStateTime.set_from_etnode(z)
            if z.tag == 'lastStateTime':
                self.lastStateTime.set_from_etnode(z)
            if z.tag == 'stateVectorTimeSpacing':
                self.stateVectorTimeSpacing = float(z.text)
            if z.tag == 'positionAccuracyMargin':
                self.positionAccuracyMargin = float(z.text)
            if z.tag == 'velocityAccuracyMargin':
                self.velocityAccuracyMargin = float(z.text)
            if z.tag == 'recProcessingTechnique':
                self.recProcessingTechnique = z.text
            if z.tag == 'recPolDegree':
                self.recPolDegree = int(z.text)
            if z.tag == 'dataGapIndicator':
                self.dataGapIndicator = float(z.text)
        return

    def __str__(self):
        retstr  = "OrbitHeader:"+sep+tab
        retstr += "generationSystem=%s"+sep+tab
        retlst  = (self.generationSystem,)
        retstr += "generationSystemVersion=%s"+sep+tab
        retlst += (self.generationSystemVersion,)
        retstr += "sensor=%s"+sep+tab
        retlst += (self.sensor,)
        retstr += "accuracy=%s"+sep+tab
        retlst += (self.accuracy,)
        retstr += "stateVectorRefFrame=%s"+sep+tab
        retlst += (self.stateVectorRefFrame,)
        retstr += "stateVectorRefTime=%s"+sep+tab
        retlst += (self.stateVectorRefTime,)
        retstr += "stateVecFormat=%s"+sep+tab
        retlst += (self.stateVecFormat,)
        retstr += "numStateVectors=%d"+sep
        retlst += (self.numStateVectors,)
        retstr += "%s"+sep
        retlst += (str(self.firstStateTime),)
        retstr += "%s"+sep
        retlst += (str(self.lastStateTime),)
        retstr += "stateVectorTimeSpacing=%-27.20g"+sep+tab
        retlst += (self.stateVectorTimeSpacing,)
        retstr += "positionAccuracyMargin=%-27.20g"+sep+tab
        retlst += (self.positionAccuracyMargin,)
        retstr += "velocityAccuracyMargin=%-27.20g"+sep+tab
        retlst += (self.velocityAccuracyMargin,)
        retstr += "recProcessingTechnique=%s"+sep+tab
        retlst += (self.recProcessingTechnique,)
        retstr += "recPolDegree=%d"+sep+tab
        retlst += (self.recPolDegree,)
        retstr += "dataGapIndicator=%-27.20g"
        retlst += (self.dataGapIndicator,)
        retstr += sep+":OrbitHeader"
        return retstr % retlst

class _StateVec(object):
    def __init__(self):
        self.maneuver = None
        self.num = None
        self.qualInd = None
        self.timeUTC = None
        self.timeGPS = None
        self.timeGPSFraction = None
        self.posX = None
        self.posY = None
        self.posZ = None
        self.velX = None
        self.velY = None
        self.velZ = None

    def set_from_etnode(self,node):
        self.maneuver = node.attrib['maneuver']
        self.num = int(node.attrib['num'])
        self.qualInd = int(node.attrib['qualInd'])
        for z in node:
            if z.tag == 'timeUTC':
                self.timeUTC = datetime.datetime.strptime(z.text,"%Y-%m-%dT%H:%M:%S.%f")
            if z.tag == 'timeGPS':
                self.timeGPS = float(z.text)
            if z.tag == 'timeGPSFraction':
                self.timeGPSFraction = float(z.text)
            if z.tag == 'posX':
                self.posX = float(z.text)
            if z.tag == 'posY':
                self.posY = float(z.text)
            if z.tag == 'posZ':
                self.posZ = float(z.text)
            if z.tag == 'velX':
                self.velX = float(z.text)
            if z.tag == 'velY':
                self.velY = float(z.text)
            if z.tag == 'velZ':
                self.velZ = float(z.text)
        return

    def __str__(self):
        retstr  = "StateVec:"+sep+tab
        retstr += "maneuver=%s"+sep+tab
        retlst  = (self.maneuver,)
        retstr += "num=%d"+sep+tab
        retlst += (self.num,)
        retstr += "qualInd=%d"+sep+tab
        retlst += (self.qualInd,)
        retstr += "timeUTC=%s"+sep+tab
        retlst += (self.timeUTC,)
        retstr += "timeGPS=%-27.20g"+sep+tab
        retlst += (self.timeGPS,)
        retstr += "timeGPSFraction=%-27.20g"+sep+tab
        retlst += (self.timeGPSFraction,)
        retstr += "posX=%-27.20g"+sep+tab+"posY=%-27.20g"+sep+tab+"posZ=%-27.20g"+sep+tab
        retlst += (self.posX,self.posY,self.posZ)
        retstr += "velX=%-27.20g"+sep+tab+"velY=%-27.20g"+sep+tab+"velZ=%-27.20g"
        retlst += (self.velX,self.velY,self.velZ)
        retstr += sep+":StateVec"
        return retstr % retlst

class _Orbit(object):
    def __init__(self):
        self.orbitHeader = _OrbitHeader()
        self.stateVec = ()

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'orbitHeader':
                self.orbitHeader.set_from_etnode(z)
            if z.tag == 'stateVec':
                self.stateVec += (_StateVec(),)
                self.stateVec[-1].set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "Orbit:"+sep
        retstr += "%s"
        retlst  = (self.orbitHeader,)
        for s in self.stateVec:
            retstr += sep+"%s"
            retlst += (str(s),)
        retstr += sep+":Orbit"
        return retstr % retlst

class _AttitudeData(object):
    def __init__(self):
        self.antsteerInd = None
        self.maneuver = None
        self.num = None
        self.qualInd = None
        self.timeUTC = None
        self.timeGPS = None
        self.timeGPSFraction = None
        self.q0 = None
        self.q1 = None
        self.q2 = None
        self.q3 = None

    def set_from_etnode(self,node):
        self.maneuver = node.attrib['antsteerInd']
        self.maneuver = node.attrib['maneuver']
        self.num = int(node.attrib['num'])
        self.qualInd = int(node.attrib['qualInd'])
        for z in node:
            if z.tag == 'timeUTC':
                self.timeUTC = z.text
            if z.tag == 'timeGPS':
                self.timeGPS = float(z.text)
            if z.tag == 'timeGPSFraction':
                self.timeGPSFraction = float(z.text)
            if z.tag == 'q0':
                self.q0 = float(z.text)
            if z.tag == 'q1':
                self.q1 = float(z.text)
            if z.tag == 'q2':
                self.q2 = float(z.text)
            if z.tag == 'q3':
                self.q3 = float(z.text)
        return

    def __str__(self):
        retstr  = "AttitudeData:"+sep+tab
        retstr += "antsteerInd=%s"+sep+tab
        retlst  = (self.antsteerInd,)
        retstr += "maneuver=%s"+sep+tab
        retlst += (self.maneuver,)
        retstr += "num=%d"+sep+tab
        retlst += (self.num,)
        retstr += "qualInd=%d"+sep+tab
        retlst += (self.qualInd,)
        retstr += "timeUTC=%s"+sep+tab
        retlst += (self.timeUTC,)
        retstr += "timeGPS=%-27.20g"+sep+tab
        retlst += (self.timeGPS,)
        retstr += "timeGPSFraction=%-27.20g"+sep+tab
        retlst += (self.timeGPSFraction,)
        retstr += "q0=%-27.20g"+sep+tab+"q1=%-27.20g"+sep+tab+"q2=%-27.20g"+sep+tab+"q3=%-27.20g"
        retlst += (self.q0,self.q1,self.q2,self.q3)
        retstr += sep+":AttitudeData"
        return retstr % retlst

class _FirstAttitudeTime(object):
    def __init__(self):
        self.firstAttitudeTimeUTC = None
        self.firstAttitudeTimeGPS = None
        self.firstAttitudeTimeGPSFraction = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'firstAttitudeTimeUTC':
                self.firstAttitudeTimeUTC = z.text
            if z.tag == 'firstAttitudeTimeGPS':
                self.firstAttitudeTimeGPS = float(z.text)
            if z.tag == 'firstAttitudeTimeGPSFraction':
                self.firstAttitudeTimeGPSFraction = float(z.text)

    def __str__(self):
        retstr  = "FirstAttitudeTime:"+sep+tab
        retstr += "firstAttitudeTimeUTC=%s"+sep+tab
        retlst  = (self.firstAttitudeTimeUTC,)
        retstr += "firstAttitudeTimeGPS=%-27.20g"+sep+tab
        retlst += (self.firstAttitudeTimeGPS,)
        retstr += "firstAttitudeTimeGPSFraction=%-27.20g"
        retlst += (self.firstAttitudeTimeGPSFraction,)
        retstr += sep+":FirstAttitudeTime"
        return retstr % retlst

class _LastAttitudeTime(object):
    def __init__(self):
        self.lastAttitudeTimeUTC = None
        self.lastAttitudeTimeGPS = None
        self.lastAttitudeTimeGPSFraction = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'lastAttitudeTimeUTC':
                self.lastAttitudeTimeUTC = z.text
            if z.tag == 'lastAttitudeTimeGPS':
                self.lastAttitudeTimeGPS = float(z.text)
            if z.tag == 'lastAttitudeTimeGPSFraction':
                self.lastAttitudeTimeGPSFraction = float(z.text)

    def __str__(self):
        retstr  = "LastAttitudeTime:"+sep+tab
        retstr += "lastAttitudeTimeUTC=%s"+sep+tab
        retlst  = (self.lastAttitudeTimeUTC,)
        retstr += "lastAttitudeTimeGPS=%-27.20g"+sep+tab
        retlst += (self.lastAttitudeTimeGPS,)
        retstr += "lastAttitudeTimeGPSFraction=%-27.20g"
        retlst += (self.lastAttitudeTimeGPSFraction,)
        retstr += sep+":LastAttitudeTime"
        return retstr % retlst

class _AttitudeDataRefFrame(object):
    def __init__(self):
        self.FromFrame = None
        self.ToFrame = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'FromFrame':
                self.FromFrame = z.text
            if z.tag == 'ToFrame':
                self.ToFrame = z.text
        return

    def __str__(self):
        retstr  = "AttitudeDataRefFrame"+sep+tab
        retstr += "FromFrame=%s"+sep+tab
        retlst  = (self.FromFrame,)
        retstr += "ToFrame=%s"
        retlst += (self.ToFrame,)
        retstr += sep+":AttitudeDataRefFrame"
        return retstr % retlst

class _AttitudeHeader(object):
    def __init__(self):
        self.generationSystem = None
        self.generationSystemVersion = None
        self.sensor = None
        self.accuracy = None
        self.attitudeDataRefFrames = _AttitudeDataRefFrame()
        self.attitudeDataRefTime = None
        self.attitudeDataFormat = None
        self.numRecords = None
        self.firstAttitudeTime = _FirstAttitudeTime()
        self.lastAttitudeTime = _LastAttitudeTime()
        self.attitudeDataTimeSpacing = None
        self.accuracyMargin = None
        self.recInterpolTechnique = None
        self.recInterpolPolDegree = None
        self.dataGapIndicator = None
        self.steeringLawIndicator = None

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'generationSystem':
                self.generationSystem = z.text
                self.generationSystemVersion = z.attrib['version']
            if z.tag == 'sensor':
                self.sensor = z.text
            if z.tag == 'accuracy':
                self.accuracy = z.text
            if z.tag == 'attitudeDataRefFrame':
                self.attitudeDataRefFrame = z.text
            if z.tag == 'attitudeDataRefTime':
                self.attitudeDataRefTime = z.text
            if z.tag == 'attitudeDataFormat':
                self.attitudeDataFormat = z.text
            if z.tag == 'numRecords':
                self.numRecords = int(z.text)
            if z.tag == 'firstAttitudeTime':
                self.firstAttitudeTime.set_from_etnode(z)
            if z.tag == 'lastAttitudeTime':
                self.lastAttitudeTime.set_from_etnode(z)
            if z.tag == 'attitudeDataTimeSpacing':
                self.attitudeDataTimeSpacing = float(z.text)
            if z.tag == 'accuracyMargin':
                self.accuracyMargin = float(z.text)
            if z.tag == 'recInterpolTechnique':
                self.recInterpolTechnique = z.text
            if z.tag == 'recInterpolPolDegree':
                self.recInterpolPolDegree = int(z.text)
            if z.tag == 'dataGapIndicator':
                self.dataGapIndicator = float(z.text)
            if z.tag == 'steeringLawIndicator':
                self.steeringLawIndicator = z.text
        return

    def __str__(self):
        retstr  = "AttitudeHeader:"+sep+tab
        retstr += "generationSystem=%s"+sep+tab
        retlst  = (self.generationSystem,)
        retstr += "generationSystemVersion=%s"+sep+tab
        retlst += (self.generationSystemVersion,)
        retstr += "sensor=%s"+sep+tab
        retlst += (self.sensor,)
        retstr += "accuracy=%s"+sep
        retlst += (self.accuracy,)
        retstr += "%s"
        retlst += (str(self.attitudeDataRefFrames),)
        retstr += "attitudeDataRefTime=%s"+sep+tab
        retlst += (self.attitudeDataRefTime,)
        retstr += "attitudeDataFormat=%s"+sep+tab
        retlst += (self.attitudeDataFormat,)
        retstr += "numRecords=%d"+sep
        retlst += (self.numRecords,)
        retstr += "%s"+sep
        retlst += (str(self.firstAttitudeTime),)
        retstr += "%s"+sep+tab
        retlst += (str(self.lastAttitudeTime),)
        retstr += "attitudeDataTimeSpacing=%-27.20g"+sep+tab
        retlst += (self.attitudeDataTimeSpacing,)
        retstr += "accuracyMargin=%-27.20g"
        retlst += (self.accuracyMargin,)
        retstr += "recInterpolTechnique=%s"+sep+tab
        retlst += (self.recInterpolTechnique,)
        retstr += "recInterpolPolDegree=%d"+sep+tab
        retlst += (self.recInterpolPolDegree,)
        retstr += "dataGapIndicator=%-27.20g"+sep+tab
        retlst += (self.dataGapIndicator,)
        retstr += "steeringLawIndicator=%s"
        retlst += (self.steeringLawIndicator,)
        retstr += sep+":AttitudeHeader"
        return retstr % retlst

class _Attitude(object):
    def __init__(self):
        self.attitudeHeader = _AttitudeHeader()
        self.attitudeData = ()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'attitudeHeader':
                self.attitudeHeader.set_from_etnode(z)
            if z.tag == 'attitudeData':
                self.attitudeData += (_AttitudeData(),)
                self.attitudeData[-1].set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "Attitude:"+sep+tab
        retstr += "%s"
        retlst  = (self.attitudeHeader,)
        for a in self.attitudeData:
            retstr += sep+"%s"
            retlst += (str(a),)
        retstr += sep+":Attitude"
        return retstr % retlst

############################################################
# Instrument                                               #
############################################################

class _Instrument(object):
    def __init__(self):
        self.instrumentInfoCoordinateType = None
        self.radarParameters = _RadarParameters()
        self.settings = _InstrumentSettings()

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'instrumentInfoCoordinateType':
                self.instrumentInfoCoordinateType = z.text
            if z.tag == 'radarParameters':
                self.radarParameters.set_from_etnode(z)
            if z.tag == 'settings':
                self.settings.set_from_etnode(z)

    def __str__(self):
        retstr  = "Instrument:"+sep+tab
        retlst  = ()
        retstr += "instrumentInfoCoordinateType=%s"+sep
        retlst += (self.instrumentInfoCoordinateType,)
        retstr += "%s"+sep
        retlst += (str(self.radarParameters),)
        retstr += "%s"
        retlst += (str(self.settings),)
        retstr += sep+":Instrument"
        return retstr % retlst

class _RadarParameters(object):
    def __init__(self):
        self.centerFrequency = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'centerFrequency':
                self.centerFrequency = float(z.text)
        return

    def __str__(self):
        retstr  = "RadarParameters:"+sep+tab
        retstr += "centerFrequency=%-27.20g"
        retlst  = (self.centerFrequency,)
        retstr += sep+":RadarParameters"
        return retstr % retlst

class _RxGainSetting(object):
    def __init__(self):
        self.startTimeUTC = None
        self.stopTimeUTC = None
        self.rxGain = None
        self.rxGainCode = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'startTimeUTC':
                self.startTimeUTC = z.text
            if z.tag == 'stopTimeUTC':
                self.stopTimeUTC = z.text
            if z.tag == 'rxGain':
                self.rxGain = float(z.text)
                self.rxGainCode = int(z.attrib['code'])
        return

    def __str__(self):
        retstr  = "RxGainSetting:"+sep+tab
        retlst  = ()
        retstr += "startTimeUTC=%s"+sep+tab
        retlst += (self.startTimeUTC,)
        retstr += "stopTimeUTC=%s"+sep+tab
        retlst += (self.stopTimeUTC,)
        retstr += "rxGain=%-27.20g"+sep+tab
        retlst += (self.rxGain,)
        retstr += "rsGainCode=%d"
        retlst += (self.rxGainCode,)
        retstr += sep+":RxGainSetting"
        return retstr % retlst

class _DataSegment(object):
    def __init__(self):
        self.segmentID = None
        self.startTimeUTC = None
        self.stopTimeUTC = None
        self.numberOfRows = None
        return

    def set_from_etnode(self,node):
        self.segmentID = int(node.attrib['segmentID'])
        for z in node:
            if z.tag == 'startTimeUTC':
                self.startTimeUTC = z.text
            if z.tag == 'stopTimeUTC':
                self.stopTimeUTC = z.text
            if z.tag == 'numberOfRows':
                self.numberOfRows = int(z.text)
        return

    def __str__(self):
        retstr  = "DataSegment:"+sep+tab
        retlst  = ()
        retstr += "segmentID=%d"+sep+tab
        retlst += (self.segmentID,)
        retstr += "startTimeUTC=%s"+sep+tab
        retlst += (self.startTimeUTC,)
        retstr += "stopTimeUTC=%s"+sep+tab
        retlst += (self.stopTimeUTC,)
        retstr += "numberOfRows=%d"
        retlst += (self.numberOfRows,)
        retstr += sep+":DataSegment"
        return retstr % retlst

class _SettingRecord(object):
    def __init__(self):
        self.dataSegment = _DataSegment()
        self.PRF = None
        self.PRFcode = None
        self.echoWindowPosition = None
        self.echoWindowPositionCode = None
        self.echowindowLength = None
        self.echowindowLengthCode = None
        self.pulseType = None
        self.echoIndex = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'dataSegment':
                self.dataSegment.set_from_etnode(z)
            if z.tag == 'PRF':
                self.PRF = float(z.text)
                self.PRFcode = int(z.attrib['code'])
            if z.tag == 'echoWindowPosition':
                self.echoWindowPosition = int(z.text)
                self.echoWindowPositionCode = int(z.attrib['code'])
            if z.tag == 'echowindowLength':
                self.echowindowLength = float(z.text)
                self.echowindowLengthCode = int(z.attrib['code'])
            if z.tag == 'pulseType':
                self.pulseType = z.text
            if z.tag == 'echoIndex':
                self.echoIndex = int(z.text)
        return

    def __str__(self):
        retstr  = "SettingRecord:"+sep
        retlst  = ()
        retstr += "%s"+sep+tab
        retlst += (str(self.dataSegment),)
        retstr += "PRF=%-27.20g"+sep+tab
        retlst += (self.PRF,)
        retstr += "PRFcode=%d"+sep+tab
        retlst += (self.PRFcode,)
        retstr += "echoWindowPosition=%d"+sep+tab
        retlst += (self.echoWindowPosition,)
        retstr += "echoWindowPositionCode=%d"+sep+tab
        retlst += (self.echoWindowPositionCode,)
        retstr += "echowindowLength=%-27.20g"+sep+tab
        retlst += (self.echowindowLength,)
        retstr += "echowindowLengthCode=%d"+sep+tab
        retlst += (self.echowindowLengthCode,)
        retstr += "pulseType=%s"+sep+tab
        retlst += (self.pulseType,)
        retstr += "echoIndex=%d"
        retlst += (self.echoIndex,)
        retstr += sep+":SettingRecord"
        return retstr % retlst

class _InstrumentSettings(object):
    def __init__(self):
        self.polLayer = None
        self.DRAoffset = None
        self.beamID = None
        self.numberOfRxGainChanges = None
        self.rxGainSetting = ()
        self.quantisationID = None
        self.quantisationControl = None
        self.rxBandwidth = None
        self.rxBandwidthCode = None
        self.RSF = None
        self.RSFcode = None
        self.numberOfPRFChanges = None
        self.numberOfEchoWindowPositionChanges = None
        self.numberOfEchoWindowLengthChanges = None
        self.numberOfSettingRecords = None
        self.settingRecord = ()

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'polLayer':
                self.polLayer = z.text
            if z.tag == 'DRAoffset':
                self.DRAoffset = z.text
            if z.tag == 'beamID':
                self.beamID = z.text
            if z.tag == 'numberOfRxGainChanges':
                self.numberOfRxGainChanges = int(z.text)
            if z.tag == 'rxGainSetting':
                self.rxGainSetting += (_RxGainSetting(),)
                self.rxGainSetting[-1].set_from_etnode(z)
            if z.tag == 'quantisationID':
                self.quantisationID = z.text
            if z.tag == 'quantisationControl':
                self.quantisationControl = z.text
            if z.tag == 'rxBandwidth':
                self.rxBandwidth = float(z.text)
                self.rxBandwidthCode = int(z.attrib['code'])
            if z.tag == 'RSF':
                self.RSF = float(z.text)
                self.RSFcode = int(z.attrib['code'])
            if z.tag == 'numberOfPRFChanges':
                self.numberOfPRFChanges = int(z.text)
            if z.tag == 'numberOfEchoWindowPositionChanges':
                self.numberOfEchoWindowPositionChanges = int(z.text)
            if z.tag == 'numberOfEchoWindowLengthChanges':
                self.numberOfEchoWindowLengthChanges = int(z.text)
            if z.tag == 'numberOfSettingRecords':
                self.numberOfSettingRecords = int(z.text)
            if z.tag == 'settingRecord':
                self.settingRecord += (_SettingRecord(),)
                self.settingRecord[-1].set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "Settings:"+sep+tab
        retlst  = ()
        retstr += "polLayer=%s"+sep+tab
        retlst += (self.polLayer,)
        retstr += "DRAoffset=%s"+sep+tab
        retlst += (self.DRAoffset,)
        retstr += "beamID=%s"+sep+tab
        retlst += (self.beamID,)
        retstr += "numberOfRxGainChanges=%d"
        retlst += (self.numberOfRxGainChanges,)
        for x in self.rxGainSetting:
            retstr += sep+"%s"
            retlst += (str(x),)
        retstr += sep+tab+"quantisationID=%s"+sep+tab
        retlst += (self.quantisationID,)
        retstr += "quantisationControl=%s"+sep+tab
        retlst += (self.quantisationControl,)
        retstr += "rxBandwidth=%-27.20g"+sep+tab
        retlst += (self.rxBandwidth,)
        retstr += "rxBandwidthCode=%d"+sep+tab
        retlst += (self.rxBandwidthCode,)
        retstr += "RSF=%-27.20g"+sep+tab
        retlst += (self.RSF,)
        retstr += "RSFcode=%d"+sep+tab
        retlst += (self.RSFcode,)
        retstr += "numberOfPRFChanges=%d"+sep+tab
        retlst += (self.numberOfPRFChanges,)
        retstr += "numberOfEchoWindowPositionChanges=%d"+sep+tab
        retlst += (self.numberOfEchoWindowPositionChanges,)
        retstr += "numberOfEchoWindowLengthChanges=%d"+sep+tab
        retlst += (self.numberOfEchoWindowLengthChanges,)
        retstr += "numberOfSettingRecords=%d"
        retlst += (self.numberOfSettingRecords,)
        for x in self.settingRecord:
            retstr += sep+"%s"
            retlst += (str(x),)
        retstr += sep+":Settings"
        return retstr % retlst

############################################################
# Instrument                                               #
############################################################

class _Processing(object):
    def __init__(self):
        self.geometry = _ProcessingGeometry()
        self.doppler = _ProcessingDoppler()
        self.processingParameter = _ProcessingParameter()
#        self.processingFlags = _ProcessingFlags()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'geometry':
                self.geometry.set_from_etnode(z)
            if z.tag == 'doppler':
                self.doppler.set_from_etnode(z)
            if z.tag == 'processingParameter':
                self.processingParameter.set_from_etnode(z)
#            if z.tag == 'processingFlags':
#                self.processingFlags.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "Processing:"
        retlst  = ()
        retstr += sep+"%s"
        retlst += (str(self.geometry),)
        retstr += sep+"%s"
        retlst += (str(self.doppler),)
        retstr += sep+"%s"
        retlst += (str(self.processingParameter),)
#        retstr += sep+"%s"
#        retlst += (str(self.processingFlags),)
        retstr += sep+":Processing"
        return retstr % retlst

class _ProcessingGeometry(object):
    def __init__(self):
        self.geometryCoordinateType = None
        self.velocityParameter = ()
        self.zeroDopplerVelocity = _ZeroDopplerVelocity()
        self.dopplerRate = ()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'geometryCoordinateType':
                self.geometryCoordinateType = z.text
            if z.tag == 'velocityParameter':
                self.velocityParameter += (_VelocityParameter(),)
                self.velocityParameter[-1].set_from_etnode(z)
            if z.tag == 'zeroDopplerVelocity':
                self.zeroDopplerVelocity.set_from_etnode(z)
            if z.tag == 'dopplerRate':
                self.dopplerRate += (_DopplerRate(),)
                self.dopplerRate[-1].set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "Geometry:"
        retlst  = ()
        retstr += sep+tab+"geometryCoordinateType=%s"
        retlst += (self.geometryCoordinateType,)
        for x in self.velocityParameter:
            retstr += sep+"%s"
            retlst += (str(x),)
        retstr += sep+"%s"
        retlst += (str(self.zeroDopplerVelocity),)
        for x in self.dopplerRate:
            retstr += sep+"%s"
            retlst += (str(x),)
        retstr += sep+":Geometry"
        return retstr % retlst

class _VelocityParameter(object):
    def __init__(self):
        self.timeUTC = None
        self.velocityParameterPolynomial = _VelocityParameterPolynomial()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'timeUTC':
                self.timeUTC = z.text
            if z.tag == 'velocityParameterPolynomial':
                self.velocityParameterPolynomial.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "VelocityParameter:"
        retlst  = ()
        retstr += sep+"self.timeUTC=%s"
        retlst += (self.timeUTC,)
        retstr += sep+"%s"
        retlst += (str(self.velocityParameterPolynomial),)
        retstr += sep+":VelocityParameter"
        return retstr % retlst

class _VelocityParameterPolynomial(object):
    def __init__(self):
        self.validityRangeMin = None
        self.validityRangeMax = None
        self.referencePoint = None
        self.polynomialDegree = None
        self.coefficient = []
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'validityRangeMin':
                self.validityRangeMin = float(z.text)
            if z.tag == 'validityRangeMax':
                self.validityRangeMax = float(z.text)
            if z.tag == 'referencePoint':
                self.referencePoint = float(z.text)
            if z.tag == 'polynomialDegree':
                self.polynomialDegree = int(z.text)
            if z.tag == 'coefficient':
                exponent = int(z.attrib['exponent'])
                if len(self.coefficient) < exponent+1:
                    lc = len(self.coefficient)
                    for i in range(lc,exponent+1):
                        self.coefficient.append(0.)
                self.coefficient[exponent] = float(z.text)
        return

    def __str__(self):
        retstr  = "VelocityParameterPolynomial:"
        retlst  = ()
        retstr += sep+tab+"validityRangeMin=%-27.20g"
        retlst += (self.validityRangeMin,)
        retstr += sep+tab+"validityRangeMax=%-27.20g"
        retlst += (self.validityRangeMax,)
        retstr += sep+tab+"referencePoint=%-27.20g"
        retlst += (self.referencePoint,)
        retstr += sep+tab+"polynomialDegree=%d"
        retlst += (self.polynomialDegree,)
        for x in self.coefficient:
            retstr += sep+tab+"coefficient=%-27.20g"
            retlst += (x,)
        retstr += sep+":VelocityParameterPolynomial"
        return retstr % retlst



class _ZeroDopplerVelocity(object):
    def __init__(self):
        self.velocity = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'velocity':
                self.velocity = float(z.text)
        return

    def __str__(self):
        retstr  = "ZeroDopplerVelocity:"
        retlst  = ()
        retstr += sep+tab+"velocity=%-27.20g"
        retlst += (self.velocity,)
        retstr += sep+":ZeroDopplerVelocity"
        return retstr % retlst

class _DopplerRate(object):
    def __init__(self):
        self.timeUTC = None
        self.dopplerRatePolynomial = _DopplerRatePolynomial()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'timeUTC':
                self.timeUTC = z.text
            if z.tag == 'dopplerRatePolynomial':
                self.dopplerRatePolynomial.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "DopplerRate:"
        retlst  = ()
        retstr += sep+tab+"timeUTC=%s"
        retlst += (self.timeUTC,)
        retstr += sep+"%s"
        retlst += (str(self.dopplerRatePolynomial),)
        retstr += sep+":DopplerRate"
        return retstr % retlst

class _DopplerRatePolynomial(object):
    def __init__(self):
        self.validityRangeMin = None
        self.validityRangeMax = None
        self.referencePoint = None
        self.polynomialDegree = None
        self.coefficient = []
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'validityRangeMin':
                self.validityRangeMin = float(z.text)
            if z.tag == 'validityRangeMax':
                self.validityRangeMax = float(z.text)
            if z.tag == 'referencePoint':
                self.referencePoint = float(z.text)
            if z.tag == 'polynomialDegree':
                self.polynomialDegree = int(z.text)
            if z.tag == 'coefficient':
                exponent = int(z.attrib['exponent'])
                if len(self.coefficient) < exponent+1:
                    lc = len(self.coefficient)
                    for i in range(lc,exponent+1):
                        self.coefficient.append(0.)
                self.coefficient[exponent] = float(z.text)
        return

    def __str__(self):
        retstr  = "DopplerRatePolynomial:"
        retlst  = ()
        retstr += sep+tab+"validityRangeMin=%-27.20g"
        retlst += (self.validityRangeMin,)
        retstr += sep+tab+"validityRangeMax=%-27.20g"
        retlst += (self.validityRangeMax,)
        retstr += sep+tab+"referencePoint=%-27.20g"
        retlst += (self.referencePoint,)
        retstr += sep+tab+"polynomialDegree=%d"
        retlst += (self.polynomialDegree,)
        for x in self.coefficient:
            retstr += sep+tab+"coefficient=%-27.20g"
            retlst += (x,)
        retstr += sep+":DopplerRatePolynomial"
        return retstr % retlst


class _ProcessingDoppler(object):
    def __init__(self):
        self.dopplerBasebandEstimationMethod = None
        self.dopplerGeometricEstimationMethod = None
        self.dopplerCentroidCoordinateType = None
        self.dopplerCentroid = _ProcessingDopplerCentroid()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'dopplerBasebandEstimationMethod':
                self.dopplerBasebandEstimationMethod = z.text
            if z.tag == 'dopplerGeometricEstimationMethod':
                self.dopplerGeometricEstimationMethod = z.text
            if z.tag == 'dopplerCentroidCoordinateType':
                self.dopplerCentroidCoordinateType = z.text
            if z.tag == 'dopplerCentroid':
                self.dopplerCentroid.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "Doppler:"
        retlst  = ()
        retstr += sep+"dopplerBasebandEstimationMethod=%s"
        retlst += (self.dopplerBasebandEstimationMethod,)
        retstr += sep+"dopplerGeometricEstimationMethod=%s"
        retlst += (self.dopplerGeometricEstimationMethod,)
        retstr += sep+"dopplerCentroidCoordinateType=%s"
        retlst += (self.dopplerCentroidCoordinateType,)
        retstr += sep+"%s"
        retlst += (str(self.dopplerCentroid),)
        retstr += sep+":Doppler"
        return retstr % retlst

class _ProcessingDopplerCentroid(object):
    def __init__(self):
        self.layerIndex = None
        self.polLayer = None
        self.DRAoffset = None
        self.beamID = None
        self.polLayerDopplerOffset = None
        self.numberOfBlocks = None
        self.numberOfRejectedBlocks = None
        self.numberOfDopplerRecords = 27
        self.dopplerRecordAzimuthSpacing = None
        self.dopplerEstimate = ()
        return

    def set_from_etnode(self,node):
        self.layerIndex = int(node.attrib['layerIndex'])
        for z in node:
            if z.tag == 'polLayer':
                self.polLayer = z.text
            if z.tag == 'DRAoffset':
                self.DRAoffset = z.text
            if z.tag == 'beamID':
                self.beamID = z.text
            if z.tag == 'polLayerDopplerOffset':
                self.polLayerDopplerOffset = float(z.text)
            if z.tag == 'numberOfBlocks':
                self.numberOfBlocks = int(z.text)
            if z.tag == 'numberOfRejectedBlocks':
                self.numberOfRejectedBlocks = int(z.text)
            if z.tag == 'numberOfDopplerRecords':
                self.numberOfDopplerRecords = int(z.text)
            if z.tag == 'dopplerRecordAzimuthSpacing':
                self.dopplerRecordAzimuthSpacing = float(z.text)
            if z.tag == 'dopplerEstimate':
                self.dopplerEstimate += (_DopplerEstimate(),)
                self.dopplerEstimate[-1].set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "DopplerCentroid:"
        retlst  = ()
        retstr += sep+"layerIndex=%d"
        retlst += (self.layerIndex,)
        retstr += sep+"polLayer=%s"
        retlst += (self.polLayer,)
        retstr += sep+"DRAoffset=%s"
        retlst += (self.DRAoffset,)
        retstr += sep+"beamID=%s"
        retlst += (self.beamID,)
        retstr += sep+"polLayerDopplerOffset=%-27.20g"
        retlst += (self.polLayerDopplerOffset,)
        retstr += sep+"numberOfBlocks=%d"
        retlst += (self.numberOfBlocks,)
        retstr += sep+"numberOfRejectedBlocks=%d"
        retlst += (self.numberOfRejectedBlocks,)
        retstr += sep+"numberOfDopplerRecords=%d"
        retlst += (self.numberOfDopplerRecords,)
        retstr += sep+"dopplerRecordAzimuthSpacing%-27.20g"
        retlst += (self.dopplerRecordAzimuthSpacing,)
        for x in self.dopplerEstimate:
            retstr += sep+"%s"
            retlst += (str(x),)
        retstr += sep+":DopplerCentroid"
        return retstr % retlst

class _DopplerEstimate(object):
    def __init__(self):
        self.timeUTC = None
        self.dopplerAtMidRange = None
        self.basebandDoppler = _BasebandDoppler()
        self.geometricDopplerFlag = None
        self.geometricDoppler = _GeometricDoppler()
        self.dopplerAmbiguity = None
        self.dopplerConsistencyFlag = None
        self.dopplerEstimateConfidence = None
        self.combinedDoppler = _CombinedDoppler()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'timeUTC':
                self.timeUTC = z.text
            if z.tag == 'dopplerAtMidRange':
                self.dopplerAtMidRange = float(z.text)
            if z.tag == 'basebandDoppler':
                self.basebandDoppler.set_from_etnode(z)
            if z.tag == 'geometricDopplerFlag':
                self.geometricDopplerFlag = z.text
            if z.tag == 'geometricDoppler':
                self.geometricDoppler.set_from_etnode(z)
            if z.tag == 'dopplerAmbiguity':
                self.dopplerAmbiguity = int (z.text)
            if z.tag == 'dopplerConsistencyFlag':
                self.dopplerConsistencyFlag = z.text
            if z.tag == 'dopplerEstimateConfidence':
                self.dopplerEstimateConfidence = z.text
            if z.tag == 'combinedDoppler':
                self.combinedDoppler.set_from_etnode(z)
        return

    def __strt__(self):
        retstr  = "DopplerEstimate:"
        retlst  = ()
        retstr += sep+tab+"timeUTC=%s"
        retlst += (self.timeUTC,)
        retstr += sep+tab+"dopplerAtMidRange=%-27.20g"
        retlst += (self.dopplerAtMidRange,)
        retstr += sep+"%s"
        retlst += (str(self.basebandDoppler),)
        retstr += sep+tab+"geometricDopplerFlag=%s"
        retstr += (self.geometricDopplerFlag,)
        retstr += sep+"%s"
        retlst += (str(self.geometricDoppler),)
        retstr += sep+tab+"dopplerAmbiguity=%d"
        retlst += (self.dopplerAmbiguity,)
        retstr += sep+tab+"dopplerConsistencyFlag=%s"
        retlst += (self.dopplerConsistencyFlag,)
        retstr += sep+tab+"dopplerEstimateConfidence=%-27.20g"
        retlst += (self.dopplerEstimateConfidence,)
        retstr += sep+"%s"
        retlst += (str(self.combinedDoppler),)
        retstr += sep+":DopplerEstimate"
        return

class _BasebandDoppler(object):
    def __init__(self):
        self.validityRangeMin = None
        self.validityRangeMax = None
        self.referencePoint = None
        self.polynomialDegree = None
        self.coefficient = []
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'validityRangeMin':
                self.validityRangeMin = float(z.text)
            if z.tag == 'validityRangeMax':
                self.validityRangeMax = float(z.text)
            if z.tag == 'referencePoint':
                self.referencePoint = float(z.text)
            if z.tag == 'polynomialDegree':
                self.polynomialDegree = int(z.text)
            if z.tag == 'coefficient':
                exponent = int(z.attrib['exponent'])
                if len(self.coefficient) < exponent+1:
                    lc = len(self.coefficient)
                    for i in range(lc,exponent+1):
                        self.coefficient.append(0.)
                self.coefficient[exponent] = float(z.text)
        return

    def __str__(self):
        retstr  = "BasebandDoppler:"
        retlst  = ()
        retstr += sep+tab+"validityRangeMin=%-27.20g"
        retlst += (self.validityRangeMin,)
        retstr += sep+tab+"validityRangeMax=%-27.20g"
        retlst += (self.validityRangeMax,)
        retstr += sep+tab+"referencePoint=%-27.20g"
        retlst += (self.referencePoint,)
        retstr += sep+tab+"polynomialDegree=%d"
        retlst += (self.polynomialDegree,)
        for x in self.coefficient:
            retstr += sep+tab+"coefficient=%-27.20g"
            retlst += (x,)
        retstr += sep+":BasebandDoppler"
        return retstr % retlst

class _GeometricDoppler(object):
    def __init__(self):
        self.validityRangeMin = None
        self.validityRangeMax = None
        self.referencePoint = None
        self.polynomialDegree = None
        self.coefficient = []
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'validityRangeMin':
                self.validityRangeMin = float(z.text)
            if z.tag == 'validityRangeMax':
                self.validityRangeMax = float(z.text)
            if z.tag == 'referencePoint':
                self.referencePoint = float(z.text)
            if z.tag == 'polynomialDegree':
                self.polynomialDegree = int(z.text)
            if z.tag == 'coefficient':
                exponent = int(z.attrib['exponent'])
                if len(self.coefficient) < exponent+1:
                    lc = len(self.coefficient)
                    for i in range(lc,exponent+1):
                        self.coefficient.append(0.)
                self.coefficient[exponent] = float(z.text)
        return

    def __str__(self):
        retstr  = "GeometricDoppler:"
        retlst  = ()
        retstr += sep+tab+"validityRangeMin=%-27.20g"
        retlst += (self.validityRangeMin,)
        retstr += sep+tab+"validityRangeMax=%-27.20g"
        retlst += (self.validityRangeMax,)
        retstr += sep+tab+"referencePoint=%-27.20g"
        retlst += (self.referencePoint,)
        retstr += sep+tab+"polynomialDegree=%d"
        retlst += (self.polynomialDegree,)
        for x in self.coefficient:
            retstr += sep+tab+"coefficient=%-27.20g"
            retlst += (x,)
        retstr += sep+":GeometricDoppler"
        return retstr % retlst


class _CombinedDoppler(object):
    def __init__(self):
        self.validityRangeMin = None
        self.validityRangeMax = None
        self.referencePoint = None
        self.polynomialDegree = None
        self.coefficient = []
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'validityRangeMin':
                self.validityRangeMin = float(z.text)
            if z.tag == 'validityRangeMax':
                self.validityRangeMax = float(z.text)
            if z.tag == 'referencePoint':
                self.referencePoint = float(z.text)
            if z.tag == 'polynomialDegree':
                self.polynomialDegree = int(z.text)
            if z.tag == 'coefficient':
                exponent = int(z.attrib['exponent'])
                if len(self.coefficient) < exponent+1:
                    lc = len(self.coefficient)
                    for i in range(lc,exponent+1):
                        self.coefficient.append(0.)
                self.coefficient[exponent] = float(z.text)
        return

    def __str__(self):
        retstr  = "CombinedDoppler:"
        retlst  = ()
        retstr += sep+tab+"validityRangeMin=%-27.20g"
        retlst += (self.validityRangeMin,)
        retstr += sep+tab+"validityRangeMax=%-27.20g"
        retlst += (self.validityRangeMax,)
        retstr += sep+tab+"referencePoint=%-27.20g"
        retlst += (self.referencePoint,)
        retstr += sep+tab+"polynomialDegree=%d"
        retlst += (self.polynomialDegree,)
        for x in self.coefficient:
            retstr += sep+tab+"coefficient=%-27.20g"
            retlst += (x,)
        retstr += sep+":CombinedDoppler"
        return retstr % retlst

class _ProcessingParameter(object):
    def __init__(self):
        self.beamID = None
        self.processingInfoCoordinateType = None
        self.rangeLooks = None
        self.azimuthLooks = None
        self.rangeLookBandwidth = None
        self.azimuthLookBandwidth = None
        self.totalProcessedRangeBandwidth = None
        self.totalProcessedAzimuthBandwidth = None
        self.rangeWindowID = None
        self.rangeWindowCoefficient = None
        self.rangeCompression = _RangeCompression()
        self.correctedInstrumentDelay = _CorrectedInstrumentDelay()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'beamID':
                self.beamID = z.text
            if z.tag == 'processingInfoCoordinateType':
                self.processingInfoCoordinateType = z.text
            if z.tag == 'rangeLooks':
                self.rangeLooks = float(z.text)
            if z.tag == 'azimuthLooks':
                self.azimuthLooks = float(z.text)
            if z.tag == 'rangeLookBandwidth':
                self.rangeLookBandwidth = float(z.text)
            if z.tag == 'azimuthLookBandwidth':
                self.azimuthLookBandwidth = float(z.text)
            if z.tag == 'totalProcessedRangeBandwidth':
                self.totalProcessedRangeBandwidth = float(z.text)
            if z.tag == 'totalProcessedAzimuthBandwidth':
                self.totalProcessedAzimuthBandwidth = float(z.text)
            if z.tag == 'rangeWindowID':
                self.rangeWindowID = z.text
            if z.tag == 'rangeWindowCoefficient':
                self.rangeWindowCoefficient = float(z.text)
            if z.tag == 'rangeCompression':
                self.rangeCompression.set_from_etnode(z)
            if z.tag == 'correctedInstrumentDelay':
                self.correctedInstrumentDelay.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "ProcessingParameter:"
        retlst  = ()
        retstr += sep+tab+"beamID=%s"
        retlst += (self.beamID,)
        retstr += sep+tab+"processingInfoCoordinateType=%s"
        retlst += (self.processingInfoCoordinateType,)
        retstr += sep+tab+"rangeLooks%-27.20g"
        retlst += (self.rangeLooks,)
        retstr += sep+tab+"azimuthLooks=%-27.20g"
        retlst += (self.azimuthLooks,)
        retstr += sep+tab+"rangeLookBandwidth=%-27.20g"
        retlst += (self.rangeLookBandwidth,)
        retstr += sep+tab+"azimuthLookBandwidth=%-27.20g"
        retlst += (self.azimuthLookBandwidth,)
        retstr += sep+tab+"totalProcessedRangeBandwidth=%-27.20g"
        retlst += (self.totalProcessedRangeBandwidth,)
        retstr += sep+tab+"totalProcessedAzimuthBandwidth=%-27.20g"
#       print type(self.totalProcessedAzimuthBandwidth)
        retlst += (self.totalProcessedAzimuthBandwidth,)
        retstr += sep+tab+"rangeWindowID=%s"
        retlst += (self.rangeWindowID,)
        retstr += sep+tab+"rangeWindowCoefficient=%-27.20g"
        retlst += (self.rangeWindowCoefficient,)
        retstr += sep+"%s"
        retlst += (str(self.rangeCompression),)
        retstr += sep+"%s"
        retlst += (str(self.correctedInstrumentDelay),)
        retstr += sep+":ProcessingParameter"
        return retstr % retlst

class _RangeCompression(object):
    def __init__(self):
        self.segmentInfo = _RCSegmentInfo()
        self.chirps = _RCChirps()
        return

    def set_from_etnode(self,node):
       for z in node:
            if z.tag == 'segmentInfo':
                self.segmentInfo.set_from_etnode(z)
            if z.tag == 'chirps':
                self.chirps.set_from_etnode(z)
       return

    def __str__(self):
        retstr  = "RangeCompression:"
        retlst  = ()
        retstr += sep+"%s"
        retlst += (str(self.segmentInfo),)
        retstr += sep+"%s"
        retlst += (str(self.chirps),)
        retstr += sep+":RangeCompression"
        return retstr % retlst


class _RCSegmentInfo(object):
    def __init__(self):
        self.polLayer = None
        self.dataSegment = _RCDataSegment()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'polLayer':
                self.polLayer = z.text
            if z.tag == 'dataSegment':
                self.dataSegment.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "SegmentInfo:"
        retlst  = ()
        retstr += sep+tab+"polLayer=%s"
        retlst += (self.polLayer,)
        retstr += sep+"%s"
        retlst += (str(self.dataSegment),)
        retstr += sep+":SegmentInfo"
        return retstr % retlst

class _RCDataSegment(object):
    def __init__(self):
        self.startTimeUTC = None
        self.stopTimeUTC = None
        self.numberOfRows = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'startTimeUTC':
                self.startTimeUTC = z.text
            if z.tag == 'stopTimeUTC':
                self.stopTimeUTC = z.text
            if z.tag == 'numberOfRows':
                self.numberOfRows = int(z.text)
        return

    def __str__(self):
        retstr  = "DataSegment:"
        retlst  = ()
        retstr += sep+tab+"startTimeUTC=%s"
        retlst += (self.startTimeUTC,)
        retstr += sep+tab+"stopTimeUTC=%s"
        retlst += (self.stopTimeUTC,)
        retstr += sep+tab+"numberOfRows=%d"
        retlst += (self.numberOfRows,)
        retstr += sep+":DataSegment"
        return retstr % retlst

class _RCChirps(object):
    def __init__(self):
        self.referenceChirp = _RCReferenceChirp()
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'referenceChirp':
                self.referenceChirp.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "Chirps:"
        retlst  = ()
        retstr += sep+"%s"
        retlst += (str(self.referenceChirp),)
        retstr += sep+":Chirps"
        return retstr % retlst

class _RCReferenceChirp(object):
    def __init__(self):
        self.pulseCode = None
        self.pulseType = None
        self.chirpDesignator = None
        self.chirpSlope = None
        self.pulseLength = None
        self.pulseBandwidth = None
        self.centerFrequency = None
        self.amplitude = _RCChirpAmplitude()
        self.phase = _RCChirpPhase()
        return

    def set_from_etnode(self,node):
        self.pulseCode = int(node.attrib['pulseCode'])
        for z in node:
            if z.tag == 'pulseType':
                self.pulseType = z.text
            if z.tag == 'chirpDesignator':
                self.chirpDesignator = z.text
            if z.tag == 'chirpSlope':
                self.chirpSlope = z.text
            if z.tag == 'pulseLength':
                self.pulseLength = float(z.text)
            if z.tag == 'pulseBandwidth':
                self.pulseBandwidth = float(z.text)
            if z.tag == 'centerFrequency':
                self.centerFrequency = float(z.text)
            if z.tag == 'amplitude':
                self.amplitude.set_from_etnode(z)
            if z.tag == 'phase':
                self.phase.set_from_etnode(z)
        return

    def __str__(self):
        retstr  = "ReferenceChirp:"
        retlst  = ()
        retstr += sep+tab+"pulseCode=%d"
        retlst += (self.pulseCode,)
        retstr += sep+tab+"pulseType=%s"
        retlst += (self.pulseType,)
        retstr += sep+tab+"chirpDesignator=%s"
        retlst += (self.chirpDesignator,)
        retstr += sep+tab+"chirpSlope=%s"
        retlst += (self.chirpSlope,)
        retstr += sep+tab+"pulseLength=%-27.20g"
        retlst += (self.pulseLength,)
        retstr += sep+tab+"pulseBandwidth=%-27.20g"
        retlst += (self.pulseBandwidth,)
        retstr += sep+tab+"centerFrequency=%-27.20g"
        retlst += (self.centerFrequency,)
        retstr += sep+"%s"
        retlst += (str(self.amplitude),)
        retstr += sep+"%s"
        retlst += (str(self.phase),)
        retstr += sep+":ReferenceChirp"
        return retstr % retlst

class _RCChirpAmplitude(object):
    def __init__(self):
        self.validityRangeMin = None
        self.validityRangeMax = None
        self.referencePoint = None
        self.polynomialDegree = None
        self.coefficient = []
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'validityRangeMin':
                self.validityRangeMin = float(z.text)
            if z.tag == 'validityRangeMax':
                self.validityRangeMax = float(z.text)
            if z.tag == 'referencePoint':
                self.referencePoint = float(z.text)
            if z.tag == 'polynomialDegree':
                self.polynomialDegree = int(z.text)
            if z.tag == 'coefficient':
                exponent = int(z.attrib['exponent'])
                if len(self.coefficient) < exponent+1:
                    lc = len(self.coefficient)
                    for i in range(lc,exponent+1):
                        self.coefficient.append(0.)
                self.coefficient[exponent] = float(z.text)
        return

    def __str__(self):
        retstr  = "Amplitude:"
        retlst  = ()
        retstr += sep+tab+"validityRangeMin=%-27.20g"
        retlst += (self.validityRangeMin,)
        retstr += sep+tab+"validityRangeMax=%-27.20g"
        retlst += (self.validityRangeMax,)
        retstr += sep+tab+"referencePoint=%-27.20g"
        retlst += (self.referencePoint,)
        retstr += sep+tab+"polynomialDegree=%d"
        retlst += (self.polynomialDegree,)
        for x in self.coefficient:
            retstr += sep+tab+"coefficient=%-27.20g"
            retlst += (x,)
        retstr += sep+":Amplitude"
        return retstr % retlst

class _RCChirpPhase(object):
    def __init__(self):
        self.validityRangeMin = None
        self.validityRangeMax = None
        self.referencePoint = None
        self.polynomialDegree = None
        self.coefficient = []
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'validityRangeMin':
                self.validityRangeMin = float(z.text)
            if z.tag == 'validityRangeMax':
                self.validityRangeMax = float(z.text)
            if z.tag == 'referencePoint':
                self.referencePoint = float(z.text)
            if z.tag == 'polynomialDegree':
                self.polynomialDegree = int(z.text)
            if z.tag == 'coefficient':
                exponent = int(z.attrib['exponent'])
                if len(self.coefficient) < exponent+1:
                    lc = len(self.coefficient)
                    for i in range(lc,exponent+1):
                        self.coefficient.append(0.)
                self.coefficient[exponent] = float(z.text)
        return

    def __str__(self):
        retstr  = "Phase:"
        retlst  = ()
        retstr += sep+tab+"validityRangeMin=%-27.20g"
        retlst += (self.validityRangeMin,)
        retstr += sep+tab+"validityRangeMax=%-27.20g"
        retlst += (self.validityRangeMax,)
        retstr += sep+tab+"referencePoint=%-27.20g"
        retlst += (self.referencePoint,)
        retstr += sep+tab+"polynomialDegree=%d"
        retlst += (self.polynomialDegree,)
        for x in self.coefficient:
            retstr += sep+tab+"coefficient=%-27.20g"
            retlst += (x,)
        retstr += sep+":Phase"
        return retstr % retlst

class _CorrectedInstrumentDelay(object):
    def __init__(self):
        self.polLayer = None
        self.DRAoffset = None
        self.totalTimeDelay = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'polLayer':
                self.polLayer = z.text
            if z.tag == 'DRAoffset':
                self.DRAoffset = z.text
            if z.tag == 'totalTimeDelay':
                self.totalTimeDelay = float(z.text)
        return

    def __str__(self):
        retstr  = "CorrectedInstrumentDelay:"
        retlst  = ()
        retstr += sep+tab+"polLayer=%s"
        retlst += (self.polLayer,)
        retstr += sep+tab+"DRAoffset=%s"
        retlst += (self.DRAoffset,)
        retstr += sep+tab+"totalTimeDelay=%-27.20g"
        retlst += (self.totalTimeDelay,)
        return retstr % retlst

class _ProcessingFlags(object):
    def __init__(self):
        self.RXGainCorrectedFlag = None
        self.DRAChannelSyncFlag = None
        self.DRAChannelDemixingPerformedFlag = None
        self.hybridCouplerCorrectedFlag = None
        self.chirpDriftCorrectedFlag = None
        self.chirpReplicaUsedFlag = None
        self.geometricDopplerUsedFlag = None
        self.noiseCorrectedFlag = None
        self.rangeSpreadingLossCorrectedFlag = None
        self.scanSARBeamCorrectedFlag = None
        self.spotLightBeamCorrectedFlag = None
        self.azimuthPatternCorrectedFlag = None
        self.elevationPatternCorrectedFlag = None
        self.polarisationCorrectedFlag = None
        self.detectedFlag = None
        self.multiLookedFlag = None
        self.propagationEffectsCorrectedFlag = None
        self.geocodedFlag = None
        self.incidenceAngleMaskGeneratedFlag = None
        self.nominalProcessingPerformedFlag = None
        return

# Extras

class _File(object):
    def __init__(self):
        self.location = _FileLocation()
        self.size = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'location':
                self.location.set_from_etnode(z)
            if z.tag == 'size':
                self.size = int(z.text)
        return

    @property
    def file(self):
        return self.location.filename # can be updated to include host + path

    def __str__(self):
        retstr  = "File:"
        retlst  = ()
        retstr += sep+"%s"
        retlst += (str(self.file),)
        retstr += sep+tab+"size=%d"
        retlst += (self.size,)
        retstr += sep+":File"
        return retstr % retlst

class _FileLocation(object):
    def __init__(self):
        self.host = None
        self.path = None
        self.filename = None
        return

    def set_from_etnode(self,node):
        for z in node:
            if z.tag == 'host':
                self.host = z.text
            if z.tag == 'path':
                self.path = z.text
            if z.tag == 'filename':
                self.filename = z.text
        return

    def __str__(self):
        retstr  = "Location:"
        retlst  = ()
        retstr += sep+"host=%s"
        retlst += (self.host,)
        retstr += sep+tab+"path=%s"
        retlst += (self.path,)
        retstr += sep+tab+"filename=%s"
        retlst += (self.filename,)
        retstr += sep+":Location"
        return retstr % retlst
