#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Brett George
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






import logging
import isceobj
import mroipac
from mroipac.baseline.Baseline import Baseline
import copy
import os

logger = logging.getLogger('isce.insar.runPreprocessor')

def runPreprocessor(self):

    from .Factories import isRawSensor, isZeroDopplerSLC, getDopplerMethod    

    ###Unpack reference
    sensor = copy.deepcopy(self.reference)

    dirname = sensor.output

    if self.referenceSensorName is None:
        self.referenceSensorName = self.sensorName

    israwdata = isRawSensor(self.referenceSensorName)


    if self.referenceDopplerMethod is None:
        mdop = getDopplerMethod(self.referenceSensorName)
    else:
        mdop = self.referenceDopplerMethod
    
    referencedop = isceobj.Doppler.createDoppler(mdop)

    if israwdata:
        print('Reference data is in RAW format. Adding _raw to output name.')
        sensor.output = os.path.join(dirname + '_raw', os.path.basename(dirname)+'.raw')
        os.makedirs(os.path.dirname(sensor.output), exist_ok=True)
        #sensor._resampleFlag = 'single2dual'
        reference = make_raw(sensor, referencedop)

        ###Weird handling here because of way make_raw is structured
        ###DOPIQ uses weird dict to store coeffs
        if mdop == 'useDOPIQ':
            #reference._dopplerVsPixel = [referencedop.quadratic[x]*reference.PRF for x in ['a','b','c']]
            reference.frame._dopplerVsPixel = [referencedop.quadratic[x]*reference.PRF for x in ['a','b','c']]

        if self._insar.referenceRawProduct is None:
            self._insar.referenceRawProduct = dirname + '_raw.xml'

        self._insar.saveProduct(reference.frame, self._insar.referenceRawProduct)

    else:
        print('Reference data is in SLC format. Adding _slc to output name.')
        iszerodop = isZeroDopplerSLC(self.referenceSensorName) 
        sensor.output =  os.path.join(dirname + '_slc', os.path.basename(dirname)+'.slc')

        os.makedirs(os.path.dirname(sensor.output), exist_ok=True)

        reference = make_raw(sensor, referencedop)
        
        if self._insar.referenceSlcProduct is None:
            self._insar.referenceSlcProduct = dirname + '_slc.xml'

        if iszerodop:
            self._insar.referenceGeometrySystem = 'Zero Doppler'
        else:
            self._insar.referenceGeometrySystem = 'Native Doppler'

        self._insar.saveProduct(reference.frame, self._insar.referenceSlcProduct)


    ###Unpack secondary
    sensor = copy.deepcopy(self.secondary)
    dirname = sensor.output

    if self.secondarySensorName is None:
        self.secondarySensorName = self.sensorName

    israwdata = isRawSensor(self.secondarySensorName)

    if self.secondaryDopplerMethod is None:
        sdop = getDopplerMethod( self.secondarySensorName)
    else:
        sdop = self.secondaryDopplerMethod

    secondarydop = isceobj.Doppler.createDoppler(sdop)

    if israwdata:
        print('Secondary data is in RAW format. Adding _raw to output name.')
        sensor.output = os.path.join(dirname + '_raw', os.path.basename(dirname)+'.raw')

        os.makedirs(os.path.dirname(sensor.output), exist_ok=True)

        secondary = make_raw(sensor, secondarydop)

        ###Weird handling here because of make_raw structure
        ###DOPIQ uses weird dict to store coefficients
        if sdop == 'useDOPIQ':
            #secondary._dopplerVsPixel = [secondarydop.quadratic[x]*secondary.PRF for x in ['a','b','c']]
            secondary.frame._dopplerVsPixel = [secondarydop.quadratic[x]*secondary.PRF for x in ['a','b','c']]

        if self._insar.secondaryRawProduct is None:
            self._insar.secondaryRawProduct = dirname + '_raw.xml'

        self._insar.saveProduct(secondary.frame, self._insar.secondaryRawProduct)

    else:
        print('Secondary data is in SLC format. Adding _slc to output name.')
        iszerodop = isZeroDopplerSLC(self.secondarySensorName)
        sensor.output =  os.path.join(dirname + '_slc', os.path.basename(dirname)+'.slc')

        os.makedirs( os.path.dirname(sensor.output), exist_ok=True)

        secondary = make_raw(sensor, secondarydop)
        
        if self._insar.secondarySlcProduct is None:
            self._insar.secondarySlcProduct = dirname + '_slc.xml'

        
        if iszerodop:
            self._insar.secondaryGeometrySystem = 'Zero Doppler'
        else:
            self._insar.secondaryGeometrySystem = 'Native Doppler'

        self._insar.saveProduct(secondary.frame, self._insar.secondarySlcProduct)
    

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    frame = reference.frame
    instrument = frame.getInstrument()
    platform = instrument.getPlatform()

    catalog.addInputsFrom(reference.sensor, 'reference.sensor')
    catalog.addItem('width', frame.numberOfSamples, 'reference')
    catalog.addItem('iBias', instrument.getInPhaseValue(), 'reference')
    catalog.addItem('qBias', instrument.getQuadratureValue(), 'reference')
    catalog.addItem('range_sampling_rate', instrument.getRangeSamplingRate(), 'reference')
    catalog.addItem('prf', instrument.getPulseRepetitionFrequency(), 'reference')
    catalog.addItem('pri', 1.0/instrument.getPulseRepetitionFrequency(), 'reference')
    catalog.addItem('pulse_length', instrument.getPulseLength(), 'reference')
    catalog.addItem('chirp_slope', instrument.getChirpSlope(), 'reference')
    catalog.addItem('wavelength', instrument.getRadarWavelength(), 'reference')
    catalog.addItem('lookSide', platform.pointingDirection, 'reference')
    catalog.addInputsFrom(frame, 'reference.frame')
    catalog.addInputsFrom(instrument, 'reference.instrument')
    catalog.addInputsFrom(platform, 'reference.platform')
    catalog.addInputsFrom(frame.orbit, 'reference.orbit')

    frame = secondary.frame
    instrument = frame.getInstrument()
    platform = instrument.getPlatform()

    catalog.addInputsFrom(secondary.sensor, 'secondary.sensor')
    catalog.addItem('width', frame.numberOfSamples, 'secondary')
    catalog.addItem('iBias', instrument.getInPhaseValue(), 'secondary')
    catalog.addItem('qBias', instrument.getQuadratureValue(), 'secondary')
    catalog.addItem('range_sampling_rate', instrument.getRangeSamplingRate(), 'secondary')
    catalog.addItem('prf', instrument.getPulseRepetitionFrequency(), 'secondary')
    catalog.addItem('pri', 1.0/instrument.getPulseRepetitionFrequency(), 'secondary')
    catalog.addItem('pulse_length', instrument.getPulseLength(), 'secondary')
    catalog.addItem('chirp_slope', instrument.getChirpSlope(), 'secondary')
    catalog.addItem('wavelength', instrument.getRadarWavelength(), 'secondary')
    catalog.addItem('lookSide', platform.pointingDirection, 'secondary')
    catalog.addInputsFrom(frame, 'secondary.frame')
    catalog.addInputsFrom(instrument, 'secondary.instrument')
    catalog.addInputsFrom(platform, 'secondary.platform')
    catalog.addInputsFrom(frame.orbit, 'secondary.orbit')

    catalog.printToLog(logger, "runPreprocessor")
    self._insar.procDoc.addAllFromCatalog(catalog)

def make_raw(sensor, doppler):
    from make_raw import make_raw

    objMakeRaw = make_raw()
    objMakeRaw(sensor=sensor, doppler=doppler)

    return objMakeRaw

def initRawImage(makeRawObj):
    from isceobj.Image import createSlcImage
    from isceobj.Image import createRawImage
    #the "raw" image in same case is an slc.
    #for now let's do it in this way. probably need to make this a factory
    #instantiated based on the sensor type
    imageType = makeRawObj.frame.getImage()
    if isinstance(imageType, createRawImage().__class__):
        filename = makeRawObj.frame.getImage().getFilename()
        bytesPerLine = makeRawObj.frame.getImage().getXmax()
        goodBytes = makeRawObj.frame.getImage().getXmax() - makeRawObj.frame.getImage().getXmin()
        logger.debug("bytes_per_line: %s" % (bytesPerLine))
        logger.debug("good_bytes_per_line: %s" % (goodBytes))
        objRaw = createRawImage()
        objRaw.setFilename(filename)

        objRaw.setNumberGoodBytes(goodBytes)
        objRaw.setWidth(bytesPerLine)
        objRaw.setXmin(makeRawObj.frame.getImage().getXmin())
        objRaw.setXmax(bytesPerLine)
    elif(isinstance(imageType,createSlcImage().__class__)):
        objRaw = createSlcImage()
        filename = makeRawObj.frame.getImage().getFilename()
        bytesPerLine = makeRawObj.frame.getImage().getXmax()
        objRaw.setFilename(filename)
        objRaw.setWidth(bytesPerLine)
        objRaw.setXmin(makeRawObj.frame.getImage().getXmin())
        objRaw.setXmax(bytesPerLine)
    return objRaw
