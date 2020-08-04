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
from isceobj.Util.decorators import use_api
logger = logging.getLogger('isce.insar.runPreprocessor')
@use_api
def runPreprocessor(self):
    reference = make_raw(self.reference, self.referencedop)
    self.insar.rawReferenceIQImage = reference.iqImage
    secondary = make_raw(self.secondary, self.secondarydop)
    self.insar.rawSecondaryIQImage = secondary.iqImage
    self._insar.numberRangeBins = reference.frame.numberRangeBins
    #add raw images to main object
    referenceRaw = initRawImage(reference)
    self._insar.setReferenceRawImage(referenceRaw)
    secondaryRaw = initRawImage(secondary)
    self._insar.setSecondaryRawImage(secondaryRaw)

    #add frames to main  object
    self._insar.setReferenceFrame(reference.frame)
    self._insar.setSecondaryFrame(secondary.frame)

    #add doppler to main object
    self._insar.setReferenceDoppler(reference.getDopplerValues())
    self._insar.setSecondaryDoppler(secondary.getDopplerValues())

    #add squints to main object
    self._insar.setReferenceSquint(reference.getSquint())
    self._insar.setSecondarySquint(secondary.getSquint())

    #add look direction
    self._insar.setLookSide(reference.frame.getInstrument().getPlatform().pointingDirection)

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    frame = self._insar.getReferenceFrame()
    instrument = frame.getInstrument()
    platform = instrument.getPlatform()

    planet = platform.getPlanet()
    catalog.addInputsFrom(planet, 'planet')
    catalog.addInputsFrom(planet.get_elp(), 'planet.ellipsoid')

    catalog.addInputsFrom(reference.sensor, 'reference.sensor')
    catalog.addItem('width', referenceRaw.getWidth(), 'reference')
    catalog.addItem('xmin', referenceRaw.getXmin(), 'reference')
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

    frame = self._insar.getSecondaryFrame()
    instrument = frame.getInstrument()
    platform = instrument.getPlatform()

    catalog.addInputsFrom(secondary.sensor, 'secondary.sensor')
    catalog.addItem('width', secondaryRaw.getWidth(), 'secondary')
    catalog.addItem('xmin', secondaryRaw.getXmin(), 'secondary')
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


    optlist = ['all', 'top', 'middle', 'bottom']
    success=False
    baseLocation = None

    for option in optlist:
        baseObj = Baseline()
        baseObj.configure()
        baseObj.baselineLocation = option
        baseObj.wireInputPort(name='referenceFrame',object=self._insar.getReferenceFrame())
        baseObj.wireInputPort(name='secondaryFrame',object=self._insar.getSecondaryFrame())
        try:
            baseObj.baseline()
            success=True
            baseLocation=option
        except:
            print('Baseline computation with option {0} Failed'.format(option))
            pass

        if success:
            break

    if not success:
        raise Exception('Baseline computation failed with all possible options. Images may not overlap.')

    catalog.addItem('horizontal_baseline_top', baseObj.hBaselineTop, 'baseline')
    catalog.addItem('horizontal_baseline_rate', baseObj.hBaselineRate, 'baseline')
    catalog.addItem('horizontal_baseline_acc', baseObj.hBaselineAcc, 'baseline')
    catalog.addItem('vertical_baseline_top', baseObj.vBaselineTop, 'baseline')
    catalog.addItem('vertical_baseline_rate', baseObj.vBaselineRate, 'baseline')
    catalog.addItem('vertical_baseline_acc', baseObj.vBaselineAcc, 'baseline')
    catalog.addItem('perp_baseline_top', baseObj.pBaselineTop, 'baseline')
    catalog.addItem('perp_baseline_bottom', baseObj.pBaselineBottom, 'baseline')
    catalog.addItem('baseline_location', baseLocation, 'baseline')

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
