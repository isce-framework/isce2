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
    master = make_raw(self.master, self.masterdop)
    self.insar.rawMasterIQImage = master.iqImage
    slave = make_raw(self.slave, self.slavedop)
    self.insar.rawSlaveIQImage = slave.iqImage
    self._insar.numberRangeBins = master.frame.numberRangeBins
    #add raw images to main object
    masterRaw = initRawImage(master)
    self._insar.setMasterRawImage(masterRaw)
    slaveRaw = initRawImage(slave)
    self._insar.setSlaveRawImage(slaveRaw)

    #add frames to main  object
    self._insar.setMasterFrame(master.frame)
    self._insar.setSlaveFrame(slave.frame)

    #add doppler to main object
    self._insar.setMasterDoppler(master.getDopplerValues())
    self._insar.setSlaveDoppler(slave.getDopplerValues())

    #add squints to main object
    self._insar.setMasterSquint(master.getSquint())
    self._insar.setSlaveSquint(slave.getSquint())

    #add look direction
    self._insar.setLookSide(master.frame.getInstrument().getPlatform().pointingDirection)

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    frame = self._insar.getMasterFrame()
    instrument = frame.getInstrument()
    platform = instrument.getPlatform()

    planet = platform.getPlanet()
    catalog.addInputsFrom(planet, 'planet')
    catalog.addInputsFrom(planet.get_elp(), 'planet.ellipsoid')

    catalog.addInputsFrom(master.sensor, 'master.sensor')
    catalog.addItem('width', masterRaw.getWidth(), 'master')
    catalog.addItem('xmin', masterRaw.getXmin(), 'master')
    catalog.addItem('iBias', instrument.getInPhaseValue(), 'master')
    catalog.addItem('qBias', instrument.getQuadratureValue(), 'master')
    catalog.addItem('range_sampling_rate', instrument.getRangeSamplingRate(), 'master')
    catalog.addItem('prf', instrument.getPulseRepetitionFrequency(), 'master')
    catalog.addItem('pri', 1.0/instrument.getPulseRepetitionFrequency(), 'master')
    catalog.addItem('pulse_length', instrument.getPulseLength(), 'master')
    catalog.addItem('chirp_slope', instrument.getChirpSlope(), 'master')
    catalog.addItem('wavelength', instrument.getRadarWavelength(), 'master')
    catalog.addItem('lookSide', platform.pointingDirection, 'master')
    catalog.addInputsFrom(frame, 'master.frame')
    catalog.addInputsFrom(instrument, 'master.instrument')
    catalog.addInputsFrom(platform, 'master.platform')
    catalog.addInputsFrom(frame.orbit, 'master.orbit')

    frame = self._insar.getSlaveFrame()
    instrument = frame.getInstrument()
    platform = instrument.getPlatform()

    catalog.addInputsFrom(slave.sensor, 'slave.sensor')
    catalog.addItem('width', slaveRaw.getWidth(), 'slave')
    catalog.addItem('xmin', slaveRaw.getXmin(), 'slave')
    catalog.addItem('iBias', instrument.getInPhaseValue(), 'slave')
    catalog.addItem('qBias', instrument.getQuadratureValue(), 'slave')
    catalog.addItem('range_sampling_rate', instrument.getRangeSamplingRate(), 'slave')
    catalog.addItem('prf', instrument.getPulseRepetitionFrequency(), 'slave')
    catalog.addItem('pri', 1.0/instrument.getPulseRepetitionFrequency(), 'slave')
    catalog.addItem('pulse_length', instrument.getPulseLength(), 'slave')
    catalog.addItem('chirp_slope', instrument.getChirpSlope(), 'slave')
    catalog.addItem('wavelength', instrument.getRadarWavelength(), 'slave')
    catalog.addItem('lookSide', platform.pointingDirection, 'slave')
    catalog.addInputsFrom(frame, 'slave.frame')
    catalog.addInputsFrom(instrument, 'slave.instrument')
    catalog.addInputsFrom(platform, 'slave.platform')
    catalog.addInputsFrom(frame.orbit, 'slave.orbit')


    optlist = ['all', 'top', 'middle', 'bottom']
    success=False
    baseLocation = None

    for option in optlist:
        baseObj = Baseline()
        baseObj.configure()
        baseObj.baselineLocation = option
        baseObj.wireInputPort(name='masterFrame',object=self._insar.getMasterFrame())
        baseObj.wireInputPort(name='slaveFrame',object=self._insar.getSlaveFrame())
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
