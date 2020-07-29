#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Kosal Khun, Marco Lavalle
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Comment: Adapted from InsarProc/runPreprocessor.py
import os
import sys
import logging
import isceobj
from make_raw import make_raw
from isceobj import Doppler, Sensor
from isceobj.Image import createSlcImage
from isceobj.Image import createRawImage
from mroipac.baseline.Baseline import Baseline

logger = logging.getLogger('isce.isceProc.runPreprocessor')

def runPreprocessor(self):
    doppler = Doppler.createDoppler(self.dopplerMethod)
    sensorname = self.sensorName

    for sceneid in self._isce.selectedScenes:
        scene = self._isce.srcFiles[sceneid]
        self._isce.frames[sceneid] = {}
        self._isce.dopplers[sceneid] = {}
        self._isce.rawImages[sceneid] = {}
        self._isce.iqImages[sceneid] = {}
        self._isce.squints[sceneid] = {}
        for pol in self._isce.selectedPols:
            sid = self._isce.formatname(sceneid, pol)
            rawfile = os.path.join(self.getoutputdir(sceneid),
                self._isce.formatname(sceneid, pol, 'raw'))

            if not 'uavsar_rpi' in sensorname.lower():
                sensor = getsensorobj(scene, pol, rawfile, sensorname, sceneid)
            else:
                #uavsar_rpi requires that we name a 'reference' and a 'secondary'
                #this sensor is strictly pairwise processing
                name = 'reference' if sceneid == self._isce.refScene else 'secondary'
                sensor = getsensorobj(scene, pol, rawfile, sensorname, name)

            catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
            rawobj = run(sensor, doppler, catalog=catalog, sceneid=sid) ##actual processing
            self._isce.frames[sceneid][pol] = rawobj.getFrame() ##add frames to main object
            self._isce.dopplers[sceneid][pol] = rawobj.getDopplerValues() ##add dopplers to main object
            self._isce.squints[sceneid][pol] = rawobj.getSquint()

            self._isce.procDoc.addAllFromCatalog(catalog)

            rawimage = initRawImage(rawobj)
            if rawobj.frame.image.imageType == 'slc': ##it's a slc image
                slcfile = rawfile[:-3] + 'raw' #ML 21-8-2014 changed slc to raw
                os.system("ln -s "+os.path.join(os.getcwd(), rawobj.frame.image.filename)+" "+slcfile)
#                os.rename(rawfile, slcfile)
                self._isce.slcImages[sceneid] = {} #ML 21-8-2014
                self._isce.slcImages[sceneid][pol] = rawimage
            else: ##it's a real raw image
                self._isce.rawImages[sceneid][pol] = rawimage ##add raw images to main object
                self._isce.iqImages[sceneid][pol] = rawobj.getIQImage() ##add iqImages to main object

    # KK 2013-12-12: calculate baselines for selected pairs
    for sceneid1, sceneid2 in self._isce.selectedPairs:
        pair = (sceneid1, sceneid2)
        frame1 = self._isce.frames[sceneid1][self._isce.refPol]
        frame2 = self._isce.frames[sceneid2][self._isce.refPol]
        sid = self._isce.formatname(pair)
        catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
        getBaseline(frame1, frame2, catalog=catalog, sceneid=sid)
        self._isce.procDoc.addAllFromCatalog(catalog)
    # KK

    # KK 2013-12-12: get refFrame from refScene/refPol
    refFrame = self._isce.frames[self._isce.refScene][self._isce.refPol]
    self._isce.numberRangeBins = refFrame.numberRangeBins
    self._isce.lookSide = refFrame.getInstrument().getPlatform().pointingDirection
    # KK

    if sensorname == 'ALOS':
        self._isce.transmit = sensor.transmit
        self._isce.receive = sensor.receive


# KK 2013-12-12: calculate baselines between 2 frames
def getBaseline(frame1, frame2, catalog=None, sceneid='NO_ID'):
    optlist = ['all', 'top', 'middle', 'bottom']
    success = False
    baseLocation = None
    for option in optlist:
        baseObj = Baseline()
        baseObj.configure()
        baseObj.baselineLocation = option
        baseObj.wireInputPort(name='referenceFrame',object=frame1)
        baseObj.wireInputPort(name='secondaryFrame',object=frame2)
        try:
            baseObj.baseline()
            success = True
            baseLocation = option
        except:
            logger.debug(('runPreprocessor.getBaseline '+
                          'option "{0}" failed'.format(option)))
            pass
        if success:
            logger.debug(('runPreprocessor.getBaseline: '+
                          'option "{0}" success'.format(option)))
            break
    if not success:
        raise Exception('Baseline computation failed with all possible options. Images may not overlap.')

    if catalog is not None:
        catalog.addItem('horizontal_baseline_top', baseObj.hBaselineTop, sceneid)
        catalog.addItem('horizontal_baseline_rate', baseObj.hBaselineRate, sceneid)
        catalog.addItem('horizontal_baseline_acc', baseObj.hBaselineAcc, sceneid)
        catalog.addItem('vertical_baseline_top', baseObj.vBaselineTop, sceneid)
        catalog.addItem('vertical_baseline_rate', baseObj.vBaselineRate, sceneid)
        catalog.addItem('vertical_baseline_acc', baseObj.vBaselineAcc, sceneid)
        catalog.addItem('perp_baseline_top', baseObj.pBaselineTop, sceneid)
        catalog.addItem('perp_baseline_bottom', baseObj.pBaselineBottom, sceneid)
# KK


def getsensorobj(scene, pol, output, sensorname, name):
    polkey = reformatscene(scene, pol, sensorname) ##change pol key to imagefile/xml/hdf5 depending on sensor
    #scene['output'] = output
    sensor = Sensor.createSensor(sensorname, name)
    sensor._ignoreMissing = True
    sensor.catalog = scene
    sensor.configure()
    setattr(sensor, polkey, scene[polkey]) #ML 21-8-2014
    setattr(sensor, 'output', output) #ML 21-8-2014
    #sensor.initRecursive(scene, {}) ##populate sensor
    del scene[polkey]
    #del scene['output']
    return sensor


def reformatscene(scenedict, pol, sensorname):
    imageKey = { ##key corresponding to the image file, according to each sensor's dictionaryOfVariables
        'ALOS': 'imagefile',
        'COSMO_SKYMED': 'hdf5',
        'ENVISAT': 'imagefile',
        'ERS': 'imagefile', #KK 2013-11-16
        'JERS': 'imagefile',
        'RADARSAT1': 'imagefile',
        'RADARSAT2': 'xml',
        'TERRASARX': 'xml',
        'GENERIC': 'hdf5',
        'ERS_ENVI': 'imagefile', #KK 2013-11-26 (ers in envi format)
        'UAVSAR_RPI':'annotationfile',
        'UAVSAR_STACK':'annotationfile',
        'SENTINEL1A':'tiff',
        'SAOCOM':'tiff'
        }
    try:
        key = imageKey[sensorname.upper()]
    except KeyError:
        sys.exit("Unknown sensorname '%s'" % sensorname)
    else:
        scenedict[key] = scenedict[pol]
        return key



def run(sensor, doppler, catalog=None, sceneid='NO_ID'):
    """
    Extract raw image from sensor.
    """

    objMakeRaw = make_raw()
    objMakeRaw(sensor=sensor, doppler=doppler)

    if catalog is not None:
        rawImage = initRawImage(objMakeRaw)

        frame = objMakeRaw.getFrame()
        instrument = frame.getInstrument()
        platform = instrument.getPlatform()
        orbit = frame.getOrbit()

        planet = platform.getPlanet()
        catalog.addInputsFrom(planet, 'planet')
        catalog.addInputsFrom(planet.get_elp(), 'planet.ellipsoid')

        catalog.addInputsFrom(sensor, 'sensor')
        catalog.addItem('width', rawImage.getWidth(), sceneid)
        catalog.addItem('xmin', rawImage.getXmin(), sceneid)
        catalog.addItem('iBias', instrument.getInPhaseValue(), sceneid)
        catalog.addItem('qBias', instrument.getQuadratureValue(), sceneid)
        catalog.addItem('range_sampling_rate', instrument.getRangeSamplingRate(), sceneid)
        catalog.addItem('prf', instrument.getPulseRepetitionFrequency(), sceneid)
        catalog.addItem('pri', 1.0/instrument.getPulseRepetitionFrequency(), sceneid)
        catalog.addItem('pulse_length', instrument.getPulseLength(), sceneid)
        catalog.addItem('chirp_slope', instrument.getChirpSlope(), sceneid)
        catalog.addItem('wavelength', instrument.getRadarWavelength(), sceneid)
        catalog.addItem('lookSide', platform.pointingDirection, sceneid) #KK 2013-12-12
        catalog.addInputsFrom(frame, '%s.frame' % sceneid)
        catalog.addInputsFrom(instrument, '%s.instrument' % sceneid)
        catalog.addInputsFrom(platform, '%s.platform' % sceneid)
        catalog.addInputsFrom(orbit, '%s.orbit' % sceneid)

        catalog.printToLog(logger, "runPreprocessor: %s" % sceneid)

    return objMakeRaw


def initRawImage(makeRawObj):
    """
    Create a rawImage object from a makeRaw object.
    """
    #the "raw" image in some cases is an slc.
    #probably need to make this a factory
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
