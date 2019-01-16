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
# Authors: Kosal Khun, Marco Lavalle
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Comment: Adapted from InsarProc/runUnwrappSnaphu.py
# giangi: taken Piyush code for snaphu and adapted

import logging
import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from isceobj.Constants import SPEED_OF_LIGHT
import os

logger = logging.getLogger('isce.isceProc.runUnwrap')

def runUnwrap(self, costMode='DEFO', initMethod='MST', defomax=4.0, initOnly=False):
    infos = {}
    for attribute in ['topophaseFlatFilename', 'unwrappedIntFilename', 'coherenceFilename', 'averageHeight', 'topo', 'peg']:
        infos[attribute] = getattr(self._isce, attribute)

    for sceneid1, sceneid2 in self._isce.selectedPairs:
        pair = (sceneid1, sceneid2)
        for pol in self._isce.selectedPols:
            frame1 = self._isce.frames[sceneid1][pol]
            intImage = self._isce.resampIntImages[pair][pol]
            width = intImage.width
            sid = self._isce.formatname(pair, pol)
            infos['outputPath'] = os.path.join(self.getoutputdir(sceneid1, sceneid2), sid)
            run(frame1, width, costMode, initMethod, defomax, initOnly, infos, sceneid=sid)


def run(frame1, width, costMode, initMethod, defomax, initOnly, infos, sceneid='NO_ID'):
    logger.info("Unwrapping interferogram using Snaphu %s: %s" % (initMethod, sceneid))
    topo = infos['topo']
    wrapName = infos['outputPath'] + '.' + infos['topophaseFlatFilename']
    unwrapName = infos['outputPath'] + '.' + infos['unwrappedIntFilename']
    corrfile  = infos['outputPath'] + '.' + infos['coherenceFilename']
    altitude = infos['averageHeight']
    wavelength = frame1.getInstrument().getRadarWavelength()
    earthRadius = infos['peg'].radiusOfCurvature

    rangeLooks = topo.numberRangeLooks
    azimuthLooks = topo.numberAzimuthLooks

    azres = frame1.platform.antennaLength/2.0
    azfact = topo.numberAzimuthLooks *azres / topo.azimuthSpacing

    rBW = frame1.instrument.pulseLength * frame1.instrument.chirpSlope
    rgres = abs(SPEED_OF_LIGHT / (2.0 * rBW))
    rngfact = rgres/topo.slantRangePixelSpacing

    corrLooks = topo.numberRangeLooks * topo.numberAzimuthLooks/(azfact*rngfact)
    maxComponents = 20

    snp = Snaphu()
    snp.setInitOnly(initOnly)
    snp.setInput(wrapName)
    snp.setOutput(unwrapName)
    snp.setWidth(width)
    snp.setCostMode(costMode)
    snp.setEarthRadius(earthRadius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(corrfile)
    snp.setInitMethod(initMethod)
    snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(maxComponents)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(rangeLooks)
    snp.setAzimuthLooks(azimuthLooks)
    snp.prepare()
    snp.unwrap()

    ######Render XML
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setAccessMode('read')
    outImage.imageType = 'unw'
    outImage.bands = 2
    outImage.scheme = 'BIL'
    outImage.dataType = 'FLOAT'
    outImage.finalizeImage()
    outImage.renderHdr()

    #####Check if connected components was created
    if snp.dumpConnectedComponents:
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName+'.conncomp')
        connImage.setWidth(width)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        connImage.finalizeImage()
        connImage.renderHdr()


def runUnwrapMcf(self):
    runUnwrap(self, costMode='SMOOTH', initMethod='MCF', defomax=2, initOnly=True)
