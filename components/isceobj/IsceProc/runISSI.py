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



# Comment: Adapted from contrib/ISSI/FR.py
import sys
import os
import math
from contrib.ISSI.FR import FR
from ISSI import Focuser
from make_raw import make_raw
from mroipac.geolocate.Geolocate import Geolocate

import logging
logger = logging.getLogger('isce.isceProc.runISSI')


def runISSI(self, opList):
    for sceneid in self._isce.selectedScenes:
        raws = {}
        slcFiles = {}
        for pol in ['hh', 'hv', 'vh', 'vv']:
            raws[pol] = make_raw()
            raws[pol].frame = self._isce.frames[sceneid][pol]
            slcFiles[pol] = self._isce.slcImages[sceneid][pol]
        focuser = Focuser(hh=raws['hh'], hv=raws['hv'], vh=raws['vh'], vv=raws['vv'])
        focuser.filter = self.FR_filter
        focuser.filterSize = (int(self.FR_filtersize_x), int(self.FR_filtersize_y))
        focuser.logger = logger

        outputs = {}
        for fname in [self._isce.frOutputName, self._isce.tecOutputName, self._isce.phaseOutputName]:
            outputs[fname] = os.path.join(self.getoutputdir(sceneid), self._isce.formatname(sceneid, ext=fname+'.slc'))

        hhFile = slcFiles['hh']
        issiobj = FR(hhFile=hhFile.filename,
                     hvFile=slcFiles['hv'].filename,
                     vhFile=slcFiles['vh'].filename,
                     vvFile=slcFiles['vv'].filename,
                     lines=hhFile.length,
                     samples=hhFile.width,
                     frOutput=outputs[self._isce.frOutputName],
                     tecOutput=outputs[self._isce.tecOutputName],
                     phaseOutput=outputs[self._isce.phaseOutputName])

        if 'polcal' in opList: ## polarimetric calibration
            issiobj.polarimetricCorrection(self._isce.transmit, self._isce.receive)
            for pol, fname in zip(['hh', 'hv', 'vh', 'vv'], [issiobj.hhFile, issiobj.hvFile, issiobj.vhFile, issiobj.vvFile]):
                self._isce.slcImages[sceneid][pol].filename = fname

        if 'fr' in opList: ## calculate faraday rotation
            frame = self._isce.frames[self._isce.refScene][self._isce.refPol]
            if frame.getImage().byteOrder != sys.byteorder[0]:
                logger.info("Will swap bytes")
                swap = True
            else:
                logger.info("Will not swap bytes")
                swap = False

            issiobj.calculateFaradayRotation(filter=focuser.filter, filterSize=focuser.filterSize, swap=swap)
            aveFr = issiobj.getAverageFaradayRotation()
            logger.info("Image Dimensions %s: %s x %s" % (sceneid, issiobj.samples,issiobj.lines))
            logger.info("Average Faraday Rotation %s: %s rad (%s deg)" % (sceneid, aveFr, math.degrees(aveFr)))

        if 'tec' in opList:
            date = focuser.hhObj.frame.getSensingStart()
            corners, lookAngles = focuser.calculateCorners()
            lookDirections = focuser.calculateLookDirections()
            fc = focuser.hhObj.frame.getInstrument().getRadarFrequency()
            meankdotb = issiobj.frToTEC(date, corners, lookAngles, lookDirections, fc)
            logger.info("Mean k.B value %s: %s" % (sceneid, meankdotb))

        if 'phase' in opList:
            fc = focuser.hhObj.frame.getInstrument().getRadarFrequency()
            issiobj.tecToPhase(fc)

