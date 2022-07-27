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



# Comment: Adapted from InsarProc/runPrepareResamps.py
import math
import logging

from isceobj.Constants import SPEED_OF_LIGHT

logger = logging.getLogger('isce.isceProc.runPrepareResamps')

def runPrepareResamps(self, rgLooks=None, azLooks=None):
    refScene = self._isce.refScene
    refPol = self._isce.refPol

    orbit = self._isce.orbits[refScene][refPol]
    frame = self._isce.frames[refScene][refPol]
    peg = self._isce.peg
    slcImage = self._isce.slcImages[refScene][refPol]

    time, schPosition, schVelocity, offset = orbit._unpackOrbit()
    s2 = schPosition[0][0]
    s2_2 = schPosition[1][0]

    lines = self._isce.numberPatches * self._isce.numberValidPulses
    self._isce.numberResampLines = lines

    fs = frame.getInstrument().getRangeSamplingRate()
    dr = (SPEED_OF_LIGHT / (2 * fs))
    self._isce.slantRangePixelSpacing = dr

    widthSlc = slcImage.getWidth()

    radarWavelength = frame.getInstrument().getRadarWavelength()

    rc = peg.getRadiusOfCurvature()
    ht = self._isce.averageHeight
    r0 = frame.getStartingRange()

    range = r0 + (widthSlc / 2 * dr)

    costheta = (2*rc*ht+ht*ht-range*range)/-2/rc/range
    sininc = math.sqrt(1 - (costheta * costheta))

    posting = self.posting
    grndpixel = dr / sininc

    if rgLooks:
        looksrange = rgLooks
    else:
        looksrange = int(posting/grndpixel+0.5)

    if azLooks:
        looksaz = azLooks
    else:
        looksaz = int(round(posting/(s2_2 - s2)))

    if (looksrange < 1):
        logger.warning("Number range looks less than zero, setting to 1")
        looksrange = 1
    if (looksaz < 1):
        logger.warning("Number azimuth looks less than zero, setting to 1")
        looksaz = 1

    self._isce.numberAzimuthLooks = looksaz
    self._isce.numberRangeLooks = looksrange
