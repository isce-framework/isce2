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


from isceobj.Constants import SPEED_OF_LIGHT

logger = logging.getLogger('isce.insar.runPrepareResamps')

def runPrepareResamps(self, rangeLooks=None, azLooks=None):
    import math
    secondaryOrbit = self.insar.secondaryOrbit
    referenceFrame = self.insar.referenceFrame
    peg = self.insar.peg
    referenceSlcImage = self.insar.referenceSlcImage
    time2, schPosition2, schVelocity2, offset2 = secondaryOrbit._unpackOrbit()
    
    s2 = schPosition2[0][0]
    s2_2 = schPosition2[1][0]
    
    valid_az_samples =  self.insar.numberValidPulses
    numPatches = self.insar.numberPatches
    lines = numPatches * valid_az_samples 
    
    fs = referenceFrame.getInstrument().getRangeSamplingRate()
    dr = (SPEED_OF_LIGHT / (2 * fs))
    
    self._insar.setSlantRangePixelSpacing(dr)
    
#    widthSlc = max(self._insar.getReferenceSlcImage().getWidth(), self._insar.getSecondarySlcImage().getWidth())
    widthSlc = self._insar.getReferenceSlcImage().getWidth()
    
    radarWavelength = referenceFrame.getInstrument().getRadarWavelength()
    
    rc = peg.getRadiusOfCurvature()  
    ht = self._insar.getAverageHeight()
    r0 = referenceFrame.getStartingRange()
    
    range = r0 + (widthSlc / 2 * dr)
    
    costheta = (2*rc*ht+ht*ht-range*range)/-2/rc/range
    sininc = math.sqrt(1 - (costheta * costheta))
    
    posting = self.posting
    grndpixel = dr / sininc
    
    if rangeLooks:
        looksrange=rangeLooks
    else:
        looksrange=int(posting/grndpixel+0.5)

    if azLooks:
        looksaz=azLooks
    else:
        looksaz=int(round(posting/(s2_2 - s2)))
    
    if (looksrange < 1):
        logger.warning("Number range looks less than zero, setting to 1")
        looksrange = 1
    if (looksaz < 1):
        logger.warning("Number azimuth looks less than zero, setting to 1")
        looksaz = 1

    self._insar.setNumberAzimuthLooks(looksaz) 
    self._insar.setNumberRangeLooks(looksrange) 
    self._insar.setNumberResampLines(lines) 
    

    #jng at one point this will go in the defaults of the self._insar calss
    numFitCoeff = 6
    self._insar.setNumberFitCoefficients(numFitCoeff) 
