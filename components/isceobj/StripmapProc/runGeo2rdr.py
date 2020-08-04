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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






import isceobj
import stdproc
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj.Util.Polynomial import Polynomial
from isceobj.Util.Poly2D import Poly2D
from isceobj.Constants import SPEED_OF_LIGHT
import logging
import numpy as np
import datetime
import os

logger = logging.getLogger('isce.insar.runGeo2rdr') 

def runGeo2rdr(self):
    from zerodop.geo2rdr import createGeo2rdr
    from isceobj.Planet.Planet import Planet

    logger.info("Running geo2rdr")

    info = self._insar.loadProduct( self._insar.secondarySlcCropProduct)

    offsetsDir = self.insar.offsetsDirname 
    os.makedirs(offsetsDir, exist_ok=True)

    grdr = createGeo2rdr()
    grdr.configure()

    planet = info.getInstrument().getPlatform().getPlanet()
    grdr.slantRangePixelSpacing = info.getInstrument().getRangePixelSize()
    grdr.prf = info.PRF #info.getInstrument().getPulseRepetitionFrequency()
    grdr.radarWavelength = info.getInstrument().getRadarWavelength()
    grdr.orbit = info.getOrbit()
    grdr.width = info.getImage().getWidth()
    grdr.length = info.getImage().getLength()

    grdr.wireInputPort(name='planet', object=planet)
    grdr.lookSide =  info.instrument.platform.pointingDirection

    grdr.setSensingStart(info.getSensingStart())
    grdr.rangeFirstSample = info.startingRange
    grdr.numberRangeLooks = 1
    grdr.numberAzimuthLooks = 1


    if self.insar.secondaryGeometrySystem.lower().startswith('native'):
        p = [x/info.PRF for x in info._dopplerVsPixel]
    else:
        p = [0.]

    grdr.dopplerCentroidCoeffs = p
    grdr.fmrateCoeffs = [0.]

    ###Input and output files
    grdr.rangeOffsetImageName = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
    grdr.azimuthOffsetImageName = os.path.join(offsetsDir, self.insar.azimuthOffsetFilename)

    latFilename = os.path.join(self.insar.geometryDirname, self.insar.latFilename + '.full')
    lonFilename = os.path.join(self.insar.geometryDirname, self.insar.lonFilename + '.full')
    heightFilename = os.path.join(self.insar.geometryDirname, self.insar.heightFilename + '.full')

    demImg = isceobj.createImage()
    demImg.load(heightFilename + '.xml')
    demImg.setAccessMode('READ')
    grdr.demImage = demImg

    latImg = isceobj.createImage()
    latImg.load(latFilename + '.xml')
    latImg.setAccessMode('READ')
    grdr.latImage = latImg

    lonImg = isceobj.createImage()
    lonImg.load(lonFilename + '.xml')
    lonImg.setAccessMode('READ')

    grdr.lonImage = lonImg
    grdr.outputPrecision = 'DOUBLE'
        
    grdr.geo2rdr()

    return
