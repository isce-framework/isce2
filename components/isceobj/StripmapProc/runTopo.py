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
logger = logging.getLogger('isce.insar.runTopo') 

def runTopo(self):
    from zerodop.topozero import createTopozero
    from isceobj.Planet.Planet import Planet

    logger.info("Running topo")

    #IU.copyAttributes(demImage, objDem)
    geometryDir = self.insar.geometryDirname

    os.makedirs(geometryDir, exist_ok=True)


    demFilename = self.verifyDEM()
    objDem = isceobj.createDemImage()
    objDem.load(demFilename + '.xml')

    info = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    intImage = info.getImage()


    planet = info.getInstrument().getPlatform().getPlanet()
    topo = createTopozero()

    topo.slantRangePixelSpacing = 0.5 * SPEED_OF_LIGHT / info.rangeSamplingRate
    topo.prf = info.PRF
    topo.radarWavelength = info.radarWavelegth
    topo.orbit = info.orbit
    topo.width = intImage.getWidth()
    topo.length = intImage.getLength()
    topo.wireInputPort(name='dem', object=objDem)
    topo.wireInputPort(name='planet', object=planet)
    topo.numberRangeLooks = 1
    topo.numberAzimuthLooks = 1
    topo.lookSide = info.getInstrument().getPlatform().pointingDirection
    topo.sensingStart = info.getSensingStart()
    topo.rangeFirstSample = info.startingRange

    topo.demInterpolationMethod='BIQUINTIC'
    topo.latFilename = os.path.join(geometryDir, self.insar.latFilename + '.full')
    topo.lonFilename = os.path.join(geometryDir, self.insar.lonFilename + '.full')
    topo.losFilename = os.path.join(geometryDir, self.insar.losFilename + '.full')
    topo.heightFilename = os.path.join(geometryDir, self.insar.heightFilename + '.full')
#    topo.incFilename = os.path.join(info.outdir, 'inc.rdr')
#    topo.maskFilename = os.path.join(info.outdir, 'mask.rdr')


    ####Doppler adjustment
    dop = [x/1.0 for x in info._dopplerVsPixel]
     
    doppler = Poly2D()
    doppler.setWidth(topo.width // topo.numberRangeLooks)
    doppler.setLength(topo.length // topo.numberAzimuthLooks)

    if self._insar.referenceGeometrySystem.lower().startswith('native'):
        doppler.initPoly(rangeOrder = len(dop)-1, azimuthOrder=0, coeffs=[dop])
    else:
        doppler.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.]])

    topo.polyDoppler = doppler

    topo.topo()


    # Record the inputs and outputs
    from isceobj.Catalog import recordInputsAndOutputs
    recordInputsAndOutputs(self._insar.procDoc, topo, "runTopo",
                           logger, "runTopo")


    self._insar.estimatedBbox = [topo.minimumLatitude, topo.maximumLatitude,
                                topo.minimumLongitude, topo.maximumLongitude]
    return topo
