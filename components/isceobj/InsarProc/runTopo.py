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
from contrib.demUtils.SWBDStitcher import SWBDStitcher

import logging
logger = logging.getLogger('isce.insar.runTopo') 

def runTopo(self):
    logger.info("Running topo")


    objMocompbaseline = self.insar.mocompBaseline
    objFormSlc1  =  self.insar.formSLC1

    #objDem = isceobj.createDemImage()
    #demImage = self.insar.demImage

    #IU.copyAttributes(demImage, objDem)
    objDem = self.insar.demImage.clone()

    topoIntImage = self._insar.getTopoIntImage()
    #intImage = isceobj.createIntImage()
    #IU.copyAttributes(topoIntImage, intImage)
    intImage = topoIntImage.clone()
    intImage.setAccessMode('read')

    posIndx = 1
    mocompPosition1 = objFormSlc1.getMocompPosition()



    planet = self.insar.referenceFrame.getInstrument().getPlatform().getPlanet()
    prf1 = self.insar.referenceFrame.getInstrument().getPulseRepetitionFrequency()
    
    objTopo = stdproc.createTopo()
    objTopo.wireInputPort(name='peg', object=self.insar.peg)
    objTopo.wireInputPort(name='frame', object=self.insar.referenceFrame)
    objTopo.wireInputPort(name='planet', object=planet)
    objTopo.wireInputPort(name='dem', object=objDem)
    objTopo.wireInputPort(name='interferogram', object=intImage)
    objTopo.wireInputPort(name='referenceslc', object = self.insar.formSLC1) #Piyush
    
    centroid = self.insar.dopplerCentroid.getDopplerCoefficients(inHz=False)[0]
    objTopo.setDopplerCentroidConstantTerm(centroid)

    v = self.insar.procVelocity
    h = self.insar.averageHeight


    objTopo.setBodyFixedVelocity(v)
    objTopo.setSpacecraftHeight(h)

    objTopo.setReferenceOrbit(mocompPosition1[posIndx]) 

    # Options
    objTopo.setNumberRangeLooks(self.insar.numberRangeLooks)
    objTopo.setNumberAzimuthLooks(self.insar.numberAzimuthLooks)
    objTopo.setNumberIterations(self.insar.topophaseIterations)
    objTopo.setHeightSchFilename(self.insar.heightSchFilename)
    objTopo.setHeightRFilename(self.insar.heightFilename)
    objTopo.setLatFilename(self.insar.latFilename)
    objTopo.setLonFilename(self.insar.lonFilename)
    objTopo.setLosFilename(self.insar.losFilename)

    if self.insar.is_mocomp is None:
        self.insar.get_is_mocomp()

    objTopo.setISMocomp(self.insar.is_mocomp)
    #set the tag used in the outfile. each message is precided by this tag
    #is the writer is not of "file" type the call has no effect
    objTopo.stdWriter = self._writer_set_file_tags("topo", "log",
                                                   "err", "out")
    objTopo.setLookSide(self.insar._lookSide)
    objTopo.topo()

    # Record the inputs and outputs
    from isceobj.Catalog import recordInputsAndOutputs
    recordInputsAndOutputs(self._insar.procDoc, objTopo, "runTopo",
                           logger, "runTopo")

    self._insar.setTopo(objTopo)
    if self.insar.applyWaterMask:
        sw = SWBDStitcher()
        sw.toRadar(self.insar.wbdImage.filename,self.insar.latFilename,
                   self.insar.lonFilename,self.insar.waterMaskImageName)

    return objTopo
