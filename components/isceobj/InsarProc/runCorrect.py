#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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



import logging

import isceobj
import stdproc
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU


logger = logging.getLogger('isce.insar.runCorrect') 


def runCorrect(self):
    logger.info("Running correct")

    objMocompbaseline = self.insar.mocompBaseline
    objFormSlc1  =  self.insar.formSLC1

    topoIntImage = self.insar.topoIntImage
    intImage = isceobj.createIntImage()
    #just pass the image object to Correct and it will handle the creation
    # and deletion of the actual image pointer  
    IU.copyAttributes(topoIntImage, intImage)

    posIndx = 1
    mocompPosition1 = objFormSlc1.mocompPosition

    planet = self.insar.referenceFrame.instrument.platform.planet
    prf1 = self.insar.referenceFrame.instrument.PRF
    objCorrect = stdproc.createCorrect()
    objCorrect.wireInputPort(name='peg', object=self.insar.peg)
    objCorrect.wireInputPort(name='frame', object=self.insar.referenceFrame)
    objCorrect.wireInputPort(name='planet', object=planet)
    objCorrect.wireInputPort(name='interferogram', object=intImage)
    objCorrect.wireInputPort(name='referenceslc', object=self.insar.formSLC1) #Piyush
    # Average velocity and height measurements       
    v = self.insar.procVelocity
    h = self.insar.averageHeight
    objCorrect.setBodyFixedVelocity(v)
    objCorrect.setSpacecraftHeight(h)
    # Need the reference orbit from Formslc       
    objCorrect.setReferenceOrbit(mocompPosition1[posIndx]) 
    objCorrect.setMocompBaseline(objMocompbaseline.baseline) 
    sch12 = objMocompbaseline.getSchs()
    objCorrect.setSch1(sch12[0])
    objCorrect.setSch2(sch12[1])
    sc = objMocompbaseline.sc
    objCorrect.setSc(sc)
    midpoint = objMocompbaseline.midpoint
    objCorrect.setMidpoint(midpoint)
    objCorrect.setLookSide(self.insar._lookSide)


    objCorrect.setNumberRangeLooks(self.insar.numberRangeLooks)
    objCorrect.setNumberAzimuthLooks(self.insar.numberAzimuthLooks)
    objCorrect.setTopophaseMphFilename(self.insar.topophaseMphFilename)
    objCorrect.setTopophaseFlatFilename(self.insar.topophaseFlatFilename)
    objCorrect.setHeightSchFilename(self.insar.heightSchFilename)

    objCorrect.setISMocomp(self.insar.is_mocomp)
    #set the tag used in the outfile. each message is precided by this tag
    #is the writer is not of "file" type the call has no effect
    objCorrect.stdWriter = self._writer_set_file_tags("correct",
                                                      "log", "err", "out")
    
    objCorrect()#.correct()

    # Record the inputs and outputs
    from isceobj.Catalog import recordInputsAndOutputs
    recordInputsAndOutputs(self.insar.procDoc, objCorrect, "runCorrect", 
                           logger, "runCorrect")


    return objCorrect

