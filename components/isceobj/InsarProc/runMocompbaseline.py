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
import stdproc
import isceobj

logger = logging.getLogger('isce.insar.runMocompbaseline')

# index of the position in the  mocompPosition array (the 0 element is the
# time)
posIndx = 1

def runMocompbaseline(self):
    logger.info("Calculating Baseline")
    ellipsoid = self._insar.getReferenceFrame().getInstrument().getPlatform().getPlanet().get_elp()
    # schPositions computed in orbit2sch
    # objFormSlc's  created during formSlc

    h = self.insar.averageHeight
    objFormSlc1  =  self.insar.formSLC1
    objFormSlc2  =  self.insar.formSLC2
    mocompPosition1 = objFormSlc1.getMocompPosition()
    mocompIndex1 = objFormSlc1.getMocompIndex()
    mocompPosition2 = objFormSlc2.getMocompPosition()
    mocompIndex2 = objFormSlc2.getMocompIndex()

    objMocompbaseline = stdproc.createMocompbaseline()

    objMocompbaseline.setMocompPosition1(mocompPosition1[posIndx])
    objMocompbaseline.setMocompPositionIndex1(mocompIndex1)
    objMocompbaseline.setMocompPosition2(mocompPosition2[posIndx])
    objMocompbaseline.setMocompPositionIndex2(mocompIndex2)

    objMocompbaseline.wireInputPort(name='referenceOrbit',
                                    object=self.insar.referenceOrbit)
    objMocompbaseline.wireInputPort(name='secondaryOrbit',
                                    object=self.insar.secondaryOrbit)
    objMocompbaseline.wireInputPort(name='ellipsoid', object=ellipsoid)
    objMocompbaseline.wireInputPort(name='peg', object=self.insar.peg)
    objMocompbaseline.setHeight(h)

    #set the tag used in the outfile. each message is precided by this tag
    #is the writer is not of "file" type the call has no effect
    self._stdWriter.setFileTag("mocompbaseline", "log")
    self._stdWriter.setFileTag("mocompbaseline", "err")
    self._stdWriter.setFileTag("mocompbaseline", "out")
    objMocompbaseline.setStdWriter(self._stdWriter)

    objMocompbaseline.mocompbaseline()

    # Record the inputs and outputs
    from isceobj.Catalog import recordInputsAndOutputs
    recordInputsAndOutputs(self._insar.procDoc, objMocompbaseline,
                           "runMocompbaseline",
                           logger, "runMocompbaseline")

    self.insar.mocompBaseline = objMocompbaseline
    return None
