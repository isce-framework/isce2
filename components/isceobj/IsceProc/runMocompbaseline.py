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



# Comment: Adapted from InsarProc/runMocompbaseline.py
import logging
import stdproc
import isceobj

logger = logging.getLogger('isce.isceProc.runMocompbaseline')


def runMocompbaseline(self):
    refPol = self._isce.refPol
    averageHeight = self._isce.averageHeight
    peg = self._isce.peg
    stdWriter = self._stdWriter
    for sceneid1, sceneid2 in self._isce.selectedPairs:
        pair = (sceneid1, sceneid2)
        objFormSlc1 = self._isce.formSLCs[sceneid1][refPol]
        objFormSlc2 = self._isce.formSLCs[sceneid2][refPol]
        orbit1 = self._isce.orbits[sceneid1][refPol]
        orbit2 = self._isce.orbits[sceneid2][refPol]
        frame1 = self._isce.frames[sceneid1][refPol]
        ellipsoid = frame1.getInstrument().getPlatform().getPlanet().get_elp()
        catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
        sid = self._isce.formatname(pair)
        objMocompbaseline = run(objFormSlc1, objFormSlc2, orbit1, orbit2, ellipsoid, averageHeight, peg, stdWriter, catalog=catalog, sceneid=sid)
        self._isce.mocompBaselines[pair] = objMocompbaseline


# index of the position in the  mocompPosition array
# (the 0 element is the time)
posIndx = 1


def run(objFormSlc1, objFormSlc2, orbit1, orbit2, ellipsoid, averageHeight, peg, stdWriter, catalog=None, sceneid='NO_ID'):
    logger.info("Calculating Baseline: %s" % sceneid)

    # schPositions computed in orbit2sch
    # objFormSlc's  created during formSlc

    mocompPosition1 = objFormSlc1.getMocompPosition()
    mocompIndex1 = objFormSlc1.getMocompIndex()
    mocompPosition2 = objFormSlc2.getMocompPosition()
    mocompIndex2 = objFormSlc2.getMocompIndex()

    objMocompbaseline = stdproc.createMocompbaseline()

    objMocompbaseline.setMocompPosition1(mocompPosition1[posIndx])
    objMocompbaseline.setMocompPositionIndex1(mocompIndex1)
    objMocompbaseline.setMocompPosition2(mocompPosition2[posIndx])
    objMocompbaseline.setMocompPositionIndex2(mocompIndex2)

    objMocompbaseline.wireInputPort(name='referenceOrbit', object=orbit1)
    objMocompbaseline.wireInputPort(name='secondaryOrbit', object=orbit2)
    objMocompbaseline.wireInputPort(name='ellipsoid', object=ellipsoid)
    objMocompbaseline.wireInputPort(name='peg', object=peg)
    objMocompbaseline.setHeight(averageHeight)

    #set the tag used in the outfile. each message is precided by this tag
    #is the writer is not of "file" type the call has no effect
    objMocompbaseline.stdWriter = stdWriter.set_file_tags("mocompbaseline",
                                                          "log",
                                                          "err",
                                                          "out")

    objMocompbaseline.mocompbaseline()

    if catalog is not None:
        # Record the inputs and outputs
        isceobj.Catalog.recordInputsAndOutputs(catalog, objMocompbaseline,
                                               "runMocompbaseline.%s" % sceneid,
                                               logger,
                                               "runMocompbaseline.%s" % sceneid)


    return objMocompbaseline
