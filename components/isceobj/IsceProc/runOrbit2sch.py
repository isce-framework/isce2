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



# Comment: Adapted from InsarProc/runOrbit2sch.py
import logging
import stdproc
import isceobj

logger = logging.getLogger('isce.isceProc.runOrbit2sch')

def runOrbit2sch(self):
    planet = self._isce.planet
    peg = self._isce.peg
    pegHavg = self._isce.averageHeight
    stdWriter = self._stdWriter
    for sceneid in self._isce.selectedScenes:
        for pol in self._isce.selectedPols:
            frame = self._isce.frames[sceneid][pol]
            orbit = self._isce.orbits[sceneid][pol]
            catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
            sid = self._isce.formatname(sceneid, pol)
            orbit, velocity = run(orbit, peg, pegHavg, planet, stdWriter, catalog=catalog, sceneid=sid)
            self._isce.orbits[sceneid][pol] = orbit ##update orbit
            self._isce.pegProcVelocities[sceneid][pol] = velocity ##update velocity



def run(orbit, peg, pegHavg, planet, stdWriter, catalog=None, sceneid='NO_ID'):
    """
    Convert orbit to SCH.
    """
    logger.info("Converting the orbit to SCH coordinates: %s" % sceneid)

    objOrbit2sch = stdproc.createOrbit2sch(averageHeight=pegHavg)
    objOrbit2sch.stdWriter = stdWriter.set_file_tags("orbit2sch",
                                                     "log",
                                                     "err",
                                                     "log")

    objOrbit2sch(planet=planet, orbit=orbit, peg=peg)
    if catalog:
        isceobj.Catalog.recordInputsAndOutputs(catalog, objOrbit2sch,
                                               "runOrbit2sch." + sceneid,
                                               logger,
                                               "runOrbit2sch." + sceneid)


    #Piyush
    ####The heights and the velocities need to be updated now.
    (ttt, ppp, vvv, rrr) = objOrbit2sch.orbit._unpackOrbit()
    procVelocity = vvv[len(vvv)//2][0]

    return objOrbit2sch.orbit, procVelocity
