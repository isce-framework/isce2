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



# Comment: Adapted from InsarProc/runEstimateHeights.py
import logging
import stdproc
import isceobj

logger = logging.getLogger('isce.isceProc.runEstimateHeights')

def runEstimateHeights(self):
    for sceneid in self._isce.selectedScenes:
        self._isce.fdHeights[sceneid] = {}
        for pol in self._isce.selectedPols:
            frame = self._isce.frames[sceneid][pol]
            orbit = self._isce.orbits[sceneid][pol]
            catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
            sid = self._isce.formatname(sceneid, pol)
            chv = run(frame, orbit, catalog=catalog, sceneid=sid)
            self._isce.procDoc.addAllFromCatalog(catalog)
            self._isce.fdHeights[sceneid][pol] = chv.height


def run(frame, orbit, catalog=None, sceneid='NO_ID'):
    """
    Estimate heights from orbit.
    """
    (time, position, velocity, offset) = orbit._unpackOrbit()

    half = len(position)//2 - 1
    xyz = position[half]
    sch = frame._ellipsoid.xyz_to_sch(xyz)

    chv = stdproc.createCalculateFdHeights()
#    planet = frame.getInstrument().getPlatform().getPlanet()
#    chv(frame=frame, orbit=orbit, planet=planet)
    chv.height = sch[2]

    if catalog is not None:
        isceobj.Catalog.recordInputsAndOutputs(catalog, chv,
                                               "runEstimateHeights.CHV.%s" % sceneid,
                                               logger,
                                               "runEstimateHeights.CHV.%s" % sceneid)
    return chv
