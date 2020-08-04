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
    from isceobj.Catalog import recordInputsAndOutputs
    chv = []
    for frame, orbit, tag in zip((self._insar.getReferenceFrame(),
                                  self._insar.getSecondaryFrame()),
                                 (self.insar.referenceOrbit,
                                  self.insar.secondaryOrbit),
                                 ('reference', 'secondary')):

        (time, position, velocity, offset) = orbit._unpackOrbit()

        half = len(position)//2 - 1
        xyz = position[half]
        import math
        sch = frame._ellipsoid.xyz_to_sch(xyz)

        chv.append(stdproc.createCalculateFdHeights())
        chv[-1].height = sch[2]

        recordInputsAndOutputs(self.procDoc, chv[-1],
                               "runEstimateHeights.CHV_"+tag, logger,
                               "runEstimateHeights.CHV_"+tag)

    self.insar.fdH1, self.insar.fdH2 = [item.height for item in chv]
    return None
