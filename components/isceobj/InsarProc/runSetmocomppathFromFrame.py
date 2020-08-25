#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import logging
import stdproc
logger = logging.getLogger('isce.insar.runSetmocomppath')
from isceobj.Catalog import recordInputsAndOutputs

def averageHeightAboveElp(planet, peg, orbit):
    elp = planet.get_elp()
    elp.setSCH(peg.latitude, peg.longitude, peg.heading)
    t, posXYZ, velXYZ, offset = orbit._unpackOrbit()
    hsum = 0.
    for xyz in posXYZ:
        llh = elp.xyz_to_llh(xyz)
        hsum += llh[2]
    print("averageHeightAboveElp: hsum, len(posXYZ), havg = ",
        hsum, len(posXYZ), havg)
    return hsum/len(posXYZ)

def sVelocityAtMidOrbit(planet, peg, orbit):
    elp = planet.get_elp()
    elp.setSCH(peg.latitude, peg.longitude, peg.heading)
    t, posXYZ, velXYZ, offset = orbit._unpackOrbit()
    sch, vsch = elp.xyzdot_to_schdot(
        posXYZ[len(posXYZ)/2+1], velXYZ[len(posXYZ)/2+1])
    print("sVelocityAtPeg: len(posXYZ)/2., vsch = ",
          len(posXYZ)/2+1, vsch)
    return vsch[0]

def runSetmocomppath(self, peg=None):
    """
    Set the peg point, mocomp heights, and mocomp velocities.
    From information provided in the sensor object
    Possible named input peg (in degrees) is used to set the peg
    rather than using the one given in the Frame.
    """

    planet = (
        self._insar.getReferenceFrame().getInstrument().getPlatform().getPlanet())
    referenceOrbit = self._insar.getReferenceOrbit()
    secondaryOrbit = self._insar.getSecondaryOrbit()

    if peg:
        #If the input peg is set, then use it
        self._insar.setPeg(peg)
        logger.info("Using the given peg = %r", peg)
        self._insar.setFirstAverageHeight(
            averageHeightAboveElp(planet, peg, referenceOrbit))
        self._insar.setSecondAverageHeight(
            averageHeightAboveElp(planet, peg, secondaryOrbit))
        self._insar.setFirstProcVelocity(
            sVelocityAtMidOrbit(planet, peg, referenceOrbit))
        self._insar.setSecondProcVelocity(
            sVelocityAtMidOrbit(planet, peg, secondaryOrbit))
#        recordInputsAndOutputs(self._insar.procDoc, peg, "peg",
#            logger, "runSetmocomppath")
        return

    logger.info("Selecting peg points from frames")

    pegpts = []
    pegpts.append(self._insar.getReferenceFrame().peg)
    pegpts.append(self._insar.getReferenceFrame().peg)
    peg = averagePeg(pegpts, planet)
    self._insar.setPeg(peg)

    self._insar.setFirstAverageHeight(
        self._insar.getReferenceFrame().platformHeight)
    self._insar.setSecondAverageHeight(
        self._insar.getSecondaryFrame().platformHeight)
    self._insar.setFirstProcVelocity(
        self._insar.getReferenceFrame().procVelocity)
    self._insar.setSecondProcVelocity(
        self._insar.getSecondaryFrame().procVelocity)

