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
logger = logging.getLogger('isce.insar.runSetmocomppath')

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
    from isceobj.Location.Peg import Peg
    from stdproc.orbit.pegManipulator import averagePeg
    from isceobj.Catalog import recordInputsAndOutputs

    logger.info("Selecting individual peg points")

    planet = self._insar.getReferenceFrame().getInstrument().getPlatform().getPlanet()
    referenceOrbit = self._insar.getReferenceOrbit()
    secondaryOrbit = self._insar.getSecondaryOrbit()

    if peg:
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

        return


    pegpts = []

    for orbitObj, order in zip((referenceOrbit, secondaryOrbit)
                                ,('First', 'Second')):
        objGetpeg = stdproc.createGetpeg()
        if peg:
            objGetpeg.setPeg(peg)

        objGetpeg.wireInputPort(name='planet', object=planet)
        objGetpeg.wireInputPort(name='Orbit', object=orbitObj)
        self._stdWriter.setFileTag("getpeg", "log")
        self._stdWriter.setFileTag("getpeg", "err")
        self._stdWriter.setFileTag("getpeg", "out")
        objGetpeg.setStdWriter(self._stdWriter)
        logger.info('Peg points are computed for individual SAR scenes.')
        objGetpeg.estimatePeg()
        pegpts.append(objGetpeg.getPeg())

        recordInputsAndOutputs(self._insar.procDoc, objGetpeg, "getpeg", \
                    logger, "runSetmocomppath")
        #Piyush
        # I set these values here for the sake of continuity, but they need to be updated
        # in orbit2sch as the correct peg point is not yet known
        getattr(self._insar,'set%sAverageHeight'%(order))(objGetpeg.getAverageHeight())
        getattr(self._insar,'set%sProcVelocity'%(order))(objGetpeg.getProcVelocity())


    logger.info('Combining individual peg points.')
    peg = averagePeg(pegpts, planet)

    if self.pegSelect.upper() == 'REFERENCE':
        logger.info('Using reference info for peg point')
        self._insar.setPeg(pegpts[0])
    elif self.pegSelect.upper() == 'SECONDARY':
        logger.info('Using secondary infor for peg point')
        self._insar.setPeg(pegpts[1])
    elif self.pegSelect.upper() == 'AVERAGE':
        logger.info('Using average peg point')
        self._insar.setPeg(peg)
    else:
        raise Exception('Unknown peg selection method')

