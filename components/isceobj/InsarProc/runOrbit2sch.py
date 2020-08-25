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
import copy

logger = logging.getLogger('isce.insar.runOrbit2sch')


def runOrbit2sch(self):
    from isceobj.Catalog import recordInputsAndOutputs
    import numpy
    logger.info("Converting the orbit to SCH coordinates")

    # Piyush
    ####We don't know the correct SCH heights yet.
    ####Not computing average peg height yet.
    peg = self.insar.peg
    pegHavg = self.insar.averageHeight
    planet = self.insar.planet

#    if self.pegSelect.upper() == 'REFERENCE':
#        pegHavg = self.insar.getFirstAverageHeight()
#    elif self.pegSelect.upper() == 'SECONDARY':
#        pegHavg = self.insar.getSecondAverageHeight()
#    elif self.pegSelect.upper() == 'AVERAGE':
#        pegHavg = self.insar.averageHeight
#    else:
#        raise Exception('Unknown peg selection method: ', self.pegSelect)

    referenceOrbit = self.insar.referenceOrbit
    secondaryOrbit = self.insar.secondaryOrbit

    objOrbit2sch1 = stdproc.createOrbit2sch(averageHeight=pegHavg)
    objOrbit2sch1.stdWriter = self.stdWriter.set_file_tags("orbit2sch",
                                                           "log",
                                                           "err",
                                                           "log")
    objOrbit2sch2 = stdproc.createOrbit2sch(averageHeight=pegHavg)
    objOrbit2sch2.stdWriter = self.stdWriter

    ## loop over reference/secondary orbits
    for obj, orb, tag, order in zip((objOrbit2sch1, objOrbit2sch2),
                                    (self.insar.referenceOrbit, self.insar.secondaryOrbit),
                                    ('reference', 'secondary'),
                                    ('First', 'Second')):
        obj(planet=planet, orbit=orb, peg=peg)
        recordInputsAndOutputs(self.insar.procDoc, obj,
                               "runOrbit2sch." + tag,
                               logger,
                               "runOrbit2sch." + tag)

        #equivalent to self.insar.referenceOrbit =
        setattr(self.insar,'%sOrbit'%(tag), obj.orbit)

        #Piyush
        ####The heights and the velocities need to be updated now.
        (ttt, ppp, vvv, rrr) = obj.orbit._unpackOrbit()

        #equivalent to self.insar.setFirstAverageHeight()
        # SCH heights replacing the earlier llh heights
        # getattr(self.insar,'set%sAverageHeight'%(order))(numpy.sum(numpy.array(ppp),axis=0)[2] /(1.0*len(ppp)))

        #equivalent to self.insar.setFirstProcVelocity()
        getattr(self.insar,'set%sProcVelocity'%(order))(vvv[len(vvv)//2][0])

    return None

