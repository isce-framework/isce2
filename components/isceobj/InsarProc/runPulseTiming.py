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



import datetime
import logging

from isceobj.Orbit.Orbit import Orbit

logger = logging.getLogger('isce.insar.runPulseTiming')

def runPulseTiming(self):
    reference = self.insar.referenceFrame
    secondary = self.insar.secondaryFrame
    # add orbits to main object -law of demeter pls.
    self.insar.referenceOrbit = pulseTiming(reference, self.insar.procDoc, 'reference')
    self.insar.secondaryOrbit = pulseTiming(secondary, self.insar.procDoc, 'secondary')
    return None

def pulseTiming(frame, catalog, which):
    logger.info("Pulse Timing")
    numberOfLines = frame.getNumberOfLines()
    prf = frame.getInstrument().getPulseRepetitionFrequency()
    pri = 1.0 / prf
    startTime = frame.getSensingStart()
    orbit = frame.getOrbit()
    pulseOrbit = Orbit(name=which+'orbit')
    startTimeUTC0 = (startTime -
                     datetime.datetime(startTime.year,
                                       startTime.month,startTime.day)
                     )
    timeVec = [pri*i +
               startTimeUTC0.seconds +
               10**-6*startTimeUTC0.microseconds for i in range(numberOfLines)
               ]
    catalog.addItem("timeVector", timeVec, "runPulseTiming.%s" % which)
    for i in range(numberOfLines):
        dt = i * pri
        time = startTime + datetime.timedelta(seconds=dt)
        sv = orbit.interpolateOrbit(time, method='hermite')
        pulseOrbit.addStateVector(sv)
        
    return pulseOrbit
