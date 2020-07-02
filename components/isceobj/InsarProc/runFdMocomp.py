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
import sys
logger = logging.getLogger('isce.insar.runFdMocomp')

## Mapping from use_dop kewword to f(referenceDop, secondaryDrop)
USE_DOP = {'AVERAGE' : lambda x, y: (x+y)/2.,
           'REFERENCE': lambda x, y: x,
           'SECONDARY': lambda x, y: y}

def runFdMocomp(self, use_dop="average"):
    """
    Calculate motion compenstation correction for Doppler centroid
    """
    H1 = self.insar.fdH1
    H2 = self.insar.fdH2
    peg = self.insar.peg
    lookSide = self.insar._lookSide
    referenceOrbit = self.insar.referenceOrbit
    secondaryOrbit = self.insar.secondaryOrbit
    rangeSamplingRate = (
        self.insar.getReferenceFrame().instrument.rangeSamplingRate)
    rangePulseDuration = (
        self.insar.getSecondaryFrame().instrument.pulseLength)
    chirpExtension = self.insar.chirpExtension
    chirpSize = int(rangeSamplingRate * rangePulseDuration)
   
    number_range_bins = self.insar.numberRangeBins
   
    referenceCentroid = self.insar.referenceDoppler.fractionalCentroid
    secondaryCentroid = self.insar.secondaryDoppler.fractionalCentroid
    logger.info("Correcting Doppler centroid for motion compensation")


    result = []
    for centroid, frame, orbit, H in zip((referenceCentroid, secondaryCentroid),
                                      (self.insar.referenceFrame,
                                       self.insar.secondaryFrame),
                                         (referenceOrbit, secondaryOrbit),
                                         (H1, H2)
                                      ):
        fdmocomp = stdproc.createFdMocomp()
        fdmocomp.wireInputPort(name='frame', object=frame)
        fdmocomp.wireInputPort(name='peg', object=peg)
        fdmocomp.wireInputPort(name='orbit', object=orbit)
        fdmocomp.setWidth(number_range_bins)
        fdmocomp.setSatelliteHeight(H)
        fdmocomp.setDopplerCoefficients([centroid, 0.0, 0.0, 0.0])
        fdmocomp.setLookSide(lookSide)
        fdmocomp.fdmocomp()
        result.append( fdmocomp.dopplerCentroid )
        pass

    referenceDopplerCorrection, secondaryDopplerCorrection = result

#    print referenceDopplerCorrection, secondaryDopplerCorrection
#    use_dop = "F"
    try:
        fd = USE_DOP[use_dop.upper()](referenceDopplerCorrection,
                                      secondaryDopplerCorrection)
    except KeyError:
        print("Unrecognized use_dop option.  use_dop = ",use_dop)
        print("Not found in dictionary:",USE_DOP.keys())
        sys.exit(1)
        pass
    
    logger.info("Updated Doppler Centroid: %s" % (fd))
    return fd



