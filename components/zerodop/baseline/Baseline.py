#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import math
import datetime
import logging
from iscesys.Component.Component import Component, Port
from isceobj.Orbit.Orbit import StateVector
import numpy as np 

BASELINE_LOCATION = Component.Parameter('baselineLocation',
        public_name = 'BASELINE_LOCATION',
        default = 'all',
        type=str,
        mandatory=False,
        doc = 'Location at which to compute baselines - "all" implies top, middle, bottom of reference image, "top" implies near start of reference image, "bottom" implies at bottom of reference image, "middle" implies near middle of reference image. To be used in case there is a large shift between images.')



class Baseline(Component):

    family = 'baseline'
    logging_name = 'isce.zerodop.baseline'

    parameter_list = (BASELINE_LOCATION,)

    # Calculate the baseline components between two frames
    def baseline(self):

        from isceobj.Util.geo.ellipsoid import Ellipsoid
        from isceobj.Planet.Planet import Planet
        for port in self.inputPorts:
            port()
            
        planet = Planet(pname='Earth')
        refElp = Ellipsoid(a=planet.ellipsoid.a, e2=planet.ellipsoid.e2, model='WGS84')


        if self.baselineLocation.lower() == 'all':
            print('Using entire span of image for estimating baselines')
            referenceTime = [self.referenceFrame.getSensingStart(),self.referenceFrame.getSensingMid(),self.referenceFrame.getSensingStop()]
        elif self.baselineLocation.lower() == 'middle':
            print('Estimating baselines around center of reference image')
            referenceTime = [self.referenceFrame.getSensingMid() - datetime.timedelta(seconds=1.0), self.referenceFrame.getSensingMid(), self.referenceFrame.getSensingMid() + datetime.timedelta(seconds=1.0)]

        elif self.baselineLocation.lower() == 'top':
            print('Estimating baselines at top of reference image')
            referenceTime =  [self.referenceFrame.getSensingStart(), self.referenceFrame.getSensingStart() + datetime.timedelta(seconds=1.0), self.referenceFrame.getSensingStart() + datetime.timedelta(seconds=2.0)]
        elif self.baselineLocation.lower() == 'bottom':
            print('Estimating baselines at bottom of reference image')
            referenceTime =  [self.referenceFrame.getSensingStop() - datetime.timedelta(seconds=2.0), self.referenceFrame.getSensingStop() - datetime.timedelta(seconds=1.0), self.referenceFrame.getSensingStop()]
        else:
            raise Exception('Unknown baseline location: {0}'.format(self.baselineLocation))


        s = [0., 0., 0.]
        bpar = []
        bperp = []
        azoff = []
        rgoff = []

        for i in range(3):
            # Calculate the Baseline at the start of the scene, mid-scene, and the end of the scene
            # First, get the position and velocity at the start of the scene
            # Calculate the distance moved since the last baseline point
            s[i] = (referenceTime[i] - referenceTime[0]).total_seconds()
            

            referenceSV = self.referenceOrbit.interpolateOrbit(referenceTime[i], method='hermite')
            rng = self.startingRange1
            target = self.referenceOrbit.pointOnGround(referenceTime[i], rng, side=self.referenceFrame.getInstrument().getPlatform().pointingDirection)

            secondaryTime, slvrng = self.secondaryOrbit.geo2rdr(target) 
            secondarySV = self.secondaryOrbit.interpolateOrbit(secondaryTime, method='hermite')

            targxyz = np.array(refElp.LLH(target[0], target[1], target[2]).ecef().tolist())
            mxyz = np.array(referenceSV.getPosition())
            mvel = np.array(referenceSV.getVelocity())
            sxyz = np.array(secondarySV.getPosition())
            mvelunit = mvel / np.linalg.norm(mvel)
            sxyz = sxyz - np.dot ( sxyz-mxyz, mvelunit) * mvelunit

            aa = np.linalg.norm(sxyz-mxyz)

            costheta = (rng*rng + aa*aa - slvrng*slvrng)/(2.*rng*aa)

#            print(aa, costheta)
            bpar.append(aa*costheta)

            perp = aa * np.sqrt(1 - costheta*costheta)
            direction = np.sign(np.dot( np.cross(targxyz-mxyz, sxyz-mxyz), mvel))
            bperp.append(direction*perp)

            ####Azimuth offset
            slvaz = (secondaryTime - self.secondaryFrame.sensingStart).total_seconds() * self.prf2
            masaz = s[i] * self.prf1
            azoff.append(slvaz - masaz)

            ####Range offset
            slvrg =  (slvrng - self.startingRange2)/self.rangePixelSize2
            masrg = (rng - self.startingRange1) / self.rangePixelSize1
            rgoff.append(slvrg - masrg)

       
#        print(bpar)
#        print(bperp)

        #Calculating baseline
        parBaselinePolynomialCoefficients = np.polyfit(s,bpar,2)
        perpBaselinePolynomialCoefficients = np.polyfit(s,bperp,2)
        
        # Populate class attributes 
        self.BparMean = parBaselinePolynomialCoefficients[-1]
        self.BparRate = parBaselinePolynomialCoefficients[1]
        self.BparAcc = parBaselinePolynomialCoefficients[0]
        self.BperpMean = perpBaselinePolynomialCoefficients[-1]
        self.BperpRate = perpBaselinePolynomialCoefficients[1]
        self.BperpAcc = perpBaselinePolynomialCoefficients[0]

        delta = (self.referenceFrame.getSensingStart() - referenceTime[0]).total_seconds()
        self.BparTop = np.polyval(parBaselinePolynomialCoefficients, delta)
        self.BperpTop = np.polyval(perpBaselinePolynomialCoefficients, delta)

        delta = (self.referenceFrame.getSensingStop() - referenceTime[0]).total_seconds()
        self.BparBottom = np.polyval(parBaselinePolynomialCoefficients, delta)
        self.BperpBottom = np.polyval(perpBaselinePolynomialCoefficients, delta)
       
        return azoff, rgoff
            
    def setReferenceRangePixelSize(self,pixelSize):
        self.rangePixelSize1 = pixelSize
        return

    def setSecondaryRangePixelSize(self,pixelSize):
        self.rangePixelSize2 = pixelSize
        return

    def setReferenceStartingRange(self,range):
        self.startingRange1 = range
        return

    def setSecondaryStartingRange(self,range):
        self.startingRange2 = range
        return

    def setReferencePRF(self,prf):
        self.prf1 = prf
        return

    def setSecondaryPRF(self,prf):
        self.prf2 = prf
        return
    
    def getHBaselineTop(self):
        return self.hBaselineTop

    def getHBaselineRate(self):
        return self.hBaselineRate

    def getHBaselineAcc(self):
        return self.hBaselineAcc

    def getVBaselineTop(self):
        return self.vBaselineTop

    def getVBaselineRate(self):
        return self.vBaselineRate

    def getVBaselineAcc(self):
        return self.vBaselineAcc

    def getPBaselineTop(self):
        return self.pBaselineTop

    def getPBaselineBottom(self):
        return self.pBaselineBottom



        
    def addReferenceFrame(self):
        frame = self._inputPorts.getPort(name='referenceFrame').getObject()
        self.startingRange1 = frame.getStartingRange()
        self.prf1 = frame.getInstrument().getPulseRepetitionFrequency()
        self.rangePixelSize1 = frame.getInstrument().getRangePixelSize()
        self.referenceOrbit = frame.getOrbit()
        self.referenceFrame = frame

    def addSecondaryFrame(self):
        frame = self._inputPorts.getPort(name='secondaryFrame').getObject()
        self.startingRange2 = frame.getStartingRange()
        self.secondaryOrbit = frame.getOrbit()
        self.prf2 = frame.getInstrument().getPulseRepetitionFrequency()
        self.rangePixelSize2 = frame.getInstrument().getRangePixelSize()
        self.secondaryFrame = frame
        
    def __init__(self, name=''):
        super(Baseline, self).__init__(family=self.__class__.family, name=name)
        self.referenceOrbit = None
        self.secondaryOrbit = None
        self.referenceFrame = None
        self.secondaryFrame = None
        self.rangePixelSize1 = None
        self.rangePixelSize2 = None
        self.startingRange1 = None
        self.startingRange2 = None
        self.prf1 = None
        self.prf2 = None
        self.lookSide = None
        self.BparMean = None
        self.BparRate = None
        self.BparAcc = None
        self.BperpMean = None
        self.BperpRate = None
        self.BperpAcc = None
        self.BperpTop = None
        self.BperpBottom = None
        self.BparTop = None
        self.BperpBottom = None
        self.logger = logging.getLogger('isce.zerodop.baseline')
        self.createPorts()
        
        # Satisfy the old Component
        self.dictionaryOfOutputVariables = {}        
        self.dictionaryOfVariables = {}        
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        return None

    def createPorts(self):
        
        # Set input ports
        # It looks like we really need two orbits, a time, range and azimuth pixel sizes
        # the two starting ranges, a planet, and the two prfs
        # These provide the orbits
        # These provide the range and azimuth pixel sizes, starting ranges, 
        # satellite heights and times for the first lines
        referenceFramePort = Port(name='referenceFrame',method=self.addReferenceFrame)  
        secondaryFramePort = Port(name='secondaryFrame',method=self.addSecondaryFrame)       
        self._inputPorts.add(referenceFramePort)
        self._inputPorts.add(secondaryFramePort)
        return None

        
    def __str__(self):
        retstr = "Initial Baseline estimates \n"
        retlst = ()
        retstr += "Parallel Baseline Top: %s\n"
        retlst += (self.BparTop,)
        retstr += "Perpendicular Baseline Top: %s\n"
        retlst += (self.BperpTop,)
        retstr += "Parallel Baseline Bottom: %s\n"
        retlst += (self.BparBottom,)
        retstr += "Perpendicular Baseline Bottom: %s \n"
        retlst += (self.BperpBottom,)
        return retstr % retlst      
