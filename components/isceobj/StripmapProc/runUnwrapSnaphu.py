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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






# giangi: taken Piyush code for snaphu and adapted

import sys
import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from isceobj.Constants import SPEED_OF_LIGHT
import copy
import os

def runSnaphu(self, igramSpectrum = "full", costMode = None,initMethod = None, defomax = None, initOnly = None):

    if costMode is None:
        costMode   = 'DEFO'
    
    if initMethod is None:
        initMethod = 'MST'
    
    if  defomax is None:
        defomax = 4.0
    
    if initOnly is None:
        initOnly = False
   
    print("igramSpectrum: ", igramSpectrum)

    if igramSpectrum == "full":
        ifgDirname = self.insar.ifgDirname

    elif igramSpectrum == "low":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferogram unwrapping')
            return
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)

    elif igramSpectrum == "high":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferogram unwrapping')
            return
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)


    wrapName = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename)

    if '.flat' in wrapName:
        unwrapName = wrapName.replace('.flat', '.unw')
    elif '.int' in wrapName:
        unwrapName = wrapName.replace('.int', '.unw')
    else:
        unwrapName = wrapName + '.unw'

    corName = os.path.join(ifgDirname , self.insar.coherenceFilename)

    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)
    wavelength = referenceFrame.getInstrument().getRadarWavelength()
    img1 = isceobj.createImage()
    img1.load(wrapName + '.xml')
    width = img1.getWidth()
    #width      = self.insar.resampIntImage.width

    orbit = referenceFrame.orbit
    prf = referenceFrame.PRF
    elp = copy.copy(referenceFrame.instrument.platform.planet.ellipsoid)
    sv = orbit.interpolate(referenceFrame.sensingMid, method='hermite')
    hdg = orbit.getHeading()
    llh = elp.xyz_to_llh(sv.getPosition())
    elp.setSCH(llh[0], llh[1], hdg)

    earthRadius = elp.pegRadCur
    sch, vsch = elp.xyzdot_to_schdot(sv.getPosition(), sv.getVelocity())
    azimuthSpacing = vsch[0] * earthRadius / ((earthRadius + sch[2]) *prf)


    earthRadius = elp.pegRadCur
    altitude   = sch[2]
    rangeLooks = 1  # self.numberRangeLooks #insar.topo.numberRangeLooks
    azimuthLooks = 1 # self.numberAzimuthLooks #insar.topo.numberAzimuthLooks

    if not self.numberAzimuthLooks:
        self.numberAzimuthLooks = 1

    if not self.numberRangeLooks:
        self.numberRangeLooks = 1

    azres = referenceFrame.platform.antennaLength/2.0
    azfact = self.numberAzimuthLooks * azres / azimuthSpacing

    rBW = referenceFrame.instrument.pulseLength * referenceFrame.instrument.chirpSlope
    rgres = abs(SPEED_OF_LIGHT / (2.0 * rBW))
    rngfact = rgres/referenceFrame.getInstrument().getRangePixelSize()

    corrLooks = self.numberRangeLooks * self.numberAzimuthLooks/(azfact*rngfact) 
    maxComponents = 20

    snp = Snaphu()
    snp.setInitOnly(initOnly)
    snp.setInput(wrapName)
    snp.setOutput(unwrapName)
    snp.setWidth(width)
    snp.setCostMode(costMode)
    snp.setEarthRadius(earthRadius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(corName)
    snp.setInitMethod(initMethod)
    #snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(maxComponents)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(rangeLooks)
    snp.setAzimuthLooks(azimuthLooks)
    snp.setCorFileFormat('FLOAT_DATA')
    snp.prepare()
    snp.unwrap()
    ######Render XML
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setAccessMode('read')
    outImage.renderHdr()
    outImage.renderVRT()
    #####Check if connected components was created
    if snp.dumpConnectedComponents:
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName+'.conncomp')
        #At least one can query for the name used
        self.insar.connectedComponentsFilename = unwrapName+'.conncomp'
        connImage.setWidth(width)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        connImage.renderHdr()
        connImage.renderVRT()

    return

'''
def runUnwrapMcf(self):
    runSnaphu(self, igramSpectrum = "full", costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    runSnaphu(self, igramSpectrum = "low", costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    runSnaphu(self, igramSpectrum = "high", costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    return
'''

def runUnwrap(self, igramSpectrum = "full"):

    runSnaphu(self, igramSpectrum = igramSpectrum, costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)

    return


