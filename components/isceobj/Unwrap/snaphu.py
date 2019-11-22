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




import sys
import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from iscesys.Component.Component import Component
from isceobj.Constants import SPEED_OF_LIGHT


class snaphu(Component):
    '''Specific connector from an insarApp object to a Snaphu object.'''
    def __init__(self, obj):

        basename = obj.insar.topophaseFlatFilename
        self.wrapName = basename
        self.unwrapName = basename.replace('.flat', '.unw')

        self.wavelength = obj.insar.masterFrame.getInstrument().getRadarWavelength()
        self.width      = obj.insar.resampIntImage.width 
        self.costMode   = 'DEFO'
        self.initMethod = 'MST'
        self.earthRadius = obj.insar.peg.radiusOfCurvature 
        self.altitude   = obj.insar.averageHeight
        self.corrfile  = obj.insar.getCoherenceFilename()
        self.rangeLooks = obj.insar.topo.numberRangeLooks
        self.azimuthLooks = obj.insar.topo.numberAzimuthLooks

        azres = obj.insar.masterFrame.platform.antennaLength/2.0
        azfact = azres / obj.insar.topo.azimuthSpacing

        rBW = obj.insar.masterFrame.instrument.pulseLength * obj.insar.masterFrame.instrument.chirpSlope
        rgres = abs(SPEED_OF_LIGHT / (2.0 * rBW))
        rngfact = rgres/obj.insar.topo.slantRangePixelSpacing

        self.corrLooks = obj.insar.topo.numberRangeLooks * obj.insar.topo.numberAzimuthLooks/(azfact*rngfact) 
        self.maxComponents = 20
        self.defomax = 4.0

    def unwrap(self):
        snp = Snaphu()
        snp.setInput(self.wrapName)
        snp.setOutput(self.unwrapName)
        snp.setWidth(self.width)
        snp.setCostMode(self.costMode)
        snp.setEarthRadius(self.earthRadius)
        snp.setWavelength(self.wavelength)
        snp.setAltitude(self.altitude)
        snp.setCorrfile(self.corrfile)
        snp.setInitMethod(self.initMethod)
        snp.setCorrLooks(self.corrLooks)
        snp.setMaxComponents(self.maxComponents)
        snp.setDefoMaxCycles(self.defomax)
        snp.setRangeLooks(self.rangeLooks)
        snp.setAzimuthLooks(self.azimuthLooks)
        snp.prepare()
        snp.unwrap()

        ######Render XML
        outImage = isceobj.Image.createImage()
        outImage.setFilename(self.unwrapName)
        outImage.setWidth(self.width)
        outImage.bands = 2
        outImage.scheme = 'BIL'
        outImage.imageType='unw'
        outImage.dataType='FLOAT'
        outImage.setAccessMode('read')
        outImage.createImage()
        outImage.finalizeImage()
        outImage.renderHdr()
