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

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU

logger = logging.getLogger('isce.insar.runFormSLCTSX')

def runFormSLC(self, patchSize=None, goodLines=None, numPatches=None):
    #NOTE tested the formslc() as a stand alone by passing the same inputs
    #computed in Howard terraSAR.py. The differences here arises from the
    #differences in the orbits when using the same orbits the results are very
    #close jng this will make the second term in coarseAz in offsetprf equal
    #zero. we do so since for tsx there is no such a term. Need to ask
    #confirmation
    self.insar.setPatchSize(self.insar.numberValidPulses)
    # the below value is zero because of we just did above, but just want to be
    #  explicit in the definition of is_mocomp
    self.is_mocomp = self.insar.get_is_mocomp

    v = self.insar.procVelocity
    h = self.insar.averageHeight
    imageSlc1 =  self.insar.referenceRawImage
    imSlc1 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc1, imSlc1)
    imSlc1.setAccessMode('read')
    imSlc1.createImage()
    formSlc1 = stdproc.createFormSLC(self.sensorName)

    formSlc1.setBodyFixedVelocity(v)
    formSlc1.setSpacecraftHeight(h)
    formSlc1.wireInputPort(name='doppler',
                           object = self.insar.dopplerCentroid)
    formSlc1.wireInputPort(name='peg', object=self.insar.peg)
    formSlc1.wireInputPort(name='frame', object=self.insar.referenceFrame)
    formSlc1.wireInputPort(name='orbit', object=self.insar.referenceOrbit)
    formSlc1.wireInputPort(name='slcInImage', object=imSlc1)
    formSlc1.wireInputPort(name='planet',
        object=self.insar.referenceFrame.instrument.platform.planet)
    self._stdWriter.setFileTag("formslcTSX", "log")
    self._stdWriter.setFileTag("formslcTSX", "err")
    self._stdWriter.setFileTag("formslcTSX", "out")
    formSlc1.setStdWriter(self._stdWriter)
    formSlc1.setLookSide(self.insar._lookSide)


#    self.insar.setReferenceSlcImage(formSlc1.formslc())
    self.insar.referenceSlcImage = formSlc1()

    imageSlc2 =  self.insar.secondaryRawImage
    imSlc2 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc2, imSlc2)
    imSlc2.setAccessMode('read')
    imSlc2.createImage()
    formSlc2 = stdproc.createFormSLC(self.sensorName)

    formSlc2.setBodyFixedVelocity(v)
    formSlc2.setSpacecraftHeight(h)
    formSlc2.wireInputPort(name='doppler',
                           object=self.insar.dopplerCentroid)
    formSlc2.wireInputPort(name='peg', object=self.insar.peg)
    formSlc2.wireInputPort(name='frame', object=self.insar.secondaryFrame)
    formSlc2.wireInputPort(name='orbit', object=self.insar.secondaryOrbit)
    formSlc2.wireInputPort(name='slcInImage', object=imSlc2)
    formSlc2.wireInputPort(name='planet',
        object=self.insar.secondaryFrame.instrument.platform.planet)

    self._stdWriter.setFileTag("formslcTSX", "log")
    self._stdWriter.setFileTag("formslcTSX", "err")
    self._stdWriter.setFileTag("formslcTSX", "out")
    formSlc2.setStdWriter(self._stdWriter)
    formSlc2.setLookSide(self.insar._lookSide)
#    self.insar.setSecondarySlcImage(formSlc2.formslc())
    self.insar.secondarySlcImage = formSlc2()
    self.insar.setNumberPatches(
        imSlc1.getLength()/float(self.insar.numberValidPulses)
        )
    imSlc1.finalizeImage()
    imSlc2.finalizeImage()
    self.insar.setFormSLC1(formSlc1)
    self.insar.setFormSLC2(formSlc2)
