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
import isceobj

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU

logger = logging.getLogger('isce.insar.runShadecpx2rg')

def runShadecpx2rg(self):

    imageAmp = self._insar.getResampAmpImage()
    widthAmp = imageAmp.getWidth()
    endian = self._insar.getMachineEndianness()

    filenameSimAmp = self._insar.getSimAmpImageName()
    objSimAmp = isceobj.createImage()
    widthSimAmp = widthAmp
    objSimAmp.initImage(filenameSimAmp,'read',widthSimAmp,'FLOAT')
    
    imageSimAmp = isceobj.createImage()
    IU.copyAttributes(objSimAmp, imageSimAmp)
    self._insar.setSimAmpImage(imageSimAmp) 
    objSimAmp.setAccessMode('write')
    objSimAmp.createImage()
    filenameHt = self._insar.getHeightFilename()
    widthHgtImage = widthAmp # they have same width by construction
    objHgtImage = isceobj.createImage()
    objHgtImage.initImage(filenameHt,'read',widthHgtImage,'FLOAT')
    imageHgt = isceobj.createImage()
    IU.copyAttributes(objHgtImage, imageHgt)
    self._insar.setHeightTopoImage(imageHgt) 
    
    objHgtImage.createImage()
   
    logger.info("Running Shadecpx2rg")
    objShade = isceobj.createSimamplitude()
    #set the tag used in the outfile. each message is precided by this tag
    #is the writer is not of "file" type the call has no effect
    self._stdWriter.setFileTag("simamplitude", "log")
    self._stdWriter.setFileTag("simamplitude", "err")
    self._stdWriter.setFileTag("simamplitude", "out")
    objShade.setStdWriter(self._stdWriter)

    shade = self._insar.getShadeFactor()

    objShade.simamplitude(objHgtImage, objSimAmp, shade=shade)

    # Record the inputs and outputs
    from isceobj.Catalog import recordInputsAndOutputs
    recordInputsAndOutputs(self._insar.procDoc, objShade, "runSimamplitude", \
                  logger, "runSimamplitude")
   
    objHgtImage.finalizeImage()
    objSimAmp.finalizeImage()
    objSimAmp.renderHdr()
