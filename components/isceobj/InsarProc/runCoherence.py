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
import operator
import isceobj


from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from mroipac.correlation.correlation import Correlation
from isceobj.Util.decorators import use_api

logger = logging.getLogger('isce.insar.runCoherence')

## mapping from algorithm method to Correlation instance method name
CORRELATION_METHOD = {
    'phase_gradient' : operator.methodcaller('calculateEffectiveCorrelation'),
    'cchz_wave' : operator.methodcaller('calculateCorrelation')
    }

@use_api
def runCoherence(self, method="phase_gradient"):
                          
    logger.info("Calculating Coherence")

    # Initialize the amplitude
#    resampAmpImage =  self.insar.resampAmpImage
#    ampImage = isceobj.createAmpImage()
#    IU.copyAttributes(resampAmpImage, ampImage)
#    ampImage.setAccessMode('read')
#    ampImage.createImage()
#    ampImage = self.insar.getResampOnlyAmp().copy(access_mode='read')
    ampImage = isceobj.createImage()
    ampImage.load( self.insar.getResampOnlyAmp().filename + '.xml')
    ampImage.setAccessMode('READ')
    ampImage.createImage()
    
    # Initialize the flattened inteferogram
    topoflatIntFilename = self.insar.topophaseFlatFilename
    intImage = isceobj.createImage()
    intImage.load ( self.insar.topophaseFlatFilename + '.xml')
    intImage.setAccessMode('READ')
    intImage.createImage()

#    widthInt = self.insar.resampIntImage.getWidth()
#    intImage.setFilename(topoflatIntFilename)
#    intImage.setWidth(widthInt)
#    intImage.setAccessMode('read')
#    intImage.createImage()

    # Create the coherence image
    cohFilename = topoflatIntFilename.replace('.flat', '.cor')
    cohImage = isceobj.createOffsetImage()
    cohImage.setFilename(cohFilename)
    cohImage.setWidth(intImage.width)
    cohImage.setAccessMode('write')
    cohImage.createImage()

    cor = Correlation()
    cor.configure()
    cor.wireInputPort(name='interferogram', object=intImage)
    cor.wireInputPort(name='amplitude', object=ampImage)
    cor.wireOutputPort(name='correlation', object=cohImage)
   
    cohImage.finalizeImage()
    intImage.finalizeImage()
    ampImage.finalizeImage()

    cor.calculateCorrelation()
#    try:
#        CORRELATION_METHOD[method](cor)
#    except KeyError:
#        print("Unrecognized correlation method")
#        sys.exit(1)
#        pass
    return None
