#!/usr/bin/env python3 

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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import numpy as np
import isce
from isceobj import createImage
import os
def runMaskImages(self):
    if self.insar.applyWaterMask:
        corrName  = self.insar.coherenceFilename
        wrapName = self.insar.topophaseFlatFilename
        maskName = self.insar.waterMaskImageName
        ampName = self.insar.resampOnlyAmpName
        prefix = self.insar.unmaskedPrefix
        newCorrName = prefix + '_' + corrName
        newWrapName =  prefix + '_' + wrapName
        newAmpName =  prefix + '_' + ampName
    
        os.system('cp -r ' + corrName + ' ' + newCorrName)
        os.system('cp -r ' + wrapName + ' ' + newWrapName)
        os.system('cp -r ' + ampName + ' ' + newAmpName)
    
        corrImage = createImage()
        corrImage.load(corrName+'.xml')
        corrmap = np.memmap(corrName,corrImage.toNumpyDataType(),'r+',
                            shape=(corrImage.bands*corrImage.coord2.coordSize,corrImage.coord1.coordSize))
        wrapImage = createImage()
        wrapImage.load(wrapName+'.xml')
        wrapmap = np.memmap(wrapName,wrapImage.toNumpyDataType(),'r+',
                            shape=(wrapImage.coord2.coordSize,wrapImage.coord1.coordSize))
        maskImage = createImage()
        maskImage.load(maskName+'.xml')
        maskmap = np.memmap(maskName,maskImage.toNumpyDataType(),'r',
                            shape=(maskImage.coord2.coordSize,maskImage.coord1.coordSize))
        ampImage = createImage()
        ampImage.load(ampName+'.xml')
        ampmap = np.memmap(ampName,ampImage.toNumpyDataType(),'r+',
                            shape=(ampImage.coord2.coordSize,ampImage.bands*ampImage.coord1.coordSize))
        #NOTE:thre is a bug in the calculation of lat.rd and lon.rdr so the two have one more line 
        #then the corr and wrap images. Add some logic to remove potential extra line
        lastLine = min(wrapmap.shape[0],maskmap.shape[0])
        #corr file is a 2 bands BIL scheme so multiply each band
        corrmap[:corrImage.bands*lastLine:2,:] = corrmap[:corrImage.bands*lastLine:2,:]*maskmap[:lastLine,:]
        corrmap[1:corrImage.bands*lastLine:2,:] = corrmap[1:corrImage.bands*lastLine:2,:]*maskmap[:lastLine,:]
        wrapmap[:lastLine,:] = wrapmap[:lastLine,:]*maskmap[:lastLine,:]
        ampmap[0:lastLine,::2] = ampmap[0:lastLine,::2]*maskmap[:lastLine,:]
        ampmap[0:lastLine,1::2] = ampmap[0:lastLine,1::2]*maskmap[:lastLine,:]
    
        #change the filename in the metadata and then save the xml file for the unmasked images
        corrImage.filename = newCorrName
        corrImage.dump(newCorrName+'.xml')
        wrapImage.filename = newWrapName
        wrapImage.dump(newWrapName+'.xml')
        ampImage.filename = newAmpName
        ampImage.dump(newAmpName+'.xml')
