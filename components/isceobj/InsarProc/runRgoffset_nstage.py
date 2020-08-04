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



import logging
import isceobj
import mroipac
import numpy
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj import Constants as CN
from mroipac.ampcor.NStage import NStage
logger = logging.getLogger('isce.insar.runRgoffset')

def runRgoffset(self):
    from isceobj.Catalog import recordInputs,recordOutputs

    coarseAcross = 0
    coarseDown = 0
    numLocationAcross = self._insar.getNumberLocationAcross()
    numLocationDown = self._insar.getNumberLocationDown()
    firstAc = self._insar.getFirstSampleAcross()
    firstDn = self._insar.getFirstSampleDown()

    ampImage = self._insar.getResampAmpImage()
    secondaryWidth = ampImage.getWidth()
    secondaryLength = ampImage.getLength()
    objAmp = isceobj.createSlcImage()
    objAmp.dataType = 'CFLOAT'
    objAmp.bands = 1
    objAmp.setFilename(ampImage.getFilename())
    objAmp.setAccessMode('read')
    objAmp.setWidth(secondaryWidth)
    objAmp.createImage()

    simImage = self._insar.getSimAmpImage()
    referenceWidth = simImage.getWidth()
    objSim = isceobj.createImage()
    objSim.setFilename(simImage.getFilename())
    objSim.dataType = 'FLOAT'
    objSim.setWidth(referenceWidth)
    objSim.setAccessMode('read')
    objSim.createImage()
    referenceLength = simImage.getLength()


    nStageObj = NStage(name='insarapp_intsim_nstage')
    nStageObj.configure()
    nStageObj.setImageDataType1('real')
    nStageObj.setImageDataType2('complex')

    if nStageObj.acrossGrossOffset is None:
        nStageObj.setAcrossGrossOffset(0)

    if nStageObj.downGrossOffset is None:
        nStageObj.setDownGrossOffset(0)


    # Record the inputs
    recordInputs(self._insar.procDoc,
                nStageObj,
                "runRgoffset",
                logger,
                "runRgoffset")

    nStageObj.nstage(slcImage1=objSim,slcImage2=objAmp)

    recordOutputs(self._insar.procDoc,
                    nStageObj,
                    "runRgoffset",
                    logger,
                    "runRgoffset")

    offField = nStageObj.getOffsetField()

    # save the input offset field for the record
    self._insar.setOffsetField(offField)
    self._insar.setRefinedOffsetField(offField)
