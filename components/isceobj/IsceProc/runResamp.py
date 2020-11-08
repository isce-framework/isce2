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
# Authors: Kosal Khun, Marco Lavalle
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Comment: Adapted from InsarProc/runResamp.py
import os
import logging
import stdproc
import isceobj

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU

logger = logging.getLogger('isce.isceProc.runResamp')

def runResamp(self):
    stdWriter = self._stdWriter
    resampName = self._isce.resampImageName
    dopplerCentroid = self._isce.dopplerCentroid
    numFitCoeff = self._isce.numberFitCoefficients
    azLooks = self._isce.numberAzimuthLooks
    rgLooks = self._isce.numberRangeLooks
    lines = self._isce.numberResampLines
    pixelSpacing = self._isce.slantRangePixelSpacing

    outresamp = "resamp" #only resamp

    for sceneid1, sceneid2 in self._isce.pairsToCoreg:
        pair = (sceneid1, sceneid2)
        self._isce.resampIntImages[pair] = {}
        self._isce.resampAmpImages[pair] = {}
        offsetField = self._isce.refinedOffsetFields[pair]
        for pol in self._isce.selectedPols:
            imageSlc1 = self._isce.slcImages[sceneid1][pol]
            imageSlc2 = self._isce.slcImages[sceneid2][pol]
            frame1 = self._isce.frames[sceneid1][pol]
            instrument = frame1.getInstrument()
            catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
            sid = self._isce.formatname(pair, pol)
            resampFilename = os.path.join(self.getoutputdir(sceneid1, sceneid2), self._isce.formatname(pair, pol, resampName))
            imageInt, imageAmp, imageResamp2 = run(imageSlc1, imageSlc2, instrument, offsetField, resampFilename, azLooks, rgLooks, lines, dopplerCentroid, numFitCoeff, pixelSpacing, stdWriter, catalog=catalog, sceneid=sid, output=outresamp)
            self._isce.resampIntImages[pair][pol] = imageInt
            self._isce.resampAmpImages[pair][pol] = imageAmp
            if imageResamp2 is not None: #update resampled slc
                self._isce.slcImages[sceneid2][pol] = imageResamp2


def run(imageSlc1, imageSlc2, instrument, offsetField, resampName, azLooks, rgLooks, lines, dopplerCentroid, numFitCoeff, pixelSpacing, stdWriter, catalog=None, sceneid='NO_ID', output="all"):
    logger.info("Resampling interferogram: %s" % sceneid)

    output = output.replace(" ", "") #remove all spaces in output
    if output == "all":
        output = ["intamp", "resamp"]
    else:
        output = output.split(",") #get a list from comma-separated text


    objSlc1 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc1, objSlc1)
    objSlc1.setAccessMode('read')
    objSlc1.createImage()

    objSlc2 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc2, objSlc2)
    objSlc2.setAccessMode('read')
    objSlc2.createImage()

    slcWidth = imageSlc1.getWidth()
    intWidth = int(slcWidth / rgLooks)

    if "resamp" in output:
        logger.info("Will output resampled slc")
        objResampSlc2 = isceobj.createSlcImage()
        objResampSlc2.setFilename(objSlc2.getFilename().replace('.slc', '.resamp.slc')) #replace .slc by .resamp.slc
        objResampSlc2.setWidth(slcWidth)
        imageResamp2 = isceobj.createSlcImage()
        IU.copyAttributes(objResampSlc2, imageResamp2)
        objResampSlc2.setAccessMode('write')
        objResampSlc2.createImage()
    else:
        objResampSlc2 = None
        imageResamp2 = None

    if "intamp" in output:
        logger.info("Will output resampled interferogram and amplitude: %s" % sceneid)
        resampAmp = resampName + '.amp'
        resampInt = resampName + '.int'

        objInt = isceobj.createIntImage()
        objInt.setFilename(resampInt)
        objInt.setWidth(intWidth)
        imageInt = isceobj.createIntImage()
        IU.copyAttributes(objInt, imageInt)
        objInt.setAccessMode('write')
        objInt.createImage()

        objAmp = isceobj.createAmpImage()
        objAmp.setFilename(resampAmp)
        objAmp.setWidth(intWidth)
        imageAmp = isceobj.createAmpImage()
        IU.copyAttributes(objAmp, imageAmp)
        objAmp.setAccessMode('write')
        objAmp.createImage()
    else:
        objInt = None
        imageInt = None
        objAmp = None
        imageAmp = None

    dopplerCoeff = dopplerCentroid.getDopplerCoefficients(inHz=False)

    objResamp = stdproc.createResamp()
    objResamp.setNumberLines(lines)
    objResamp.setNumberFitCoefficients(numFitCoeff)
    objResamp.setNumberAzimuthLooks(azLooks)
    objResamp.setNumberRangeLooks(rgLooks)
    objResamp.setSlantRangePixelSpacing(pixelSpacing)
    objResamp.setDopplerCentroidCoefficients(dopplerCoeff)

    objResamp.wireInputPort(name='offsets', object=offsetField)
    objResamp.wireInputPort(name='instrument', object=instrument)
    #set the tag used in the outfile. each message is precided by this tag
    #is the writer is not of "file" type the call has no effect
    objResamp.stdWriter = stdWriter.set_file_tags("resamp",
                                                  "log",
                                                  "err",
                                                  "out")
    objResamp.resamp(objSlc1, objSlc2, objInt, objAmp, objResampSlc2)

    if catalog is not None:
        # Record the inputs and outputs
        isceobj.Catalog.recordInputsAndOutputs(catalog, objResamp,
                                               "runResamp.%s" % sceneid,
                                               logger,
                                               "runResamp.%s" % sceneid)

    for obj in [objInt, objAmp, objSlc1, objSlc2, objResampSlc2]:
        if obj is not None:
            obj.finalizeImage()

    return imageInt, imageAmp, imageResamp2
