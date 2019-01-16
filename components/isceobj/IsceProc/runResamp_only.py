#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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



# Comment: Adapted from InsarProc/runResamp_only.py
import logging
import stdproc
import isceobj
import os

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU

logger = logging.getLogger('isce.isceProc.runResamp_only')

def runResamp_only(self):
    infos = {}
    for attribute in ['dopplerCentroid', 'resampOnlyImageName', 'numberFitCoefficients', 'slantRangePixelSpacing']:
        infos[attribute] = getattr(self._isce, attribute)

    stdWriter = self._stdWriter

    pair = self._isce.pairsToCoreg[0]
    offsetField = self._isce.refinedOffsetFields[pair]

    for sceneid1, sceneid2 in self._isce.selectedPairs:
        pair = (sceneid1, sceneid2)
        self._isce.resampOnlyImages[pair] = {}
        self._isce.resampOnlyAmps[pair] = {}
        for pol in self._isce.selectedPols:
            imageInt = self._isce.resampIntImages[pair][pol]
            imageAmp = self._isce.resampAmpImages[pair][pol]
            frame1 = self._isce.frames[sceneid1][pol]
            instrument = frame1.getInstrument()
            sid = self._isce.formatname(pair, pol)
            infos['outputPath'] = os.path.join(self.getoutputdir(sceneid1, sceneid2), sid)
            catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
            objIntOut, objAmpOut = run(imageInt, imageAmp, instrument, offsetField, infos, stdWriter, catalog=catalog, sceneid=sid)
            self._isce.resampOnlyImages[pair][pol] = objIntOut
            self._isce.resampOnlyAmps[pair][pol] = objAmpOut


def run(imageInt, imageAmp, instrument, offsetField, infos, stdWriter, catalog=None, sceneid='NO_ID'):
    logger.info("Running Resamp_only: %s" % sceneid)

    objInt = isceobj.createIntImage()
    objIntOut = isceobj.createIntImage()
    IU.copyAttributes(imageInt, objInt)
    IU.copyAttributes(imageInt, objIntOut)
    outIntFilename = infos['outputPath'] + '.' + infos['resampOnlyImageName']
    objInt.setAccessMode('read')
    objIntOut.setFilename(outIntFilename)

    objIntOut.setAccessMode('write')
    objInt.createImage()
    objIntOut.createImage()

    objAmp = isceobj.createAmpImage()
    objAmpOut = isceobj.createAmpImage()
    IU.copyAttributes(imageAmp, objAmp)
    IU.copyAttributes(imageAmp, objAmpOut)
    outAmpFilename = outIntFilename.replace('int', 'amp')
    objAmp.setAccessMode('read')
    objAmpOut.setFilename(outAmpFilename)

    objAmpOut.setAccessMode('write')
    objAmp.createImage()
    objAmpOut.createImage()

    numRangeBin = objInt.getWidth()
    lines = objInt.getLength()


    dopplerCoeff = infos['dopplerCentroid'].getDopplerCoefficients(inHz=False)

    objResamp = stdproc.createResamp_only()

    objResamp.setNumberLines(lines)
    objResamp.setNumberFitCoefficients(infos['numberFitCoefficients'])
    objResamp.setSlantRangePixelSpacing(infos['slantRangePixelSpacing'])
    objResamp.setNumberRangeBin(numRangeBin)
    objResamp.setDopplerCentroidCoefficients(dopplerCoeff)

    objResamp.wireInputPort(name='offsets', object=offsetField)
    objResamp.wireInputPort(name='instrument', object=instrument)
    #set the tag used in the outfile. each message is precided by this tag
    #if the writer is not of "file" type the call has no effect
    objResamp.stdWriter = stdWriter.set_file_tags("resamp_only",
                                                  "log",
                                                  "err",
                                                  "out")

    objResamp.resamp_only(objInt, objIntOut, objAmp, objAmpOut)

    if catalog is not None:
        # Record the inputs and outputs
        isceobj.Catalog.recordInputsAndOutputs(catalog, objResamp,
                                               "runResamp_only.%s" % sceneid,
                                               logger,
                                               "runResamp_only.%s" % sceneid)
    objInt.finalizeImage()
    objIntOut.finalizeImage()
    objAmp.finalizeImage()
    objAmpOut.finalizeImage()

    return objIntOut, objAmpOut
