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



# Comment: Adapted from InsarProc/runUnwrapGrass.py
import logging
import isceobj
from iscesys.Component.Component import Component
from mroipac.grass.grass import Grass
import os
# giangi: taken Piyush code grass.py and adapted

logger = logging.getLogger('isce.isceProc.runUnwrap')

def runUnwrap(self):
    infos = {}
    for attribute in ['topophaseFlatFilename', 'unwrappedIntFilename', 'coherenceFilename']:
        infos[attribute] = getattr(self._isce, attribute)

    for sceneid1, sceneid2 in self._isce.selectedPairs:
        pair = (sceneid1, sceneid2)
        for pol in self._isce.selectedPols:
            intImage = self._isce.resampIntImages[pair][pol]
            width = intImage.width
            sid = self._isce.formatname(pair, pol)
            infos['outputPath'] = os.path.join(self.getoutputdir(sceneid1, sceneid2), sid)
            run(width, infos, sceneid=sid)


def run(width, infos, sceneid='NO_ID'):
    logger.info("Unwrapping interferogram using Grass: %s" % sceneid)
    wrapName   = infos['outputPath'] + '.' + infos['topophaseFlatFilename']
    unwrapName = infos['outputPath'] + '.' + infos['unwrappedIntFilename']
    corName    = infos['outputPath'] + '.' + infos['coherenceFilename']

    with isceobj.contextIntImage(
        filename=wrapName,
        width=width,
        accessMode='read') as intImage:

        with isceobj.contextOffsetImage(
            filename=corName,
            width = width,
            accessMode='read') as cohImage:

            with isceobj.contextUnwImage(
                filename=unwrapName,
                width = width,
                accessMode='write') as unwImage:

                grs=Grass(name='insarapp_grass')
                grs.configure()
                grs.wireInputPort(name='interferogram',
                    object=intImage)
                grs.wireInputPort(name='correlation',
                    object=cohImage)
                grs.wireInputPort(name='unwrapped interferogram',
                    object=unwImage)
                grs.unwrap()
                unwImage.renderHdr()

