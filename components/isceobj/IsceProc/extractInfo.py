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



# Comment: Adapted from InsarProc/extractInfo.py by Brett George
from contrib.frameUtils.FrameInfoExtractor import FrameInfoExtractor
import logging
logger = logging.getLogger('isce.isceProc.ExtractInfo')


def extractInfo(self, frames):
    FIE = FrameInfoExtractor()
    infos = []
    for frame in frames:
        infos.append(FIE.extractInfoFromFrame(frame))

    mainInfo = infos[0]
    mainInfo.sensingStart = [ info.sensingStart for info in infos ]
    mainInfo.sensingStop = [ info.sensingStop for info in infos ]

    # for stitched frames do not make sense anymore
    bbs = [ info.getBBox() for info in infos ]
    bbxy = {}
    for x in range(4):
        bbxy[x] = {}
        for y in range(2):
            bbxy[x][y] = [ bb[x][y] for bb in bbs ]
    latEarlyNear = bbxy[0][0][0]
    latLateNear = bbxy[2][0][0]

    #figure out which one is the bottom
    if latEarlyNear > latLateNear:
        #early is the top
        ret = []
        # the calculation computes the minimum bbox. it is not exact, but given
        # the approximation in the estimate of the corners, it's ok
        for x, op1, op2 in zip(range(4), (min, min, max, max), (max, min, max, min)):
            ret.append([op1(bbxy[x][0]), op2(bbxy[x][1])])
    else:
        # late is the top
        ret = []
        for x, op1, op2 in zip(range(4), (max, max, min, min), (max, min, max, min)):
            ret.append([op1(bbxy[x][0]), op2(bbxy[x][1])])

    mainInfo.bbox = ret
    return mainInfo
    # the track should be the same for all

