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



import isceobj.Catalog
import logging
logger = logging.getLogger('isce.insar.extractInfo')

def extractInfo(self, reference, secondary):
    from contrib.frameUtils.FrameInfoExtractor import FrameInfoExtractor
    FIE = FrameInfoExtractor()
    referenceInfo = FIE.extractInfoFromFrame(reference)
    secondaryInfo = FIE.extractInfoFromFrame(secondary)
    referenceInfo.sensingStart = [referenceInfo.sensingStart, secondaryInfo.sensingStart]
    referenceInfo.sensingStop = [referenceInfo.sensingStop, secondaryInfo.sensingStop]
    # for stitched frames do not make sense anymore
    mbb = referenceInfo.getBBox()
    sbb = secondaryInfo.getBBox()
    latEarlyNear = mbb[0][0]
    latLateNear = mbb[2][0]
 
    #figure out which one is the bottom
    if latEarlyNear > latLateNear: 
        #early is the top
        ret = []
        # the calculation computes the minimum bbox. it is not exact, bu given
        # the approximation in the estimate of the corners, it's ok 
        ret.append([min(mbb[0][0], sbb[0][0]), max(mbb[0][1], sbb[0][1])])
        ret.append([min(mbb[1][0], sbb[1][0]), min(mbb[1][1], sbb[1][1])])
        ret.append([max(mbb[2][0], sbb[2][0]), max(mbb[2][1], sbb[2][1])])
        ret.append([max(mbb[3][0], sbb[3][0]), min(mbb[3][1], sbb[3][1])])
    else:
        # late is the top
        ret = []
        ret.append([max(mbb[0][0], sbb[0][0]), max(mbb[0][1], sbb[0][1])])
        ret.append([max(mbb[1][0], sbb[1][0]), min(mbb[1][1], sbb[1][1])])
        ret.append([min(mbb[2][0], sbb[2][0]), max(mbb[2][1], sbb[2][1])])
        ret.append([min(mbb[3][0], sbb[3][0]), min(mbb[3][1], sbb[3][1])])
    
    referenceInfo.bbox = ret
    return referenceInfo
    # the track should be the same for both

