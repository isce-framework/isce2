#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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



from __future__ import print_function
from iscesys.Component.Component import Component,Port
from isceobj.Location.Offset import OffsetField,Offset
from isceobj.Util import offoutliers
from isceobj.Util.decorators import dov, pickled, logged


@pickled
class Offoutliers(Component):

    logging_name = "stdproc.offoutliers"

    dictionaryOfVariables = {
        'DISTANCE' : ['distance', float, True],
        }
    dictionaryOfOutputVariables = {
        'INDEX_ARRAY' : 'indexArray',
        'AVERAGE_OFFSET_DOWN' : 'averageOffsetDown',
        'AVERAGE_OFFSET_ACROSS' : 'averageOffsetAcross'
        }


    @logged
    def __init__(self):
        self.snrThreshold = 0
        self.indexArray = []
        self.dim1_indexArray = None
        self.indexArraySize = None
        self.averageOffsetDown = None
        self.averageOffsetAcross = None
        self.numPoints = None
        self.locationAcross = []
        self.dim1_locationAcross = None
        self.locationAcrossOffset = []
        self.dim1_locationAcrossOffset = None
        self.locationDown = []
        self.dim1_locationDown = None
        self.locationDownOffset = []
        self.dim1_locationDownOffset = None
        self.distance = None
        self.sig = []
        self.dim1_sig = None
        self.snr = []
        self.dim1_snr = None
        super(Offoutliers, self).__init__()
        return None

    def createPorts(self):
        self.inputPorts['offsets'] = self.addOffsets
        return None

    def offoutliers(self):
        for port in self._inputPorts:
            port()

        self.numPoints = len(self.locationAcross)
        self.dim1_indexArray = self.numPoints
        self.allocateArrays()
        self.setState()
        offoutliers.offoutliers_Py()
        self.indexArraySize = offoutliers.getIndexArraySize_Py()
        self.getState()
        self.deallocateArrays()

    def setState(self):
        offoutliers.setStdWriter_Py(int(self.stdWriter))
        offoutliers.setNumberOfPoints_Py(int(self.numPoints))
        offoutliers.setLocationAcross_Py(self.locationAcross,
                                         self.dim1_locationAcross)
        offoutliers.setLocationAcrossOffset_Py(self.locationAcrossOffset,
                                               self.dim1_locationAcrossOffset)
        offoutliers.setLocationDown_Py(self.locationDown,
                                       self.dim1_locationDown)
        offoutliers.setLocationDownOffset_Py(self.locationDownOffset,
                                             self.dim1_locationDownOffset)
        offoutliers.setDistance_Py(self.distance)
        offoutliers.setSign_Py(self.sig, self.dim1_sig)
        offoutliers.setSNR_Py(self.snr, self.dim1_snr)

    def setNumberOfPoints(self, var):
        self.numPoints = int(var)

    def setLocationAcross(self, var):
        self.locationAcross = var

    def setLocationAcrossOffset(self, var):
        self.locationAcrossOffset = var

    def setLocationDown(self, var):
        self.locationDown = var

    def setLocationDownOffset(self, var):
        self.locationDownOffset = var

    def setDistance(self, var):
        self.distance = float(var)

    def setSign(self, var):
        """I think that this is actually the sigma, not the sign"""
        self.sig = var

    def setSNR(self, var):
        self.snr = var

    def setSNRThreshold(self, var):
        self.snrThreshold = var

    def getState(self):
        #Notice that we allocated a larger size since it was not known a priori, but when we retrieve the data we only retrieve the valid ones
        self.indexArray = offoutliers.getIndexArray_Py(self.indexArraySize)
        self.averageOffsetDown = offoutliers.getAverageOffsetDown_Py()
        self.averageOffsetAcross = offoutliers.getAverageOffsetAcross_Py()

    def getIndexArray(self):
        return self.indexArray

    def getAverageOffsetDown(self):
        return self.averageOffsetDown

    def getAverageOffsetAcross(self):
        return self.averageOffsetAcross

    def getRefinedLocations(self):
        indxA = self.indexArray
        numArrays = 6
        retList = [[0]*len(indxA) for i in range(numArrays)]
        for j in range(len(retList[0])):
            retList[0][j] = self.locationAcross[indxA[j]]
            retList[1][j] = self.locationAcrossOffset[indxA[j]]
            retList[2][j] = self.locationDown[indxA[j]]
            retList[3][j] = self.locationDownOffset[indxA[j]]
            retList[4][j] = self.snr[indxA[j]]
            retList[5][j] = self.sig[indxA[j]]

        return retList

    def getRefinedOffsetField(self):
        offsets = OffsetField()

        indxA = self.indexArray
        for j in range(len(indxA)):
            offset = Offset()
            across = self.locationAcross[indxA[j]]
            down = self.locationDown[indxA[j]]
            acrossOffset = self.locationAcrossOffset[indxA[j]]
            downOffset = self.locationDownOffset[indxA[j]]
            snr = self.snr[indxA[j]]
            #sign = self.sig[indxA[j]]
            offset.setCoordinate(across,down)
            offset.setOffset(acrossOffset,downOffset)
            offset.setSignalToNoise(snr)
            offsets.addOffset(offset)

        return offsets

    def allocateArrays(self):
        if self.dim1_indexArray is None:
            self.dim1_indexArray = len(self.indexArray)

        if not self.dim1_indexArray:
            print("Error. Trying to allocate zero size array")
            raise Exception

        offoutliers.allocate_indexArray_Py(self.dim1_indexArray)

        if self.dim1_locationAcross is None:
            self.dim1_locationAcross = len(self.locationAcross)

        if not self.dim1_locationAcross:
            print("Error. Trying to allocate zero size array")
            raise Exception

        offoutliers.allocate_xd_Py(self.dim1_locationAcross)

        if self.dim1_locationAcrossOffset is None:
            self.dim1_locationAcrossOffset = len(self.locationAcrossOffset)

        if not self.dim1_locationAcrossOffset:
            print("Error. Trying to allocate zero size array")
            raise Exception

        offoutliers.allocate_acshift_Py(self.dim1_locationAcrossOffset)

        if self.dim1_locationDown is None:
            self.dim1_locationDown = len(self.locationDown)

        if not self.dim1_locationDown:
            print("Error. Trying to allocate zero size array")
            raise Exception

        offoutliers.allocate_yd_Py(self.dim1_locationDown)

        if self.dim1_locationDownOffset is None:
            self.dim1_locationDownOffset = len(self.locationDownOffset)

        if not self.dim1_locationDownOffset:
            print("Error. Trying to allocate zero size array")
            raise Exception

        offoutliers.allocate_dnshift_Py(self.dim1_locationDownOffset)

        if (self.dim1_sig is None):
            self.dim1_sig = len(self.sig)

        if (not self.dim1_sig):
            print("Error. Trying to allocate zero size array")
            raise Exception

        offoutliers.allocate_sig_Py(self.dim1_sig)

        if self.dim1_snr is None:
            self.dim1_snr = len(self.snr)

        if not self.dim1_snr:
            print("Error. Trying to allocate zero size array")
            raise Exception

        offoutliers.allocate_s_Py(self.dim1_snr)

    def deallocateArrays(self):
        offoutliers.deallocate_indexArray_Py()
        offoutliers.deallocate_xd_Py()
        offoutliers.deallocate_acshift_Py()
        offoutliers.deallocate_yd_Py()
        offoutliers.deallocate_dnshift_Py()
        offoutliers.deallocate_sig_Py()
        offoutliers.deallocate_s_Py()

    def addOffsets(self):
        offsets = self._inputPorts.getPort('offsets').getObject()
        if offsets:
            # First, cull the offsets using the SNR provided
            culledOffsets = offsets.cull(self.snrThreshold)
            try:
                for offset in culledOffsets:
                    across, down = offset.getCoordinate()
                    acrossOffset, downOffset = offset.getOffset()
                    snr = offset.getSignalToNoise()
                    self.locationAcross.append(across)
                    self.locationDown.append(down)
                    self.locationAcrossOffset.append(acrossOffset)
                    self.locationDownOffset.append(downOffset)
                    self.snr.append(snr)
                    self.sig.append(1.0) # Sigmas used in the inversion
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError("Unable to wire Offset port")
            pass
        pass
    pass
