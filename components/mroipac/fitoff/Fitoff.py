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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
from iscesys.Component.Component import Component,Port
from isceobj.Location.Offset import OffsetField,Offset
from mroipac.fitoff import fitoff
from isceobj.Util.decorators import dov, pickled, logged


@pickled
class Fitoff(Component):
    
    logging_name = "mroipac.fitoff"

    dictionaryOfVariables = { 
        'NUMBER_OF_SIGMAS' : ['nSigma', float, True],
        'MAX_RMS' : ['maxRMS', float, True],
        'NUM_POINTS' : ['numPoints', int, True],
        'MIN_ITER': ['minIter', int, True],
        'MAX_ITER': ['maxIter', int, True],
        'MIN_PONTS': ['minPoints', int, True],
        }
    dictionaryOfOutputVariables = {
        'AFFINE_TRANSFORM' : 'affineTransform',
        'AVERAGE_OFFSET_DOWN' : 'averageOffsetDown',
        'AVERAGE_OFFSET_ACROSS' : 'averageOffsetAcross'
        }


    @dov
    @logged
    def __init__(self):
        super(Fitoff, self).__init__()
        self.numPoints = 0
        self.maxRMS = 0.08
        self.nSigma = 1.5
        self.minPoints = 50
        self.minIter = 3 
        self.maxIter = 30
        self.useL1norm = True
        self.affineTransform = []
        self.averageOffsetDown = None
        self.averageOffsetAcross = None
        self.numPoints = None
        self.locationAcross = []
        self.locationAcrossOffset = []
        self.locationDown = []
        self.locationDownOffset = []
        self.distance = None
        self.snr = []
        self.cov_across = []
        self.cov_down = []
        self.cov_cross = []
        self.numRefined = None
        self.refinedOffsetField = None
        self.createPorts()
#        self.stdWriter = None
        return None

    def createPorts(self):
        self._inputPorts.add( Port(name='offsets',method=self.addOffsets) )
        return None

    def fitoff(self):
        for port in self._inputPorts:
            method = port.getMethod()
            method()
            
        self.numPoints = len(self.locationAcross)
        self.allocateArrays()

        self.setState()
        fitoff.fitoff_Py()
        self.getState()
        self.deallocateArrays()

    def setState(self):
        fitoff.setStdWriter_Py(int(self.stdWriter))
        fitoff.setLocationAcross_Py(self.locationAcross,
                                         self.numPoints)
        fitoff.setLocationAcrossOffset_Py(self.locationAcrossOffset,
                                               self.numPoints)
        fitoff.setLocationDown_Py(self.locationDown,
                                       self.numPoints)
        fitoff.setLocationDownOffset_Py(self.locationDownOffset,
                                             self.numPoints)
        fitoff.setSNR_Py(self.snr, self.numPoints)
        fitoff.setCovDown_Py(self.cov_down, self.numPoints)
        fitoff.setCovAcross_Py(self.cov_across, self.numPoints)
        fitoff.setCovCross_Py(self.cov_cross, self.numPoints)
        fitoff.setMaxRms_Py(self.maxRMS)
        fitoff.setNSig_Py(self.nSigma)
        fitoff.setMinPoint_Py(self.minPoints)
        fitoff.setL1normFlag_Py(int(self.useL1norm))
        fitoff.setMinIter_Py(self.minIter)
        fitoff.setMaxIter_Py(self.maxIter)

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

    def setCov_Across(self, var):
        self.cov_across = var

    def setCov_Down(self, var):
        self.covDown = var

    def setCov_Cross(self,var):
        self.cov_cross = var

    def setNSigma(self, var):
        self.nSigma = var

    def setMaxRMS(self, var):
        self.maxRms = var

    def setSNR(self, var):
        self.snr = var

    def setMinPoints(self, var):
        self.minPoints = var

#    def stdWriter(self, var):
#        self.stdWriter = var

    def getState(self):
        #Notice that we allocated a larger size since it was not known a priori, but when we retrieve the data we only retrieve the valid ones
        self.affineVec = fitoff.getAffineVector_Py()
        self.averageOffsetAcross = self.affineVec[4]
        self.averageOffsetDown = self.affineVec[5]
        self.numRefined = fitoff.getNumberOfRefinedOffsets_Py()
        retList = fitoff.getRefinedOffsetField_Py(self.numRefined)

        self.refinedOffsetField = OffsetField()
        for value in retList:
            oneoff = Offset(value[0],
                            value[1],
                            value[2],
                            value[3],
                            value[4],
                            value[5],
                            value[6],
                            value[7])
            self.refinedOffsetField.addOffset(oneoff)

        return

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
            offset.setCoordinate(across,down)
            offset.setOffset(acrossOffset,downOffset)
            offset.setSignalToNoise(snr)
            offsets.addOffset(offset)
        
        return offsets
        
    def allocateArrays(self):
        if self.numPoints is None:
            self.numPoints = len(self.locationAcross)

        fitoff.setNumberLines_Py(int(self.numPoints))
        fitoff.allocateArrays_Py(int(self.numPoints))
        return

    def deallocateArrays(self):
        fitoff.deallocateArrays_Py()

    def addOffsets(self):
        offsets = self._inputPorts.getPort('offsets').getObject()
        if offsets:
            try:
                for offset in offsets:
                    across, down = offset.getCoordinate()
                    acrossOffset, downOffset = offset.getOffset()
                    snr = offset.getSignalToNoise()
                    cova, covd, covx = offset.getCovariance()
                    self.locationAcross.append(across)
                    self.locationDown.append(down)                
                    self.locationAcrossOffset.append(acrossOffset)
                    self.locationDownOffset.append(downOffset)
                    self.snr.append(snr)
                    self.cov_across.append(cova) # Sigmas used in the inversion
                    self.cov_down.append(covd)
                    self.cov_cross.append(covx)
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError("Unable to wire Offset port")
            pass
        pass
    pass

