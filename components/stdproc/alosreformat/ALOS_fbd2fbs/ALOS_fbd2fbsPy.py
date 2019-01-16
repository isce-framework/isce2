#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2011 California Institute of Technology. ALL RIGHTS RESERVED.
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
from iscesys.Compatibility import Compatibility
from isceobj import Constants as CN
from stdproc.alosreformat.ALOS_fbd2fbs import ALOS_fbd2fbs

class ALOS_fbd2fbsPy(Component):

    def ALOS_fbd2fbs(self):
        for port in self.inputPorts:
            port()

        self.setState()
        ALOS_fbd2fbs.ALOS_fbd2fbs_Py()
        self.updateValues()
        return None

    def run(self):
        self.ALOS_fbd2fbs()

    def setState(self):
        ALOS_fbd2fbs.setNumberGoodBytes_Py(int(self.numberGoodBytes))
        ALOS_fbd2fbs.setNumberBytesPerLine_Py(int(self.numberBytesPerLine))
        ALOS_fbd2fbs.setNumberLines_Py(int(self.numberLines))
        ALOS_fbd2fbs.setFirstSample_Py(int(self.firstSample))
        ALOS_fbd2fbs.setInPhaseValue_Py(float(self.inPhaseValue))
        ALOS_fbd2fbs.setQuadratureValue_Py(float(self.quadratureValue))
        ALOS_fbd2fbs.setInputFilename_Py(str(self.inputFilename))
        ALOS_fbd2fbs.setOutputFilename_Py(str(self.outputFilename))
        return None

    ## TODO:fix harcoded values
    def updateValues(self):
        self.quadratureValue = 63.5
        self.inPhaseValue = 63.5
        self.rangeChirpExtensionPoints *= 2.0
        fbssamp = 2*int(self.numberGoodBytes/2)  #EMG - self.firstSample)
        self.numberGoodBytes = 2*(fbssamp) #EMG + self.firstSample)
        self.bytesPerLine = 2*(fbssamp + self.firstSample)
        self.rangeSamplingRate *= 2
        self.rangePixelSize /=2 

    def updateFrame(self,frame):
        frame.getImage().setXmax(self.bytesPerLine)
        frame.getImage().setWidth(self.bytesPerLine)
        frame.getImage().setFilename(self.outputFilename)
        frame.setNumberOfSamples(self.bytesPerLine)
        instrument = frame.getInstrument()
        instrument.setInPhaseValue(self.inPhaseValue)
        instrument.setQuadratureValue(self.quadratureValue)
        instrument.setRangeSamplingRate(self.rangeSamplingRate)
        instrument.setChirpSlope(self.chirpSlope)       
        instrument.setRangePixelSize(self.rangePixelSize)

    def setRangeSamplingRate(self,var):
        self.rangeSamplingRate = float(var)
    
    def setRangeChirpExtensionPoints(self,var):
        self.rangeChirpExtensionPoints = float(var)
    
    def setNumberGoodBytes(self,var):
        self.numberGoodBytes = int(var)
        return None

    def setNumberBytesPerLine(self,var):
        self.numberBytesPerLine = int(var)
        return None

    def setNumberLines(self,var):
        self.numberLines = int(var)
        return None
    
    def setFirstSample(self,var):
        self.firstSample = int(var)
        return None

    def setInPhaseValue(self,var):
        self.inPhaseValue = float(var)
        return None

    def setQuadratureValue(self,var):
        self.quadratureValue = float(var)
        return None

    def setInputFilename(self,var):
        self.inputFilename = str(var)
        return None

    def setOutputFilename(self,var):
        self.outputFilename = str(var)
        return None
 
    def getRangeSamplingRate(self):
        return self.rangeSamplingRate
    
    def getRangeChirpExtensionPoints(self):
        return self.rangeChirpExtensionPoints
    
    def getNumberGoodBytes(self):
        return self.numberGoodBytes

    def getNumberBytesPerLine(self):
        return self.numberBytesPerLine

    def getChirpSlope(self):
        return self.chirpSlope

    def getInPhaseValue(self):
        return self.inPhaseValue

    def getQuadratureValue(self):
        return self.quadratureValue

    def addFrame(self):
        frame = self._inputPorts.getPort('frame').getObject()
        if (frame):
            try:
                self.numberLines = frame.getNumberOfLines()
                self.numberBytesPerLine = frame.getImage().getXmax()
                self.firstSample = frame.getImage().getXmin()/2
                self.numberGoodBytes = frame.getImage().getXmax() - frame.getImage().getXmin()
                instrument = frame.getInstrument()
                self.inPhaseValue = instrument.getInPhaseValue()
                self.quadratureValue = instrument.getQuadratureValue()
                self.rangeSamplingRate = instrument.getRangeSamplingRate()
                self.rangePixelSize = instrument.getRangePixelSize()
                self.chirpSlope = instrument.getChirpSlope()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    
    logging_name = "stdproc.alosreformat.ALOS_fbd2fbs"
    def __init__(self):
        super(ALOS_fbd2fbsPy, self).__init__()
        self.rangeChirpExtensionPoints = 0
        self.rangeSamplingRate = None
        self.numberGoodBytes = None
        self.numberBytesPerLine = None
        self.numberLines = None
        self.firstSample = None
        self.chirpSlope = None
        self.inPhaseValue = None
        self.quadratureValue = None
        self.rangePixelSize = None
        self.inputFilename = ''
        self.outputFilename = ''
        self.dictionaryOfVariables = { 
            'NUMBER_RANGE_BIN' : ['self.numberRangeBin', 'int','mandatory'], 
            'NUMBER_GOOD_BYTES' : ['self.numberGoodBytes', 'int','mandatory'], 
            'RANGE_SAMPLING_RATE' : ['self.rangeSamplingRate', 'float','mandatory'], 
            'RANGE_CHIRP_EXTENSION_POINTS':['self.rangeChirpExtensionPoints','float','mandatory'], 
            'NUMBER_BYTES_PER_LINE' : ['self.numberBytesPerLine', 'int','mandatory'], 
            'FIRST_SAMPLE' : ['self.firstSample', 'int','mandatory'], 
            'NUMBER_LINES' : ['self.numberLines', 'int','mandatory'], 
            'INPHASE_VALUE' : ['self.inPhaseValue', 'float','mandatory'], 
            'QUADRATURE_VALUE' : ['self.quadratureValue', 'float','mandatory'], 
            'INPUT_FILENAME' : ['self.inputFilename', 'str','mandatory'], 
            'OUTPUT_FILENAME' : ['self.outputFilename', 'str','mandatory'] 
            }
        self.dictionaryOfOutputVariables = {                                            
            'NUMBER_GOOD_BYTES' : 'self.numberGoodBytes', 
            'RANGE_SAMPLING_RATE' : 'self.rangeSamplingRate', 
            'RANGE_CHIRP_EXTENSION_POINTS' : 'self.rangeChirpExtensionPoints', 
            'NUMBER_BYTES_PER_LINE' : 'self.numberBytesPerLine', 
            'CHIRP_SLOPE' : 'self.chirpSlope', 
            'INPHASE_VALUE' : 'self.inPhaseValue', 
            'QUADRATURE_VALUE' : 'self.quadratureValue'                                      }
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        typePos = 2
        for key , val in self.dictionaryOfVariables.items():
            if val[typePos] == 'mandatory':
                self.mandatoryVariables.append(key)
            elif val[typePos] == 'optional':
                self.optionalVariables.append(key)
            else:
                print('Error. Variable can only be optional or mandatory')
                raise Exception
        return None

    def createPorts(self):
        framePort = Port(name='frame',method=self.addFrame)
        self._inputPorts.add(framePort)
        return None

    pass
