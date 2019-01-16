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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import logging
import math
import random
import isceobj
from isceobj.Scene.Frame import Frame
from iscesys.Component.Component import Component, Port
from mroipac.dopiq import dopiq
from isceobj.Util.mathModule import MathModule

YMIN = Component.Parameter(
    'startLine',
    public_name='YMIN',
    default=1,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


XMIN = Component.Parameter(
    'lineHeaderLength',
    public_name='XMIN',
    default=0,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


I_BIAS = Component.Parameter(
    'mean',
    public_name='I_BIAS',
    default=0,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


WIDTH = Component.Parameter(
    'lineLength',
    public_name='WIDTH',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


XMAX = Component.Parameter(
    'lastSample',
    public_name='XMAX',
    default=0,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


PRF = Component.Parameter(
    'prf',
    public_name='PRF',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


FILE_LENGTH = Component.Parameter(
    'numberOfLines',
    public_name='FILE_LENGTH',
    default=0,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


DOPPLER = Component.Parameter(
    'fractionalDoppler',
    public_name='DOPPLER',
    default=[],
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)


class DopIQ(Component):


    parameter_list = (
                      YMIN,
                      XMIN,
                      I_BIAS,
                      WIDTH,
                      XMAX,
                      PRF,
                      FILE_LENGTH,
                      DOPPLER
                     )


    logging_name = "isce.DopIQ"
    family = 'dopiq'

    def __init__(self,family='',name=''):
        super(DopIQ, self).__init__(family if family else  self.__class__.family, name=name)
#        self.logger = logging.getLogger(
        self.rawImage = ''
        self.rawFilename = ''
        self.dim1_doppler = None     
        self.pixelIndex = []
        self.linear = {}
        self.quadratic = {}   #insarApp

        self.coeff_list = []  #roiApp
        
#        self.createPorts()
                                      
        
        return None

    def createPorts(self):
        # Create Input Ports
        instrumentPort = Port(name="instrument",method=self.addInstrument,
                              doc="An object that has getPulseRepetitionFrequency() and getInPhaseValue() methods")
        framePort = Port(name="frame",method=self.addFrame,
                         doc="An object that has getNumberOfSamples() and getNumberOfLines() methods")
        imagePort = Port(name="image",method=self.addImage,
                         doc="An object that has getXmin() and getXmax() methods")                
        self._inputPorts.add(instrumentPort)
        self._inputPorts.add(framePort)
        self._inputPorts.add(imagePort)
        

    def addInstrument(self):
        instrument = self._inputPorts.getPort('instrument').getObject()
        if (instrument):
            try:            
                self.prf = instrument.getPulseRepetitionFrequency()
                self.mean = instrument.getInPhaseValue()
            except AttributeError:
                self.logger.error("Object %s requires a getPulseRepetitionFrequency() and getInPhaseValue() method" % (instrument.__class__))
        
    def addFrame(self):
        frame = self._inputPorts.getPort('frame').getObject()
        if(frame):
            try:                                
                self.numberOfLines = frame.getNumberOfLines()
            except AttributeError:
                self.logger.error("Object %s requires a getNumberOfSamples() and getNumberOfLines() method" % (frame.__class__))
        
    def addImage(self):
        image = self._inputPorts.getPort('image').getObject()
        if(image):
            try:
                self.rawFilename = image.getFilename()
                self.lineHeaderLength = image.getXmin()
                self.lastSample = image.getXmax()
                self.lineLength = self.lastSample
            except AttributeError:
                self.logger.error("Object %s requires getXmin(), getXmax() and getFilename() methods" % (image.__class__))                    

    def setRawfilename(self,filename):
        self.rawFilename = filename

    def setPRF(self,prf):
        self.prf = float(prf)

    def setMean(self,mean):
        self.mean = float(mean)

    def setLineLength(self,length):
        self.lineLength = int(length)

    def setLineHeaderLength(self,length):
        self.lineHeaderLength = int(length)

    def setLastSample(self,length):
        self.lastSample = int(length)

    def setNumberOfLines(self,lines):
        self.numberOfLines = int(lines)

    def setStartLine(self,start):
        if (start < 1):
            raise ValueError("START_LINE must be greater than 0")
        self.startLine = int(start)

    ##
    # Return the doppler estimates in Hz/prf as a function of range bin
    def getDoppler(self):        
        return self.fractionalDoppler

    def calculateDoppler(self,rawImage=None):
        self.activateInputPorts()

        rawCreatedHere = False
        if (rawImage == None):
            rawImage = self.createRawImage()
            rawCreateHere = True
        rawImagePt= rawImage.getImagePointer()        
        self.setState()
        self.allocateArrays()
        dopiq.dopiq_Py(rawImagePt)
        self.getState()        
        self.deallocateArrays()        
        if(rawCreatedHere):
            rawImage.finalizeImage()
          
    def createRawImage(self):
        # Check file name
        width = self.lineLength        
        objRaw = isceobj.createRawImage()
        objRaw.initImage(self.rawFilename,'read',width)
        objRaw.createImage()                
        return objRaw
    
    def setState(self):
        # Set up the stuff needed for dopiq
        dopiq.setPRF_Py(self.prf)
        dopiq.setNumberOfLines_Py(self.numberOfLines)
        dopiq.setMean_Py(self.mean)
        dopiq.setLineLength_Py(int(self.lineLength))
        dopiq.setLineHeaderLength_Py(self.lineHeaderLength)
        dopiq.setLastSample_Py(int(self.lastSample))
        dopiq.setStartLine_Py(self.startLine)
        self.dim1_doppler = int((self.lastSample - self.lineHeaderLength)/2)
        
    def getState(self):
        self.fractionalDoppler = dopiq.getDoppler_Py(self.dim1_doppler)        
        
    def allocateArrays(self):
        if (self.dim1_doppler == None):
            self.dim1_doppler = len(self.fractionalDoppler)
            
        if (not self.dim1_doppler):
            self.logger.error("Error. Trying to allocate zero size array")

            raise Exception
        
        dopiq.allocate_doppler_Py(self.dim1_doppler)
        
    def deallocateArrays(self):
        dopiq.deallocate_doppler_Py()

    def _wrap(self):
        """Wrap the Doppler values"""
        wrapCount = 0*5;
        noiseLevel = 0*0.7;
 
        for i in range(len(self.fractionalDoppler)):
            if ( wrapCount != 0 ):
                self.fractionalDoppler[i] += wrapCount + i * wrapCount / len(self.fractionalDoppler)

            if( noiseLevel != 0 ):
                self.fractionalDoppler[i] += 1 + noiseLevel/2 - random.random(noiseLevel)

            self.fractionalDoppler[i] -= int(self.fractionalDoppler[i])
 
    def _unwrap(self):
        """Unwrapping"""
         
        averageLength=10
        firstDop = 0
         
        lastValues = []
        unw = [None]*len(self.fractionalDoppler)
         
        for i in range(averageLength-1):
            lastValues.append(firstDop)
         
        for i in range(len(self.fractionalDoppler)):
            predicted = sum(lastValues) / len(lastValues)
            ambiguity = predicted - self.fractionalDoppler[i]
            ambiguity = int(ambiguity)
            unw[i] = self.fractionalDoppler[i] + ambiguity
         
        if ( len(lastValues) >= (averageLength-1)):
            lastValues.pop(0)
        lastValues.append(unw[i])
         
        return unw
         
    def _cullPoints(self,pixels,unw):
        """Remove points greater than 3 standard deviations from the line fit"""

        slope = self.linear['b']
        intercept = self.linear['a']
        stdDev = self.linear['stdDev'] 
        numCulled = 0
        newPixel = []
        newUnw = []
 
        for i in range(len(pixels)):	
            fit      = intercept + slope * pixels[i]
            residual = unw[i] - fit
            if ( math.fabs(residual) < 3*stdDev ):
                newPixel.append(pixels[i])
                newUnw.append(unw[i])
            else:
                numCulled += 1
         
        return newPixel, newUnw
 
    def fitDoppler(self):
        """Read in a Doppler file, remove outliers and then perform a quadratic fit"""
        self._wrap()
        unw = self._unwrap()
        self.pixelIndex = range(len(self.fractionalDoppler))
        (self.linear['a'], self.linear['b'], self.linear['stdDev']) = MathModule.linearFit(self.pixelIndex, unw)        
        (pixels, unw) = self._cullPoints(self.pixelIndex,unw)
        (self.linear['a'], self.linear['b'], self.linear['stdDev']) = MathModule.linearFit(pixels, unw)        
        (pixels, unw) = self._cullPoints(pixels,unw)
  
        (a,b,c) = MathModule.quadraticFit(pixels,unw)
        self.quadratic['a'] = a  
        self.quadratic['b'] = b  
        self.quadratic['c'] = c


        self.coeff_list = [a,b,c]
