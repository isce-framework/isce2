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



import isceobj
from iscesys.Component.Component import Component, Port
from mroipac.doppler import doppler

class Doppler(Component):

    def __init__(self):
        super(Doppler, self).__init__()
        self.slcImage = ''
        self.slcFilename = ''        
        self.lines = None
        self.startLine = 1
        self.samples = None
        self.dim1_r_fd = None     
        self.r_fd = []
        self.createPorts()
        self.dictionaryOfVariables = {
            'WIDTH': ['self.samples','int','mandatory'],
            'YMIN': ['self.startLine','int','mandatory'],
            'FILE_LENGTH': ['self.Lines','int','mandatory']}
        
        self.dictionaryOfOutputVariables= {'R_FD': 'self.r_fd'}
        self.descriptionOfVariables = {}
        return None
        
    def createPorts(self):
        # Create Input Ports
        framePort = Port(name='frame',method=self.addFrame)
        instrumentPort = Port(name='instrument',method=self.addInstrument)
        imagePort = Port(name='image',method=self.addImage)
        self._inputPorts.add(framePort)
        self._inputPorts.add(imagePort)
        self._inputPorts.add(instrumentPort)
        return None

    def addFrame(self):
        frame = self._inputPorts.getPort(name='frame').getObject()
        if (frame):
            try:
                self.samples = frame.getNumberOfSamples()
                self.lines = frame.getNumberOfLines()
            except AttributeError:
                print( "Object %s requires getNumberOfSamples() and getNumberOfLines() methods" % frame.__class__)
        
    def addImage(self):
        image = self._inputPorts.getPort(name='image').getObject()
        if (image):
            try:
                self.slcFilename = image.getFilename()
            except AttributeError:
                print ("Object %s requires a getFilename() methods" % image.__class__)
                
    def addInstrument(self):
        pass

    def setSLCfilename(self,filename):
        self.slcFilename = filename
    
    def setSamples(self,length):
        self.samples = int(length)

    def setLines(self,lines):
        self.lines = int(lines)

    def setStartLine(self,start):
        if (start < 1):
            raise ValueError("START_LINE must be greater than 0")
        self.startLine = int(start)

    def getDoppler(self):        
        return self.r_fd

    def calculateDoppler(self,slcImage=None):
        for port in self._inputPorts:
            method = port.getMethod()
            method()
        slcCreatedHere = False
        if (slcImage == None):
            slcImage = self.createSlcImage()
            slcCreateHere = True
        slcImagePt= slcImage.getImagePointer()        
        self.setState()
        self.allocateArrays()
        doppler.doppler_Py(slcImagePt)
        self.getState()        
        self.deallocateArrays()        
        if(slcCreatedHere):
            slcImage.finalizeImage()
          
    def createSlcImage(self):
        # Check file name
        width = self.samples       
        from isceobj.Image.SlcImage import SlcImage
        objRaw = SlcImage()
        objRaw.initImage(self.slcFilename,'read','l',width)
        objRaw.createImage()                
        return objRaw
    
    def setState(self):
        # Set up the stuff needed for doppler        
        doppler.setLines_Py(self.lines)        
        doppler.setSamples_Py(self.samples)        
        doppler.setStartLine_Py(self.startLine)
        self.dim1_r_fd = int(self.samples)
        
    def getState(self):
        self.r_fd = doppler.get_r_fd_Py(self.dim1_r_fd)        
        
    def allocateArrays(self):
        if (self.dim1_r_fd == None):
            self.dim1_r_fd = len(self.r_fd)
            
        if (not self.dim1_r_fd):
            print("Error. Trying to allocate zero size array")

            raise Exception
        
        doppler.allocate_r_fd_Py(self.dim1_r_fd)
        
    def deallocateArrays(self):
        doppler.deallocate_r_fd_Py()
