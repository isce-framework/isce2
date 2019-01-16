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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
import sys
import logging
import os
import math
from iscesys.Component.Component import Component, Port
from iscesys.Compatibility import Compatibility
from isceobj.Doppler import calc_dop
import isceobj
from isceobj.Util.decorators import pickled, logged, port

HEADER = Component.Parameter(
    'header',
    public_name='HEADER',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


FIRST_LINE = Component.Parameter(
    'firstLine',
    public_name='FIRST_LINE',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


QOFFSET = Component.Parameter(
    'Qoffset',
    public_name='QOFFSET',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


WIDTH = Component.Parameter(
    'width',
    public_name='WIDTH',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


LAST_LINE = Component.Parameter(
    'lastLine',
    public_name='LAST_LINE',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


IOFFSET = Component.Parameter(
    'Ioffset',
    public_name='IOFFSET',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


RAW_FILENAME = Component.Parameter(
    'rawFilename',
    public_name='RAW_FILENAME',
    default='',
    type=str,
    mandatory=False,
    intent='input',
    doc=''
)


RNG_DOPPLER = Component.Parameter(
    'rngDoppler',
    public_name='RNG_DOPPLER',
    default=[],
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)


FD = Component.Parameter(
    'fd',
    public_name='FD',
    default=None,
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)

@pickled
class Calc_dop(Component):


    parameter_list = (
                      HEADER,
                      FIRST_LINE,
                      QOFFSET,
                      WIDTH,
                      LAST_LINE,
                      IOFFSET,
                      RAW_FILENAME
                     )



    logging_name = 'isceobj.Doppler.Calc_dop'
    

    
    family = 'calc_dop'
    @logged
    def __init__(self,family='',name=''):
        super(Calc_dop, self).__init__(family if family else  self.__class__.family, name=name)
        self.dim1_rngDoppler = None
        self.quadratic = {}   #insarapp
        self.coeff_list = None  #roiapp
        self.initOptionalAndMandatoryLists()
        self.createPorts()
        return None

    def createPorts(self):
        instrumentPort = Port(name="instrument",
                              method=self.addInstrument,
                              doc=(
                "An object that has getPulseRepetitionFrequency() and "+
                "getInPhaseValue() methods"
                ))
        framePort = Port(name="frame",
                         method=self.addFrame,
                         doc=(
                "An object that has getNumberOfSamples() and " +
                " etNumberOfLines() methods")
                         )
        imagePort = Port(name="image",
                         method=self.addImage,
                         doc=(
                "An object that has getXmin() and getXmax() methods"
                )
                         )
        self.inputPorts.add(instrumentPort)
        self.inputPorts.add(framePort)
        self.inputPorts.add(imagePort)
        return None

    def calculateDoppler(self, rawImage=None):
        self.activateInputPorts()

        rawCreatedHere = False
        if rawImage is None:
            self.rawImage = self.createRawImage()
            rawCreateHere = True
        else:
            self.rawImage = rawImage
            pass
        rawAccessor = self.rawImage.getImagePointer()
        self.setDefaults()
        self.rngDoppler = [0]*int((self.width - self.header)/2)
        self.allocateArrays()
        self.setState()
        calc_dop.calc_dop_Py(rawAccessor)
        self.getState()
        self.deallocateArrays()
        if rawCreatedHere:
            self.rawImage.finalizeImage()
            pass
        return None

    def createRawImage(self):
        # Check file name
        width = self.width        
        objRaw = isceobj.createRawImage()
        objRaw.initImage(self.rawFilename, 'read', width)
        objRaw.createImage()                
        return objRaw

    def fitDoppler(self):
#no fit is done. just keeping common interface with DopIQ
        self.quadratic['a'] = self.fd # for now use only zero order term 
        self.quadratic['b'] = 0  
        self.quadratic['c'] = 0
   
        self.coeff_list = [self.fd,0.,0.]
    def setDefaults(self):
        if self.firstLine is None:
            self.firstLine = 100
            self.logger.info('Variable  FIRST_LINE has been set  equal the defualt value %i' % (self.firstLine))
        if self.lastLine is None:
            self.lastLine = self.rawImage.getLength() - 200
            self.logger.info('Variable  LAST_LINE has been set  equal the default value imageLength - 200 = %i' % (self.lastLine))
        if self.header is None:
            self.header = 0
            self.logger.info('Variable  HEADER has been set  equal the default value %i' % (self.header))


    @port('__complex__')
    def addInstrument(self):
        z = complex(self.instrument)
        self.Ioffset, self.Qoffset = (z.real, z.imag)
        

    @port('numberOfLines')
    def addFrame(self):
        self.numberOfLines = self.frame.numberOfLines
        pass

    @port(None)
    def addImage(self):
        self.rawFilename = self.image.getFilename()
        self.header = self.image.getXmin()
        self.width = self.image.getXmax() - self.header
        return None


    def setState(self):
        calc_dop.setHeader_Py(int(self.header))
        calc_dop.setWidth_Py(int(self.width))
        calc_dop.setLastLine_Py(int(self.lastLine))
        calc_dop.setFirstLine_Py(int(self.firstLine))
        calc_dop.setIoffset_Py(float(self.Ioffset))
        calc_dop.setQoffset_Py(float(self.Qoffset))
        return None

    def setFilename(self, var):
        self.rawFilename = var

    def setHeader(self, var):
        self.header = int(var)
        return

    def setWidth(self, var):
        self.width = int(var)
        return

    def setLastLine(self, var):
        self.lastLine = int(var)
        return

    def setFirstLine(self, var):
        self.firstLine = int(var)
        return

    def setIoffset(self, var):
        self.Ioffset = float(var)
        return

    def setQoffset(self, var):
        self.Qoffset = float(var)
        return

    def getState(self):
        self.rngDoppler = calc_dop.getRngDoppler_Py(self.dim1_rngDoppler)
        self.fd = calc_dop.getDoppler_Py()
        return

    def getRngDoppler(self):
        return self.rngDoppler

    def getDoppler(self):
        return self.fd

    def allocateArrays(self):
        if self.dim1_rngDoppler is None:
            self.dim1_rngDoppler = len(self.rngDoppler)
            pass
        if not self.dim1_rngDoppler:
            print("Error. Trying to allocate zero size array")
            raise Exception

        calc_dop.allocate_rngDoppler_Py(self.dim1_rngDoppler)
        return 

    def deallocateArrays(self):
        calc_dop.deallocate_rngDoppler_Py()
        return 

    pass

        
                
        
                
        
                
        
                
