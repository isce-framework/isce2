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
import os
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from contrib.demUtils import upsampledem
from isceobj.Image import createDemImage
from iscesys.StdOEL.StdOELPy import create_writer
import sys

class UpsampleDem(Component):
    '''
    Component for upsampling DEMs.
    '''
    interpolationMethods = {'AKIMA' : 0,
                           'BIQUINTIC' : 1}

    #####   NOTE deltas are in arcsec
    def upsampledem(self, demImage = None, xFactor=None, yFactor=None):
        '''
        The driver.
        '''

        if demImage is not None:
            self.wireInputPort(name='demImage', object=demImage)

        for item in self._inputPorts:
            item()

        if xFactor is not None:
            self.setXFactor(xFactor)

        if yFactor is not None:
            self.setYFactor(yFactor)

        self.setDefaults()
        self.createImages()
        

        inAccessor = self._inImage.getImagePointer()
        outAccessor = self._outImage.getImagePointer()
        self.setState()

        intpKey = self.interpolationMethods[self.method.upper()]
        upsampledem.upsampledem_Py(inAccessor,outAccessor, intpKey)
        self._inImage.finalizeImage()
        self._outImage.finalizeImage()
        self.createXmlMetadata()
        return


    def createXmlMetadata(self):
        from isceobj.Image import createDemImage
        
        demImage = createDemImage()
        demImage.dataType = 'FLOAT'
        outname = self._outputFilename
        demImage.initImage(outname,'read',self._outWidth)
        length = demImage.getLength()
        deltaLon = self._deltaLongitude/(3600.0*self.xFactor) 
        deltaLat = self._deltaLatitude/(3600.0*self.yFactor)

        dictProp = {'Coordinate1':{'size':self._outWidth,'startingValue':self._startLongitude,'delta':deltaLon},'Coordinate2':{'size':length,'startingValue':self._startLatitude,'delta':deltaLat},'FILE_NAME':outname}
        if self.reference:
            dictProp['REFERENCE'] = self.reference
        #no need to pass the dictionaryOfFacilities since init will use the default one
        demImage.init(dictProp)
        demImage.renderHdr()
        self._image = demImage
        return

    def setDefaults(self):
        '''
        Set up default values.
        '''

        if (self._xFactor is None) and (self._yFactor is None):
            raise Exception('Oversampling factors not defined.')

        if self._xFactor and (self._yFactor is None):
            print('YFactor not defined. Set same as XFactor.')
            self._yFactor = self._xFactor

        if self.yFactor and (self.xFactor is None):
            print('XFactor not defined. Set same as YFactor.')
            self._xFactor = self._yFactor

        if (self._xFactor==1) and (self._yFactor==1):
            raise Exception('No oversampling requested.')

        if self._outputFilename is None:
            self._outputFilename = self._inputFilename+'_ovs_%d_%d'%(self._yFactor, self._xFactor)

        if (self._width is None) or (self._numberLines is None):
            raise Exception('Input Dimensions undefined.')

        self._outWidth = (self._width-1)*self._xFactor + 1
        self._outNumberLines = (self._numberLines-1)*self._yFactor + 1

        if self._stdWriter is None:
            self._stdWriter = create_writer("log", "", True, filename="upsampledem.log")

        if self.method is None:
            self.method = 'BIQUINTIC'
        else:
            if self.method.upper() not in list(self.interpolationMethods.keys()):
                raise Exception('Interpolation method must be one of ' + str(list(self.interpolationMethods.keys())))

        return


    def setState(self):
        upsampledem.setStdWriter_Py(int(self.stdWriter))
        upsampledem.setWidth_Py(int(self.width))
        upsampledem.setXFactor_Py(int(self.xFactor))
        upsampledem.setYFactor_Py(int(self.yFactor))
        upsampledem.setNumberLines_Py(int(self.numberLines))
        upsampledem.setPatchSize_Py(int(self.patchSize))

        return

    def createImages(self):
        #the fortran code use to read in short, convert to float and convert back to short.
        #let's use the image api and teh casters to do that
        if (self._inImage is None) or (self._inImage.dataType.upper() != 'FLOAT'):
            print('Creating input Image')
            inImage = createDemImage()
            if self._inType.upper() == 'SHORT':
                inImage.initImage(self.inputFilename,'read',self.width,'SHORT',caster='ShortToFloatCaster')
            elif self._inType.upper() == 'INT':
                inImage.initImage(self.inputFilename, 'read',self.width,'INT', caster='IntToFloatCaster')
            else:
                inImage.initImage(self.inputFilename, 'read', self.width, 'FLOAT')

            inImage.createImage()
            self._inImage = inImage
        else:
            if self._inImage.width != self.width:
                raise Exception('Input Image width inconsistency.')

            if self._inImage.length != self.numberLines:
                raise Exception('Input Image length inconsistency.')

            if self._inImage.getImagePointer() is None:
                self._inImage.createImage()

        if self._outImage is None:
            outImage = createDemImage()
            #manages float and writes out short
            outImage.initImage(self.outputFilename,'write',self._outWidth,'FLOAT')
            outImage.createImage()
            self._outImage = outImage
        else:
            if self._outImage.width != self._outWidth:
                raise Exception('Output Image width inconsistency.')

            if self._outImage.length != self._outNumberLines:
                raise Exception('Output Image length inconsistency.')
        
        return
    
    def setInputFilename(self,var):
        self._inputFilename = var

    def setOutputFilename(self,var):
        self._outputFilename = var
    
    def setWidth(self,var):
        self._width = int(var)
        return

    def setNumberLines(self,var):
        self._numberLines = int(var)
        return

    def setStartLatitude(self,var):
        self._startLatitude = float(var)
        return

    def setStartLongitude(self,var):
        self._startLongitude = float(var)
        return

    def setDeltaLatitude(self,var):
        self._deltaLatitude = float(var)
        return

    def setDeltaLongitude(self,var):
        self._deltaLongitude = float(var)
        return
    
    def setReference(self, var):
        self._reference = str(var)
        return

    def setPatchSize(self, var):
        self._patchSize = int(var)
        return

    def setXFactor(self, var):
        self._xFactor = int(var)
        return

    def setYFactor(self, var):
        self._yFactor = int(var)
        return

    def getInputFilename(self):
        return self._inputFilename 

    def getOutputFilename(self):
        return self._outputFilename 
    
    def getWidth(self):
        return self._width 

    def getNumberLines(self):
        return self._numberLines
        

    def getStartLatitude(self):
        return self._startLatitude 
         

    def getStartLongitude(self):
        return self._startLongitude 
         

    def getDeltaLatitude(self):
        return self._deltaLatitude 
         

    def getDeltaLongitude(self):
        return self._deltaLongitude

    def getReference(self):
        return self._reference

    def getPatchSize(self):
        return self._patchSize

    def getXFactor(self):
        return self._xFactor

    def getYFactor(self):
        return self._yFactor

    def addDemImage(self):
        dem = self._inputPorts['demImage']
        if dem:
            self._inImage = dem
            try:
                self._inputFilename = dem.filename
                self._width = dem.width
                self._inType = dem.dataType
                self._numberLines = dem.length
                self._startLongitude = dem.coord1.coordStart
                self._startLatitude = dem.coord2.coordStart
                self._deltaLongitude = dem.coord1.coordDelta*3600
                self._deltaLatitude = dem.coord2.coordDelta*3600
                self._reference = dem.reference
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError
     
        
    def __init__(self, stdWriter=None):
        super(UpsampleDem,self).__init__()
        self._inputFilename = ''
        #if not provided it assumes that we want to overwrite the input
        self._outputFilename = ''
        self._width = None
        self._inType = None
        self._outWidth = None
        self._xFactor = None
        self._yFactor = None
        self._numberLines = None
        self._outNumberLines = None
        self._patchSize = 64
        self._inImage = None
        self._outImage = None
        self._stdWriter = stdWriter
        self._startLongitude = None
        self._startLatitude = None
        self._deltaLatitude = None
        self._deltaLongitude = None
        self._reference = None
        demImagePort = Port(name='demImage', method=self.addDemImage)
        self.method = None

        self._inputPorts.add(demImagePort)
        self.dictionaryOfVariables = { 
                        'WIDTH' : ['width', 'int','mandatory'], 
                        'NUMBER_LINES' : ['numberLines','int','mandatory'], 
                        'INPUT_FILENAME' : ['inputFilename', 'str','mandatory'], 
                        'OUTPUT_FILENAME' : ['outputFilename', 'str','optional'], 
                        'XFACTOR' : ['xFactor', 'float','mandatory'], 
                        'YFACTOR' : ['yFactor', 'float','mandatory'], 
                        'START_LATITUDE' : ['startLatitude', 'float', 'mandatory'], 
                        'START_LONGITUDE' : ['startLongitude', 'float', 'mandatory'],
                        'DELTA_LONGITUDE' : ['deltaLongitude', 'float', 'mandatory'],
                        'DELTA_LATITUDE' : ['deltaLatitude', 'float', 'mandatory'],
                        'REFERENCE' : ['reference', 'str', 'mandatory']
                        }
        self.dictionaryOfOutputVariables = {}
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        self.initOptionalAndMandatoryLists()
        return

    inputFilename = property(getInputFilename,setInputFilename)
    outputFilename = property(getOutputFilename,setOutputFilename)
    width = property(getWidth,setWidth)
    numberLines = property(getNumberLines, setNumberLines)
    xFactor = property(getXFactor,setXFactor)
    yFactor = property(getYFactor,setYFactor)
    startLatitude = property(getStartLatitude, setStartLatitude)
    startLongitude = property(getStartLongitude, setStartLongitude)
    deltaLatitude = property(getDeltaLatitude, setDeltaLatitude)
    deltaLongitude = property(getDeltaLongitude, setDeltaLongitude)
    reference = property(getReference, setReference)
    patchSize = property(getPatchSize, setPatchSize)
    
    pass
