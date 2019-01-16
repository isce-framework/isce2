#!/usr/bin/env python3

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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




from __future__ import print_function
import isce
import isceobj
import os
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from isceobj.Image.Image import Image
from stdproc.model.addsubmodel import addsubmodel

class AddSubModel(Component):
    '''
    Class for dealing with projecting data to LOS.
    Takes a 3-channel ENU file in meters and projects it to LOS in radians.
    '''

    def addsubmodel(self, inImage=None, modelImage = None, outImage=None):
        '''
        The driver.
        '''
        for port in self._inputPorts:
            port()

        if modelImage is not None:
            self.modelImage = modelImage
            self._modelFilename = modelImage.filename

        if self.modelImage is None:
            self.logger.error("Model Image is not set.")
            raise Exception

        if inImage is not None:
            self.inImage = inImage
            self._inputFilename = inImage.filename

        if self.inImage is None:
            self.logger.error("LOS Image is not set.")

        if outImage is not None:
            self.outImage = outImage
            self._outputFilename = outImage.filename

        self.setDefaults()
        self.createImages()


        self.setState()
        modelAccessor = self.modelImage.getImagePointer()
        inAccessor = self.inImage.getImagePointer()
        outAccessor = self.outImage.getImagePointer()

        if inImage.dataType.upper().startswith('C'):
            if modelImage.dataType.upper().startswith('C'):
                addsubmodel.cpxCpxProcess(self._ptr, inAccessor, modelAccessor, outAccessor)
            else:
                addsubmodel.cpxUnwProcess(self._ptr, inAccessor, modelAccessor, outAccessor)
        else:
            addsubmodel.unwUnwProcess(self._ptr, inAccessor, modelAccessor, outAccessor)

        self.getState()

        self.destroyImages()
        self.outImage.renderHdr()

    def setDefaults(self):
        '''
        Check if everything is properly wired.
        '''
        if self._outputFilename is None:
            self._outputFilename = self._inputFilename + '.corrected'

        if self._flip is None:
            self._flip = False

        if self._scaleFactor is None:
            self._scaleFactor = 1.0

        if self._width is None:
            self._width = self.inImage.width

        if self._numberLines is None:
            self._numberLines = self.inImage.length

        return

    def setState(self):
        '''
        Set the C++ class values.
        '''
        addsubmodel.setDims(self._ptr, self._width, self._numberLines)
        addsubmodel.setFlip(self._ptr, int(self._flip))
        addsubmodel.setScaleFactor(self._ptr, self._scaleFactor)
        return

    def getState(self):
        pass

    def createImages(self):
        '''
        Create output if its missing.
        '''
        self.inImage.createImage()

        if (self.outImage is None):
            self.outImage = createImage()
            accessMode = 'write'
            dataType = self.inImage.dataType
            bands = 1
            scheme = 'BIP'
            width = self._width
            self.outImage.init(self._outputFilename,accessMode,width,
                    dataType,bands=bands,scheme=scheme)
            self.outImage.createFile(self._numberLines)
        else:
            if (self.outImage.width != self._width):
                print('Output and Input images have different widths')
                raise Exception

            self.outImage.createImage()


        return

    def destroyImages(self):
        self.outImage.finalizeImage()
        self.modelImage.finalizeImage()
        self.inImage.finalizeImage()
        return

    def addModelImage(self):
        model = self._inputPorts['modelImage']
        if model:
            try:
                self._modelFilename = model.filename

            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

            if self._width is None:
                self._width = model.width
                self._numberLines = model.length
            else:
                if (self._width != model.width) or (self._numberLines != model.length):
                    raise ValueError('Model Image size mismatch')

            if model.bands != 1:
                raise ValueError('Image with 1 Band expected for input model. Got image with %d bands.'%(model.bands))
            self.modelImage = model

        return

    def addInputImage(self):
        inp = self._inputPorts['inputImage']
        if inp:
            try:
                self._inputFilename = inp.filename
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

            if self._width is None:
                self._width = inp.width
                self._numberLines = inp.length
            else:
                if (self._width != inp.width) or (self._numberLines != inp.length):
                    raise ValueError('Input Image size mismatch')

            if (inp.bands != 1):
                raise ValueError('Single band image expected for inputImage. Got %d band image'%(inp.bands))

            self.inImage = inp

        return


    def setModelFilename(self,var):
        self._modelFilename = str(var)
        return

    def setInputFilename(self,var):
        self._inputFilename = str(var)
        return

    def setOutputFilename(self,var):
        self._outputFilename = str(var)
        return

    def setNumberLines(self,var):
        self._numberLines = int(var)
        return

    def setWidth(self,var):
        self._width = int(var)
        return

    def setScaleFactor(self,var):
        self._scaleFactor = float(var)
        return

    def setFlip(self,var):
        self._flip = bool(var)
        return

    def getModelFilename(self):
        return self._modelFilename

    def getInputFilename(self):
        return self._inputFilename

    def getOutputFilename(self):
        return self._outputFilename

    def getNumberLines(self):
        return self._numberLines

    def getWidth(self):
        return self._width

    def getScaleFactor(self):
        return self._scaleFactor

    def getFlip(self):
        return self._flip

    def __init__(self):
        super(AddSubModel,self).__init__()
        self._inputFilename = ''
        self._outputFilename = ''
        self._modelFilename = ''
        self._scaleFactor = None
        self._flip = None
        self._width = None
        self._numberLines = None

        self._ptr = addsubmodel.createaddsubmodel()

        self.modelImage = None
        self.inImage = None
        self.outImage = None

        inputImagePort = Port(name='modelImage', method=self.addModelImage)
        self._inputPorts.add(inputImagePort)

        inImagePort = Port(name='inputImage', method=self.addInputImage)
        self._inputPorts.add(inImagePort)

        self.dictionaryOfVariables = { \
                'WIDTH' : ['width', 'int', 'mandatory'], \
                'INPUT' : ['inputFilename', 'str', 'mandatory'], \
                'OUTPUT' : ['outputFilename', 'str', 'mandatory'] }

        self.dictionaryOfOutputVariables = {}
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        self.initOptionalAndMandatoryLists()
        return

    def __del__(self):
        '''
        Destructor.
        '''
        addsubmodel.destroyaddsubmodel(self._ptr)
        pass


    inputFilename = property(getInputFilename,setInputFilename)
    outputFilename = property(getOutputFilename,setOutputFilename)
    modelFilename = property(getModelFilename,setModelFilename)
    numberLines = property(getNumberLines,setNumberLines)
    width = property(getWidth,setWidth)
    scaleFactor = property(getScaleFactor,setScaleFactor)
    flip = property(getFlip,setFlip)

    pass


if __name__ == '__main__':

    def load_pickle(step='correct'):
        import cPickle

        insarObj = cPickle.load(open('PICKLE/{0}'.format(step), 'rb'))
        return insarObj

    iobj = load_pickle()
    wid = iobj.topo.width
    lgt = iobj.topo.length

    print('Creating model object')
    objModel = isceobj.createImage()
    objModel.setFilename('topophase.mph')
    objModel.setWidth(wid)
    objModel.setLength(lgt)
    objModel.setAccessMode('read')
    objModel.dataType='CFLOAT'
    objModel.bands = 1
    objModel.createImage()

    print('Creating los object')
    objLos = isceobj.createImage()
    objLos.setFilename('resampOnlyImage.int')
    objLos.setAccessMode('read')
    objLos.bands = 1
    objLos.dataType = 'CFLOAT'
    objLos.setWidth(wid)
    objLos.setLength(lgt)
    objLos.createImage()

    print('Creating output object')
    objOut = isceobj.createImage()
    objOut.setFilename('model.rdr')
    objOut.setAccessMode('write')
    objOut.dataType = 'CFLOAT'
    objOut.setWidth(wid)
    objOut.createImage()


    model = AddSubModel()
    model.setWidth(wid)
    model.setNumberLines(lgt)
    model.setScaleFactor(1.0)
    model.setFlip(True)
    model.addsubmodel(modelImage=objModel, inImage=objLos, outImage=objOut)
