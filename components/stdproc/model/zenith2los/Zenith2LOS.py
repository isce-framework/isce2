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
from stdproc.model.zenith2los import zenith2los

class Zenith2LOS(Component):
    '''
    Class for dealing with projecting data to LOS.
    Takes a 1-channel zenith delay file in meters and projects it to LOS in radians.
    '''

    def zenith2los(self, modelImage = None, latImage=None, lonImage=None, losImage=None, outImage=None):
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

        if losImage is not None:
            self.losImage = losImage

        if self.losImage is None:
            self.logger.error("LOS Image is not set.")

        if lonImage is not None:
            self.lonImage = lonImage

        if latImage is not None:
            self.latImage = latImage

        if outImage is not None:
            self.outImage = outImage
            self._outputFilename = outImage.filename

        self.setDefaults()
        self.createImages()

        modelAccessor = self.modelImage.getImagePointer()

        if (self.lonImage is not None) and (self.latImage is not None):
            lonAccessor = self.lonImage.getImagePointer()
            latAccessor = self.latImage.getImagePointer()
        else:
            lonAccessor = 0
            latAccessor = 0
    
        losAccessor = self.losImage.getImagePointer()
        outAccessor = self.outImage.getImagePointer()

        self.setState()

        zenith2los.zenith2los(self._ptr, modelAccessor,latAccessor,lonAccessor,losAccessor,outAccessor)

        self.getState()

        self.destroyImages()
        self.outImage.renderHdr()

    def setDefaults(self):
        '''
        Check if everything is properly wired.
        '''

        if self._outputFilename is None:
            self._outputFilename = self._modelFilename + '.los'

        if (self.lonImage is None) or (self.latImage is None):
            self._width = self._geowidth
            self._numberLines = self._geoNumberLines
        else:
            if (self.lonImage.width != self.latImage.width):
                print('Lon and Lat Images have different widths')
                raise Exception

            if (self.lonImage.length != self.lonImage.length):
                print('Lon and Lat Images have different lengths')
                raise Exception

        if (self.losImage.width != self._width):
            print('LOS and output images have different widths')
            raise Exception

        if (self.losImage.length < self._numberLines):
            print('LOS and output images have different lengths')
            raise Exception

        if self._scaleFactor is None:
            self._scaleFactor = 1.0

        #####To return LOS in meters
        if self._wavelength is None:
            print('Wavelength not set')
            raise Exception

        if (self._startLatitude is None) or (self._deltaLatitude is None):
            print('Latitude information incomplete.')
            raise Exception

        if (self._startLongitude is None) or (self._deltaLongitude is None):
            print('Longitude information incomplete.')
            raise Exception

        return

    def setState(self):
        '''
        Set the C++ class values.
        '''
        zenith2los.setDims(self._ptr, self._width, self._numberLines)
        zenith2los.setGeoDims(self._ptr, self._geoWidth, self._geoNumberLines)
        zenith2los.setWavelength(self._ptr, self._wavelength)
        zenith2los.setScaleFactor(self._ptr, self._scaleFactor)
        zenith2los.setLatitudeInfo(self._ptr, self._startLatitude, self._deltaLatitude)
        zenith2los.setLongitudeInfo(self._ptr, self._startLongitude, self._deltaLongitude)
        return

    def getState(self):
        pass

    def createImages(self):
        '''
        Create output if its missing.
        '''
        if (self.outImage is None):
            self.outImage = createImage()
            accessMode = 'write'
            dataType = 'FLOAT'
            bands = 1
            scheme = 'BIP'
            width = self._width
            self.outImage.init(self._outputFilename,accessMode,width,
                    dataType,bands=bands,scheme=scheme)
            self.outImage.createFile(self._numberLines)
        else:
            if (self.outImage.width != self._width):
                print('Output and LOS images have different widths')
                raise Exception

            self.outImage.createImage()

        self.losImage.createImage()

        if (self.lonImage  is not None) and (self.latImage is not None):
            self.lonImage.createImage()
            self.latImage.createImage()

        return

    def destroyImages(self):
        self.outImage.finalizeImage()
        self.modelImage.finalizeImage()
        self.losImage.finalizeImage()

        if self.lonImage:
            self.lonImage.finalizeImage()

        if self.latImage:
            self.latImage.finalizeImage()
    

    def addModelImage(self):
        model = self._inputPorts['modelImage']
        if model:
            try:
                self._modelFilename = model.filename
                self._outputFilename = self._modelFilename + '.los'
                self._geoWidth = model.width
                self._geoNumberLines = model.length
                self._startLongitude = model.coord1.coordStart
                self._startLatitude = model.coord2.coordStart
                self._deltaLongitude = model.coord1.coordDelta
                self._deltaLatitude = model.coord2.coordDelta

            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

            if model.bands != 3:
                raise ValueError('Image with 3 Bands expected for input model. Got image with %d bands.'%(model.bands))
            self.modelImage = model

        return

    def addLatImage(self):
        lat = self._inputPorts['latImage']
        if lat:
            if self._width is None:
                self._width = lat.width
                self._numberLines = lat.length
            else:
                if (self._width != lat.width) or (self._numberLines != lat.length):
                    raise ValueError('Input Lat Image size mismatch')

            if (lat.bands != 1):
                raise ValueError('Single band image expected for Lat. Got %d band image'%(lat.bands))

            self.latImage = lat

        return

    def addLonImage(self):
        lon = self._inputPorts['lonImage']
        if lon:
            if self._width is None:
                self._width = lon.width
                self._numberLines = lon.length
            else:
                if (self._width != lon.width) or (self._numberLines != lon.length):
                    raise ValueError('Input Lon Image size mismatch')

            if (lon.bands != 1):
                raise ValueError('Single band image expected for Lon. Got %d band image'%(lon.bands))

            self.lonImage = lon

        return

    def addLosImage(self):
        los = self._inputPorts['losImage']
        if los:
            if self._width is None:
                self._width = los.width
                self._numberLines = los.length
            else:
                if (self._width != los.width) or (self._numberLines != los.length):
                    raise ValueError('Input Lon Image size mismatch')

            if (los.bands != 2):
                raise ValueError('Single band image expected for Lon. Got %d band image'%(los.bands))

            self.losImage = los

        return



    def setInputFilename(self,var):
        self._modelFilename = str(var)
        return

    def setOutputFilename(self,var):
        self._outputFilename = str(var)
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

    def setNumberLines(self,var):
        self._numberLines = int(var)
        return

    def setWidth(self,var):
        self._width = int(var)
        return

    def setGeoNumberLines(self,var):
        self._geoNumberLines = int(var)
        return

    def setGeoWidth(self,var):
        self._geoWidth = int(var)
        return

    def setScaleFactor(self,var):
        self._scaleFactor = float(var)
        return

    def setWavelength(self,var):
        self._wavelength = float(var)
        return

    def getInputFilename(self):
        return self._modelFilename

    def getOutputFilename(self):
        return self._outputFilename

    def getStartLatitude(self):
        return self._startLatitude

    def getStartLongitude(self):
        return self._startLongitude

    def getDeltaLatitude(self):
        return self._deltaLatitude

    def getDeltaLongitude(self):
        return self._deltaLongitude

    def getNumberLines(self):
        return self._numberLines

    def getWidth(self):
        return self._width

    def getScaleFactor(self):
        return self._scaleFactor

    def getWavelength(self):
        return self._wavelength

    def __init__(self):
        super(Zenith2LOS,self).__init__()
        self._modelFilename = ''
        self._outputFilename = ''
        self._startLatitude = None
        self._startLongitude = None
        self._deltaLatitude = None
        self._deltaLongitude = None
        self._geoNumberLines = None
        self._scaleFactor = None
        self._geoWidth = None
        self._wavelength = None
        self._width = None
        self._numberLines = None

        self._ptr = zenith2los.createZenith2LOS()

        self.modelImage = None
        self.latImage = None
        self.lonImage = None
        self.losImage = None
        self.outImage = None

        inputImagePort = Port(name='modelImage', method=self.addModelImage)
        self._inputPorts.add(inputImagePort)

        latImagePort = Port(name='latImage', method=self.addLatImage)
        self._inputPorts.add(latImagePort)

        lonImagePort = Port(name='lonImage', method=self.addLonImage)
        self._inputPorts.add(lonImagePort)

        losImagePort = Port(name='losImage', method=self.addLosImage)
        self._inputPorts.add(losImagePort)

        self.dictionaryOfVariables = { \
                'WIDTH' : ['width', 'int', 'mandatory'], \
                'INPUT' : ['modelFilename', 'str', 'mandatory'], \
                'OUTPUT' : ['outputFilename', 'str', 'mandatory'], \
                'START_LATITUDE' : ['startLatitude', 'float', 'mandatory'], \
                'START_LONGITUDE' : ['startLongitude', 'float', 'mandatory'], \
                'DELTA_LONGITUDE' : ['deltaLongitude', 'float', 'mandatory']}

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
        zenith2los.destroyZenith2LOS(self._ptr)
        pass


    modelFilename = property(getInputFilename,setInputFilename)
    outputFilename = property(getOutputFilename,setOutputFilename)
    startLatitude = property(getStartLatitude,setStartLatitude)
    startLongitude = property(getStartLongitude,setStartLongitude)
    deltaLatitude = property(getDeltaLatitude,setDeltaLatitude)
    deltaLongitude = property(getDeltaLongitude,setDeltaLongitude)
    numberLines = property(getNumberLines,setNumberLines)
    width = property(getWidth,setWidth)
    scaleFactor = property(getScaleFactor,setScaleFactor)
    wavelength = property(getWavelength,setWavelength)

    pass


if __name__ == '__main__':
    
    import numpy as np

    def load_pickle(step='correct'):
        import cPickle

        insarObj = cPickle.load(open('PICKLE/{0}'.format(step), 'rb'))
        return insarObj

    wid = 401
    lgt = 401
    data = np.zeros((lgt,wid), dtype=np.float32)
    data[:,:] = 1.0

    data.tofile('model.enu')

    print('Creating model object')
    objModel = isceobj.createImage()
    objModel.setFilename('model.enu')
    objModel.setWidth(wid)
    objModel.setAccessMode('read')
    objModel.dataType='FLOAT'
    objModel.bands = 1
    objModel.createImage()


    ####insarApp related
    iobj = load_pickle()
    topo = iobj.getTopo()

    startLat = topo.maximumLatitude + 0.5
    startLon = topo.minimumLatitude - 0.5
    deltaLat = (topo.minimumLatitude - topo.maximumLatitude-1.0)/(1.0*lgt)
    deltaLon = (topo.maximumLongitude - topo.minimumLongitude + 1.0) / (1.0*wid)

    print('Creating lat object')
    objLat = isceobj.createImage()
    objLat.setFilename(topo.latFilename)
    objLat.setAccessMode('read')
    objLat.dataType = 'FLOAT'
    objLat.setWidth(topo.width)
    objLat.createImage()

    print('Creating lon object')
    objLon = isceobj.createImage()
    objLon.setFilename(topo.lonFilename)
    objLon.setAccessMode('read')
    objLon.dataType = 'FLOAT'
    objLon.setWidth(topo.width)
    objLon.createImage()

    print('Creating los object')
    objLos = isceobj.createImage()
    objLos.setFilename('los.rdr')
    objLos.setAccessMode('read')
    objLos.bands = 2
    objLos.scheme = 'BIL'
    objLos.dataType = 'FLOAT'
    objLos.setWidth(topo.width)
    objLos.createImage()

    print('Creating output object')
    objOut = isceobj.createImage()
    objOut.setFilename('model.rdr')
    objOut.setAccessMode('write')
    objOut.dataType = 'FLOAT'
    objOut.setWidth(topo.width)
    objOut.createImage()


    model = Zenith2LOS()
    model.setWidth(topo.width)
    model.setNumberLines(topo.length)
    model.setGeoWidth(wid)
    model.setGeoNumberLines(lgt)
    model.setStartLatitude(startLat)
    model.setDeltaLatitude(deltaLat)
    model.setStartLongitude(startLon)
    model.setDeltaLongitude(deltaLon)
    model.setScaleFactor(1.0)
    model.setWavelength(4*np.pi)
    model.zenith2los(modelImage=objModel, latImage=objLat, lonImage=objLon, losImage=objLos, outImage=objOut)
