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
from contrib.demUtils import correct_geoid_i2_srtm
from isceobj.Image.Image import Image

CONVERSION_TYPE = Component.Parameter(
    '_conversionType',
    public_name='CONVERSION_TYPE',
    default=-1,
    type=int,
    mandatory=True,
    intent='input',
    doc='The reference conversion. If -1  EGM96 -> WGS84, if 1 WGS84 -> EGM96.'
)


DELTA_LATITUDE = Component.Parameter(
    '_deltaLatitude',
    public_name='DELTA_LATITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Latitude increment of the DEM.'
)


DELTA_LONGITUDE = Component.Parameter(
    '_deltaLongitude',
    public_name='DELTA_LONGITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Longitude increment of the DEM.'
)


GEOID_FILENAME = Component.Parameter(
    '_geoidFilename',
    public_name='GEOID_FILENAME',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'egm96geoid.dat'),
    type=str,
    mandatory=True,
    intent='input',
    doc='Geoid filename name used for the conversion. Default egm96geoid.dat.'
)


INPUT_FILENAME = Component.Parameter(
    '_inputFilename',
    public_name='INPUT_FILENAME',
    default='',
    type=str,
    mandatory=True,
    intent='input',
    doc='Name of the DEM file.'
)


OUTPUT_FILENAME = Component.Parameter(
    '_outputFilename',
    public_name='OUTPUT_FILENAME',
    default='',
    type=str,
    mandatory=False,
    intent='input',
    doc='Name of the output filename.'
)


OVERWRITE_INPUT_FILE_FLAG = Component.Parameter(
    '_overwriteInputFileFlag',
    public_name='OVERWRITE_INPUT_FILE_FLAG',
    default=False,
    type=str,
    mandatory=False,
    intent='input',
    doc='Flag that if set overwrites the input file with the result of the conversion.'
)


START_LATITUDE = Component.Parameter(
    '_startLatitude',
    public_name='START_LATITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Starting latitude.'
)


START_LONGITUDE = Component.Parameter(
    '_startLongitude',
    public_name='START_LONGITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Starting longitude.'
)


WIDTH = Component.Parameter(
    '_width',
    public_name='WIDTH',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Width of the DEM.'
)

NULL_IS_WATER = Component.Parameter(
        '_nullIsWater',
        public_name='NULL_IS_WATER',
        default=True,
        type = bool,
        mandatory = True,
        intent = 'input',
        doc = 'Treat null values as water/ invalid data')


class Correct_geoid_i2_srtm(Component):


    parameter_list = (
                      GEOID_FILENAME,
                      DELTA_LATITUDE,
                      START_LONGITUDE,
                      OVERWRITE_INPUT_FILE_FLAG,
                      DELTA_LONGITUDE,
                      START_LATITUDE,
                      CONVERSION_TYPE,
                      INPUT_FILENAME,
                      WIDTH,
                      OUTPUT_FILENAME,
                      NULL_IS_WATER
                     )


    
    ## This is how it is used, so I amde a call.
    def __call__(self, dem_image, ctype=-1, ext='.wgs84'):
        self.wireInputPort(name='demImage', object=dem_image)
        self.conversionType = ctype
        #make sure it write in the cwd and not from where the input file was read 
        self.outputFilename = self._outputFilename if self._outputFilename else os.path.basename(dem_image.filename) + ext
        self.correct_geoid_i2_srtm()
        self.createXmlMetadata()
        return self.image


    #####
    #####   NOTE deltas are in arcsec
    def correct_geoid_i2_srtm(self):
        for item in self._inputPorts:
            item()
        inImage,outImage = self.createImages()
        inAccessor = inImage.getImagePointer()
        outAccessor = outImage.getImagePointer()

        self.setState()
        correct_geoid_i2_srtm.correct_geoid_i2_srtm_Py(inAccessor,outAccessor)
        inImage.finalizeImage()
        outImage.finalizeImage()
        if(self.overwriteInputFileFlag):
            import shutil
            shutil.move(self.outputFilename, self.inputFilename)
            self.outputFilename = self.inputFilename
        return


    def createXmlMetadata(self):
        from isceobj.Image import createDemImage
        
        demImage = createDemImage()
        
        outname = self._outputFilename
        demImage.initImage(outname,'read',self._width,self._dataType)
        length = demImage.getLength()
        deltaLon = self._deltaLongitude
        deltaLat = self._deltaLatitude 

        dictProp = {'REFERENCE':self.reference,'Coordinate1': \
                   {'size':self._width,'startingValue':self._startLongitude,'delta':deltaLon}, \
                   'Coordinate2':{'size':length,'startingValue':self._startLatitude, \
                   'delta':deltaLat},'FILE_NAME':outname}
        #no need to pass the dictionaryOfFacilities since init will use the default one
        demImage.init(dictProp)
        demImage.renderHdr()
        demImage.metadatalocation = self.outputFilename + '.xml'

        self._image = demImage


    def setState(self):
        correct_geoid_i2_srtm.setStdWriter_Py(int(self.stdWriter))
        correct_geoid_i2_srtm.setWidth_Py(int(self.width))
        correct_geoid_i2_srtm.setStartLatitude_Py(float(self.startLatitude))
        correct_geoid_i2_srtm.setStartLongitude_Py(float(self.startLongitude))
        correct_geoid_i2_srtm.setDeltaLatitude_Py(float(self.deltaLatitude))
        correct_geoid_i2_srtm.setDeltaLongitude_Py(float(self.deltaLongitude))
        correct_geoid_i2_srtm.setNumberLines_Py(int(self._numberLines))
        correct_geoid_i2_srtm.setConversionType_Py(int(self.conversionType))
        correct_geoid_i2_srtm.setGeoidFilename_Py(self.geoidFilename)
        correct_geoid_i2_srtm.setNullIsWater_Py(int(self.nullIsWater))

        return

    def createImages(self):
        #the fortran code used to read in short, convert to float and convert back to short.
        #let's use the image api and the casters to do that. The image in input can be of any
        # comptible type
        inImage = self._dem.clone()
        #reads short and convert to float
        inImage.initImage(self.inputFilename,'read',self.width,self.dataType)
        #create a suitable caster from self.dataType to self._dataTypeBindings
        inImage.setCaster('read',self._dataTypeBindings)
        inImage.createImage()
        self._numberLines = inImage.getLength()
        outImage = Image()
        #if name not provided assume overwrite of input
        import random  
        if(not self.outputFilename):
            self.outputFilename = os.path.basename(self.inputFilename) + str(int(random.random()*100000)) #add 6 digit random number to input filename 
            self.overwriteInputFileFlag = True
        #manages float and writes out short
        outImage.initImage(self.outputFilename,'write',self.width,self.dataType)
        outImage.metadatalocation = self.outputFilename

        #create a suitable caster from self._dataTypeBindings to self.dataType
        outImage.setCaster('write',self._dataTypeBindings)
        outImage.createImage()
        return inImage,outImage
    
    def setInputFilename(self,var):
        self._inputFilename = var

    def setOutputFilename(self,var):
        self._outputFilename = var
    
    def setOverwriteInputFileFlag(self,var):
        self._overwriteInputFileFlag = var

    def setWidth(self,var):
        self._width = int(var)
        return
    
    def setDataType(self,var):
        self._dataType = var
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

    def setConversionType(self,var):
        self._conversionType = int(var)
        return

    def setGeoidFilename(self,var):
        self._geoidFilename = str(var)
        return

    def setReference(self,var):
        self._reference = var
        return

    def setNullIsWater(self, var):
        self._nullIsWater = var
        return

    def getInputFilename(self):
        return self._inputFilename 

    def getOutputFilename(self):
        return self._outputFilename 
    
    def getOverwriteInputFileFlag(self):
        return self._overwriteInputFileFlag 

    def getWidth(self):
        return self._width 
        
    def getDataType(self):
        return self._dataType 

    def getStartLatitude(self):
        return self._startLatitude 
         

    def getStartLongitude(self):
        return self._startLongitude 
         

    def getDeltaLatitude(self):
        return self._deltaLatitude 
         

    def getDeltaLongitude(self):
        return self._deltaLongitude 
         

    def getConversionType(self):
        return self._conversionType 
         

    def getGeoidFilename(self):
        return self._geoidFilename  
         
    def getImage(self):
        return self._image
    
    def getReference(self):
        if self._conversionType == -1:
            self._reference = 'WGS84'
        else:
            self._reference = 'EGM96'
        return self._reference

    def getNullIsWater(self):
        return self._nullIsWater
     
    def addDemImage(self):
        dem = self._inputPorts['demImage']
        if dem:
            try:
                self._dem = dem.clone()
                self._inputFilename = dem.filename
                self._width = dem.width
                self._dataType = dem.dataType
                self._startLongitude = dem.coord1.coordStart
                self._startLatitude = dem.coord2.coordStart
                self._deltaLongitude = dem.coord1.coordDelta
                self._deltaLatitude = dem.coord2.coordDelta
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError
     
        
    family = 'correct_geoid_i2_srtm'

    def __init__(self, stdWriter=None,family='',name=''):
        super(Correct_geoid_i2_srtm, self).__init__(family if family else  self.__class__.family, name=name)
        #if not provided it assumes that we want to overwrite the input
        self._numberLines = None
        self._image = None
        self._reference = None
        if(stdWriter):
            self._stdWriter = stdWriter
        else:
            from iscesys.StdOEL.StdOELPy import create_writer
            self._stdWriter = create_writer("log", "", True, filename="insar.log")
        self._dataType = None
        self._dem = None
        self._dataTypeBindings = 'FLOAT'
        demImagePort = Port(name='demImage', method=self.addDemImage)

        self._inputPorts.add(demImagePort)
        #make sure that the .dat file is in the executing directory
        self.initOptionalAndMandatoryLists()
        return

    reference = property(getReference,setReference)
    image = property(getImage)
    inputFilename = property(getInputFilename,setInputFilename)
    outputFilename = property(getOutputFilename,setOutputFilename)
    overwriteInputFileFlag = property(getOverwriteInputFileFlag,setOverwriteInputFileFlag)
    width = property(getWidth,setWidth)  
    dataType = property(getDataType,setDataType)  
    startLatitude = property(getStartLatitude,setStartLatitude)
    startLongitude = property(getStartLongitude,setStartLongitude)
    deltaLatitude = property(getDeltaLatitude,setDeltaLatitude)
    deltaLongitude = property(getDeltaLongitude,setDeltaLongitude)
    conversionType = property(getConversionType,setConversionType)
    geoidFilename = property(getGeoidFilename,setGeoidFilename)
    nullIsWater = property(getNullIsWater, setNullIsWater)

    pass
