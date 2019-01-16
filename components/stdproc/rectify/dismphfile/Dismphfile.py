#!/usr/bin/env python3 

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
import sys
import os
import math
from iscesys.Component.Component import Component
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from stdproc.rectify.dismphfile import dismphfile
from isceobj.Image.StreamImage import StreamImage
class Dismphfile(Component):

    def dismphfile(self,imageIn = None,imageOut = None):
        if not (imageIn == None):
            self.imageIn = imageIn
        
        if (self.imageIn == None):
            print("Error. Input image is not set.")
            raise Exception

        self.imageIn.createImage()
        self.accessorIn = self.imageIn.getImagePointer()
        createdHere = False
        if(isinstance(imageOut,str)):
            self.createOutputImage(imageOut)
            createdHere = True
        elif not (imageOut == None):
            self.imageOut = imageOut
        
        if (self.imageOut == None):
            print("Error. Output image is not set.")
            raise Exception
        
        self.imageOut.createImage()
        self.accessorOut = self.imageOut.getImagePointer()

        self.setDefaults()
        self.setState()
        dismphfile.dismphfile_Py(self.accessorIn,self.accessorOut)
        if(createdHere):
            self.imageOut.finalizeImage()
        self.createKmlFile()
        
        return



    def createOutputImage(self,imageOut):

        accessmode = 'write'
        width = 1
        objImg = StreamImage()
        datatype = 'BYTE'
        endian = 'l' #does not matter since single byte data
        objImg.initImage(imageOut,accessmode,datatype,endian)
        # it actually creates the C++ object
        objImg.createImage()
        self.imageOut = objImg


    def setState(self):
        dismphfile.setLength_Py(int(self.length))
        dismphfile.setFirstLine_Py(int(self.firstLine))
        dismphfile.setNumberLines_Py(int(self.numberLines))
        dismphfile.setFlipFlag_Py(int(self.flipFlag))
        dismphfile.setScale_Py(float(self.scale))
        dismphfile.setExponent_Py(float(self.exponent))

        return

    def setDefaults(self):
        if(self.length == None):
            self.length = self.imageIn.getWidth()
        if(self.numberLines == None):
            self.numberLines = self.imageIn.getLength()
        if(self.scale == None):
            self.scale = 0.6
        if(self.exponent == None):
            self.exponent = 0.3
        if(self.flipFlag == None):
            self.flipFlag = 0
        if(self.firstLine == None):
            self.firstLine = 1
        if(self.title == ''):
            self.title = self.imageOut.getFilename().split('.')[0] #remove possible extension

        if(self.kmlFilename == ''):
            self.kmlFilename = self.imageOut.getFilename().split('.')[0] #remove possible extension
            self.kmlFilename += '.kml'

    def setInputImage(self,imageIn):
        self.imageIn = imageIn
    
    def setOutputImage(self,imageOut):
        self.imageOut = imageOut
    
    def setBoundingBox(self,bb):
        self.minLat = bb[0]
        self.maxLat = bb[1]
        self.minLon = bb[2]
        self.maxLon = bb[3]

    def setKmlFilename(self,filename):
        self.kmlFilename = filename
    
    def setDescription(self,description):
        self.description = description

    def setMinimumLatitude(self,minLat):
        self.minLat = minLat

    def setMinimumLongitude(self,minLon):
        self.minLon = minLon
    
    def setMaximumLatitude(self,maxLat):
        self.maxLat = maxLat

    def setMaximumLongitude(self,maxLon):
        self.maxLon = maxLon
   
    def setTitle(self,title):
        self.title = title

    def setFirstLine(self,var):
        self.firstLine = int(var)
        return
    
    def setNumbersLines(self,var):
        self.numberLines = int(var)
        return

    # use length only beacuse of the namenclature adopted in the fortran code
    def setWidth(self,var):
        self.lenght = int(var)
        return


    def setFlipFlag(self,var):
        self.flipFlag = int(var)
        return

    def setScale(self,var):
        self.scale = float(var)
        return

    def setExponent(self,var):
        self.exponent = float(var)
        return


    def createKmlFile(self):
        outname = self.imageOut.getFilename()
        f=open(self.kmlFilename,'w')
        f.write('<?xml version="1.0" encoding="UTF-8"?>'+"\n")
        f.write('<kml xmlns="http://earth.google.com/kml/2.2">'+"\n")
        f.write('<GroundOverlay>'+"\n")
        f.write('    <name>' + self.title + '</name>'+"\n")
        f.write('    <description>' +  self.description + '</description>'+"\n")
        f.write('    <Icon>'+"\n")
        f.write('          <href>' + outname + '</href>'+"\n")
        f.write('    </Icon>'+"\n")
        f.write('    <LatLonBox>'+"\n")
        f.write('        <north> '+str(self.minLat)+' </north>'+"\n")
        f.write('        <south> '+str(self.maxLat)+' </south>'+"\n")
        f.write('        <east> '+str(self.maxLon)+' </east>'+"\n")
        f.write('        <west> '+str(self.minLon)+' </west>'+"\n")
        f.write('    </LatLonBox>'+"\n")
        f.write('</GroundOverlay>'+"\n")
        f.write('</kml>'+"\n")
        f.close()




    def __init__(self):
        Component.__init__(self)
        self.accessorIn = None
        self.accessorOut = None
        self.length = None
        self.firstLine = None
        self.numberLines = None
        self.flipFlag = None
        self.scale = None
        self.exponent = None
        self.minLat = None
        self.maxLat = None
        self.minLon = None
        self.maxLon = None
        self.title = ''
        self.kmlFilename = ''
        self.description = ''
        self.dictionaryOfVariables = { \
                                      'LENGTH' : ['self.length', 'int','mandatory'], \
                                      'FIRST_LINE' : ['self.firstLine', 'int','optional'], \
                                      'NUMBER_LINES' : ['self.numberLines', 'int','optional'], \
                                      'FLIP_FLAG' : ['self.flipFlag', 'int','optional'], \
                                      'SCALE' : ['self.scale', 'float','optional'], \
                                      'EXPONENT' : ['self.exponent', 'float','optional'], \
                                      'MIN_LAT' : ['self.minLat', 'float','mandatory'], \
                                      'MAX_LAT' : ['self.maxLat', 'float','mandatory'], \
                                      'MIN_LON' : ['self.minLon', 'float','mandatory'], \
                                      'MAX_LON' : ['self.maxLon', 'float','mandatory'], \
                                      'TITLE' : ['self.title', 'str','optional'], \
                                      'KLM_FILENAME' : ['self.klmFilename', 'str','optional'], \
                                      'DESCRIPTION' : ['self.description', 'str','optional'], \
                                      }
        self.dictionaryOfOutputVariables = {}
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
        return





#end class




if __name__ == "__main__":
    sys.exit(main())
