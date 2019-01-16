#!/usr/bin/env python3 

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





import sys
import os
import math
import numpy as np
import logging
from iscesys.Component.Component import Component,Port
from stdproc.stdproc.resamp_slc import resamp_slc
from isceobj.Util import combinedlibmodule as CL
import isceobj
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj.Util import Poly2D

class Resamp_slc(Component):

    interpolationMethods = { 'SINC' : 0,
                             'BILINEAR' : 1,
                             'BICUBIC'  : 2,
                             'NEAREST'  : 3,
                             'AKIMA'    : 4,
                             'BIQUINTIC': 5}

    def resamp_slc(self, imageIn=None, imageOut=None):
        for port in self.inputPorts:
            port()

        if imageIn is not None:
            self.imageIn = imageIn
        
        if self.imageIn is None:
            self.logger.error("Input slc image not set.")
            raise Exception


        if imageOut is not None:
            self.imageOut = imageOut


        if self.imageOut is None:
            self.logger.error("Output slc image not set.")
            raise Exception
        
        self.setDefaults()
        self.createImages()
        self.setState()
        resamp_slc.setRangeCarrier_Py(self.rangeCarrierAccessor)
        resamp_slc.setAzimuthCarrier_Py(self.azimuthCarrierAccessor)
        resamp_slc.setRangeOffsetsPoly_Py(self.rangeOffsetsAccessor)
        resamp_slc.setAzimuthOffsetsPoly_Py(self.azimuthOffsetsAccessor)
        resamp_slc.setDopplerPoly_Py(self.dopplerAccessor)
        resamp_slc.resamp_slc_Py(self.imageInAccessor,self.imageOutAccessor,self.residualAzimuthAccessor, self.residualRangeAccessor)
        self.destroyImages()

        return

    def createImages(self):
        if self.imageIn._accessor is None:
            self.imageIn.createImage()

        self.imageInAccessor = self.imageIn.getImagePointer()

        if self.imageOut._accessor is None:
            self.imageOut.createImage()

        self.imageOutAccessor = self.imageOut.getImagePointer()

        if self.rangeCarrierPoly is not None:
            self.rangeCarrierAccessor = self.rangeCarrierPoly.exportToC()
        else:
            print('No Range Carrier provided.')
            print('Assuming zero range carrier.')
            poly = Poly2D.Poly2D()
            poly.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.]])
            self.rangeCarrierAccessor = poly.exportToC()

        if self.azimuthCarrierPoly is not None:
            self.azimuthCarrierAccessor = self.azimuthCarrierPoly.exportToC()
        else:
            poly = Poly2D.Poly2D()
            poly.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.]])
            self.azimuthCarrierAccessor = poly.exportToC()

            print('No Azimuth Carrier provided.')
            print('Assuming zero azimuth carrier.')

        if self.rangeOffsetsPoly is not None:
            self.rangeOffsetsAccessor = self.rangeOffsetsPoly.exportToC()
        else:
            print('No range offset polynomial provided')
            poly = Poly2D.Poly2D()
            poly.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.]])
            self.rangeOffsetsAccessor = poly.exportToC()

        if self.azimuthOffsetsPoly is not None:
            self.azimuthOffsetsAccessor  = self.azimuthOffsetsPoly.exportToC()
        else:
            print('No azimuth offset polynomial provided')
            poly = Poly2D.Poly2D()
            poly.initPoly(rangeOrder=0, azimuthOrder=0, coeffs = [[0.]])
            self.azimuthOffsetsAccessor = poly.exportToC()

        if self.residualRangeImage is not None:
            if self.residualRangeImage._accessor is None:
                self.residualRangeImage.setCaster('read', 'DOUBLE')
                self.residualRangeImage.createImage()

            self.residualRangeAccessor = self.residualRangeImage.getImagePointer()
        else:
            self.residualRangeAccessor = 0

        if self.residualAzimuthImage is not None:
            if self.residualAzimuthImage._accessor is None:
                self.residualAzimuthImage.setCaster('read', 'DOUBLE')
                self.residualAzimuthImage.createImage()

            self.residualAzimuthAccessor = self.residualAzimuthImage.getImagePointer()
        else:
            self.residualAzimuthAccessor = 0

        if self.dopplerPoly is not None:
            self.dopplerAccessor = self.dopplerPoly.exportToC()
        else:
            print('No doppler polynomial provided')
            print('Assuming zero doppler centroid')
            poly = Poly2D.Poly2D()
            poly.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.]])
            self.dopplerAccessor = poly.exportToC()


    def destroyImages(self):
        CL.freeCPoly2D(self.rangeCarrierAccessor)
        CL.freeCPoly2D(self.azimuthCarrierAccessor)
        CL.freeCPoly2D(self.rangeOffsetsAccessor)
        CL.freeCPoly2D(self.azimuthOffsetsAccessor)
        CL.freeCPoly2D(self.dopplerAccessor)
        if self.residualRangeImage is not None:
            self.residualRangeImage.finalizeImage()

        if self.residualAzimuthImage is not None:
            self.residualAzimuthImage.finalizeImage()

        self.imageIn.finalizeImage()
        self.imageOut.finalizeImage()

        return

    def setDefaults(self):
        if self.inputLines is None:
            self.inputLines = self.imageIn.getLength()
            self.logger.warning('The variable INPUT_LINES has been set to the default value %d which is the number of lines in the slc image.' % (self.inputLines)) 
       
        if self.inputWidth is None:
            self.inputWidth = self.imageIn.getWidth()
            self.logger.warning('The variable INPUT_WIDTH has been set to the default value %d which is the width of the slc image.' % (self.inputWidth))

        if self.inputWidth != self.imageIn.getWidth():
            raise Exception('Width of input image {0} does not match specified width {1}'.format(self.imageIn.getWidth(), self.inputWidth))

        if self.startingRange is None:
            self.startingRange = 0.0

        if self.referenceStartingRange is None:
            self.referenceStartingRange = self.startingRange

        if self.referenceSlantRangePixelSpacing is None:
            self.referenceSlantRangePixelSpacing = self.slantRangePixelSpacing

        if self.referenceWavelength is None:
            self.referenceWavelength = self.radarWavelength
        
        if self.outputLines is None:
            self.outputLines = self.imageOut.getLength()
            self.logger.warning('The variable OUTPUT_LINES has been set to the default value %d which is the number of lines in the slc image.'%(self.outputLines))

        if self.outputWidth is None:
            self.outputWidth = self.imageOut.getWidth()
            self.logger.warning('The variable OUTPUT_WIDTH has been set to the default value %d which is the width of the slc image.'%(self.outputWidth))


        if (self.outputWidth != self.imageOut.getWidth()):
            raise Exception('Width of output image {0} does not match specified width {1}'.format(self.imageOut.getWidth(), self.outputWidth))

        if self.imageIn.dataType.upper().startswith('C'):
            self.isComplex = True
        else:
            self.isComplex = False

        
        if self.imageIn.getBands() > 1:
            raise Exception('The code currently is setup to resample single band images only')

        
        if self.method is None:
            if self.isComplex:
                self.method = 'SINC'
            else:
                self.method = 'BILINEAR'

        if self.flatten is None:
            self.logger.warning('No flattening requested')
            self.flatten = False

        return


    def setState(self):
        resamp_slc.setInputWidth_Py(int(self.inputWidth))
        resamp_slc.setInputLines_Py(int(self.inputLines))
        resamp_slc.setOutputWidth_Py(int(self.outputWidth))
        resamp_slc.setOutputLines_Py(int(self.outputLines))
        resamp_slc.setRadarWavelength_Py(float(self.radarWavelength))
        resamp_slc.setSlantRangePixelSpacing_Py(float(self.slantRangePixelSpacing))

        ###Introduced for dealing with data with different range sampling frequencies
        resamp_slc.setReferenceWavelength_Py(float(self.referenceWavelength))
        resamp_slc.setStartingRange_Py(float(self.startingRange))
        resamp_slc.setReferenceStartingRange_Py(float(self.referenceStartingRange))
        resamp_slc.setReferenceSlantRangePixelSpacing_Py(float(self.referenceSlantRangePixelSpacing))

        intpKey = self.interpolationMethods[self.method.upper()]
        resamp_slc.setMethod_Py(int(intpKey))
        resamp_slc.setIsComplex_Py(int(self.isComplex))
        resamp_slc.setFlatten_Py(int(self.flatten))

        return


    def setInputWidth(self,var):
        self.inputWidth = int(var)
        return

    def setInputLines(self, var):
        self.inputLines = int(var)
        return
    
    def setOutputWidth(self, var):
        self.outputWidth = int(var)
        return

    def setOutputLines(self,var):
        self.outputLines = int(var)
        return

    def setRadarWavelength(self,var):
        self.radarWavelength = float(var)
        return

    def setSlantRangePixelSpacing(self,var):
        self.slantRangePixelSpacing = float(var)
        return

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.stdproc.resamp_slc')
        return

    def addOffsets(self):
        from isceobj.Util.Poly2D import Poly2D
        offsets = self._inputPorts['offsets']
        if offsets:
            polys = offsets.getFitPolynomials()
            self.azimuthOffsetsPoly = polys[0]
            self.rangeOffsetsPoly = polys[1]

    def addSlc(self):
        from isceobj.Util import Poly2D
        from isceobj.Constants import SPEED_OF_LIGHT

        formslc = self._inputPorts['slc']
        if (formslc):

            ####Set up azimuth carrier information
            coeffs = []
            coeffs.append([2*np.pi*val for val in formslc.dopplerCentroidCoefficients])

            self.dopplerPoly = Poly2D.Poly2D()
            self.dopplerPoly.initPoly(rangeOrder=len(formslc.dopplerCentroidCoefficients)-1, azimuthOrder=0, coeffs=coeffs)
       
            ######Setup range carrier information
            delr = 0.5*SPEED_OF_LIGHT / formslc.rangeSamplingRate
            self.slantRangePixelSpacing = delr

            self.radarWavelength = formslc.radarWavelength

#            coeffs = [[0.0, -4 * np.pi * delr/self.radarWavelength]]
#            self.rangeCarrierPoly = Poly2D.Poly2D()
#            self.rangeCarrierPoly.initPoly(rangeOrder=1, azimuthOrder=0, coeffs=coeffs)
        
            img = isceobj.createImage()
            IU.copyAttributes(formslc.slcImage, img)
            img.setAccessMode('read')
            self.imageIn = img

    def addReferenceImage(self):
        refImg = self._inputPorts['reference']
        if (refImg):
            self.outputWidth = refImg.getWidth()
            self.outputLines = refImg.getLength()

    def __init__(self):
        Component.__init__(self)
        self.inputWidth = None
        self.inputLines = None
        self.outputWidth = None
        self.outputLines = None
        self.radarWavelength = None
        self.slantRangePixelSpacing = None
        self.azimuthOffsetsPoly = None
        self.azimuthOffsetsAccessor = None
        self.rangeOffsetsPoly = None
        self.rangeOffsetsAccessor = None
        self.rangeCarrierPoly = None
        self.rangeCarrierAccessor = None
        self.azimuthCarrierPoly = None
        self.azimuthCarrierAccessor = None
        self.residualRangeImage = None
        self.residualAzimuthImage = None
        self.residualRangeAccessor = None
        self.residualAzimuthAccessor = None
        self.dopplerPoly = None
        self.dopplerAccessor = None
        self.isComplex = None
        self.method = None
        self.flatten = None
        self.startingRange = None
        self.referenceWavelength = None
        self.referenceStartingRange = None
        self.referenceSlantRangePixelSpacing = None

        self.logger = logging.getLogger('isce.stdproc.resamp_slc')
       
        offsetPort = Port(name='offsets', method=self.addOffsets)
        slcPort = Port(name='slc', method=self.addSlc)
        referencePort = Port(name='reference', method=self.addReferenceImage)

        self._inputPorts.add(offsetPort)
        self._inputPorts.add(slcPort)
        self._inputPorts.add(referencePort)

        self.dictionaryOfVariables = { \
                                      'INPUT_WIDTH' : ['self.inputWidth', 'int','mandatory'], \
                                      'INPUT_LINES' : ['self.inputLines', 'int','optional'], \
                                      'OUTPUT_LINES' : ['self.outputLines', 'int', 'optional'], \
                                      'OUTPUT_WIDTH' : ['self.outputWidth', 'int', 'optional'], \
                                      'RADAR_WAVELENGTH' : ['self.radarWavelength', 'float','mandatory'], \
                                      'SLANT_RANGE_PIXEL_SPACING' : ['self.slantRangePixelSpacing', 'float','mandatory'], \
                                      }
        
        self.dictionaryOfOutputVariables = { }

        return





#end class




if __name__ == "__main__":
    sys.exit(main())
