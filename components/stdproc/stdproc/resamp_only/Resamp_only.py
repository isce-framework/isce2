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
from iscesys.Component.Component import Component, Port
from iscesys.Compatibility import Compatibility
from stdproc.stdproc.resamp_only import resamp_only

class Resamp_only(Component):

    def resamp_only(self, imageIntIn, imageIntOut, imageAmpIn, imageAmpOut):
        for port in self.inputPorts:
            port()

        if not (imageIntIn == None):
            self.imageIntIn = imageIntIn
        if not (imageAmpIn == None):
            self.imageAmpIn = imageAmpIn
        
        if (self.imageIntIn == None):
            self.logger.error("Input interferogram image not set.")
            raise Exception
        if (self.imageAmpIn == None):
            self.logger.error("Input amplitude image not set.")
            raise Exception
        
        if not (imageIntOut == None):
            self.imageIntOut = imageIntOut
        if not (imageAmpOut == None):
            self.imageAmpOut = imageAmpOut

        if (self.imageIntOut == None):
            self.logger.error("Output interferogram image not set.")
            raise Exception
        if (self.imageAmpOut == None):
            self.logger.error("Output amplitude image not set.")
            raise Exception
        
        self.setDefaults()
        #preallocate the two arrays that are returned
        self.azimuthCarrier = [0]*self.numberRangeBin
        self.rangeCarrier = [0]*self.numberRangeBin

        self.imageIntInAccessor = self.imageIntIn.getImagePointer()
        self.imageIntOutAccessor = self.imageIntOut.getImagePointer()
        self.imageAmpInAccessor = self.imageAmpIn.getImagePointer()
        self.imageAmpOutAccessor = self.imageAmpOut.getImagePointer()
        self.computeSecondLocation()
        self.allocateArrays()
        self.setState()
        resamp_only.resamp_only_Py(self.imageIntInAccessor,self.imageIntOutAccessor, self.imageAmpInAccessor, self.imageAmpOutAccessor)
        self.getState()
        self.deallocateArrays()
        self.imageIntOut.finalizeImage()
        self.imageAmpOut.finalizeImage()
        self.imageIntOut.renderHdr()
        self.imageAmpOut.renderHdr()

        return

    def setDefaults(self):
        if (self.numberLines == None):
            self.numberLines = self.imageIntIn.getLength()
            self.logger.warning('The variable NUMBER_LINES has been set to the default value %d which is the number of lines in the slc image.' % (self.numberLines)) 
       
        if (self.numberRangeBin == None):
            self.numberRangeBin = self.imageIntIn.getWidth()
            self.logger.warning('The variable NUMBER_RANGE_BIN has been set to the default value %d which is the width of the slc image.' % (self.numberRangeBin))

        if (self.numberFitCoefficients == None):
            self.numberFitCoefficients = 6
            self.logger.warning('The variable NUMBER_FIT_COEFFICIENTS has been set to the default value %s' % (self.numberFitCoefficients)) 
        
        if (self.firstLineOffset == None):
            self.firstLineOffset = 1
            self.logger.warning('The variable FIRST_LINE_OFFSET has been set to the default value %s' % (self.firstLineOffset)) 

    def computeSecondLocation(self):
#this part was previously done in the fortran code
        self.locationAcross2 = [0]*len(self.locationAcross1)
        self.locationAcrossOffset2 = [0]*len(self.locationAcross1)
        self.locationDown2 = [0]*len(self.locationAcross1)
        self.locationDownOffset2 = [0]*len(self.locationAcross1)
        self.snr2 = [0]*len(self.locationAcross1)
        for i in range(len(self.locationAcross1)):
            self.locationAcross2[i] = self.locationAcross1[i] + self.locationAcrossOffset1[i]
            self.locationAcrossOffset2[i] = self.locationAcrossOffset1[i]
            self.locationDown2[i] = self.locationDown1[i] + self.locationDownOffset1[i]
            self.locationDownOffset2[i] = self.locationDownOffset1[i]
            self.snr2[i] = self.snr1[i]

    def setState(self):
        resamp_only.setStdWriter_Py(int(self.stdWriter))
        resamp_only.setNumberFitCoefficients_Py(int(self.numberFitCoefficients))
        resamp_only.setNumberRangeBin_Py(int(self.numberRangeBin))
        resamp_only.setNumberLines_Py(int(self.numberLines))
        resamp_only.setFirstLineOffset_Py(int(self.firstLineOffset))
        resamp_only.setRadarWavelength_Py(float(self.radarWavelength))
        resamp_only.setSlantRangePixelSpacing_Py(float(self.slantRangePixelSpacing))
        resamp_only.setDopplerCentroidCoefficients_Py(self.dopplerCentroidCoefficients, self.dim1_dopplerCentroidCoefficients)
        resamp_only.setLocationAcross1_Py(self.locationAcross1, self.dim1_locationAcross1)
        resamp_only.setLocationAcrossOffset1_Py(self.locationAcrossOffset1, self.dim1_locationAcrossOffset1)
        resamp_only.setLocationDown1_Py(self.locationDown1, self.dim1_locationDown1)
        resamp_only.setLocationDownOffset1_Py(self.locationDownOffset1, self.dim1_locationDownOffset1)
        resamp_only.setSNR1_Py(self.snr1, self.dim1_snr1)
        resamp_only.setLocationAcross2_Py(self.locationAcross2, self.dim1_locationAcross2)
        resamp_only.setLocationAcrossOffset2_Py(self.locationAcrossOffset2, self.dim1_locationAcrossOffset2)
        resamp_only.setLocationDown2_Py(self.locationDown2, self.dim1_locationDown2)
        resamp_only.setLocationDownOffset2_Py(self.locationDownOffset2, self.dim1_locationDownOffset2)
        resamp_only.setSNR2_Py(self.snr2, self.dim1_snr2)

        return

    def setNumberFitCoefficients(self,var):
        self.numberFitCoefficients = int(var)
        return

    def setNumberRangeBin(self,var):
        self.numberRangeBin = int(var)
        return

    def setNumberLines(self,var):
        self.numberLines = int(var)
        return

    def setFirstLineOffset(self,var):
        self.firstLineOffset = int(var)
        return

    def setRadarWavelength(self,var):
        self.radarWavelength = float(var)
        return

    def setSlantRangePixelSpacing(self,var):
        self.slantRangePixelSpacing = float(var)
        return

    def setDopplerCentroidCoefficients(self,var):
        self.dopplerCentroidCoefficients = var
        return

    def setLocationAcross1(self,var):
        self.locationAcross1 = var
        return

    def setLocationAcrossOffset1(self,var):
        self.locationAcrossOffset1 = var
        return

    def setLocationDown1(self,var):
        self.locationDown1 = var
        return

    def setLocationDownOffset1(self,var):
        self.locationDownOffset1 = var
        return

    def setSNR1(self,var):
        self.snr1 = var
        return

    def setLocationAcross2(self,var):
        self.locationAcross2 = var
        return

    def setLocationAcrossOffset2(self,var):
        self.locationAcrossOffset2 = var
        return

    def setLocationDown2(self,var):
        self.locationDown2 = var
        return

    def setLocationDownOffset2(self,var):
        self.locationDownOffset2 = var
        return

    def setSNR2(self,var):
        self.snr2 = var
        return

    ## Not a getter
    def getState(self):
        self.azimuthCarrier = resamp_only.getAzimuthCarrier_Py(self.dim1_azimuthCarrier)
        self.rangeCarrier = resamp_only.getRangeCarrier_Py(self.dim1_rangeCarrier)

        return

    def getAzimuthCarrier(self):
        return self.azimuthCarrier

    def getRangeCarrier(self):
        return self.rangeCarrier

    def allocateArrays(self):
        if (self.dim1_dopplerCentroidCoefficients == None):
            self.dim1_dopplerCentroidCoefficients = len(self.dopplerCentroidCoefficients)

        if (not self.dim1_dopplerCentroidCoefficients):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_dopplerCoefficients_Py(self.dim1_dopplerCentroidCoefficients)

        if (self.dim1_locationAcross1 == None):
            self.dim1_locationAcross1 = len(self.locationAcross1)

        if (not self.dim1_locationAcross1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_r_ranpos_Py(self.dim1_locationAcross1)

        if (self.dim1_locationAcrossOffset1 == None):
            self.dim1_locationAcrossOffset1 = len(self.locationAcrossOffset1)

        if (not self.dim1_locationAcrossOffset1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_r_ranoff_Py(self.dim1_locationAcrossOffset1)

        if (self.dim1_locationDown1 == None):
            self.dim1_locationDown1 = len(self.locationDown1)

        if (not self.dim1_locationDown1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_r_azpos_Py(self.dim1_locationDown1)

        if (self.dim1_locationDownOffset1 == None):
            self.dim1_locationDownOffset1 = len(self.locationDownOffset1)

        if (not self.dim1_locationDownOffset1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_r_azoff_Py(self.dim1_locationDownOffset1)

        if (self.dim1_snr1 == None):
            self.dim1_snr1 = len(self.snr1)

        if (not self.dim1_snr1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_r_sig_Py(self.dim1_snr1)

        if (self.dim1_locationAcross2 == None):
            self.dim1_locationAcross2 = len(self.locationAcross2)

        if (not self.dim1_locationAcross2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_r_ranpos2_Py(self.dim1_locationAcross2)

        if (self.dim1_locationAcrossOffset2 == None):
            self.dim1_locationAcrossOffset2 = len(self.locationAcrossOffset2)

        if (not self.dim1_locationAcrossOffset2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_r_ranoff2_Py(self.dim1_locationAcrossOffset2)

        if (self.dim1_locationDown2 == None):
            self.dim1_locationDown2 = len(self.locationDown2)

        if (not self.dim1_locationDown2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_r_azpos2_Py(self.dim1_locationDown2)

        if (self.dim1_locationDownOffset2 == None):
            self.dim1_locationDownOffset2 = len(self.locationDownOffset2)

        if (not self.dim1_locationDownOffset2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_r_azoff2_Py(self.dim1_locationDownOffset2)

        if (self.dim1_snr2 == None):
            self.dim1_snr2 = len(self.snr2)

        if (not self.dim1_snr2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_r_sig2_Py(self.dim1_snr2)

        if (self.dim1_azimuthCarrier == None):
            self.dim1_azimuthCarrier = len(self.azimuthCarrier)

        if (not self.dim1_azimuthCarrier):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_azimuthCarrier_Py(self.dim1_azimuthCarrier)

        if (self.dim1_rangeCarrier == None):
            self.dim1_rangeCarrier = len(self.rangeCarrier)

        if (not self.dim1_rangeCarrier):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_only.allocate_rangeCarrier_Py(self.dim1_rangeCarrier)
        return

    def deallocateArrays(self):
        resamp_only.deallocate_dopplerCoefficients_Py()
        resamp_only.deallocate_r_ranpos_Py()
        resamp_only.deallocate_r_ranoff_Py()
        resamp_only.deallocate_r_azpos_Py()
        resamp_only.deallocate_r_azoff_Py()
        resamp_only.deallocate_r_sig_Py()
        resamp_only.deallocate_r_ranpos2_Py()
        resamp_only.deallocate_r_ranoff2_Py()
        resamp_only.deallocate_r_azpos2_Py()
        resamp_only.deallocate_r_azoff2_Py()
        resamp_only.deallocate_r_sig2_Py()
        resamp_only.deallocate_azimuthCarrier_Py()
        resamp_only.deallocate_rangeCarrier_Py()

        return

    def addInstrument(self):
        instrument = self._inputPorts.getPort('instrument').getObject()
        if(instrument):
            try:
                self.radarWavelength = instrument.getRadarWavelength()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError("Unable to wire instrument port")


    def addOffsets(self):
        offsets = self._inputPorts.getPort('offsets').getObject()
        if(offsets):
            try:
                for offset in offsets:
                    (across,down) = offset.getCoordinate()
                    (acrossOffset,downOffset) = offset.getOffset()
                    snr = offset.getSignalToNoise()
                    self.locationAcross1.append(across)
                    self.locationDown1.append(down)                
                    self.locationAcrossOffset1.append(acrossOffset)
                    self.locationDownOffset1.append(downOffset)
                    self.snr1.append(snr)
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError("Unable to wire Offset port")

    logging_name = 'isce.stdproc.resamp_only'

    def __init__(self):
        super(Resamp_only, self).__init__()
        self.numberFitCoefficients = None
        self.numberRangeBin = None
        self.numberLines = None
        self.firstLineOffset = None
        self.radarWavelength = None
        self.slantRangePixelSpacing = None
        self.dopplerCentroidCoefficients = []
        self.dim1_dopplerCentroidCoefficients = None
        self.locationAcross1 = []
        self.dim1_locationAcross1 = None
        self.locationAcrossOffset1 = []
        self.dim1_locationAcrossOffset1 = None
        self.locationDown1 = []
        self.dim1_locationDown1 = None
        self.locationDownOffset1 = []
        self.dim1_locationDownOffset1 = None
        self.snr1 = []
        self.dim1_snr1 = None
        self.locationAcross2 = []
        self.dim1_locationAcross2 = None
        self.locationAcrossOffset2 = []
        self.dim1_locationAcrossOffset2 = None
        self.locationDown2 = []
        self.dim1_locationDown2 = None
        self.locationDownOffset2 = []
        self.dim1_locationDownOffset2 = None
        self.snr2 = []
        self.dim1_snr2 = None
        self.azimuthCarrier = []
        self.dim1_azimuthCarrier = None
        self.rangeCarrier = []
        self.dim1_rangeCarrier = None
        self.dictionaryOfVariables = { 
            'NUMBER_FIT_COEFFICIENTS' : ['self.numberFitCoefficients', 'int','optional'], 
            'NUMBER_RANGE_BIN' : ['self.numberRangeBin', 'int','mandatory'], 
            'NUMBER_LINES' : ['self.numberLines', 'int','optional'], 
            'FIRST_LINE_OFFSET' : ['self.firstLineOffset', 'int','optional'], 
            'RADAR_WAVELENGTH' : ['self.radarWavelength', 'float','mandatory'], 
            'SLANT_RANGE_PIXEL_SPACING' : ['self.slantRangePixelSpacing', 'float','mandatory'], 
            'DOPPLER_CENTROID_COEFFICIENTS' : ['self.dopplerCentroidCoefficients', 'float','mandatory'], 
            'LOCATION_ACROSS1' : ['self.locationAcross1', 'float','mandatory'], 
            'LOCATION_ACROSS_OFFSET1' : ['self.locationAcrossOffset1', 'float','mandatory'], 
            'LOCATION_DOWN1' : ['self.locationDown1', 'float','mandatory'], 
            'LOCATION_DOWN_OFFSET1' : ['self.locationDownOffset1', 'float','mandatory'], 
            'SNR1' : ['self.snr1', 'float','mandatory'], 
            'LOCATION_ACROSS2' : ['self.locationAcross2', 'float','mandatory'], 
            'LOCATION_ACROSS_OFFSET2' : ['self.locationAcrossOffset2', 'float','mandatory'], 
            'LOCATION_DOWN2' : ['self.locationDown2', 'float','mandatory'], 
            'LOCATION_DOWN_OFFSET2' : ['self.locationDownOffset2', 'float','mandatory'], 
            'SNR2' : ['self.snr2', 'float','mandatory'] 
            }
        
        self.dictionaryOfOutputVariables = { 
            'AZIMUTH_CARRIER' : 'self.azimuthCarrier',
            'RANGE_CARRIER' : 'self.rangeCarrier' 
            }

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
        offsetPort = Port(name='offsets',method=self.addOffsets)
        instrumentPort = Port(name='instrument',method=self.addInstrument)
        self._inputPorts.add(offsetPort)
        self._inputPorts.add(instrumentPort)
        return None

    pass
