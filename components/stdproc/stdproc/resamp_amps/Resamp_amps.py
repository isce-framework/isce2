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
from iscesys.Component.Component import Component
from iscesys.Compatibility import Compatibility
from stdproc.stdproc.resamp_amps import resamp_amps

class Resamp_amps(Component):

    def resamp_amps(self,imageIn,imageOut):
        if not (imageIn == None):
            self.imageIn = imageIn
        
        if (self.imageIn == None):
            self.logger.error("Input slc image not set.")
            raise Exception
        if not (imageOut == None):
            self.imageOut = imageOut
        if (self.imageOut == None):
            self.logger.error("Output slc image not set.")
            raise Exception
        self.setDefaults()
        self.imageInAccessor = self.imageIn.getLineAccessorPointer()
        self.imageOutAccessor = self.imageOut.getLineAccessorPointer()
        self.computeSecondLocation()    
        self.allocateArrays()
        self.setState()
        resamp_amps.resamp_amps_Py(self.imageInAccessor,self.imageOutAccessor)
        self.getState()
        self.deallocateArrays()

        return


    def setDefaults(self):
        if (self.numberLines == None):
            self.numberLines = self.image1.getFileLength()
            self.logger.warning('The variable NUMBER_LINES has been set to the default value %d which is the number of lines in the slc image.'% (self.numberLines)) 
        
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
        resamp_amps.setNumberFitCoefficients_Py(int(self.numberFitCoefficients))
        resamp_amps.setNumberRangeBin_Py(int(self.numberRangeBin))
        resamp_amps.setNumberLines_Py(int(self.numberLines))
        resamp_amps.setFirstLineOffset_Py(int(self.firstLineOffset))
        resamp_amps.setRadarWavelength_Py(float(self.radarWavelength))
        resamp_amps.setSlantRangePixelSpacing_Py(float(self.slantRangePixelSpacing))
        resamp_amps.setDopplerCentroidCoefficients_Py(self.dopplerCentroidCoefficients, self.dim1_dopplerCentroidCoefficients)
        resamp_amps.setLocationAcross1_Py(self.locationAcross1, self.dim1_locationAcross1)
        resamp_amps.setLocationAcrossOffset1_Py(self.locationAcrossOffset1, self.dim1_locationAcrossOffset1)
        resamp_amps.setLocationDown1_Py(self.locationDown1, self.dim1_locationDown1)
        resamp_amps.setLocationDownOffset1_Py(self.locationDownOffset1, self.dim1_locationDownOffset1)
        resamp_amps.setSNR1_Py(self.snr1, self.dim1_snr1)
        resamp_amps.setLocationAcross2_Py(self.locationAcross2, self.dim1_locationAcross2)
        resamp_amps.setLocationAcrossOffset2_Py(self.locationAcrossOffset2, self.dim1_locationAcrossOffset2)
        resamp_amps.setLocationDown2_Py(self.locationDown2, self.dim1_locationDown2)
        resamp_amps.setLocationDownOffset2_Py(self.locationDownOffset2, self.dim1_locationDownOffset2)
        resamp_amps.setSNR2_Py(self.snr2, self.dim1_snr2)

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

    def getState(self):
        self.ULRangeOffset = resamp_amps.getULRangeOffset_Py()
        self.ULAzimuthOffset = resamp_amps.getULAzimuthOffset_Py()
        self.URRangeOffset = resamp_amps.getURRangeOffset_Py()
        self.URAzimuthOffset = resamp_amps.getURAzimuthOffset_Py()
        self.LLRangeOffset = resamp_amps.getLLRangeOffset_Py()
        self.LLAzimuthOffset = resamp_amps.getLLAzimuthOffset_Py()
        self.LRRangeOffset = resamp_amps.getLRRangeOffset_Py()
        self.LRAzimuthOffset = resamp_amps.getLRAzimuthOffset_Py()
        self.CenterRangeOffset = resamp_amps.getCenterRangeOffset_Py()
        self.CenterAzimuthOffset = resamp_amps.getCenterAzimuthOffset_Py()

        return





    def getULRangeOffset(self):
        return self.ULRangeOffset

    def getULAzimuthOffset(self):
        return self.ULAzimuthOffset

    def getURRangeOffset(self):
        return self.URRangeOffset

    def getURAzimuthOffset(self):
        return self.URAzimuthOffset

    def getLLRangeOffset(self):
        return self.LLRangeOffset

    def getLLAzimuthOffset(self):
        return self.LLAzimuthOffset

    def getLRRangeOffset(self):
        return self.LRRangeOffset

    def getLRAzimuthOffset(self):
        return self.LRAzimuthOffset

    def getCenterRangeOffset(self):
        return self.CenterRangeOffset

    def getCenterAzimuthOffset(self):
        return self.CenterAzimuthOffset






    def allocateArrays(self):
        if (self.dim1_dopplerCentroidCoefficients == None):
            self.dim1_dopplerCentroidCoefficients = len(self.dopplerCentroidCoefficients)

        if (not self.dim1_dopplerCentroidCoefficients):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_amps.allocate_dopplerCoefficients_Py(self.dim1_dopplerCentroidCoefficients)

        if (self.dim1_locationAcross1 == None):
            self.dim1_locationAcross1 = len(self.locationAcross1)

        if (not self.dim1_locationAcross1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_amps.allocate_r_ranpos_Py(self.dim1_locationAcross1)

        if (self.dim1_locationAcrossOffset1 == None):
            self.dim1_locationAcrossOffset1 = len(self.locationAcrossOffset1)

        if (not self.dim1_locationAcrossOffset1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_amps.allocate_r_ranoff_Py(self.dim1_locationAcrossOffset1)

        if (self.dim1_locationDown1 == None):
            self.dim1_locationDown1 = len(self.locationDown1)

        if (not self.dim1_locationDown1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_amps.allocate_r_azpos_Py(self.dim1_locationDown1)

        if (self.dim1_locationDownOffset1 == None):
            self.dim1_locationDownOffset1 = len(self.locationDownOffset1)

        if (not self.dim1_locationDownOffset1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_amps.allocate_r_azoff_Py(self.dim1_locationDownOffset1)

        if (self.dim1_snr1 == None):
            self.dim1_snr1 = len(self.snr1)

        if (not self.dim1_snr1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_amps.allocate_r_sig_Py(self.dim1_snr1)

        if (self.dim1_locationAcross2 == None):
            self.dim1_locationAcross2 = len(self.locationAcross2)

        if (not self.dim1_locationAcross2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_amps.allocate_r_ranpos2_Py(self.dim1_locationAcross2)

        if (self.dim1_locationAcrossOffset2 == None):
            self.dim1_locationAcrossOffset2 = len(self.locationAcrossOffset2)

        if (not self.dim1_locationAcrossOffset2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_amps.allocate_r_ranoff2_Py(self.dim1_locationAcrossOffset2)

        if (self.dim1_locationDown2 == None):
            self.dim1_locationDown2 = len(self.locationDown2)

        if (not self.dim1_locationDown2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_amps.allocate_r_azpos2_Py(self.dim1_locationDown2)

        if (self.dim1_locationDownOffset2 == None):
            self.dim1_locationDownOffset2 = len(self.locationDownOffset2)

        if (not self.dim1_locationDownOffset2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_amps.allocate_r_azoff2_Py(self.dim1_locationDownOffset2)

        if (self.dim1_snr2 == None):
            self.dim1_snr2 = len(self.snr2)

        if (not self.dim1_snr2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_amps.allocate_r_sig2_Py(self.dim1_snr2)


        return





    def deallocateArrays(self):
        resamp_amps.deallocate_dopplerCoefficients_Py()
        resamp_amps.deallocate_r_ranpos_Py()
        resamp_amps.deallocate_r_ranoff_Py()
        resamp_amps.deallocate_r_azpos_Py()
        resamp_amps.deallocate_r_azoff_Py()
        resamp_amps.deallocate_r_sig_Py()
        resamp_amps.deallocate_r_ranpos2_Py()
        resamp_amps.deallocate_r_ranoff2_Py()
        resamp_amps.deallocate_r_azpos2_Py()
        resamp_amps.deallocate_r_azoff2_Py()
        resamp_amps.deallocate_r_sig2_Py()
        return None

    logging_name = 'isce.stdproc.resamp_amps'

    def __init__(self):
        super(Resamp_amps, self).__init__()
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
        self.ULRangeOffset = None
        self.ULAzimuthOffset = None
        self.URRangeOffset = None
        self.URAzimuthOffset = None
        self.LLRangeOffset = None
        self.LLAzimuthOffset = None
        self.LRRangeOffset = None
        self.LRAzimuthOffset = None
        self.CenterRangeOffset = None
        self.CenterAzimuthOffset = None
#        self.logger = logging.getLogger('isce.stdproc.resamp_amps')
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
            'UL_RANGE_OFFSET': 'self.ULRangeOffset',
            'UL_AZIMUTH_OFFSET' : 'self.ULAzimuthOffset',
            'UR_RANGE_OFFSET' : 'self.URRangeOffset' ,
            'UR_AZIMUTH_OFFSET' : 'self.URAzimuthOffset', 
            'LL_RANGE_OFFSET' : 'self.LLRangeOffset', 
            'LL_AZIMUTH_OFFSET' : 'self.LLAzimuthOffset', 
            'LR_RANGE_OFFSET' : 'self.LRRangeOffset', 
            'LR_AZIMUTH_OFFSET' : 'self.LRAzimuthOffset', 
            'CENTER_RANGE_OFFSET' : 'self.CenterRangeOffset',
            'CENTER_AZIMUTH_OFFSET' : 'self.CenterAzimuthOffset'
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
    pass




