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
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
from stdproc.stdproc.resamp import resamp

LOCATION_ACROSS_OFFSET2 = Component.Parameter('locationAcrossOffset2',
                                              public_name='LOCATION_ACROSS_OFFSET2',
                                              default=[],
                                              type=float,
                                              mandatory=True,
                                              intent='input',
                                              doc='')


LOCATION_ACROSS_OFFSET1 = Component.Parameter('locationAcrossOffset1',
                                              public_name='LOCATION_ACROSS_OFFSET1',
                                              default=[],
                                              type=float,
                                              mandatory=True,
                                              intent='input',
                                              doc='')


LOCATION_DOWN_OFFSET2 = Component.Parameter('locationDownOffset2',
                                            public_name='LOCATION_DOWN_OFFSET2',
                                            default=[],
                                            type=float,
                                            mandatory=True,
                                            intent='input',
                                            doc='')


LOCATION_DOWN_OFFSET1 = Component.Parameter('locationDownOffset1',
                                            public_name='LOCATION_DOWN_OFFSET1',
                                            default=[],
                                            type=float,
                                            mandatory=True,
                                            intent='input',
                                            doc='')


NUMBER_AZIMUTH_LOOKS = Component.Parameter('numberAzimuthLooks',
                                           public_name='NUMBER_AZIMUTH_LOOKS',
                                           default=None,
                                           type=int,
                                           mandatory=True,
                                           intent='input',
                                           doc='')


FLATTEN_WITH_OFFSET_FLAG = Component.Parameter('flattenWithOffsetFlag',
                                               public_name='FLATTEN_WITH_OFFSET_FLAG',
                                               default=None,
                                               type=int,
                                               mandatory=False,
                                               intent='input',
                                               doc='')


SLANT_RANGE_PIXEL_SPACING = Component.Parameter('slantRangePixelSpacing',
                                                public_name='SLANT_RANGE_PIXEL_SPACING',
                                                default=None,
                                                type=float,
                                                mandatory=True,
                                                intent='input',
                                                doc='')


SNR2 = Component.Parameter('snr2',
                           public_name='SNR2',
                           default=[],
                           type=float,
                           mandatory=True,
                           intent='inoutput',
                           doc='')


RADAR_WAVELENGTH = Component.Parameter('radarWavelength',
                                       public_name='RADAR_WAVELENGTH',
                                       default=None,
                                       type=float,
                                       mandatory=True,
                                       intent='input',
                                       doc='')


NUMBER_RANGE_LOOKS = Component.Parameter('numberRangeLooks',
                                         public_name='NUMBER_RANGE_LOOKS',
                                         default=None,
                                         type=int,
                                         mandatory=True,
                                         intent='input',
                                         doc='')


NUMBER_FIT_COEFFICIENTS = Component.Parameter('numberFitCoefficients',
                                              public_name='NUMBER_FIT_COEFFICIENTS',
                                              default=None,
                                              type=int,
                                              mandatory=False,
                                              intent='input',
                                              doc='')


SNR1 = Component.Parameter('snr1',
                           public_name='SNR1',
                           default=[],
                           type=float,
                           mandatory=True,
                           intent='input',
                           doc='')


LOCATION_ACROSS2 = Component.Parameter('locationAcross2',
                                       public_name='LOCATION_ACROSS2',
                                       default=[],
                                       type=float,
                                       mandatory=True,
                                       intent='input',
                                       doc='')


FIRST_LINE_OFFSET = Component.Parameter('firstLineOffset',
                                        public_name='FIRST_LINE_OFFSET',
                                        default=None,
                                        type=int,
                                        mandatory=False,
                                        intent='input',
                                        doc='')


LOCATION_DOWN2 = Component.Parameter('locationDown2',
                                     public_name='LOCATION_DOWN2',
                                     default=[],
                                     type=float,
                                     mandatory=True,
                                     intent='input',
                                     doc='')


LOCATION_DOWN1 = Component.Parameter('locationDown1',
                                     public_name='LOCATION_DOWN1',
                                     default=[],
                                     type=float,
                                     mandatory=True,
                                     intent='input',
                                     doc='')


NUMBER_RANGE_BIN1 = Component.Parameter('numberRangeBin1',
                                        public_name='NUMBER_RANGE_BIN1',
                                        default=None,
                                        type=int,
                                        mandatory=True,
                                        intent='input',
                                        doc='')


NUMBER_RANGE_BIN2 = Component.Parameter('numberRangeBin2',
                                        public_name='NUMBER_RANGE_BIN2',
                                        default=None,
                                        type=int,
                                        mandatory=True,
                                        intent='input',
                                        doc='')


NUMBER_LINES = Component.Parameter('numberLines',
                                   public_name='NUMBER_LINES',
                                   default=None,
                                   type=int,
                                   mandatory=True,
                                   intent='input',
                                   doc='')


LOCATION_ACROSS1 = Component.Parameter('locationAcross1',
                                       public_name='LOCATION_ACROSS1',
                                       default=[],
                                       type=float,
                                       mandatory=True,
                                       intent='input',
                                       doc='')


DOPPLER_CENTROID_COEFFICIENTS = Component.Parameter('dopplerCentroidCoefficients',
                                                    public_name='DOPPLER_CENTROID_COEFFICIENTS',
                                                    default=[],
                                                    type=float,
                                                    mandatory=True,
                                                    intent='input',
                                                    doc='')


START_LINE = Component.Parameter('startLine',
                                 public_name='START_LINE',
                                 default=None,
                                 type=int,
                                 mandatory=False,
                                 intent='input',
                                 doc='')


REFINED_LOCATION_ACROSS_OFFSET2 = Component.Parameter('acrossOffset2',
                                                      public_name='REFINED_LOCATION_ACROSS_OFFSET2',
                                                      default=[],
                                                      type=float,
                                                      mandatory=False,
                                                      intent='output',
                                                      doc='')


REFINED_LOCATION_DOWN_OFFSET2 = Component.Parameter('downOffset2',
                                                    public_name='REFINED_LOCATION_DOWN_OFFSET2',
                                                    default=[],
                                                    type=float,
                                                    mandatory=False,
                                                    intent='output',
                                                    doc='')


REFINED_LOCATION_DOWN_OFFSET1 = Component.Parameter('downOffset1',
                                                    public_name='REFINED_LOCATION_DOWN_OFFSET1',
                                                    default=[],
                                                    type=float,
                                                    mandatory=False,
                                                    intent='output',
                                                    doc='')


REFINED_LOCATION_ACROSS_OFFSET1 = Component.Parameter('acrossOffset1',
                                                      public_name='REFINED_LOCATION_ACROSS_OFFSET1',
                                                      default=[],
                                                      type=float,
                                                      mandatory=False,
                                                      intent='output',
                                                      doc='')


class Resamp(Component):


    parameter_list = (
                      LOCATION_ACROSS_OFFSET2,
                      LOCATION_ACROSS_OFFSET1,
                      LOCATION_DOWN_OFFSET2,
                      LOCATION_DOWN_OFFSET1,
                      NUMBER_AZIMUTH_LOOKS,
                      FLATTEN_WITH_OFFSET_FLAG,
                      SLANT_RANGE_PIXEL_SPACING,
                      SNR2,
                      RADAR_WAVELENGTH,
                      NUMBER_RANGE_LOOKS,
                      NUMBER_FIT_COEFFICIENTS,
                      SNR1,
                      LOCATION_ACROSS2,
                      FIRST_LINE_OFFSET,
                      LOCATION_DOWN2,
                      LOCATION_DOWN1,
                      NUMBER_RANGE_BIN1,
                      NUMBER_RANGE_BIN2,
                      NUMBER_LINES,
                      LOCATION_ACROSS1,
                      DOPPLER_CENTROID_COEFFICIENTS,
                      START_LINE,
                      REFINED_LOCATION_ACROSS_OFFSET1,
                      REFINED_LOCATION_ACROSS_OFFSET2,
                      REFINED_LOCATION_DOWN_OFFSET1,
                      REFINED_LOCATION_DOWN_OFFSET2
                     )


    
    def resamp(self, image1=None, image2=None, imageInt=None, imageAmp=None, resamp2=None): #KK 2013-11-10: added resamp2
        #KK: if imageInt, imageAmp or resamp2 is None, it will not be output
        null_pointer = 0 #KK: accessor value when image parameter is None

        for port in self._inputPorts:
            port()

        if image1 is not None:
            self.image1 = image1
        else:
            self.logger.error("First slc image not set.")
            raise Exception

        if image2 is not None:
            self.image2 = image2
        else:
            self.logger.error("Second slc image not set.")
            raise Exception

        self.resamp2 = resamp2 #KK

#KK: removed if statements...
#        if imageInt is not None:
        self.imageInt= imageInt
#        if self.imageInt == None:
#            self.logger.error("Interference image not set.")
#            raise Exception

#        if imageAmp is not None:
        self.imageAmp= imageAmp
#        if self.imageAmp is None:
#            self.logger.error("Amplitude image not set.")
#            raise Exception

        self.setDefaults()
        self.image1Accessor = self.image1.getImagePointer()
        self.image2Accessor = self.image2.getImagePointer()
        #create the int and amp file to allow random access
        length = self.numberLines
        lengthIntAmp = length//self.numberAzimuthLooks
        if self.imageInt is not None: #KK: image is really created if imageInt not None
            self.imageInt.createFile(lengthIntAmp)
            self.imageIntAccessor = self.imageInt.getImagePointer()
        else: #KK
            self.imageIntAccessor = null_pointer #KK

        if self.imageAmp is not None: #KK
            self.imageAmp.createFile(lengthIntAmp)
            self.imageAmpAccessor = self.imageAmp.getImagePointer()
        else: #KK
            self.imageAmpAccessor = null_pointer #KK

        if self.resamp2 is not None: #KK
            self.resamp2.createFile(length) #KK
            self.resamp2Accessor = self.resamp2.getImagePointer() #KK
        else: #KK
            self.resamp2Accessor = null_pointer #KK

        #remember we put the offset for the images in one array
        # so twice the length
        self.acrossOffset = [0]*(2*len(self.locationAcross1))
        self.downOffset = [0]*(2*len(self.locationAcross1))

        self.computeSecondLocation()    
        self.allocateArrays()
        self.setState()
        resamp.resamp_Py(self.image1Accessor,
                         self.image2Accessor,
                         self.imageIntAccessor,
                         self.imageAmpAccessor,
                         self.resamp2Accessor) #KK
        self.getState()
        if self.imageAmp is not None: #KK: render header only if imageAmp is not None
            self.imageAmp.bandDescription = ['amplitude slc1','amplitude slc2']
            self.imageAmp.finalizeImage()
            self.imageAmp.renderHdr()
        if self.imageInt is not None: #KK
            self.imageInt.finalizeImage()
            self.imageInt.renderHdr()
        if self.resamp2 is not None: #KK
            self.resamp2.finalizeImage()
            self.resamp2.renderHdr() #KK

        #since the across and down offsets are returned in one array,
        # just split it for each location  #should be an even number
        halfArray = len(self.acrossOffset)//2
        #remember that slicing leave out the larger extreme of the interval
        self.acrossOffset1 = self.acrossOffset[0:halfArray]
        self.acrossOffset2 = self.acrossOffset[halfArray:2*halfArray]
        self.downOffset1 = self.downOffset[0:halfArray]
        self.downOffset2 = self.downOffset[halfArray:2*halfArray]
        self.deallocateArrays()

        return


    def setDefaults(self):
        if self.numberLines is None:
            self.numberLines = self.image1.getLength()
            self.logger.warning(
                'The variable NUMBER_LINES has been set to the default value %d which is the number of lines in the slc image.'
                % self.numberLines
                ) 


        if self.numberRangeBin1 is None:
            self.numberRangeBin1 = self.image1.getWidth()
            self.logger.warning(
                'The variable NUMBER_RANGE_BIN1 has been set to the default value %d which is the width of the first slc image.'
                % self.numberRangeBin1
                )
             
        if self.numberRangeBin2 is None:
            self.numberRangeBin2 = self.image2.getWidth()
            self.logger.warning(
                'The variable NUMBER_RANGE_BIN2 has been set to the default value %d which is the width of the second slc image.'
                % self.numberRangeBin2
                ) 
        
        if self.numberFitCoefficients is None:
            self.numberFitCoefficients = 6
            self.logger.warning(
                'The variable NUMBER_FIT_COEFFICIENTS has been set to the default value %s'
                % self.numberFitCoefficients
                ) 
        
        if self.startLine is None:
            self.startLine = 1
            self.logger.warning(
                'The variable START_LINE has been set to the default value %s'
                % self.startLine
                )
             
        if self.firstLineOffset is None:
            self.firstLineOffset = 1
            self.logger.warning(
                'The variable FIRST_LINE_OFFSET has been set to the default value %s'
                % self.firstLineOffset
                ) 
        
        if self.flattenWithOffsetFlag is None:
            self.flattenWithOffsetFlag = 0
            self.logger.warning(
                'The variable FLATTEN_WITH_OFFSET_FLAG has been set to the default value %s' %
                self.flattenWithOffsetFlag
                ) 

    #this part was previously done in the fortran code        
    def computeSecondLocation(self):
        self.locationAcross2 = [0]*len(self.locationAcross1)
        self.locationAcrossOffset2 = [0]*len(self.locationAcross1)
        self.locationDown2 = [0]*len(self.locationAcross1)
        self.locationDownOffset2 = [0]*len(self.locationAcross1)
        self.snr2 = [0]*len(self.locationAcross1)
        for i in range(len(self.locationAcross1)):
            self.locationAcross2[i] = (
                self.locationAcross1[i] + self.locationAcrossOffset1[i]
                )
            self.locationAcrossOffset2[i] = self.locationAcrossOffset1[i]
            self.locationDown2[i] = (
                self.locationDown1[i] + self.locationDownOffset1[i]
                )
            self.locationDownOffset2[i] = self.locationDownOffset1[i]
            self.snr2[i] = self.snr1[i]

    def setState(self):
        resamp.setStdWriter_Py(int(self.stdWriter))
        resamp.setNumberFitCoefficients_Py(self.numberFitCoefficients)
        resamp.setNumberRangeBin1_Py(int(self.numberRangeBin1))
        resamp.setNumberRangeBin2_Py(int(self.numberRangeBin2))
        resamp.setStartLine_Py(int(self.startLine))
        resamp.setNumberLines_Py(int(self.numberLines))
        resamp.setNumberLinesImage2_Py(int(self.image2.getLength()))
        resamp.setFirstLineOffset_Py(int(self.firstLineOffset))
        resamp.setNumberRangeLooks_Py(int(self.numberRangeLooks))
        resamp.setNumberAzimuthLooks_Py(int(self.numberAzimuthLooks))
        resamp.setRadarWavelength_Py(float(self.radarWavelength))
        resamp.setSlantRangePixelSpacing_Py(float(self.slantRangePixelSpacing))
        resamp.setFlattenWithOffsetFitFlag_Py(int(self.flattenWithOffsetFlag))
        resamp.setDopplerCentroidCoefficients_Py(self.dopplerCentroidCoefficients, self.dim1_dopplerCentroidCoefficients)
        resamp.setLocationAcross1_Py(self.locationAcross1,
                                     self.dim1_locationAcross1)
        resamp.setLocationAcrossOffset1_Py(self.locationAcrossOffset1,
                                           self.dim1_locationAcrossOffset1)
        resamp.setLocationDown1_Py(self.locationDown1, self.dim1_locationDown1)
        resamp.setLocationDownOffset1_Py(self.locationDownOffset1,
                                         self.dim1_locationDownOffset1)
        resamp.setSNR1_Py(self.snr1, self.dim1_snr1)
        resamp.setLocationAcross2_Py(self.locationAcross2,
                                     self.dim1_locationAcross2)
        resamp.setLocationAcrossOffset2_Py(self.locationAcrossOffset2,
                                           self.dim1_locationAcrossOffset2)
        resamp.setLocationDown2_Py(self.locationDown2,
                                   self.dim1_locationDown2)
        resamp.setLocationDownOffset2_Py(self.locationDownOffset2,
                                         self.dim1_locationDownOffset2)
        resamp.setSNR2_Py(self.snr2, self.dim1_snr2)
        return

    def setNumberFitCoefficients(self, var):
        self.numberFitCoefficients = int(var)
        return

    def setNumberRangeBin1(self, var):
        self.numberRangeBin1 = int(var)
        return

    def setNumberRangeBin2(self, var):
        self.numberRangeBin2 = int(var)
        return

    def setStartLine(self, var):
        self.startLine = int(var)
        return

    def setNumberLines(self, var):
        self.numberLines = int(var)
        return

    def setFirstLineOffset(self, var):
        self.firstLineOffset = int(var)
        return

    def setNumberRangeLooks(self, var):
        self.numberRangeLooks = int(var)
        return

    def setNumberAzimuthLooks(self, var):
        self.numberAzimuthLooks = int(var)
        return

    def setRadarWavelength(self, var):
        self.radarWavelength = float(var)
        return

    def setSlantRangePixelSpacing(self, var):
        self.slantRangePixelSpacing = float(var)
        return

    def setFlattenWithOffsetFitFlag(self, var):
        self.flattenWithOffsetFlag = int(var)
        return

    def setDopplerCentroidCoefficients(self, var):
        self.dopplerCentroidCoefficients = var
        return

    def setLocationAcross1(self, var):
        self.locationAcross1 = var
        return

    def setLocationAcrossOffset1(self, var):
        self.locationAcrossOffset1 = var
        return

    def setLocationDown1(self, var):
        self.locationDown1 = var
        return

    def setLocationDownOffset1(self, var):
        self.locationDownOffset1 = var
        return

    def setSNR1(self, var):
        self.snr1 = var
        return

    def setLocationAcross2(self, var):
        self.locationAcross2 = var
        return

    def setLocationAcrossOffset2(self, var):
        self.locationAcrossOffset2 = var
        return

    def setLocationDown2(self, var):
        self.locationDown2 = var
        return

    def setLocationDownOffset2(self, var):
        self.locationDownOffset2 = var
        return

    def setSNR2(self, var):
        self.snr2 = var
        return

    def getState(self):
        self.acrossOffset = resamp.getLocationAcrossOffset_Py(
            self.dim1_acrossOffset
            )
        self.downOffset = resamp.getLocationDownOffset_Py(self.dim1_downOffset)
        return

    def getRefinedLocationAcrossOffset1(self):
        return self.acrossOffset1

    def getRefinedLocationDownOffset1(self):
        return self.downOffset1

    def getRefinedLocationAcrossOffset2(self):
        return self.acrossOffset2

    def getRefinedLocationDownOffset2(self):
        return self.downOffset2

    def allocateArrays(self):
        if self.dim1_dopplerCentroidCoefficients is None:
            self.dim1_dopplerCentroidCoefficients = len(
                self.dopplerCentroidCoefficients
                )

        if not self.dim1_dopplerCentroidCoefficients:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_dopplerCoefficients_Py(
            self.dim1_dopplerCentroidCoefficients
            )

        if self.dim1_locationAcross1 is None:
            self.dim1_locationAcross1 = len(self.locationAcross1)

        if not self.dim1_locationAcross1:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_r_ranpos_Py(self.dim1_locationAcross1)

        if self.dim1_locationAcrossOffset1 is None:
            self.dim1_locationAcrossOffset1 = len(self.locationAcrossOffset1)

        if not self.dim1_locationAcrossOffset1:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_r_ranoff_Py(self.dim1_locationAcrossOffset1)

        if self.dim1_locationDown1 is None:
            self.dim1_locationDown1 = len(self.locationDown1)

        if not self.dim1_locationDown1:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_r_azpos_Py(self.dim1_locationDown1)

        if self.dim1_locationDownOffset1 is None:
            self.dim1_locationDownOffset1 = len(self.locationDownOffset1)

        if not self.dim1_locationDownOffset1:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_r_azoff_Py(self.dim1_locationDownOffset1)

        if self.dim1_snr1 is None:
            self.dim1_snr1 = len(self.snr1)

        if not self.dim1_snr1:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_r_sig_Py(self.dim1_snr1)

        if self.dim1_locationAcross2 is None:
            self.dim1_locationAcross2 = len(self.locationAcross2)

        if not self.dim1_locationAcross2:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_r_ranpos2_Py(self.dim1_locationAcross2)

        if self.dim1_locationAcrossOffset2 is None:
            self.dim1_locationAcrossOffset2 = len(self.locationAcrossOffset2)

        if not self.dim1_locationAcrossOffset2:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_r_ranoff2_Py(self.dim1_locationAcrossOffset2)

        if self.dim1_locationDown2 is None:
            self.dim1_locationDown2 = len(self.locationDown2)

        if not self.dim1_locationDown2:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_r_azpos2_Py(self.dim1_locationDown2)

        if self.dim1_locationDownOffset2 is None:
            self.dim1_locationDownOffset2 = len(self.locationDownOffset2)

        if not self.dim1_locationDownOffset2:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_r_azoff2_Py(self.dim1_locationDownOffset2)

        if self.dim1_snr2 is None:
            self.dim1_snr2 = len(self.snr2)

        if not self.dim1_snr2:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_r_sig2_Py(self.dim1_snr2)

        if self.dim1_acrossOffset is None:
            self.dim1_acrossOffset = len(self.acrossOffset)

        if not self.dim1_acrossOffset:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_acrossOffset_Py(self.dim1_acrossOffset)

        if self.dim1_downOffset is None:
            self.dim1_downOffset = len(self.downOffset)

        if not self.dim1_downOffset:
            self.logger.error("Trying to allocate zero size array")
            raise Exception

        resamp.allocate_downOffset_Py(self.dim1_downOffset)
        return

    def deallocateArrays(self):
        resamp.deallocate_dopplerCoefficients_Py()
        resamp.deallocate_r_ranpos_Py()
        resamp.deallocate_r_ranoff_Py()
        resamp.deallocate_r_azpos_Py()
        resamp.deallocate_r_azoff_Py()
        resamp.deallocate_r_sig_Py()
        resamp.deallocate_r_ranpos2_Py()
        resamp.deallocate_r_ranoff2_Py()
        resamp.deallocate_r_azpos2_Py()
        resamp.deallocate_r_azoff2_Py()
        resamp.deallocate_r_sig2_Py()
        resamp.deallocate_acrossOffset_Py()
        resamp.deallocate_downOffset_Py()
        return

    def addInstrument(self):
        instrument = self._inputPorts['instrument']
        if instrument:
            try:
                self.radarWavelength = instrument.getRadarWavelength()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError("Unable to wire instrument port")

    def addOffsets(self):
        offsets = self._inputPorts['offsets']
        if offsets:
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

    logging_name = 'isce.stdproc.resamp'
    family = 'resamp'

    def __init__(self,family='',name=''):
        super(Resamp, self).__init__(family if family else  self.__class__.family, name=name)
        
        self.image1 = None
        self.image2 = None
        self.imageInt = None
        self.imageAmp = None
        self.image1Accessor = None
        self.image2Accessor = None
        self.imageIntAccessor = None
        self.imageAmpAccessor = None
        self.dim1_dopplerCentroidCoefficients = None
        self.dim1_locationAcross1 = None
        self.dim1_locationAcrossOffset1 = None
        self.dim1_locationDown1 = None
        self.dim1_locationDownOffset1 = None
        self.dim1_snr1 = None
        self.dim1_locationAcross2 = None
        self.dim1_locationAcrossOffset2 = None
        self.dim1_locationDown2 = None
        self.dim1_locationDownOffset2 = None
        self.dim1_snr2 = None
        self.acrossOffset = []
        self.dim1_acrossOffset = None
        self.downOffset = []
        self.dim1_downOffset = None
        return

    def createPorts(self):
        offsetPort = Port(name='offsets',method=self.addOffsets)
        instrumentPort = Port(name='instrument',method=self.addInstrument)
        self._inputPorts.add(offsetPort)
        self._inputPorts.add(instrumentPort)
        return None

    pass
