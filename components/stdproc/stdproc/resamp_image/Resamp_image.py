from __future__ import print_function
from iscesys.Component.Component import Component, Port
from iscesys.Compatibility import Compatibility
from isceobj.Image import createOffsetImage
from stdproc.stdproc.resamp_image import resamp_image

DOPPLER_CENTROID_COEFFICIENTS = Component.Parameter(
    'dopplerCentroidCoefficients',
    public_name='DOPPLER_CENTROID_COEFFICIENTS',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc='Doppler centroid coefficients'
)

FIRST_LINE_OFFSET = Component.Parameter(
    'firstLineOffset',
    public_name='FIRST_LINE_OFFSET',
    default=1,
    type=int,
    mandatory=False,
    intent='input',
    doc='Line first offset'
)

LOCATION_ACROSS1 = Component.Parameter(
    'locationAcross1',
    public_name='LOCATION_ACROSS1',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc='Range locations first image.'
)

LOCATION_ACROSS2 = Component.Parameter(
    'locationAcross2',
    public_name='LOCATION_ACROSS2',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc='Range locations second image'
)


LOCATION_ACROSS_OFFSET1 = Component.Parameter(
    'locationAcrossOffset1',
    public_name='LOCATION_ACROSS_OFFSET1',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc='Range offset locations first image.'
)

LOCATION_ACROSS_OFFSET2 = Component.Parameter(
    'locationAcrossOffset2',
    public_name='LOCATION_ACROSS_OFFSET2',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc='Range offset locations second image.'
)

LOCATION_DOWN1 = Component.Parameter(
    'locationDown1',
    public_name='LOCATION_DOWN1',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc='Azimuth locations first image.'
)

LOCATION_DOWN2 = Component.Parameter(
    'locationDown2',
    public_name='LOCATION_DOWN2',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc='Azimuth locations second Image'
)


LOCATION_DOWN_OFFSET1 = Component.Parameter(
    'locationDownOffset1',
    public_name='LOCATION_DOWN_OFFSET1',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc='Azimuth'
)

LOCATION_DOWN_OFFSET2 = Component.Parameter(
    'locationDownOffset2',
    public_name='LOCATION_DOWN_OFFSET2',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc='Azimuth offset locations second image.'
)

NUMBER_FIT_COEFFICIENTS = Component.Parameter(
    'numberFitCoefficients',
    public_name='NUMBER_FIT_COEFFICIENTS',
    default=6,
    type=int,
    mandatory=False,
    intent='input',
    doc='Number of coefficients used for fit'
)


NUMBER_LINES = Component.Parameter(
    'numberLines',
    public_name='NUMBER_LINES',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Number of lines in the image'
)


NUMBER_LOOKS = Component.Parameter(
    'numberLooks',
    public_name='NUMBER_LOOKS',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Number of looks'
)


NUMBER_RANGE_BIN = Component.Parameter(
    'numberRangeBin',
    public_name='NUMBER_RANGE_BIN',
    default=None,
    type=int,
    mandatory=False,
    intent='input',
    doc='Image width'
)


RADAR_WAVELENGTH = Component.Parameter(
    'radarWavelength',
    public_name='RADAR_WAVELENGTH',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Radar wavelength'
)


SLANT_RANGE_PIXEL_SPACING = Component.Parameter(
    'slantRangePixelSpacing',
    public_name='SLANT_RANGE_PIXEL_SPACING',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Sample range pixel spacing'
)


SNR1 = Component.Parameter(
    'snr1',
    public_name='SNR1',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc='First signal to noise ratio'
)


SNR2 = Component.Parameter(
    'snr2',
    public_name='SNR2',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc='Second signal to noise ratio'
)


class Resamp_image(Component):


    parameter_list = (
                      LOCATION_ACROSS_OFFSET1,
                      LOCATION_ACROSS_OFFSET2,
                      DOPPLER_CENTROID_COEFFICIENTS,
                      FIRST_LINE_OFFSET,
                      SLANT_RANGE_PIXEL_SPACING,
                      LOCATION_DOWN1,
                      NUMBER_LINES,
                      LOCATION_ACROSS1,
                      LOCATION_DOWN_OFFSET2,
                      LOCATION_DOWN2,
                      NUMBER_LOOKS,
                      NUMBER_RANGE_BIN,
                      SNR1,
                      LOCATION_ACROSS2,
                      NUMBER_FIT_COEFFICIENTS,
                      RADAR_WAVELENGTH,
                      LOCATION_DOWN_OFFSET1,
                      SNR2
                     )


    def resamp_image(self,imageRangeOffset=None,imageAzimuthOffset=None):
        for port in self.inputPorts:
            port()

        #check if images are created. if not check if the image name has been
        #given and create it based on the info provided. other wise create
        #using default names
        rangeImageCreatedHere = False
        azimuthImageCreatedHere = False
        if not (imageRangeOffset == None):
            self.imageRangeOffset = imageRangeOffset
        if (imageRangeOffset == None):
            if (self.imageRangeOffsetName == ''):
                self.imageRangeOffsetName = 'raoff.mht'
                self.logger.warning('The imageRangeOffset has been given the default name %s' % (self.imageRangeOffsetName))

            self.imageRangeOffset = self.createImage(self.imageRangeOffsetName)
            rangeImageCreatedHere = True
        
        if not (imageAzimuthOffset == None):
            self.imageAzimuthOffset = imageAzimuthOffset
        if (imageAzimuthOffset == None):
            if (self.imageAzimuthOffsetName == ''):
                self.imageAzimuthOffsetName = 'azoff.mht'
                self.logger.warning('The imageAzimuthOffset has been given the default name %s' % (self.imageAzimuthOffsetName))

            self.imageAzimuthOffset = self.createImage(self.imageAzimuthOffsetName)        
            azimuthImageCreatedHere = True
        self.setDefaults()
        self.imageRangeOffsetAccessor = self.imageRangeOffset.getImagePointer()
        self.imageAzimuthOffsetAccessor = self.imageAzimuthOffset.getImagePointer()
        
        self.computeSecondLocation()    
        
        self.allocateArrays()
        self.setState()
        resamp_image.resamp_image_Py(self.imageRangeOffsetAccessor,self.imageAzimuthOffsetAccessor)
        self.deallocateArrays()
        
        if(rangeImageCreatedHere):
            self.imageRangeOffset.finalizeImage()

        if(azimuthImageCreatedHere):
            self.imageAzimuthOffset.finalizeImage()
        
        self.imageRangeOffset.renderHdr()
        self.imageAzimuthOffset.renderHdr()
        
        return
    
   
    def setDefaults(self):
        
        if (self.numberFitCoefficients == None):
            self.numberFitCoefficients = 6
            self.logger.warning('The variable NUMBER_FIT_COEFFICIENTS has been set to the default value %s' % (self.numberFitCoefficients)) 
        
        if (self.firstLineOffset == None):
            self.firstLineOffset = 1
            self.logger.warning('The variable FIRST_LINE_OFFSET has been set to the default value %s' % (self.firstLineOffset))
    
    def createImage(self,name):
        obj = createOffsetImage()  
        accessMode = "write"
        #dataType = "CFLOAT"
        if (self.numberRangeBin == None):
            print('Error. Cannot create default offset image if NUMBER_RANGE_BIN is not specified.')
            raise Exception
        if (self.numberLooks == None):
            print('Error. Cannot create default offset image if NUMBER_LOOKS is not specified.')
            raise Exception
        width = self.numberRangeBin/self.numberLooks
        obj.initImage(name,accessMode,width)
        obj.createImage()
        return obj

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
        resamp_image.setStdWriter_Py(int(self.stdWriter))
        resamp_image.setNumberFitCoefficients_Py(int(self.numberFitCoefficients))
        resamp_image.setNumberRangeBin_Py(int(self.numberRangeBin))
        resamp_image.setNumberLines_Py(int(self.numberLines))
        resamp_image.setNumberLooks_Py(int(self.numberLooks))
        resamp_image.setFirstLineOffset_Py(int(self.firstLineOffset))
        resamp_image.setRadarWavelength_Py(float(self.radarWavelength))
        resamp_image.setSlantRangePixelSpacing_Py(float(self.slantRangePixelSpacing))
        resamp_image.setDopplerCentroidCoefficients_Py(self.dopplerCentroidCoefficients, self.dim1_dopplerCentroidCoefficients)
        resamp_image.setLocationAcross1_Py(self.locationAcross1, self.dim1_locationAcross1)
        resamp_image.setLocationAcrossOffset1_Py(self.locationAcrossOffset1, self.dim1_locationAcrossOffset1)
        resamp_image.setLocationDown1_Py(self.locationDown1, self.dim1_locationDown1)
        resamp_image.setLocationDownOffset1_Py(self.locationDownOffset1, self.dim1_locationDownOffset1)
        resamp_image.setSNR1_Py(self.snr1, self.dim1_snr1)
        resamp_image.setLocationAcross2_Py(self.locationAcross2, self.dim1_locationAcross2)
        resamp_image.setLocationAcrossOffset2_Py(self.locationAcrossOffset2, self.dim1_locationAcrossOffset2)
        resamp_image.setLocationDown2_Py(self.locationDown2, self.dim1_locationDown2)
        resamp_image.setLocationDownOffset2_Py(self.locationDownOffset2, self.dim1_locationDownOffset2)
        resamp_image.setSNR2_Py(self.snr2, self.dim1_snr2)

        return

    def setImageRangeOffsetName(self,name):
        self.imageRangeOffsetName = name

    def setImageAzimuthOffsetName(self,name):
        self.imageAzimuthOffsetName = name

    def setNumberFitCoefficients(self,var):
        self.numberFitCoefficients = int(var)
        return

    def setNumberRangeBin(self,var):
        self.numberRangeBin = int(var)
        return

    def setNumberLines(self,var):
        self.numberLines = int(var)
        return

    def setNumberLooks(self,var):
        self.numberLooks = int(var)
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

    def allocateArrays(self):
        if (self.dim1_dopplerCentroidCoefficients == None):
            self.dim1_dopplerCentroidCoefficients = len(self.dopplerCentroidCoefficients)

        if (not self.dim1_dopplerCentroidCoefficients):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_image.allocate_dopplerCoefficients_Py(self.dim1_dopplerCentroidCoefficients)

        if (self.dim1_locationAcross1 == None):
            self.dim1_locationAcross1 = len(self.locationAcross1)

        if (not self.dim1_locationAcross1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_image.allocate_r_ranpos_Py(self.dim1_locationAcross1)

        if (self.dim1_locationAcrossOffset1 == None):
            self.dim1_locationAcrossOffset1 = len(self.locationAcrossOffset1)

        if (not self.dim1_locationAcrossOffset1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_image.allocate_r_ranoff_Py(self.dim1_locationAcrossOffset1)

        if (self.dim1_locationDown1 == None):
            self.dim1_locationDown1 = len(self.locationDown1)

        if (not self.dim1_locationDown1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_image.allocate_r_azpos_Py(self.dim1_locationDown1)

        if (self.dim1_locationDownOffset1 == None):
            self.dim1_locationDownOffset1 = len(self.locationDownOffset1)

        if (not self.dim1_locationDownOffset1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_image.allocate_r_azoff_Py(self.dim1_locationDownOffset1)

        if (self.dim1_snr1 == None):
            self.dim1_snr1 = len(self.snr1)

        if (not self.dim1_snr1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_image.allocate_r_sig_Py(self.dim1_snr1)

        if (self.dim1_locationAcross2 == None):
            self.dim1_locationAcross2 = len(self.locationAcross2)

        if (not self.dim1_locationAcross2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_image.allocate_r_ranpos2_Py(self.dim1_locationAcross2)

        if (self.dim1_locationAcrossOffset2 == None):
            self.dim1_locationAcrossOffset2 = len(self.locationAcrossOffset2)

        if (not self.dim1_locationAcrossOffset2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_image.allocate_r_ranoff2_Py(self.dim1_locationAcrossOffset2)

        if (self.dim1_locationDown2 == None):
            self.dim1_locationDown2 = len(self.locationDown2)

        if (not self.dim1_locationDown2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_image.allocate_r_azpos2_Py(self.dim1_locationDown2)

        if (self.dim1_locationDownOffset2 == None):
            self.dim1_locationDownOffset2 = len(self.locationDownOffset2)

        if (not self.dim1_locationDownOffset2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_image.allocate_r_azoff2_Py(self.dim1_locationDownOffset2)

        if (self.dim1_snr2 == None):
            self.dim1_snr2 = len(self.snr2)

        if (not self.dim1_snr2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        resamp_image.allocate_r_sig2_Py(self.dim1_snr2)


        return





    def deallocateArrays(self):
        resamp_image.deallocate_dopplerCoefficients_Py()
        resamp_image.deallocate_r_ranpos_Py()
        resamp_image.deallocate_r_ranoff_Py()
        resamp_image.deallocate_r_azpos_Py()
        resamp_image.deallocate_r_azoff_Py()
        resamp_image.deallocate_r_sig_Py()
        resamp_image.deallocate_r_ranpos2_Py()
        resamp_image.deallocate_r_ranoff2_Py()
        resamp_image.deallocate_r_azpos2_Py()
        resamp_image.deallocate_r_azoff2_Py()
        resamp_image.deallocate_r_sig2_Py()

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

    logging_name = 'isce.stdproc.resamp_image'

    family = 'resamp_image'

    def __init__(self,family='',name=''):
        super(Resamp_image, self).__init__(family if family else  self.__class__.family, name=name)
        
        self.imageRangeOffset = None
        self.imageAzimuthOffset = None
        self.imageRangeOffsetAccessor = None
        self.imageAzimuthOffsetAccessor = None
        self.imageRangeOffsetName = ''
        self.imageAzimuthOffsetName = ''
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
#        self.logger = logging.getLogger('isce.stdproc.resamp_image')
#        self.createPorts()
        
        return None

    def createPorts(self):
        offsetPort = Port(name='offsets',method=self.addOffsets)
        instrumentPort = Port(name='instrument',method=self.addInstrument)
        self._inputPorts.add(offsetPort)
        self._inputPorts.add(instrumentPort)
        return None

    pass
