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
import logging
import logging.config
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from iscesys.Compatibility import Compatibility
from isceobj.Scene.Frame import FrameMixin

## Reference Secondary Hash Table
REFERENCE_SECONDARY = {0:'reference', 1:'secondary', 'reference':'reference', 'secondary':'secondary'}

PROCEED_IF_ZERO_DEM = Component.Parameter(
    '_proceedIfZeroDem',
    public_name='proceed if zero dem',
    default=False,
    type=bool,
    mandatory=False,
    doc='Flag to apply continue processing if a dem is not available or cannot be downloaded.'
)

RESAMP_IMAGE_NAME_BASE = Component.Parameter('_resampImageName',
                                  public_name='resamp image name base',
                                  default='resampImage',
                                  type=str,
                                  mandatory=False,
                                  doc=('Base name for output interferogram and amplitude files, '+
                                  'with fixed extensions .int and .amp added')
                                 )

PEG = Component.Facility('_peg',
                          public_name='peg',
                          module='isceobj.Location.Peg',
                          factory='Peg',
                          mandatory=False,
                          doc='')

IS_MOCOMP = Component.Parameter('is_mocomp',
    public_name='is_mocomp',
    default=None,
    type=int,
    mandatory=False,
    doc=''
)

RG_IMAGE_NAME = Component.Parameter('_rgImageName',
                                    public_name='rgImageName',
                                    default='rgImage',
                                    type=str,
                                    mandatory=False,
                                    doc='')
AZRES_FACTOR = Component.Parameter('_azResFactor',
                                    public_name='azResFactor',
                                    default=1.,
                                    type=float,
                                    mandatory=False,
                                    doc='Factor that multiplies the azimuth resolution adopted in focusing.')

SIM_AMP_IMAGE_NAME = Component.Parameter('_simAmpImageName',
                                         public_name='simAmpImageName',
                                         default='simamp.rdr',
                                         type=str,
                                         mandatory=False,
                                         doc='')

APPLY_WATER_MASK = Component.Parameter(
    '_applyWaterMask',
    public_name='applyWaterMask',
    default=True,
    type=bool,
    mandatory=False,
    doc='Flag to apply water mask to images before unwrapping.'
)

WATER_MASK_IMAGE_NAME = Component.Parameter(
    '_waterMaskImageName',
    public_name='waterMaskImageName',
    default='waterMask.msk',
    type=str,
    mandatory=False,
    doc='Filename of the water body mask image in radar coordinate cropped to the interferogram size.'
)
RESAMP_ONLY_IMAGE_NAME = Component.Parameter(
    '_resampOnlyImageName',
    public_name='resampOnlyImageName',
    default='resampOnlyImage.int',
    type=str,
    mandatory=False,
    doc='Filename of the dem-resampled interferogram.'
)

RESAMP_ONLY_AMP_NAME = Component.Parameter(
    '_resampOnlyAmpName',
    public_name='resampOnlyAmpName',
    default=RESAMP_ONLY_IMAGE_NAME.default.replace('.int', '.amp'),
    type=str,
    mandatory=False,
    doc='Filename of the dem-resampled amplitudes.'
)

OFFSET_IMAGE_NAME = Component.Parameter('_offsetImageName',
                                        public_name='offsetImageName',
                                        default='Offset.mht',
                                        type=str,
                                        mandatory=False,
                                        doc='')

DEM_INIT_FILE = Component.Parameter('_demInitFile',
                                    public_name='demInitFile',
                                    default='DemImage.xml',
                                    type=str,
                                    mandatory=False,
                                    doc='')


FIRST_SAMPLE_ACROSS_PRF = Component.Parameter('_firstSampleAcrossPrf',
                                              public_name='firstSampleAcrossPrf',
                                              default=50,
                                              type=int,
                                              mandatory=False,
                                              doc='')


FIRST_SAMPLE_DOWN_PRF = Component.Parameter('_firstSampleDownPrf',
                                            public_name='firstSampleDownPrf',
                                            default=50,
                                            type=int,
                                            mandatory=False,
                                            doc='')


NUMBER_LOCATION_ACROSS_PRF = Component.Parameter('_numberLocationAcrossPrf',
                                                 public_name='numberLocationAcrossPrf',
                                                 default=40,
                                                 type=int,
                                                 mandatory=False,
                                                 doc='')


NUMBER_LOCATION_DOWN_PRF = Component.Parameter('_numberLocationDownPrf',
                                               public_name='numberLocationDownPrf',
                                               default=50,
                                               type=int,
                                               mandatory=False,
                                               doc='')

NUMBER_VALID_PULSES = Component.Parameter('_numberValidPulses',
                                          public_name='numberValidPulses',
                                          default=2048,
                                          type=int,
                                          mandatory=False,
                                          doc='')

FIRST_SAMPLE_ACROSS = Component.Parameter('_firstSampleAcross',
                                          public_name='firstSampleAcross',
                                          default=50,
                                          type=int,
                                          mandatory=False,
                                          doc='')


FIRST_SAMPLE_DOWN = Component.Parameter('_firstSampleDown',
                                        public_name='firstSampleDown',
                                        default=50,
                                        type=int,
                                        mandatory=False,
                                        doc='')


NUMBER_LOCATION_ACROSS = Component.Parameter('_numberLocationAcross',
                                             public_name='numberLocationAcross',
                                             default=40,
                                             type=int,
                                             mandatory=False,
                                             doc='')


NUMBER_LOCATION_DOWN = Component.Parameter('_numberLocationDown',
                                           public_name='numberLocationDown',
                                           default=40,
                                           type=int,
                                           mandatory=False,
                                           doc='')




TOPOPHASE_ITERATIONS = Component.Parameter('_topophaseIterations',
                                           public_name='topophaseIterations',
                                           default=25,
                                           type=int,
                                           mandatory=False,
                                           doc='')


COHERENCE_FILENAME = Component.Parameter('_coherenceFilename',
                                         public_name='coherenceFilename',
                                         default='topophase.cor',
                                         type=str,
                                         mandatory=False,
                                         doc='')


UNWRAPPED_INT_FILENAME = Component.Parameter('_unwrappedIntFilename',
                                             public_name='unwrappedIntFilename',
                                             default='filt_topophase.unw',
                                             type=str,
                                             mandatory=False,
                                             doc='')

UNWRAPPED_2STAGE_FILENAME = Component.Parameter('_unwrapped2StageFilename',
                                             public_name='unwrapped2StageFilename',
                                             default='filt_topophase_2stage.unw',
                                             type=str,
                                             mandatory=False,
                                             doc='Output File name of 2Stage unwrapper')

CONNECTED_COMPONENTS_FILENAME = Component.Parameter(
    '_connectedComponentsFilename',
    public_name='connectedComponentsFilename',
    default=None,
    type=str,
    mandatory=False,
    doc=''
)

PHSIG_FILENAME = Component.Parameter('_phsigFilename',
                                     public_name='phsigFilename',
                                     default='phsig.cor',
                                     type=str,
                                     mandatory=False,
                                     doc='')


TOPOPHASE_MPH_FILENAME = Component.Parameter('_topophaseMphFilename',
                                             public_name='topophaseMphFilename',
                                             default='topophase.mph',
                                             type=str,
                                             mandatory=False,
                                             doc='')


TOPOPHASE_FLAT_FILENAME = Component.Parameter('_topophaseFlatFilename',
                                              public_name='topophaseFlatFilename',
                                              default='topophase.flat',
                                              type=str,
                                              mandatory=False,
                                              doc='')


FILT_TOPOPHASE_FLAT_FILENAME = Component.Parameter('_filt_topophaseFlatFilename',
                                                   public_name='filt_topophaseFlatFilename',
                                                   default='filt_topophase.flat',
                                                   type=str,
                                                   mandatory=False,
                                                   doc='')


HEIGHT_FILENAME = Component.Parameter('_heightFilename',
                                      public_name='heightFilename',
                                      default='z.rdr',
                                      type=str,
                                      mandatory=False,
                                      doc='')


HEIGHT_SCH_FILENAME = Component.Parameter('_heightSchFilename',
                                          public_name='heightSchFilename',
                                          default='zsch.rdr',
                                          type=str,
                                          mandatory=False,
                                          doc='')


GEOCODE_FILENAME = Component.Parameter('_geocodeFilename',
                                       public_name='geocodeFilename',
                                       default='topophase.geo',
                                       type=str,
                                       mandatory=False,
                                       doc='')


LOS_FILENAME = Component.Parameter('_losFilename',
                                   public_name='losFilename',
                                   default='los.rdr',
                                   type=str,
                                   mandatory=False,
                                   doc='')


LAT_FILENAME = Component.Parameter('_latFilename',
                                   public_name='latFilename',
                                   default='lat.rdr',
                                   type=str,
                                   mandatory=False,
                                   doc='')


LON_FILENAME = Component.Parameter('_lonFilename',
                                   public_name='lonFilename',
                                   default='lon.rdr',
                                   type=str,
                                   mandatory=False,
                                   doc='')


DEM_CROP_FILENAME = Component.Parameter('_demCropFilename',
                                        public_name='demCropFilename',
                                        default='dem.crop',
                                        type=str,
                                        mandatory=False,
                                        doc='')


FILTER_STRENGTH = Component.Parameter('_filterStrength',
                                      public_name='filterStrength',
                                      default=0.7,
                                      type=float,
                                      mandatory=False,
                                      doc='')

NUMBER_PATCHES = Component.Parameter('_numberPatches',
                                     public_name='numberPatches',
                                     default=None,
                                     type=int,
                                     mandatory=False,
                                     doc='')


PATCH_SIZE = Component.Parameter('_patchSize',
                                 public_name='patchSize',
                                 default=8192,
                                 type=int,
                                 mandatory=False,
                                 doc='')

SECONDARY_RANGE_MIGRATION_FLAG = Component.Parameter('_secondaryRangeMigrationFlag',
     public_name='secondaryRangeMigrationFlag',
     default=None,
     type=str,
     mandatory=False,
     doc=''
)
POSTING = Component.Parameter('_posting',
                              public_name='posting',
                              default=15,
                              type=int,
                              mandatory=False,
                              doc='')


NUMBER_FIT_COEFFICIENTS = Component.Parameter('_numberFitCoefficients',
                                              public_name='numberFitCoefficients',
                                              default=6,
                                              type=int,
                                              mandatory=False,
                                              doc='')


NUMBER_LOOKS = Component.Parameter('_numberLooks',
                                   public_name='numberLooks',
                                   default=4,
                                   type=int,
                                   mandatory=False,
                                   doc='')


NUMBER_AZIMUTH_LOOKS = Component.Parameter('_numberAzimuthLooks',
                                           public_name='numberAzimuthLooks',
                                           default=1,
                                           type=int,
                                           mandatory=False,
                                           doc='')


NUMBER_RANGE_LOOKS = Component.Parameter('_numberRangeLooks',
    public_name='numberRangeLooks',
    default=None,
    type=int,
    mandatory=False,
    doc=''
)


SHADE_FACTOR = Component.Parameter('_shadeFactor',
                                   public_name='shadeFactor',
                                   default=3,
                                   type=int,
                                   mandatory=False,
                                   doc='')

#ask
REFERENCE_SQUINT = Component.Parameter('_referenceSquint',
                                    public_name='referenceSquint',
                                    default=0.,
                                    type=float,
                                    mandatory=False,
                                    doc='')

#ask
SECONDARY_SQUINT = Component.Parameter('_secondarySquint',
                                   public_name='secondarySquint',
                                   default=0.,
                                   type=float,
                                   mandatory=False,
                                   doc='')

GEOCODE_LIST = Component.Parameter('_geocode_list',
    public_name='geocode_list',
    default=[COHERENCE_FILENAME,
             UNWRAPPED_INT_FILENAME,
             PHSIG_FILENAME,
             LOS_FILENAME,
             TOPOPHASE_FLAT_FILENAME,
             FILT_TOPOPHASE_FLAT_FILENAME,
             RESAMP_ONLY_AMP_NAME,
             UNWRAPPED_2STAGE_FILENAME,
             ],
    container=list,
    type=str,
    mandatory=False,
    doc='List of files to geocode'
)
UNMASKED_PREFIX = Component.Parameter('_unmaskedPrefix',
                                   public_name='unmaskedPrefix',
                                   default='unmasked',
                                   type=str,
                                   mandatory=False,
                                   doc='Prefix prepended to the image filenames that have not been water masked')



class InsarProc(Component, FrameMixin):
    """
    This class holds the properties, along with methods (setters and getters)
    to modify and return their values.
    """

    parameter_list = (RESAMP_IMAGE_NAME_BASE,
                      IS_MOCOMP,
                      RG_IMAGE_NAME,
                      AZRES_FACTOR,
                      SIM_AMP_IMAGE_NAME,
                      APPLY_WATER_MASK,
                      WATER_MASK_IMAGE_NAME,
                      RESAMP_ONLY_IMAGE_NAME,
                      RESAMP_ONLY_AMP_NAME,
                      OFFSET_IMAGE_NAME,
                      DEM_INIT_FILE,
                      FIRST_SAMPLE_ACROSS_PRF,
                      FIRST_SAMPLE_DOWN_PRF,
                      NUMBER_LOCATION_ACROSS_PRF,
                      NUMBER_LOCATION_DOWN_PRF,
                      NUMBER_VALID_PULSES,
                      FIRST_SAMPLE_ACROSS,
                      FIRST_SAMPLE_DOWN,
                      NUMBER_LOCATION_ACROSS,
                      NUMBER_LOCATION_DOWN,
                      TOPOPHASE_ITERATIONS,
                      COHERENCE_FILENAME,
                      UNWRAPPED_INT_FILENAME,
                      CONNECTED_COMPONENTS_FILENAME,
                      PHSIG_FILENAME,
                      TOPOPHASE_MPH_FILENAME,
                      TOPOPHASE_FLAT_FILENAME,
                      FILT_TOPOPHASE_FLAT_FILENAME,
                      HEIGHT_FILENAME,
                      HEIGHT_SCH_FILENAME,
                      GEOCODE_FILENAME,
                      LOS_FILENAME,
                      LAT_FILENAME,
                      LON_FILENAME,
                      DEM_CROP_FILENAME,
                      FILTER_STRENGTH,
                      NUMBER_PATCHES,
                      PATCH_SIZE,
                      SECONDARY_RANGE_MIGRATION_FLAG,
                      POSTING,
                      NUMBER_FIT_COEFFICIENTS,
                      NUMBER_LOOKS,
                      NUMBER_AZIMUTH_LOOKS,
                      NUMBER_RANGE_LOOKS,
                      SHADE_FACTOR,
                      REFERENCE_SQUINT,
                      SECONDARY_SQUINT,
                      GEOCODE_LIST,
                      UNMASKED_PREFIX,
                      UNWRAPPED_2STAGE_FILENAME,
                      PROCEED_IF_ZERO_DEM)

    facility_list = (
                     PEG,
                     )


    family='insarcontext'

    def __init__(self, name='', procDoc=None):
        #self.updatePrivate()

        super().__init__(family=self.__class__.family, name=name)
        self.procDoc = procDoc
        return None

    def _init(self):
        """
        Method called after Parameters are configured.
        Determine whether some Parameters still have unresolved
        Parameters as their default values and resolve them.
        """

        #Determine whether the geocode_list still contains Parameters
        #and give those elements the proper value.  This will happen
        #whenever the user doesn't provide as input a geocode_list for
        #this component.
        for i, x in enumerate(self.geocode_list):
            if isinstance(x, Component.Parameter):
                y = getattr(self, getattr(x, 'attrname'))
                self.geocode_list[i] = y
        return

    def get_is_mocomp(self):
        self.is_mocomp =  int((
                self.getPatchSize() - self.getNumberValidPulses()
                )/2)
        return self.is_mocomp

    # Getters
    @property
    def proceedIfZeroDem(self):
        return self._proceedIfZeroDem
   
    def getLookSide(self):
        return self._lookSide

    def getReferenceSquint(self):
        return self._referenceSquint

    def getSecondarySquint(self):
        return self._secondarySquint

    def getFormSLC1(self):
        return self._formSLC1

    def getFormSLC2(self):
        return self._formSLC2

    def getMocompBaseline(self):
        return self._mocompBaseline

    def getTopocorrect(self):
        return self._topocorrect

    def getTopo(self):
        return self._topo

    ## to be deprecated
    def getAverageHeight(self):
        return self.averageHeight
    @property
    def averageHeight(self):
        return (self._pegH1 + self._pegH2)/2.0

    def getFirstAverageHeight(self):
        return self._pegH1

    def getSecondAverageHeight(self):
        return self._pegH2

    def getFirstFdHeight(self):
        return self._fdH1

    def getSecondFdHeight(self):
        return self._fdH2

    ## deprecate ASAP
    def getProcVelocity(self):
        return self.procVelocity
    @property
    def procVelocity(self):
        return (self._pegV1 + self._pegV2)/2.0

    # <v>, <h>
    def vh(self):
        return self.procVelocity, self.averageHeight

    def getFirstProcVelocity(self):
        return self._pegV1

    def getSecondProcVelocity(self):
        return self._pegV2

    def getReferenceFrame(self):
        return self._referenceFrame

    def getSecondaryFrame(self):
        return self._secondaryFrame

    def getReferenceOrbit(self):
        return self._referenceOrbit

    def getSecondaryOrbit(self):
        return self._secondaryOrbit

    def getReferenceDoppler(self):
        return self._referenceDoppler

    def getSecondaryDoppler(self):
        return self._secondaryDoppler

    def getPeg(self):
        return self._peg

    def getReferenceRawImage(self):
        return self._referenceRawImage

    def getSecondaryRawImage(self):
        return self._secondaryRawImage

    def getReferenceSlcImage(self):
        return self._referenceSlcImage

    def getSecondarySlcImage(self):
        return self._secondarySlcImage

    def getSimAmpImage(self):
        return self._simAmpImage

    def getRgImage(self):
        return self._rgImage

    def getResampAmpImage(self):
        return self._resampAmpImage

    def getResampIntImage(self):
        return self._resampIntImage

    def getResampOnlyImage(self):
        return self._resampOnlyImage
    def getResampOnlyAmp(self):
        return self._resampOnlyAmp

    def getTopoIntImage(self):
        return self._topoIntImage

    def getHeightTopoImage(self):
        return self._heightTopoImage

    def getOffsetAzimuthImage(self):
        return self._offsetAzimuthImage

    def getOffsetRangeImage(self):
        return self._offsetRangeImage

    def getSLC1ImageName(self):
        return self._slc1ImageName

    def getSLC2ImageName(self):
        return self._slc2ImageName

    def getSimAmpImageName(self):
        return self._simAmpImageName
    @property 
    def applyWaterMask(self):
        return self._applyWaterMask
    def getRgImageName(self):
        return self._rgImageName

    def getDemInitFile(self):
        return self._demInitFile

    def getDemImage(self):
        return self._demImage

    def getOffsetImageName(self):
        return self._offsetImageName

    def getResampImageName(self):
        return self._resampImageName
    @property
    def resampOnlyAmpName(self):
        return self._resampOnlyAmpName
    def getResampOnlyImageName(self):
        return self._resampOnlyImageName
    def getTopocorrectFlatImage(self):
        return self._topocorrectFlatImage

    def getFirstSampleAcrossPrf(self):
        return self._firstSampleAcrossPrf

    def getFirstSampleDownPrf(self):
        return self._firstSampleDownPrf

    def getNumberRangeBins(self):
        return self._numberRangeBins

    def getNumberLocationAcrossPrf(self):
        return self._numberLocationAcrossPrf

    def getNumberLocationDownPrf(self):
        return self._numberLocationDownPrf

    def getFirstSampleAcross(self):
        return self._firstSampleAcross

    def getFirstSampleDown(self):
        return self._firstSampleDown

    def getNumberLocationAcross(self):
        return self._numberLocationAcross

    def getNumberLocationDown(self):
        return self._numberLocationDown

    def getOffsetField(self):
        return self._offsetField

    def getRefinedOffsetField(self):
        return self._refinedOffsetField

    def getOffsetField1(self):
        return self._offsetField1

    def getRefinedOffsetField1(self):
        return self._refinedOffsetField1

    def getNumberValidPulses(self):
        return self._numberValidPulses

    def getNumberPatches(self):
        return self._numberPatches

    def getPatchSize(self):
        return self._patchSize

    def getMachineEndianness(self):
        return self._machineEndianness

    def getSecondaryRangeMigrationFlag(self):
        return self._secondaryRangeMigrationFlag

    def getChirpExtension(self):
        return self._chirpExtension

    def getSlantRangePixelSpacing(self):
        return self._slantRangePixelSpacing

    def getDopplerCentroid(self):
        return self._dopplerCentroid

    def getPosting(self):
        return self._posting

    def getNumberFitCoefficients(self):
        return self._numberFitCoefficients

    def getNumberLooks(self):
        return self._numberLooks

    def getNumberAzimuthLooks(self):
        return self._numberAzimuthLooks

    def getNumberRangeLooks(self):
        return self._numberRangeLooks

    def getNumberResampLines(self):
        return self._numberResampLines

    def getShadeFactor(self):
        return self._shadeFactor

    def getTopophaseFlatFilename(self):
        return self._topophaseFlatFilename

    def getFiltTopophaseFlatFilename(self):
        return self._filt_topophaseFlatFilename

    def getCoherenceFilename(self):
        return self._coherenceFilename

    def getUnwrappedIntFilename(self):
        return self._unwrappedIntFilename

    def getUnwrapped2StageFilename(self):
        return self._unwrapped2StageFilename
      
    def getConnectedComponentsFilename(self):
        return self._connectedComponentsFilename

    def getPhsigFilename(self):
        return self._phsigFilename

    def getTopophaseMphFilename(self):
        return self._topophaseMphFilename

    def getHeightFilename(self):
        return self._heightFilename

    def getHeightSchFilename(self):
        return self._heightSchFilename

    def getGeocodeFilename(self):
        return self._geocodeFilename

    def getLosFilename(self):
        return self._losFilename

    def getLatFilename(self):
        return self._latFilename

    def getLonFilename(self):
        return self._lonFilename

    def getDemCropFilename(self):
        return self._demCropFilename

    def getTopophaseIterations(self):
        return self._topophaseIterations

    def getFilterStrength(self):
        return self._filterStrength

    def getGeocodeList(self):
        return self._geocode_list

    def getRawReferenceIQImage(self):
        return self._rawReferenceIQImage

    def getRawSecondaryIQImage(self):
        return self._rawSecondaryIQImage
    @property 
    def azResFactor(self):
        return self._azResFactor
    @property
    def wbdImage(self):
        return self._wbdImage
    @property 
    def waterMaskImageName(self):
        return self._waterMaskImageName 
    @property
    def unmaskedPrefix(self):
        return self._unmaskedPrefix
    # Setters
    @proceedIfZeroDem.setter
    def proceedIfZeroDem(self,proceedIfZeroDem):
        self._proceedIfZeroDem = proceedIfZeroDem
        
    def setLookSide(self, lookSide):
        self._lookSide = lookSide

    def setReferenceSquint(self, squint):
        self._referenceSquint = squint

    def setSecondarySquint(self, squint):
        self._secondarySquint = squint

    def setFormSLC1(self, fslc):
        self._formSLC1 = fslc

    def setFormSLC2(self, fslc):
        self._formSLC2 = fslc

    def setMocompBaseline(self, mocompbl):
        self._mocompBaseline = mocompbl

    def setTopo(self, topo):
        self._topo = topo

    def setTopocorrect(self, topo):
        self._topocorrect = topo

    def setFirstAverageHeight(self, h1):
        self._pegH1 = h1

    def setSecondAverageHeight(self, h2):
        self._pegH2 = h2

    def setFirstFdHeight(self, h1):
        self._fdH1 = h1

    def setSecondFdHeight(self, h2):
        self._fdH2 = h2

    def setFirstProcVelocity(self, v1):
        self._pegV1 = v1

    def setSecondProcVelocity(self, v2):
        self._pegV2 = v2


    def setReferenceFrame(self, frame):
        self._referenceFrame = frame

    def setSecondaryFrame(self, frame):
        self._secondaryFrame = frame

    def setReferenceOrbit(self, orbit):
        self._referenceOrbit = orbit

    def setSecondaryOrbit(self, orbit):
        self._secondaryOrbit = orbit

    def setReferenceDoppler(self, doppler):
        self._referenceDoppler = doppler

    def setSecondaryDoppler(self, doppler):
        self._secondaryDoppler = doppler

    def setPeg(self, peg):
        self._peg = peg

    def setReferenceRawImage(self, image):
        self._referenceRawImage = image

    def setSecondaryRawImage(self, image):
        self._secondaryRawImage = image

    def setReferenceSlcImage(self, image):
        self._referenceSlcImage = image

    def setSecondarySlcImage(self, image):
        self._secondarySlcImage = image

    def setSimAmpImage(self, image):
        self._simAmpImage = image

    def setRgImage(self, image):
        self._rgImage = image

    def setOffsetAzimuthImage(self, image):
        self._offsetAzimuthImage = image

    def setOffsetRangeImage(self, image):
        self._offsetRangeImage = image

    def setResampAmpImage(self, image):
        self._resampAmpImage = image

    def setResampIntImage(self, image):
        self._resampIntImage = image

    def setResampOnlyImage(self, image):
        self._resampOnlyImage = image
    def setResampOnlyAmp(self, image):
        self._resampOnlyAmp = image

    def setTopoIntImage(self, image):
        self._topoIntImage = image

    def setHeightTopoImage(self, image):
        self._heightTopoImage = image

    def setSimAmpImageName(self, name):
        self._simAmpImageName = name
    @applyWaterMask.setter 
    def applyWaterMask(self,val):
        self._applyWaterMask = val
    def setSLC1ImageName(self, name):
        self._slc1ImageName = name

    def setSLC2ImageName(self, name):
        self._slc2ImageName = name

    def setRgImageName(self, name):
        self._rgImageName = name

    def setOffsetImageName(self, name):
        self._offsetImageName = name

    def setResampImageName(self, name):
        self._resampImageName = name
    def setResampOnlyImageName(self, name):
        self._resampOnlyImageName = name
    @resampOnlyAmpName.setter
    def resampOnlyAmpName(self, name):
        self._resampOnlyAmpName = name

    def setDemImage(self, image):
        self._demImage = image

    def setDemInitFile(self, init):
        self._demInitFile = init

    def setTopocorrectFlatImage(self, image):
        self._topocorrectFlatImage = image

    def setFirstSampleAcrossPrf(self, x):
        self._firstSampleAcrossPrf = x

    def setFirstSampleDownPrf(self, x):
        self._firstSampleDownPrf = x

    def setNumberRangeBins(self, x):
        self._numberRangeBins = x

    def setNumberLocationAcrossPrf(self, x):
        self._numberLocationAcrossPrf = x

    def setNumberLocationDownPrf(self, x):
        self._numberLocationDownPrf = x

    def setFirstSampleAcross(self, x):
        self._firstSampleAcross = x

    def setFirstSampleDown(self, x):
        self._firstSampleDown = x

    def setNumberLocationAcross(self, x):
        self._numberLocationAcross = x

    def setNumberLocationDown(self, x):
        self._numberLocationDown = x

    def setOffsetField(self, offsets):
        self._offsetField = offsets

    def setRefinedOffsetField(self, offsets):
        self._refinedOffsetField = offsets

    def setOffsetField1(self, offsets):
        self._offsetField1 = offsets

    def setRefinedOffsetField1(self, offsets):
        self._refinedOffsetField1 = offsets


    def setNumberValidPulses(self, x):
        self._numberValidPulses = x

    def setNumberPatches(self, x):
        self._numberPatches = x

    def setPatchSize(self, x):
        self._patchSize = x

    def setMachineEndianness(self, x):
        self._machineEndianness = x

    def setSecondaryRangeMigrationFlag(self, yorn):
        """Should be 'y' or 'n'"""
        self._secondaryRangeMigrationFlag = yorn

    def setChirpExtension(self, ext):
        """Should probably be a percentage rather than value"""
        self._chirpExtension = int(ext)
        return None
    
    @property
    def chirpExtensionPercentage(self):
        return NotImplemented
    @chirpExtensionPercentage.setter
    def chirpExtensionPercentage(self, value):
        raise AttributeError("Can only set chirpExtension")
   
    def setSlantRangePixelSpacing(self, x):
        self._slantRangePixelSpacing = x

    def setDopplerCentroid(self, x):
        self._dopplerCentroid = x

    def setPosting(self, x):
        self._posting = x

    def setNumberFitCoefficients(self, x):
        self._numberFitCoefficients = x

    def setNumberLooks(self, x):
        self._numberLooks = int(x)

    def setNumberAzimuthLooks(self, x):
        self._numberAzimuthLooks = int(x)

    def setNumberRangeLooks(self, x):
        self._numberRangeLooks = int(x)

    def setNumberResampLines(self, x):
        self._numberResampLines = int(x)

    def setShadeFactor(self, x):
        self._shadeFactor = x

    def setTopophaseFlatFilename(self, filename):
        self._topophaseFlatFilename = filename

    def setFiltTopophaseFlatFilename(self, filename):
        self._filt_topophaseFlatFilename = filename

    def setCoherenceFilename(self, filename):
        self._coherenceFilename = filename

    def setUnwrappedIntFilename(self, filename):
        self._unwrappedIntFilename = filename

    def setUnwrapped2StageFilename(self, filename):
        self._unwrapped2StageFilename= filename

    def setConnectedComponentsFilename(self,val):
        self._connectedComponentsFilename = val

    def setPhsigFilename(self, filename):
        self._phsigFilename = filename

    def setTopophaseMphFilename(self, filename):
        self._topophaseMphFilename = filename

    def setHeightFilename(self, filename):
        self._heightFilename = filename

    def setHeightSchFilename(self, filename):
        self._heightSchFilename = filename

    def setGeocodeFilename(self, filename):
        self._geocodeFilename = filename

    def setLosFilename(self, filename):
        self._losFilename = filename

    def setLatFilename(self, filename):
        self._latFilename = filename

    def setLonFilename(self, filename):
        self._lonFilename = filename

    def setDemCropFilename(self, filename):
        self._demCropFilename = filename

    def setTopophaseIterations(self, iter):
        self._topophaseIterations = iter

    def setFilterStrength(self, alpha):
        self._filterStrength = alpha

    def setGeocodeList(self,prd):
        self._geocode_list = prd

    def setRawReferenceIQImage(self,im):
        self._rawReferenceIQImage = im

    def setRawSecondaryIQImage(self,im):
        self._rawSecondaryIQImage = im
    
    @azResFactor.setter
    def azResFactor(self,val):
        self._azResFactor = val
    @wbdImage.setter
    def wbdImage(self,val):
        self._wbdImage = val
    @waterMaskImageName.setter
    def waterMaskImageName(self,val):
        self._waterMaskImageName = val
    @unmaskedPrefix.setter
    def unmaskedPrefix(self,val):
        self._unmaskedPrefix = val
    ## folowing are tbd to split formSLC.
    def _hasher(self, index, Attr):
        return getattr(self, REFERENCE_SECONDARY[index] + Attr)

    def select_frame(self, index): return self._hasher(index, 'Frame')
    def select_orbit(self, index): return self._hasher(index, 'Orbit')
    def select_doppler(self, index): return self._hasher(index, 'Doppler')
    def select_rawimage(self, index): return self._hasher(index, 'RawImage')
    def select_slcimage(self, index): return self._hasher(index, 'SlcImage')
    def select_squint(self, index): return self._hasher(index, 'SquintImage')

    def iter_orbits(self):
        return (self.select_orbit(n) for n in range(2))

    def select_swath(self, index):
        return RadarSwath(frame=self.select_frame(index),
                          orbit=self.select_orbit(index),
                          doppler=self.select_doppler(index),
                          rawimage=self.select_rawimage(index),
                          slcimage=self.select_slcimage(index),
                          squint=self.select_squint(index))

    ## This overides the _FrameMixin.frame
    @property
    def frame(self):
        return self.referenceFrame

    # Some line violate PEP008 in order to facilitate using "grep"
    # for development
    refinedOffsetField = property(getRefinedOffsetField, setRefinedOffsetField)
    offsetField = property(getOffsetField, setOffsetField)
    demCropFilename = property(getDemCropFilename, setDemCropFilename)
    referenceFrame = property(getReferenceFrame, setReferenceFrame)
    secondaryFrame = property(getSecondaryFrame, setSecondaryFrame)
    referenceOrbit = property(getReferenceOrbit, setReferenceOrbit)
    secondaryOrbit = property(getSecondaryOrbit, setSecondaryOrbit)
    referenceDoppler = property(getReferenceDoppler, setReferenceDoppler)
    secondaryDoppler = property(getSecondaryDoppler, setSecondaryDoppler)
    peg = property(getPeg, setPeg)
    pegH1 = property(getFirstAverageHeight, setFirstAverageHeight)
    pegH2 = property(getSecondAverageHeight, setSecondAverageHeight)
    fdH1 = property(getFirstFdHeight, setFirstFdHeight)
    fdH2 = property(getSecondFdHeight, setSecondFdHeight)
    pegV1 = property(getFirstProcVelocity, setFirstProcVelocity)
    pegV2 = property(getSecondProcVelocity, setSecondProcVelocity)
    referenceRawImage = property(getReferenceRawImage, setReferenceRawImage)
    secondaryRawImage = property(getSecondaryRawImage, setSecondaryRawImage)
    referenceSlcImage = property(getReferenceSlcImage, setReferenceSlcImage)
    secondarySlcImage = property(getSecondarySlcImage, setSecondarySlcImage)
    simAmpImage = property(getSimAmpImage, setSimAmpImage)
    demImage = property(getDemImage, setDemImage)
    demInitFile = property(getDemInitFile, setDemInitFile)
    rgImage = property(getRgImage, setRgImage)
    topocorrectFlatImage = property(getTopocorrectFlatImage, setTopocorrectFlatImage)
    resampAmpImage = property(getResampAmpImage, setResampAmpImage)
    resampIntImage = property(getResampIntImage, setResampIntImage)
    resampOnlyImage = property(getResampOnlyImage, setResampOnlyImage)
    topoIntImage = property(getTopoIntImage, setTopoIntImage)
    heightTopoImage = property(getHeightTopoImage, setHeightTopoImage)
    offsetAzimuthImage = property(getOffsetAzimuthImage, setOffsetAzimuthImage)
    offsetRangeImage = property(getOffsetRangeImage, setOffsetRangeImage)
    slc1ImageName = property(getSLC1ImageName, setSLC1ImageName)
    slc2ImageName = property(getSLC2ImageName, setSLC2ImageName)
    rgImageName = property(getRgImageName, setRgImageName)
    resampOnlyImageName = property(getResampOnlyImageName, setResampOnlyImageName)
    resampImageName = property(getResampImageName, setResampImageName)
    offsetImageName = property(getOffsetImageName, setOffsetImageName)
    chirpExtension = property(getChirpExtension, setChirpExtension)
    firstSampleAcrossPrf = property(getFirstSampleAcrossPrf, setFirstSampleAcrossPrf)
    firstSampleDownPrf = property(getFirstSampleDownPrf, setFirstSampleDownPrf)
    numberLocationAcrossPrf = property(getNumberLocationAcrossPrf, setNumberLocationAcrossPrf)
    numberLocationDownPrf = property(getNumberLocationDownPrf, setNumberLocationDownPrf)
    firstSampleAcross = property(getFirstSampleAcross, setFirstSampleAcross)
    firstSampleDown = property(getFirstSampleDown, setFirstSampleDown)
    numberLocationAcross = property(getNumberLocationAcross, setNumberLocationAcross)
    numberLocationDown = property(getNumberLocationDown, setNumberLocationDown)
    numberAzimuthLooks = property(getNumberAzimuthLooks, setNumberAzimuthLooks)
    numberValidPulses = property(getNumberValidPulses, setNumberValidPulses)
    numberPatches = property(getNumberPatches, setNumberPatches)
    patchSize = property(getPatchSize, setPatchSize)
    machineEndianness = property(getMachineEndianness, setMachineEndianness)
    secondaryRangeMigrationFlag = property(getSecondaryRangeMigrationFlag, setSecondaryRangeMigrationFlag)
    coherenceFilename = property(getCoherenceFilename, setCoherenceFilename)
    unwrappedIntFilename = property(getUnwrappedIntFilename, setUnwrappedIntFilename)
    unwrapped2StageFilename = property(getUnwrapped2StageFilename, setUnwrapped2StageFilename)
    connectedComponentsFilename = property(getConnectedComponentsFilename,setConnectedComponentsFilename)
    phsigFilename = property(getPhsigFilename, setPhsigFilename)
    topophaseMphFilename = property(getTopophaseMphFilename, setTopophaseMphFilename)
    topophaseFlatFilename = property(getTopophaseFlatFilename, setTopophaseFlatFilename)
    filt_topophaseFlatFilename = property(getFiltTopophaseFlatFilename, setFiltTopophaseFlatFilename)
    heightFilename = property(getHeightFilename, setHeightFilename)
    heightSchFilename = property(getHeightSchFilename, setHeightSchFilename)
    geocodeFilename = property(getGeocodeFilename, setGeocodeFilename)
    losFilename = property(getLosFilename, setLosFilename)
    latFilename = property(getLatFilename, setLatFilename)
    lonFilename = property(getLonFilename, setLonFilename)
    lookSide = property(getLookSide, setLookSide)
    topophaseIterations = property(getTopophaseIterations, setTopophaseIterations)
    slantRangePixelSpacing = property(getSlantRangePixelSpacing, setSlantRangePixelSpacing)
    dopplerCentroid = property(getDopplerCentroid, setDopplerCentroid)
    posting = property(getPosting, setPosting)
    numberLooks = property(getNumberLooks, setNumberLooks)
    numberFitCoefficients = property(getNumberFitCoefficients, setNumberFitCoefficients)
    numberAzimuthLooks = property(getNumberAzimuthLooks, setNumberAzimuthLooks)
    numberRangeLooks = property(getNumberRangeLooks, setNumberRangeLooks)
    numberResampLines = property(getNumberResampLines, setNumberResampLines)
    numberRangeBins = property(getNumberRangeBins, setNumberRangeBins)
    shadeFactor = property(getShadeFactor, setShadeFactor)
    filterStrength = property(getFilterStrength, setFilterStrength)
    formSLC1 = property(getFormSLC1, setFormSLC1)
    formSLC2 = property(getFormSLC2, setFormSLC2)
    mocompBaseline = property(getMocompBaseline, setMocompBaseline)
    topocorrect = property(getTopocorrect, setTopocorrect)
    topo = property(getTopo, setTopo)
    referenceSquint = property(getReferenceSquint, setReferenceSquint)
    secondarySquint = property(getSecondarySquint, setSecondarySquint)
    geocode_list = property(getGeocodeList, setGeocodeList)
    rawReferenceIQImage = property(getRawReferenceIQImage, setRawReferenceIQImage)
    rawSecondaryIQImage = property(getRawSecondaryIQImage, setRawSecondaryIQImage)

    pass


## Why this: the code bloat with reference this and secondary that indicates the
## design princple does not use composition, this is an attempt to
## fix that
class RadarSwath(object):
    def __init__(self,
                 frame=None,
                 orbit=None,
                 doppler=None,
                 rawimage=None,
                 slcimage=None,
                 squint=None):
        self.frame = frame
        self.orbit = orbit
        self.doppler = doppler
        self.rawimage = rawimage
        self.slcimage = slcimage
        self.squint = squint
        return None
    pass
