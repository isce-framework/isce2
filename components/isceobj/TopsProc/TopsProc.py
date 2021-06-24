#
# Author: Piyush Agram
# Copyright 2016
#

import os
import logging
import logging.config
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from iscesys.Compatibility import Compatibility


REFERENCE_SLC_PRODUCT = Component.Parameter('referenceSlcProduct',
                                public_name='reference slc product',
                                default='reference',
                                type=str,
                                mandatory=False,
                                doc='Directory name of the reference SLC product')


SECONDARY_SLC_PRODUCT = Component.Parameter('secondarySlcProduct',
                                public_name='secondary slc product',
                                default='secondary',
                                type=str,
                                mandatory=False,
                                doc='Directory name of the secondary SLC product')

COMMON_BURST_START_REFERENCE_INDEX  = Component.Parameter('commonBurstStartReferenceIndex',
                                public_name = 'common burst start reference index',
                                default = None,
                                type = int,
                                container=list,
                                mandatory = False,
                                doc = 'Reference burst start index for common bursts')

COMMON_BURST_START_SECONDARY_INDEX = Component.Parameter('commonBurstStartSecondaryIndex',
                                public_name = 'common burst start secondary index',
                                default = None,
                                type = int,
                                container=list,
                                mandatory = False,
                                doc = 'Secondary burst start index for common bursts')

NUMBER_COMMON_BURSTS = Component.Parameter('numberOfCommonBursts',
                                public_name = 'number of common bursts',
                                default = None,
                                type = int,
                                container=list,
                                mandatory = False,
                                doc = 'Number of common bursts between secondary and reference')


DEM_FILENAME = Component.Parameter('demFilename',
                                public_name='dem image name',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'Name of the dem file')

GEOMETRY_DIRNAME = Component.Parameter('geometryDirname',
                                public_name='geometry directory name',
                                default='geom_reference',
                                type=str,
                                mandatory=False,
                                doc = 'Geometry directory')

ESD_DIRNAME = Component.Parameter('esdDirname',
                                public_name = 'ESD directory name',
                                default = 'ESD',
                                type = str,
                                mandatory = False,
                                doc = 'ESD directory')


COARSE_OFFSETS_DIRECTORY = Component.Parameter('coarseOffsetsDirname',
                                public_name = 'coarse offsets directory name',
                                default = 'coarse_offsets',
                                type = str,
                                mandatory = False,
                                doc = 'coarse offsets directory name')

COARSE_COREG_DIRECTORY = Component.Parameter('coarseCoregDirname',
                                public_name = 'coarse coreg directory name',
                                default = 'coarse_coreg',
                                type = str,
                                mandatory = False,
                                doc = 'coarse coregistered slc directory name')

COARSE_IFG_DIRECTORY = Component.Parameter('coarseIfgDirname',
                                public_name = 'coarse interferogram directory name',
                                default = 'coarse_interferogram',
                                type = str,
                                mandatory = False,
                                doc = 'Coarse interferogram directory')


FINE_OFFSETS_DIRECTORY = Component.Parameter('fineOffsetsDirname',
                                public_name = 'fine offsets directory name',
                                default = 'fine_offsets',
                                type = str,
                                mandatory = False,
                                doc = 'fine offsets directory name')

FINE_COREG_DIRECTORY = Component.Parameter('fineCoregDirname',
                                public_name = 'fine coreg directory name',
                                default = 'fine_coreg',
                                type = str,
                                mandatory = False,
                                doc = 'fine coregistered slc directory name')

FINE_IFG_DIRECTORY = Component.Parameter('fineIfgDirname',
                                public_name = 'fine interferogram directory name',
                                default = 'fine_interferogram',
                                type = str,
                                mandatory = False,
                                doc = 'Fine interferogram directory')

MERGED_DIRECTORY = Component.Parameter('mergedDirname',
                                public_name = 'merged products directory name',
                                default = 'merged',
                                type = str,
                                mandatory = False,
                                doc = 'Merged product directory')

OVERLAPS_SUBDIRECTORY = Component.Parameter('overlapsSubDirname',
                                public_name = 'overlaps subdirectory name',
                                default = 'overlaps',
                                type = str,
                                mandatory = False,
                                doc = 'Overlap region processing directory')

SECONDARY_RANGE_CORRECTION = Component.Parameter('secondaryRangeCorrection',
                                public_name = 'secondary range correction',
                                default = 0.0,
                                type = float,
                                mandatory = False,
                                doc = 'Range correction in m to apply to secondary')

SECONDARY_TIMING_CORRECTION = Component.Parameter('secondaryTimingCorrection',
                                public_name = 'secondary timing correction',
                                default = 0.0,
                                type = float,
                                mandatory = False,
                                doc = 'Timing correction in secs to apply to secondary')

NUMBER_OF_SWATHS = Component.Parameter('numberOfSwaths',
                                public_name = 'number of swaths',
                                default=0,
                                type=int,
                                mandatory = False,
                                doc = 'Number of swaths')

APPLY_WATER_MASK = Component.Parameter(
    'applyWaterMask',
    public_name='apply water mask',
    default=True,
    type=bool,
    mandatory=False,
    doc='Flag to apply water mask to images before unwrapping.'
)

WATER_MASK_FILENAME = Component.Parameter(
    'waterMaskFileName',
    public_name='water mask file name',
    default='waterMask.msk',
    type=str,
    mandatory=False,
    doc='Filename of the water body mask image in radar coordinate cropped to the interferogram size.'
)


MERGED_IFG_NAME = Component.Parameter(
    'mergedIfgname',
    public_name='merged interferogram name',
    default='topophase.flat',
    type=str,
    mandatory=False,
    doc='Filename of the merged interferogram.'
)


MERGED_LOS_NAME = Component.Parameter(
        'mergedLosName',
        public_name = 'merged los name',
        default = 'los.rdr',
        type = str,
        mandatory = False,
        doc = 'Merged los file name')


COHERENCE_FILENAME = Component.Parameter('coherenceFilename',
                                         public_name='coherence name',
                                         default='phsig.cor',
                                         type=str,
                                         mandatory=False,
                                         doc='Coherence file name')

CORRELATION_FILENAME = Component.Parameter('correlationFilename',
                                        public_name='correlation name',
                                        default='topophase.cor',
                                        type=str,
                                        mandatory=False,
                                        doc='Correlation file name')

FILTERED_INT_FILENAME = Component.Parameter('filtFilename',
                                        public_name = 'filtered interferogram name',
                                        default = 'filt_topophase.flat',
                                        type = str,
                                        mandatory = False,
                                        doc = 'Filtered interferogram filename')


UNWRAPPED_INT_FILENAME = Component.Parameter('unwrappedIntFilename',
                                             public_name='unwrapped interferogram filename',
                                             default='filt_topophase.unw',
                                             type=str,
                                             mandatory=False,
                                             doc='')

UNWRAPPED_2STAGE_FILENAME = Component.Parameter('unwrapped2StageFilename',
                                             public_name='unwrapped 2Stage filename',
                                             default='filt_topophase_2stage.unw',
                                             type=str,
                                             mandatory=False,
                                             doc='Output File name of 2Stage unwrapper')

CONNECTED_COMPONENTS_FILENAME = Component.Parameter(
    'connectedComponentsFilename',
    public_name='connected component filename',
    default=None,
    type=str,
    mandatory=False,
    doc=''
)

DEM_CROP_FILENAME = Component.Parameter('demCropFilename',
                                        public_name='dem crop file name',
                                        default='dem.crop',
                                        type=str,
                                        mandatory=False,
                                        doc='')


GEOCODE_LIST = Component.Parameter('geocode_list',
    public_name='geocode list',
    default=[COHERENCE_FILENAME,
             CORRELATION_FILENAME,
             UNWRAPPED_INT_FILENAME,
             MERGED_LOS_NAME,
             MERGED_IFG_NAME,
             FILTERED_INT_FILENAME,
             UNWRAPPED_2STAGE_FILENAME,
             ],
    container=list,
    type=str,
    mandatory=False,
    doc='List of files to geocode'
)

UNMASKED_PREFIX = Component.Parameter('unmaskedPrefix',
                                   public_name='unmasked filename prefix',
                                   default='unmasked',
                                   type=str,
                                   mandatory=False,
                                   doc='Prefix prepended to the image filenames that have not been water masked')


####Adding things from topsOffsetApp for integration
OFFSET_TOP = Component.Parameter(
    'offset_top',
    public_name='Top offset location',
    default=None,
    type=int,
    mandatory=False,
    doc='Ampcor-calculated top offset location. Overridden by workflow.'
                                    )

OFFSET_LEFT = Component.Parameter(
    'offset_left',
    public_name='Left offset location',
    default=None,
    type=int,
    mandatory=False,
    doc='Ampcor-calculated left offset location. Overridden by workflow.'
                                    )

OFFSET_WIDTH = Component.Parameter(
    'offset_width',
    public_name='Offset image nCols',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of columns in the final offset field (calculated in DenseAmpcor).'
                                        )

OFFSET_LENGTH = Component.Parameter(
    'offset_length',
    public_name='Offset image nRows',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of rows in the final offset field (calculated in DenseAmpcor).'
                                        )

OFFSET_OUTPUT_FILE = Component.Parameter(
    'offsetfile',
    public_name='Offset filename',
    default='dense_offsets.bil',
    type=str,
    mandatory=False,
    doc='Filename for gross dense offsets BIL. Used in runDenseOffsets.'
                                            )

OFFSET_SNR_FILE = Component.Parameter(
        'snrfile',
        public_name='Offset SNR filename',
        default='dense_offsets_snr.bil',
        type=str,
        mandatory=False,
        doc='Filename for gross dense offsets SNR. Used in runDenseOffsets.')

OFFSET_COV_FILE = Component.Parameter(
        'covfile',
        public_name='Offset covariance filename',
        default='dense_offsets_cov.bil',
        type=str,
        mandatory=False,
        doc='Filename for gross dense offsets covariance. Used in runDenseOffsets.')

FILT_OFFSET_OUTPUT_FILE = Component.Parameter(
    'filt_offsetfile',
    public_name='Filtered offset filename',
    default='filt_dense_offsets.bil',
    type=str,
    mandatory=False,
    doc='Filename for filtered dense offsets BIL.'
                                                )

OFFSET_GEOCODE_LIST = Component.Parameter('off_geocode_list',
        public_name='offset geocode list',
        default = [OFFSET_OUTPUT_FILE,
                   OFFSET_SNR_FILE,
                   OFFSET_COV_FILE,
                   FILT_OFFSET_OUTPUT_FILE],
        container = list,
        type=str,
        mandatory=False,
        doc = 'List of files on offset grid to geocode')



class TopsProc(Component):
    """
    This class holds the properties, along with methods (setters and getters)
    to modify and return their values.
    """

    parameter_list = (REFERENCE_SLC_PRODUCT,
                      SECONDARY_SLC_PRODUCT,
                      COMMON_BURST_START_REFERENCE_INDEX,
                      COMMON_BURST_START_SECONDARY_INDEX,
                      NUMBER_COMMON_BURSTS,
                      DEM_FILENAME,
                      GEOMETRY_DIRNAME,
                      COARSE_OFFSETS_DIRECTORY,
                      COARSE_COREG_DIRECTORY,
                      COARSE_IFG_DIRECTORY,
                      FINE_OFFSETS_DIRECTORY,
                      FINE_COREG_DIRECTORY,
                      FINE_IFG_DIRECTORY,
                      OVERLAPS_SUBDIRECTORY,
                      SECONDARY_RANGE_CORRECTION,
                      SECONDARY_TIMING_CORRECTION,
                      NUMBER_OF_SWATHS,
                      ESD_DIRNAME,
                      APPLY_WATER_MASK,
                      WATER_MASK_FILENAME,
                      MERGED_DIRECTORY,
                      MERGED_IFG_NAME,
                      MERGED_LOS_NAME,
                      COHERENCE_FILENAME,
                      FILTERED_INT_FILENAME,
                      UNWRAPPED_INT_FILENAME,
                      UNWRAPPED_2STAGE_FILENAME,
                      CONNECTED_COMPONENTS_FILENAME,
                      DEM_CROP_FILENAME,
                      GEOCODE_LIST,
                      UNMASKED_PREFIX,
                      CORRELATION_FILENAME,
                      OFFSET_TOP,
                      OFFSET_LEFT,
                      OFFSET_LENGTH,
                      OFFSET_WIDTH,
                      OFFSET_OUTPUT_FILE,
                      OFFSET_SNR_FILE,
                      OFFSET_COV_FILE,
                      FILT_OFFSET_OUTPUT_FILE,
                      OFFSET_GEOCODE_LIST)

    facility_list = ()


    family='topscontext'

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

        mergedir  = self.mergedDirname
        for i, x in enumerate(self.geocode_list):
            if isinstance(x, Component.Parameter):
                y = getattr(self, getattr(x, 'attrname'))
                self.geocode_list[i] = os.path.join(mergedir, y)


        for i,x in enumerate(self.off_geocode_list):
            if isinstance(x, Component.Parameter):
                y = getattr(self, getattr(x, 'attrname'))
                self.off_geocode_list[i] = os.path.join(mergedir, y)

        return


    def loadProduct(self, xmlname):
        '''
        Load the product using Product Manager.
        '''

        from iscesys.Component.ProductManager import ProductManager as PM

        pm = PM()
        pm.configure()

        obj = pm.loadProduct(xmlname)

        return obj


    def saveProduct(self, obj, xmlname):
        '''
        Save the product to an XML file using Product Manager.
        '''

        from iscesys.Component.ProductManager import ProductManager as PM

        pm = PM()
        pm.configure()

        pm.dumpProduct(obj, xmlname)

        return None

    @property
    def referenceSlcOverlapProduct(self):
        return os.path.join(self.referenceSlcProduct, self.overlapsSubDirname)

    @property
    def coregOverlapProduct(self):
        return os.path.join(self.coarseCoregDirname, self.overlapsSubDirname)

    @property
    def coarseIfgOverlapProduct(self):
        return os.path.join(self.coarseIfgDirname, self.overlapsSubDirname)

    def commonReferenceBurstLimits(self, ind):
        return (self.commonBurstStartReferenceIndex[ind], self.commonBurstStartReferenceIndex[ind] + self.numberOfCommonBursts[ind])

    def commonSecondaryBurstLimits(self, ind):
        return (self.commonBurstStartSecondaryIndex[ind], self.commonBurstStartSecondaryIndex[ind] + self.numberOfCommonBursts[ind])


    def getMergedOrbit(self, product):
        from isceobj.Orbit.Orbit import Orbit

        ###Create merged orbit
        orb = Orbit()
        orb.configure()

        burst = product[0].bursts[0]
        #Add first burst orbit to begin with
        for sv in burst.orbit:
             orb.addStateVector(sv)


        for pp in product:
            ##Add all state vectors
            for bb in pp.bursts:
                for sv in bb.orbit:
                    if (sv.time< orb.minTime) or (sv.time > orb.maxTime):
                        orb.addStateVector(sv)

                bb.orbit = orb

        return orb



    def getInputSwathList(self, inlist):
        '''
        To be used to get list of swaths that user wants us to process.
        '''
        if len(inlist) == 0:
            return [x+1 for x in range(self.numberOfSwaths)]
        else:
            return inlist

    def getValidSwathList(self, inlist):
        '''
        Used to get list of swaths left after applying all filters  - e.g, region of interest.
        '''

        checklist = self.getInputSwathList(inlist)

        validlist = [x for x in checklist if self.numberOfCommonBursts[x-1] > 0]

        return validlist

    def hasGPU(self):
        '''
        Determine if GPU modules are available.
        '''

        flag = False
        try:
            from zerodop.GPUtopozero.GPUtopozero import PyTopozero
            from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr
            from zerodop.GPUresampslc.GPUresampslc import PyResampSlc
            flag = True
        except:
            pass

        return flag
