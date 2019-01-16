#
# Author: Piyush Agram
# Copyright 2018
#

import os
import logging
import logging.config
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from iscesys.Compatibility import Compatibility


MASTER_SLC_PRODUCT = Component.Parameter('masterSlcProduct',
                                public_name='master slc product',
                                default='master',
                                type=str,
                                mandatory=False,
                                doc='Directory name of the master SLC product')


SLAVE_SLC_PRODUCT = Component.Parameter('slaveSlcProduct',
                                public_name='slave slc product',
                                default='slave',
                                type=str,
                                mandatory=False,
                                doc='Directory name of the slave SLC product')

NUMBER_COMMON_BURSTS = Component.Parameter('numberOfCommonBursts',
                                public_name = 'number of common bursts',
                                default = None,
                                type = int,
                                container=list,
                                mandatory = False,
                                doc = 'Number of common bursts between slave and master')


DEM_FILENAME = Component.Parameter('demFilename',
                                public_name='dem image name',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'Name of the dem file')

GEOMETRY_DIRNAME = Component.Parameter('geometryDirname',
                                public_name='geometry directory name',
                                default='geom_master',
                                type=str,
                                mandatory=False, 
                                doc = 'Geometry directory')

COMMON_RANGE_SPECTRA_SLC_DIRECTORY = Component.Parameter('commonRangeSpectraSlcDirectory',
                                public_name='equalized slc directory name',
                                default='commonrangespectra_slc',
                                type=str,
                                mandatory=False,
                                doc='directory with common range spectral slcs')

RANGE_SPECTRA_OVERLAP_THRESHOLD = Component.Parameter('rangeSpectraOverlapThreshold',
                                public_name='range spectra overlap threshold',
                                default=3.0e6,
                                type=float,
                                mandatory=False,
                                doc='Minimum range spectra overlap needed')

EQUALIZED_SLC_DIRECTORY = Component.Parameter('equalizedSlcDirectory',
                                public_name='equalized slc directory',
                                default='equalized_slc',
                                type=str,
                                mandatory=False,
                                doc='Directory with equalized slcs')

BURST_SYNC_DIRECTORY = Component.Parameter('burstSyncDirectory',
                                public_name='bursy sync directory',
                                default='burst_sync',
                                type=str,
                                mandatory=False,
                                doc='Directory with burst sync information')

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

NUMBER_OF_SWATHS = Component.Parameter('numberOfSwaths',
                                public_name = 'number of swaths',
                                default=0,
                                type=int,
                                mandatory = False,
                                doc = 'Number of swaths')

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
                   FILT_OFFSET_OUTPUT_FILE],
        container = list,
        type=str,
        mandatory=False,
        doc = 'List of files on offset grid to geocode')



class ScansarProc(Component):
    """
    This class holds the properties, along with methods (setters and getters)
    to modify and return their values.
    """

    parameter_list = (MASTER_SLC_PRODUCT,
                      SLAVE_SLC_PRODUCT,
                      DEM_FILENAME,
                      GEOMETRY_DIRNAME,
                      COMMON_RANGE_SPECTRA_SLC_DIRECTORY,
                      RANGE_SPECTRA_OVERLAP_THRESHOLD,
                      EQUALIZED_SLC_DIRECTORY,
                      BURST_SYNC_DIRECTORY,
                      COARSE_OFFSETS_DIRECTORY,
                      COARSE_COREG_DIRECTORY,
                      COARSE_IFG_DIRECTORY,
                      FINE_OFFSETS_DIRECTORY,
                      FINE_COREG_DIRECTORY,
                      FINE_IFG_DIRECTORY,
                      OVERLAPS_SUBDIRECTORY,
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
                      FILT_OFFSET_OUTPUT_FILE,
                      OFFSET_GEOCODE_LIST)

    facility_list = ()


    family='scansarcontext'

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


    def getMergedOrbit(self, products):
        from isceobj.Orbit.Orbit import Orbit

        ###Create merged orbit
        orb = Orbit()
        orb.configure()

        burst = product[0]
        #Add first burst orbit to begin with
        for sv in burst.orbit:
             orb.addStateVector(sv)


        for pp in product:
            ##Add all state vectors
            for sv in pp.orbit:
                if (sv.time< orb.minTime) or (sv.time > orb.maxTime):
                    orb.addStateVector(sv)

                pp.orbit = orb

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
        
        import glob

        inswaths = glob.glob( os.path.join(self.masterSlcProduct, 's*.xml'))
        
        swaths = []
        for x in inswaths:
            swaths.append( int(os.path.splitext(os.path.basename(x))[0][-1]))
        
        return sorted(swaths)

    def hasGPU(self):
        '''
        Determine if GPU modules are available.
        '''

        flag = False
        try:
            from zerodop.GPUtopozero.GPUtopozero import PyTopozero
            from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr
            flag = True
        except:
            pass

        return flag

    def getOverlapFrequency(self, centerfreq1, bandwidth1,
                                  centerfreq2, bandwidth2):
        startfreq1 = centerfreq1 - bandwidth1 / 2.0
        endingfreq1 = centerfreq1 + bandwidth1 / 2.0

        startfreq2 = centerfreq2 - bandwidth2 / 2.0
        endingfreq2 = centerfreq2 + bandwidth2 / 2.0

        overlapfreq = []
        if startfreq2 <= startfreq1 <= endingfreq2:
            overlapfreq.append(startfreq1)

        if startfreq2 <= endingfreq1 <= endingfreq2:
            overlapfreq.append(endingfreq1)

        if startfreq1 < startfreq2 < endingfreq1:
            overlapfreq.append(startfreq2)
      
        if startfreq1 < endingfreq2 < endingfreq1:
            overlapfreq.append(endingfreq2)

        if len(overlapfreq) != 2:
            #no overlap bandwidth
            return None
        else:
            startfreq = min(overlapfreq)
            endingfreq = max(overlapfreq)
            return [startfreq, endingfreq]

    @property
    def commonRangeSpectraMasterSlcProduct(self):
        infile = self.masterSlcProduct
        if infile[-1] == os.path.sep:
            infile = infile[:-1]

        base = os.path.sep.join( infile.split(os.path.sep)[-2:])
        return os.path.join( self.commonRangeSpectraSlcDirectory, base)

    @property
    def commonRangeSpectraSlaveSlcProduct(self):
        infile = self.slaveSlcProduct
        if infile[-1] == os.path.sep:
            infile = infile[:-1]

        base = os.path.sep.join( infile.split(os.path.sep)[-2:])
        return os.path.join( self.commonRangeSpectraSlcDirectory, base)

    @property
    def equalizedMasterSlcProduct(self):
        infile = self.masterSlcProduct
        if infile[-1] == os.path.sep:
            infile = infile[:-1]

        base = os.path.sep.join(infile.split(os.path.sep)[-2:])
        return os.path.join( self.equalizedSlcDirectory, base)

    @property
    def equalizedSlaveSlcProduct(self):
        infile = self.slaveSlcProduct
        if infile[-1] == os.path.sep:
            infile = infile[:-1]

        base = os.path.sep.join(infile.split(os.path.sep)[-2:])
        return os.path.join( self.equalizedSlcDirectory, base)

    def writeBurstSyncFile(self, outfile, rgoff, azoff,
                                 nb, nc,
                                 unsynLines, synLines):
        
        with open(outfile, 'w') as fid:
            fid.write('image pair range offset: {0}\n'.format(rgoff))
            fid.write('image pair azimuth offset: {0}\n'.format(azoff))
            fid.write('number of lines in a burst: {0}\n'.format(nb))
            fid.write('number of lines in a burst cycle: {0}\n'.format(nc))
            fid.write('number of unsynchronized lines in a burst: {0}\n'.format(unsynLines))
            fid.write('burst synchronization: {0}%'.format((synLines/nb)*100.0))


