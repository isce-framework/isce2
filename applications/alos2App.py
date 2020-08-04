#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
# 


import time
import os
import sys
import logging
import logging.config

import isce
import isceobj
import iscesys
from iscesys.Component.Application import Application
from iscesys.Compatibility import Compatibility
from iscesys.Component.Configurable import SELF
from isceobj import Alos2Proc

logging.config.fileConfig(
    os.path.join(os.environ['ISCE_HOME'], 'defaults', 'logging',
        'logging.conf')
)

logger = logging.getLogger('isce.insar')


REFERENCE_DIR = Application.Parameter('referenceDir',
                                public_name='reference directory',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc="reference data directory")

SECONDARY_DIR = Application.Parameter('secondaryDir',
                                public_name='secondary directory',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc="secondary data directory")

REFERENCE_FRAMES = Application.Parameter('referenceFrames',
                                public_name = 'reference frames',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'reference frames to process')

SECONDARY_FRAMES = Application.Parameter('secondaryFrames',
                                public_name = 'secondary frames',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'secondary frames to process')

REFERENCE_POLARIZATION = Application.Parameter('referencePolarization',
                                public_name='reference polarization',
                                default='HH',
                                type=str,
                                mandatory=False,
                                doc="reference polarization to process")

SECONDARY_POLARIZATION = Application.Parameter('secondaryPolarization',
                                public_name='secondary polarization',
                                default='HH',
                                type=str,
                                mandatory=False,
                                doc="secondary polarization to process")

#for ScanSAR-stripmap, always process all swaths, 
#user's settings are overwritten
STARTING_SWATH = Application.Parameter('startingSwath',
                                public_name='starting swath',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="starting swath to process")

ENDING_SWATH = Application.Parameter('endingSwath',
                                public_name='ending swath',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="ending swath to process")

DEM = Application.Parameter('dem',
                                public_name='dem for coregistration',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='dem for coregistration file')

DEM_GEO = Application.Parameter('demGeo',
                                public_name='dem for geocoding',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='dem for geocoding file')

#this water body is used to create water body in radar coordinate used in processing
#radar-coordinate water body is created two times in runRdr2Geo.py and runLook.py, respectively
#radar-coordinate water body is used in:
#(1) determining the number of offsets in slc offset estimation, and radar/dem offset estimation
#(2) masking filtered interferogram or unwrapped interferogram
#(3) determining the number of offsets in slc residual offset estimation after geometric offset
#    computation in coregistering slcs in dense offset.
#(4) masking dense offset field
WBD = Application.Parameter('wbd',
                                public_name='water body',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='water body file')

USE_VIRTUAL_FILE = Application.Parameter('useVirtualFile',
                                public_name = 'use virtual file',
                                default=True,
                                type=bool,
                                mandatory=False,
                                doc = 'use virtual file when possible to save space')

USE_GPU = Application.Parameter('useGPU',
                                public_name='use GPU',
                                default=False,
                                type=bool,
                                mandatory=False,
                                doc='Allow App to use GPU when available')


BURST_SYNCHRONIZATION_THRESHOLD = Application.Parameter('burstSynchronizationThreshold',
                                public_name = 'burst synchronization threshold',
                                default = 75.0,
                                type=float,
                                mandatory = True,
                                doc = 'burst synchronization threshold in percentage')

CROP_SLC = Application.Parameter('cropSlc',
                                public_name = 'crop slc',
                                default=False,
                                type=bool,
                                mandatory=False,
                                doc = 'crop slcs to the overlap area (always crop for ScanSAR-stripmap)')

#for areas where no water body data available, turn this off, otherwise the program will use geometrical offset, which is not accuate enough
#if it still does not work, set "number of range offsets for slc matching" and "number of azimuth offsets for slc matching"
USE_WBD_FOR_NUMBER_OFFSETS = Application.Parameter('useWbdForNumberOffsets',
                                public_name = 'use water body to dertermine number of matching offsets',
                                default = True,
                                type = bool,
                                mandatory = False,
                                doc = 'use water body to dertermine number of matching offsets')

NUMBER_RANGE_OFFSETS = Application.Parameter('numberRangeOffsets',
                                public_name = 'number of range offsets for slc matching',
                                default = None,
                                type = int,
                                mandatory = False,
                                container = list,
                                doc = 'number of range offsets for slc matching')

NUMBER_AZIMUTH_OFFSETS = Application.Parameter('numberAzimuthOffsets',
                                public_name = 'number of azimuth offsets for slc matching',
                                default = None,
                                type = int,
                                mandatory = False,
                                container = list,
                                doc = 'number of azimuth offsets for slc matching')

NUMBER_RANGE_LOOKS1 = Application.Parameter('numberRangeLooks1',
                                public_name='number of range looks 1',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of range looks when forming interferogram")

NUMBER_AZIMUTH_LOOKS1 = Application.Parameter('numberAzimuthLooks1',
                                public_name='number of azimuth looks 1',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of azimuth looks when forming interferogram")

NUMBER_RANGE_LOOKS2 = Application.Parameter('numberRangeLooks2',
                                public_name='number of range looks 2',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of range looks for further multiple looking")

NUMBER_AZIMUTH_LOOKS2 = Application.Parameter('numberAzimuthLooks2',
                                public_name='number of azimuth looks 2',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of azimuth looks for further multiple looking")

NUMBER_RANGE_LOOKS_SIM = Application.Parameter('numberRangeLooksSim',
                                public_name='number of range looks sim',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of range looks when simulating radar image")

NUMBER_AZIMUTH_LOOKS_SIM = Application.Parameter('numberAzimuthLooksSim',
                                public_name='number of azimuth looks sim',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of azimuth looks when simulating radar image")

SWATH_OFFSET_MATCHING = Application.Parameter('swathOffsetMatching',
                                public_name = 'do matching when computing adjacent swath offset',
                                default=True,
                                type=bool,
                                mandatory=False,
                                doc = 'do matching when computing adjacent swath offset')

FRAME_OFFSET_MATCHING = Application.Parameter('frameOffsetMatching',
                                public_name = 'do matching when computing adjacent frame offset',
                                default=True,
                                type=bool,
                                mandatory=False,
                                doc = 'do matching when computing adjacent frame offset')

FILTER_STRENGTH = Application.Parameter('filterStrength',
                                public_name = 'interferogram filter strength',
                                default = 0.3,
                                type=float,
                                mandatory = True,
                                doc = 'interferogram filter strength (power spectrum filter)')

FILTER_WINSIZE = Application.Parameter('filterWinsize',
                                public_name = 'interferogram filter window size',
                                default = 32,
                                type=int,
                                mandatory = False,
                                doc = 'interferogram filter window size')

FILTER_STEPSIZE = Application.Parameter('filterStepsize',
                                public_name = 'interferogram filter step size',
                                default = 4,
                                type=int,
                                mandatory = False,
                                doc = 'interferogram filter step size')

REMOVE_MAGNITUDE_BEFORE_FILTERING = Application.Parameter('removeMagnitudeBeforeFiltering',
                                public_name = 'remove magnitude before filtering',
                                default=True,
                                type=bool,
                                mandatory=False,
                                doc = 'remove magnitude before filtering')

WATERBODY_MASK_STARTING_STEP = Application.Parameter('waterBodyMaskStartingStep',
                                public_name='water body mask starting step',
                                default='unwrap',
                                type=str,
                                mandatory=False,
                                doc='water body mask starting step: None, filt, unwrap')

GEOCODE_LIST = Application.Parameter('geocodeList',
                                public_name = 'geocode file list',
                                default=None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'geocode file list')

GEOCODE_BOUNDING_BOX = Application.Parameter('bbox',
                                public_name = 'geocode bounding box',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'geocode bounding box')

GEOCODE_INTERP_METHOD = Application.Parameter('geocodeInterpMethod',
                                public_name='geocode interpolation method',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='geocode interpolation method: sinc, bilinear, bicubic, nearest')
#####################################################################

#ionospheric correction parameters
DO_ION = Application.Parameter('doIon',
                                public_name = 'do ionospheric phase estimation',
                                default = True,
                                type = bool,
                                mandatory = False,
                                doc = 'do ionospheric phase estimation')

APPLY_ION = Application.Parameter('applyIon',
                                public_name = 'apply ionospheric phase correction',
                                default = True,
                                type = bool,
                                mandatory = False,
                                doc = 'apply ionospheric phase correction')

NUMBER_RANGE_LOOKS_ION = Application.Parameter('numberRangeLooksIon',
                                public_name='number of range looks ion',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of range looks for ionospheric correction")

NUMBER_AZIMUTH_LOOKS_ION = Application.Parameter('numberAzimuthLooksIon',
                                public_name='number of azimuth looks ion',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of azimuth looks for ionospheric correction")

MASKED_AREAS_ION = Application.Parameter('maskedAreasIon',
                                public_name = 'areas masked out in ionospheric phase estimation',
                                default = None,
                                type = int,
                                mandatory = False,
                                container = list,
                                doc = 'areas masked out in ionospheric phase estimation')

FIT_ION = Application.Parameter('fitIon',
                                public_name = 'apply polynomial fit before filtering ionosphere phase',
                                default = True,
                                type = bool,
                                mandatory = False,
                                doc = 'apply polynomial fit before filtering ionosphere phase')

FILTERING_WINSIZE_MAX_ION = Application.Parameter('filteringWinsizeMaxIon',
                                public_name='maximum window size for filtering ionosphere phase',
                                default=151,
                                type=int,
                                mandatory=False,
                                doc='maximum window size for filtering ionosphere phase')

FILTERING_WINSIZE_MIN_ION = Application.Parameter('filteringWinsizeMinIon',
                                public_name='minimum window size for filtering ionosphere phase',
                                default=41,
                                type=int,
                                mandatory=False,
                                doc='minimum window size for filtering ionosphere phase')

FILTER_SUBBAND_INT = Application.Parameter('filterSubbandInt',
                                public_name = 'filter subband interferogram',
                                default = False,
                                type = bool,
                                mandatory = False,
                                doc = 'filter subband interferogram')

FILTER_STRENGTH_SUBBAND_INT = Application.Parameter('filterStrengthSubbandInt',
                                public_name = 'subband interferogram filter strength',
                                default = 0.3,
                                type=float,
                                mandatory = True,
                                doc = 'subband interferogram filter strength (power spectrum filter)')

FILTER_WINSIZE_SUBBAND_INT = Application.Parameter('filterWinsizeSubbandInt',
                                public_name = 'subband interferogram filter window size',
                                default = 32,
                                type=int,
                                mandatory = False,
                                doc = 'subband interferogram filter window size')

FILTER_STEPSIZE_SUBBAND_INT = Application.Parameter('filterStepsizeSubbandInt',
                                public_name = 'subband interferogram filter step size',
                                default = 4,
                                type=int,
                                mandatory = False,
                                doc = 'subband interferogram filter step size')

REMOVE_MAGNITUDE_BEFORE_FILTERING_SUBBAND_INT = Application.Parameter('removeMagnitudeBeforeFilteringSubbandInt',
                                public_name = 'remove magnitude before filtering subband interferogram',
                                default=True,
                                type=bool,
                                mandatory=False,
                                doc = 'remove magnitude before filtering subband interferogram')
#####################################################################

#dense offset parameters
DO_DENSE_OFFSET = Application.Parameter('doDenseOffset',
                                public_name='do dense offset',
                                default = False,
                                type = bool,
                                mandatory = False,
                                doc = 'perform dense offset estimation')

ESTIMATE_RESIDUAL_OFFSET = Application.Parameter('estimateResidualOffset',
                                public_name='estimate residual offset after geometrical coregistration',
                                default = True,
                                type = bool,
                                mandatory = False,
                                doc = 'estimate residual offset after geometrical coregistration')

DELETE_GEOMETRY_FILES = Application.Parameter('deleteGeometryFiles',
                                public_name='delete geometry files used for dense offset estimation',
                                default = False,
                                type = bool,
                                mandatory = False,
                                doc = 'delete geometry files used for dense offset estimation')


#for the following set of matching parameters
#from: dense offset estimation window width
#to: dense offset covariance surface oversample window size
#normally we only have to set the following parameters.
#a good set of parameters other than default is:
#    <property name="dense offset estimation window width">128</property>
#    <property name="dense offset estimation window hight">128</property>
#    <property name="dense offset skip width">64</property>
#    <property name="dense offset skip hight">64</property>

OFFSET_WINDOW_WIDTH = Application.Parameter('offsetWindowWidth',
                                public_name='dense offset estimation window width',
                                default=64,
                                type=int,
                                mandatory=False,
                                doc='dense offset estimation window width')

OFFSET_WINDOW_HEIGHT = Application.Parameter('offsetWindowHeight',
                                public_name='dense offset estimation window hight',
                                default=64,
                                type=int,
                                mandatory=False,
                                doc='dense offset estimation window hight')

#NOTE: actual number of resulting correlation pixels: offsetSearchWindowWidth*2+1
OFFSET_SEARCH_WINDOW_WIDTH = Application.Parameter('offsetSearchWindowWidth',
                                public_name='dense offset search window width',
                                default=8,
                                type=int,
                                mandatory=False,
                                doc='dense offset search window width')

#NOTE: actual number of resulting correlation pixels: offsetSearchWindowHeight*2+1
OFFSET_SEARCH_WINDOW_HEIGHT = Application.Parameter('offsetSearchWindowHeight',
                                public_name='dense offset search window hight',
                                default=8,
                                type=int,
                                mandatory=False,
                                doc='dense offset search window hight')

OFFSET_SKIP_WIDTH = Application.Parameter('offsetSkipWidth',
                                public_name='dense offset skip width',
                                default=32,
                                type=int,
                                mandatory=False,
                                doc='dense offset skip width')

OFFSET_SKIP_HEIGHT = Application.Parameter('offsetSkipHeight',
                                public_name='dense offset skip hight',
                                default=32,
                                type=int,
                                mandatory=False,
                                doc='dense offset skip hight')

OFFSET_COVARIANCE_OVERSAMPLING_FACTOR = Application.Parameter('offsetCovarianceOversamplingFactor',
                                public_name='dense offset covariance surface oversample factor',
                                default=64,
                                type=int,
                                mandatory=False,
                                doc='dense offset covariance surface oversample factor')

OFFSET_COVARIANCE_OVERSAMPLING_WINDOWSIZE = Application.Parameter('offsetCovarianceOversamplingWindowsize',
                                public_name='dense offset covariance surface oversample window size',
                                default=16,
                                type=int,
                                mandatory=False,
                                doc='dense offset covariance surface oversample window size')

MASK_OFFSET_WITH_WBD = Application.Parameter('maskOffsetWithWbd',
                                public_name='mask dense offset with water body',
                                default = True,
                                type = bool,
                                mandatory = False,
                                doc = 'mask dense offset with water body')

DO_OFFSET_FILTERING = Application.Parameter('doOffsetFiltering',
                                public_name='do offset filtering',
                                default = False,
                                type = bool,
                                mandatory = False,
                                doc = 'perform dense offset filtering')

OFFSET_FILTER_WINDOWSIZE = Application.Parameter('offsetFilterWindowsize',
                                public_name='offset filter window size',
                                default=3,
                                type=int,
                                mandatory=False,
                                doc='offset filter window size')

OFFSET_FILTER_SNR_THRESHOLD = Application.Parameter('offsetFilterSnrThreshold',
                                public_name = 'offset filter snr threshold',
                                default = 0.0,
                                type=float,
                                mandatory = False,
                                doc = 'offset filter snr threshold')
#####################################################################

#system parameters
PICKLE_DUMPER_DIR = Application.Parameter('pickleDumpDir',
                                public_name='pickle dump directory',
                                default='PICKLE',
                                type=str,
                                mandatory=False,
                                doc="If steps is used, the directory in which to store pickle objects.")

PICKLE_LOAD_DIR = Application.Parameter('pickleLoadDir',
                                public_name='pickle load directory',
                                default='PICKLE',
                                type=str,
                                mandatory=False,
                                doc="If steps is used, the directory from which to retrieve pickle objects.")

RENDERER = Application.Parameter('renderer',
                                public_name='renderer',
                                default='xml',
                                type=str,
                                mandatory=True,
                                doc="Format in which the data is serialized when using steps. Options are xml (default) or pickle.")
#####################################################################

#Facility declarations
REFERENCE = Application.Facility('reference',
                                public_name='reference',
                                module='isceobj.Sensor.MultiMode',
                                factory='createSensor',
                                args=('ALOS2', 'reference'),
                                mandatory=True,
                                doc="reference component")

SECONDARY = Application.Facility('secondary',
                                public_name='secondary',
                                module='isceobj.Sensor.MultiMode',
                                factory='createSensor',
                                args=('ALOS2','secondary'),
                                mandatory=True,
                                doc="secondary component")

# RUN_UNWRAPPER = Application.Facility('runUnwrapper',
#                                 public_name='Run unwrapper',
#                                 module='isceobj.Alos2Proc',
#                                 factory='createUnwrapper',
#                                 args=(SELF(), DO_UNWRAP, UNWRAPPER_NAME),
#                                 mandatory=False,
#                                 doc="Unwrapping module")

# RUN_UNWRAP_2STAGE = Application.Facility('runUnwrap2Stage',
#                                 public_name='Run unwrapper 2 Stage',
#                                 module='isceobj.Alos2Proc',
#                                 factory='createUnwrap2Stage',
#                                 args=(SELF(), DO_UNWRAP_2STAGE, UNWRAPPER_NAME),
#                                 mandatory=False,
#                                 doc="Unwrapping module")

_INSAR = Application.Facility('_insar',
                                public_name='alos2proc',
                                module='isceobj.Alos2Proc',
                                factory='createAlos2Proc',
                                args = ('alos2AppContext',isceobj.createCatalog('alos2Proc')),
                                mandatory=False,
                                doc="Alos2Proc object")


## Common interface for all insar applications.
class Alos2InSAR(Application):
    family = 'alos2insar'
    parameter_list = (REFERENCE_DIR,
                        SECONDARY_DIR,
                        REFERENCE_FRAMES,
                        SECONDARY_FRAMES,
                        REFERENCE_POLARIZATION,
                        SECONDARY_POLARIZATION,
                        STARTING_SWATH,
                        ENDING_SWATH,
                        DEM,
                        DEM_GEO,
                        WBD,
                        USE_VIRTUAL_FILE,
                        USE_GPU,
                        BURST_SYNCHRONIZATION_THRESHOLD,
                        CROP_SLC,
                        USE_WBD_FOR_NUMBER_OFFSETS,
                        NUMBER_RANGE_OFFSETS,
                        NUMBER_AZIMUTH_OFFSETS,
                        NUMBER_RANGE_LOOKS1,
                        NUMBER_AZIMUTH_LOOKS1,
                        NUMBER_RANGE_LOOKS2,
                        NUMBER_AZIMUTH_LOOKS2,
                        NUMBER_RANGE_LOOKS_SIM,
                        NUMBER_AZIMUTH_LOOKS_SIM,
                        SWATH_OFFSET_MATCHING,
                        FRAME_OFFSET_MATCHING,
                        FILTER_STRENGTH,
                        FILTER_WINSIZE,
                        FILTER_STEPSIZE, 
                        REMOVE_MAGNITUDE_BEFORE_FILTERING,
                        WATERBODY_MASK_STARTING_STEP,
                        GEOCODE_LIST,
                        GEOCODE_BOUNDING_BOX,
                        GEOCODE_INTERP_METHOD,
                        #ionospheric correction parameters
                        DO_ION,
                        APPLY_ION,
                        NUMBER_RANGE_LOOKS_ION,
                        NUMBER_AZIMUTH_LOOKS_ION,
                        MASKED_AREAS_ION,
                        FIT_ION,
                        FILTERING_WINSIZE_MAX_ION,
                        FILTERING_WINSIZE_MIN_ION,
                        FILTER_SUBBAND_INT,
                        FILTER_STRENGTH_SUBBAND_INT,
                        FILTER_WINSIZE_SUBBAND_INT,
                        FILTER_STEPSIZE_SUBBAND_INT,
                        REMOVE_MAGNITUDE_BEFORE_FILTERING_SUBBAND_INT,
                        #dense offset parameters
                        DO_DENSE_OFFSET,
                        ESTIMATE_RESIDUAL_OFFSET,
                        DELETE_GEOMETRY_FILES,
                        OFFSET_WINDOW_WIDTH,
                        OFFSET_WINDOW_HEIGHT,
                        OFFSET_SEARCH_WINDOW_WIDTH,
                        OFFSET_SEARCH_WINDOW_HEIGHT,
                        OFFSET_SKIP_WIDTH,
                        OFFSET_SKIP_HEIGHT,
                        OFFSET_COVARIANCE_OVERSAMPLING_FACTOR,
                        OFFSET_COVARIANCE_OVERSAMPLING_WINDOWSIZE,
                        MASK_OFFSET_WITH_WBD,
                        DO_OFFSET_FILTERING,
                        OFFSET_FILTER_WINDOWSIZE,
                        OFFSET_FILTER_SNR_THRESHOLD,
                        #system parameters
                        PICKLE_DUMPER_DIR,
                        PICKLE_LOAD_DIR,
                        RENDERER)

    facility_list = (REFERENCE,
                     SECONDARY,
                     #RUN_UNWRAPPER,
                     #RUN_UNWRAP_2STAGE,
                     _INSAR)

    _pickleObj = "_insar"

    def __init__(self, family='', name='',cmdline=None):
        import isceobj
        from isceobj.Alos2Proc import Alos2Proc
        from iscesys.StdOEL.StdOELPy import create_writer

        super().__init__(
            family=family if family else  self.__class__.family, name=name,
            cmdline=cmdline)

        self._stdWriter = create_writer("log", "", True, filename="alos2insar.log")
        self._add_methods()
        self._insarProcFact = Alos2Proc
        return None



    def Usage(self):
        print("Usages: ")
        print("alos2App.py <input-file.xml>")
        print("alos2App.py --steps")
        print("alos2App.py --help")
        print("alos2App.py --help --steps")


    def _init(self):

        message =  (
            ("ISCE VERSION = %s, RELEASE_SVN_REVISION = %s,"+
             "RELEASE_DATE = %s, CURRENT_SVN_REVISION = %s") %
            (isce.__version__,
             isce.release_svn_revision,
             isce.release_date,
             isce.svn_revision)
            )
        logger.info(message)

        print(message)
        return None

    def _configure(self):

        self.insar.procDoc._addItem("ISCE_VERSION",
            "Release: %s, svn-%s, %s. Current svn-%s" %
            (isce.release_version, isce.release_svn_revision,
             isce.release_date, isce.svn_revision
            ),
            ["insarProc"]
            )

        return None

    @property
    def insar(self):
        return self._insar
    @insar.setter
    def insar(self, value):
        self._insar = value
        return None

    @property
    def procDoc(self):
        return self.insar.procDoc

    @procDoc.setter
    def procDoc(self):
        raise AttributeError(
            "Can not assign to .insar.procDoc-- but you hit all its other stuff"
            )

    def _finalize(self):
        pass

    def help(self):
        from isceobj.Sensor.MultiMode import SENSORS
        print(self.__doc__)
        lsensors = list(SENSORS.keys())
        lsensors.sort()
        print("The currently supported sensors are: ", lsensors)
        return None

    def help_steps(self):
        print(self.__doc__)
        print("A description of the individual steps can be found in the README file")
        print("and also in the ISCE.pdf document")
        return


    def renderProcDoc(self):
        self.procDoc.renderXml()

    def startup(self):
        self.help()
        self._insar.timeStart = time.time()

    def endup(self):
        self.renderProcDoc()
        self._insar.timeEnd = time.time()
        logger.info("Total Time: %i seconds" %
                    (self._insar.timeEnd-self._insar.timeStart))
        return None


    ## Add instance attribute RunWrapper functions, which emulate methods.
    def _add_methods(self):
        self.runPreprocessor = Alos2Proc.createPreprocessor(self)
        self.runDownloadDem = Alos2Proc.createDownloadDem(self)
        self.runPrepareSlc = Alos2Proc.createPrepareSlc(self)
        self.runSlcOffset = Alos2Proc.createSlcOffset(self)
        self.runFormInterferogram = Alos2Proc.createFormInterferogram(self)
        self.runSwathOffset = Alos2Proc.createSwathOffset(self)
        self.runSwathMosaic = Alos2Proc.createSwathMosaic(self)
        self.runFrameOffset = Alos2Proc.createFrameOffset(self)
        self.runFrameMosaic = Alos2Proc.createFrameMosaic(self)
        self.runRdr2Geo = Alos2Proc.createRdr2Geo(self)
        self.runGeo2Rdr = Alos2Proc.createGeo2Rdr(self)
        self.runRdrDemOffset = Alos2Proc.createRdrDemOffset(self)
        self.runRectRangeOffset = Alos2Proc.createRectRangeOffset(self)
        self.runDiffInterferogram = Alos2Proc.createDiffInterferogram(self)
        self.runLook = Alos2Proc.createLook(self)
        self.runCoherence = Alos2Proc.createCoherence(self)
        self.runIonSubband = Alos2Proc.createIonSubband(self)
        self.runIonUwrap = Alos2Proc.createIonUwrap(self)
        self.runIonFilt = Alos2Proc.createIonFilt(self)
        self.runFilt = Alos2Proc.createFilt(self)
        self.runUnwrapSnaphu = Alos2Proc.createUnwrapSnaphu(self)
        self.runGeocode = Alos2Proc.createGeocode(self)

        #for dense offset
        self.runSlcMosaic = Alos2Proc.createSlcMosaic(self)
        self.runSlcMatch = Alos2Proc.createSlcMatch(self)
        self.runDenseOffset = Alos2Proc.createDenseOffset(self)
        self.runFiltOffset = Alos2Proc.createFiltOffset(self)
        self.runGeocodeOffset = Alos2Proc.createGeocodeOffset(self)


        return None

    def _steps(self):

        self.step('startup', func=self.startup,
                     doc=("Print a helpful message and "+
                          "set the startTime of processing")
                  )

        # Run a preprocessor for the two sets of frames
        self.step('preprocess',
                  func=self.runPreprocessor,
                  doc=(
                """Preprocess the reference and secondary sensor data to raw images"""
                )
                  )

        self.step('download_dem',
                  func=self.runDownloadDem,
                  doc=(
                """download DEM and water body"""
                )
                  )

        ##Run prepare slc
        self.step('prep_slc', func=self.runPrepareSlc,
                  doc=(
                """prepare multi-mode SLC for InSAR processing"""
                )
                  )

        ##Run slc offset
        self.step('slc_offset', func=self.runSlcOffset,
                  doc=(
                """estimate offset between slc pairs"""
                )
                  )

        ##Run slc offset
        self.step('form_int', func=self.runFormInterferogram,
                  doc=(
                """form interferogram"""
                )
                  )

        self.step('swath_offset', func=self.runSwathOffset,
                  doc=(
                """estimate offset between adjacent swaths"""
                )
                  )

        self.step('swath_mosaic', func=self.runSwathMosaic,
                  doc=(
                """mosaic swaths"""
                )
                  )

        self.step('frame_offset', func=self.runFrameOffset,
                  doc=(
                """estimate offset between adjacent frames"""
                )
                  )

        self.step('frame_mosaic', func=self.runFrameMosaic,
                  doc=(
                """mosaic frames"""
                )
                  )

        self.step('rdr2geo', func=self.runRdr2Geo,
                  doc=(
                """compute lat/lon/hgt"""
                )
                  )

        self.step('geo2rdr', func=self.runGeo2Rdr,
                  doc=(
                """compute range and azimuth offsets"""
                )
                  )

        self.step('rdrdem_offset', func=self.runRdrDemOffset,
                  doc=(
                """estimate offsets between radar image and dem (simulated radar image)"""
                )
                  )

        self.step('rect_rgoffset', func=self.runRectRangeOffset,
                  doc=(
                """rectify range offset"""
                )
                  )

        self.step('diff_int', func=self.runDiffInterferogram,
                  doc=(
                """create differential interferogram"""
                )
                  )

        self.step('look', func=self.runLook,
                  doc=(
                """take looks"""
                )
                  )

        self.step('coherence', func=self.runCoherence,
                  doc=(
                """estimate coherence"""
                )
                  )

        self.step('ion_subband', func=self.runIonSubband,
                  doc=(
                """create subband interferograms for ionospheric correction"""
                )
                  )

        self.step('ion_unwrap', func=self.runIonUwrap,
                  doc=(
                """unwrap subband interferograms"""
                )
                  )

        self.step('ion_filt', func=self.runIonFilt,
                  doc=(
                """compute and filter ionospheric phase"""
                )
                  )

        self.step('filt', func=self.runFilt,
                  doc=(
                """filter interferogram"""
                )
                  )

        self.step('unwrap', func=self.runUnwrapSnaphu,
                  doc=(
                """unwrap interferogram"""
                )
                  )

        self.step('geocode', func=self.runGeocode,
                  doc=(
                """geocode final products"""
                )
                  )

        #for dense offset
        self.step('slc_mosaic', func=self.runSlcMosaic,
                  doc=(
                """mosaic slcs"""
                )
                  )

        self.step('slc_match', func=self.runSlcMatch,
                  doc=(
                """match slc pair"""
                )
                  )

        self.step('dense_offset', func=self.runDenseOffset,
                  doc=(
                """estimate offset field"""
                )
                  )

        self.step('filt_offset', func=self.runFiltOffset,
                  doc=(
                """filt offset field"""
                )
                  )

        self.step('geocode_offset', func=self.runGeocodeOffset,
                  doc=(
                """geocode offset field"""
                )
                  )


        return None

    ## Main has the common start to both insarApp and dpmApp.
    def main(self):
        self.help()

        timeStart= time.time()

        # Run a preprocessor for the two sets of frames
        self.runPreprocessor()

        self.runDownloadDem()

        self.runPrepareSlc()

        self.runSlcOffset()

        self.runFormInterferogram()

        self.runSwathOffset()

        self.runSwathMosaic()

        self.runFrameOffset()

        self.runFrameMosaic()

        self.runRdr2Geo()

        self.runGeo2Rdr()

        self.runRdrDemOffset()

        self.runRectRangeOffset()

        self.runDiffInterferogram()

        self.runLook()

        self.runCoherence()

        self.runIonSubband()

        self.runIonUwrap()

        self.runIonFilt()

        self.runFilt()

        self.runUnwrapSnaphu()

        self.runGeocode()

        #for dense offset
        self.runSlcMosaic()

        self.runSlcMatch()

        self.runDenseOffset()

        self.runFiltOffset()

        self.runGeocodeOffset()


        timeEnd = time.time()
        logger.info("Total Time: %i seconds" %(timeEnd - timeStart))

        self.renderProcDoc()

        return None


    def updateParamemetersFromUser(self):
        '''
        update these parameters in case users set them in the middle of processing
        '''

        if self.numberRangeLooks1 != None:
            self._insar.numberRangeLooks1 = self.numberRangeLooks1
        if self.numberAzimuthLooks1 != None:
            self._insar.numberAzimuthLooks1 = self.numberAzimuthLooks1

        if self.numberRangeLooks2 != None:
            self._insar.numberRangeLooks2 = self.numberRangeLooks2
        if self.numberAzimuthLooks2 != None:
            self._insar.numberAzimuthLooks2 = self.numberAzimuthLooks2

        if self.numberRangeLooksSim != None:
            self._insar.numberRangeLooksSim = self.numberRangeLooksSim
        if self.numberAzimuthLooksSim != None:
            self._insar.numberAzimuthLooksSim = self.numberAzimuthLooksSim

        if self.numberRangeLooksIon != None:
            self._insar.numberRangeLooksIon = self.numberRangeLooksIon
        if self.numberAzimuthLooksIon != None:
            self._insar.numberAzimuthLooksIon = self.numberAzimuthLooksIon

        if self.dem != None:
            self._insar.dem = self.dem
        if self.demGeo != None:
            self._insar.demGeo = self.demGeo
        if self.wbd != None:
            self._insar.wbd = self.wbd

        if self._insar.referenceDate != None and self._insar.secondaryDate != None and \
            self._insar.numberRangeLooks1 != None and self._insar.numberAzimuthLooks1 != None and \
            self._insar.numberRangeLooks2 != None and self._insar.numberAzimuthLooks2 != None:
            self._insar.setFilename(referenceDate=self._insar.referenceDate, secondaryDate=self._insar.secondaryDate, 
                nrlks1=self._insar.numberRangeLooks1, nalks1=self._insar.numberAzimuthLooks1, 
                nrlks2=self._insar.numberRangeLooks2, nalks2=self._insar.numberAzimuthLooks2)


if __name__ == "__main__":
    import sys
    insar = Alos2InSAR(name="alos2App")
    insar.configure()
    insar.run()
