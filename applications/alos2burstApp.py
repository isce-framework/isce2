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
from isceobj import Alos2burstProc

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
#radar-coordinate water body is created three times in runRdr2Geo.py, runLook.py and runLookSd.py, respectively
#radar-coordinate water body is used in:
#(1) determining the number of offsets in slc offset estimation, and radar/dem offset estimation
#(2) masking filtered interferogram or unwrapped interferogram
#(3) masking filtered interferogram or unwrapped interferogram in sd processing
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

#always remove unsynchronized signal, since no extra computation required, and it does
#improve coherence (example: indonesia_sep_2018/d25r/180927-181011_burst_3subswaths)
BURST_SYNCHRONIZATION_THRESHOLD = Application.Parameter('burstSynchronizationThreshold',
                                public_name = 'burst synchronization threshold',
                                default = 100.0,
                                type=float,
                                mandatory = True,
                                doc = 'burst synchronization threshold in percentage')

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

#these two parameters are always 1, not to be set by users
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

SWATH_PHASE_DIFF_SNAP_ION = Application.Parameter('swathPhaseDiffSnapIon',
                                public_name = 'swath phase difference snap to fixed values',
                                default = None,
                                type = bool,
                                mandatory = False,
                                container = list,
                                doc = 'swath phase difference snap to fixed values')

SWATH_PHASE_DIFF_LOWER_ION = Application.Parameter('swathPhaseDiffLowerIon',
                                public_name = 'swath phase difference of lower band',
                                default = None,
                                type = float,
                                mandatory = False,
                                container = list,
                                doc = 'swath phase difference of lower band')

SWATH_PHASE_DIFF_UPPER_ION = Application.Parameter('swathPhaseDiffUpperIon',
                                public_name = 'swath phase difference of upper band',
                                default = None,
                                type = float,
                                mandatory = False,
                                container = list,
                                doc = 'swath phase difference of upper band')

FIT_ION = Application.Parameter('fitIon',
                                public_name = 'apply polynomial fit before filtering ionosphere phase',
                                default = True,
                                type = bool,
                                mandatory = False,
                                doc = 'apply polynomial fit before filtering ionosphere phase')

FILT_ION = Application.Parameter('filtIon',
                                public_name = 'whether filtering ionosphere phase',
                                default = True,
                                type = bool,
                                mandatory = False,
                                doc = 'whether filtering ionosphere phase')

FIT_ADAPTIVE_ION = Application.Parameter('fitAdaptiveIon',
                                public_name = 'apply polynomial fit in adaptive filtering window',
                                default = True,
                                type = bool,
                                mandatory = False,
                                doc = 'apply polynomial fit in adaptive filtering window')

FILT_SECONDARY_ION = Application.Parameter('filtSecondaryIon',
                                public_name = 'whether do secondary filtering of ionosphere phase',
                                default = True,
                                type = bool,
                                mandatory = False,
                                doc = 'whether do secondary filtering of ionosphere phase')

FILTERING_WINSIZE_MAX_ION = Application.Parameter('filteringWinsizeMaxIon',
                                public_name='maximum window size for filtering ionosphere phase',
                                default=301,
                                type=int,
                                mandatory=False,
                                doc='maximum window size for filtering ionosphere phase')

FILTERING_WINSIZE_MIN_ION = Application.Parameter('filteringWinsizeMinIon',
                                public_name='minimum window size for filtering ionosphere phase',
                                default=11,
                                type=int,
                                mandatory=False,
                                doc='minimum window size for filtering ionosphere phase')

FILTERING_WINSIZE_SECONDARY_ION = Application.Parameter('filteringWinsizeSecondaryIon',
                                public_name='window size of secondary filtering of ionosphere phase',
                                default=5,
                                type=int,
                                mandatory=False,
                                doc='window size of secondary filtering of ionosphere phase')

FILTER_STD_ION = Application.Parameter('filterStdIon',
                                public_name = 'standard deviation of ionosphere phase after filtering',
                                default = None,
                                type=float,
                                mandatory = False,
                                doc = 'standard deviation of ionosphere phase after filtering')

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

#spectral diversity parameters
NUMBER_RANGE_LOOKS_SD = Application.Parameter('numberRangeLooksSd',
                                public_name='number of range looks sd',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of range looks for spectral diversity")

NUMBER_AZIMUTH_LOOKS_SD = Application.Parameter('numberAzimuthLooksSd',
                                public_name='number of azimuth looks sd',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of azimuth looks for spectral diversity")

FILTER_STRENGTH_SD = Application.Parameter('filterStrengthSd',
                                public_name = 'interferogram filter strength SD',
                                default = 0.3,
                                type=float,
                                mandatory = False,
                                doc = 'interferogram filter strength for spectral diversity')

FILTER_WINSIZE_SD = Application.Parameter('filterWinsizeSd',
                                public_name = 'interferogram filter window size SD',
                                default = 32,
                                type=int,
                                mandatory = False,
                                doc = 'interferogram filter window size for spectral diversity')

FILTER_STEPSIZE_SD = Application.Parameter('filterStepsizeSd',
                                public_name = 'interferogram filter step size SD',
                                default = 4,
                                type=int,
                                mandatory = False,
                                doc = 'interferogram filter step size for spectral diversity')

WATERBODY_MASK_STARTING_STEP_SD = Application.Parameter('waterBodyMaskStartingStepSd',
                                public_name='water body mask starting step SD',
                                default='unwrap',
                                type=str,
                                mandatory=False,
                                doc='water body mask starting step: None, filt, unwrap')

UNION_SD = Application.Parameter('unionSd',
                                public_name = 'union when combining sd interferograms',
                                default = True,
                                type = bool,
                                mandatory = False,
                                doc = 'union or intersection when combining sd interferograms')

GEOCODE_LIST_SD = Application.Parameter('geocodeListSd',
                                public_name = 'geocode file list SD',
                                default=None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'geocode file list for SD')

GEOCODE_INTERP_METHOD_SD = Application.Parameter('geocodeInterpMethodSd',
                                public_name='geocode interpolation method SD',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='geocode interpolation method for SD: sinc, bilinear, bicubic, nearest')
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
#                                 module='isceobj.Alos2burstProc',
#                                 factory='createUnwrapper',
#                                 args=(SELF(), DO_UNWRAP, UNWRAPPER_NAME),
#                                 mandatory=False,
#                                 doc="Unwrapping module")

# RUN_UNWRAP_2STAGE = Application.Facility('runUnwrap2Stage',
#                                 public_name='Run unwrapper 2 Stage',
#                                 module='isceobj.Alos2burstProc',
#                                 factory='createUnwrap2Stage',
#                                 args=(SELF(), DO_UNWRAP_2STAGE, UNWRAPPER_NAME),
#                                 mandatory=False,
#                                 doc="Unwrapping module")

_INSAR = Application.Facility('_insar',
                                public_name='alos2burstproc',
                                module='isceobj.Alos2burstProc',
                                factory='createAlos2burstProc',
                                args = ('alos2burstAppContext',isceobj.createCatalog('alos2burstProc')),
                                mandatory=False,
                                doc="Alos2burstProc object")


## Common interface for all insar applications.
class Alos2burstInSAR(Application):
    family = 'alos2burstinsar'
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
                        SWATH_PHASE_DIFF_SNAP_ION,
                        SWATH_PHASE_DIFF_LOWER_ION,
                        SWATH_PHASE_DIFF_UPPER_ION,
                        FIT_ION,
                        FILT_ION,
                        FIT_ADAPTIVE_ION,
                        FILT_SECONDARY_ION,
                        FILTERING_WINSIZE_MAX_ION,
                        FILTERING_WINSIZE_MIN_ION,
                        FILTERING_WINSIZE_SECONDARY_ION,
                        FILTER_STD_ION,
                        FILTER_SUBBAND_INT,
                        FILTER_STRENGTH_SUBBAND_INT,
                        FILTER_WINSIZE_SUBBAND_INT,
                        FILTER_STEPSIZE_SUBBAND_INT,
                        REMOVE_MAGNITUDE_BEFORE_FILTERING_SUBBAND_INT,
                        #spectral diversity parameters
                        NUMBER_RANGE_LOOKS_SD,
                        NUMBER_AZIMUTH_LOOKS_SD,
                        FILTER_STRENGTH_SD,
                        FILTER_WINSIZE_SD,
                        FILTER_STEPSIZE_SD,
                        WATERBODY_MASK_STARTING_STEP_SD,
                        UNION_SD,
                        GEOCODE_LIST_SD,
                        GEOCODE_INTERP_METHOD_SD,
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
        from isceobj.Alos2burstProc import Alos2burstProc
        from iscesys.StdOEL.StdOELPy import create_writer

        super().__init__(
            family=family if family else  self.__class__.family, name=name,
            cmdline=cmdline)

        self._stdWriter = create_writer("log", "", True, filename="alos2burstinsar.log")
        self._add_methods()
        self._insarProcFact = Alos2burstProc
        return None



    def Usage(self):
        print("Usages: ")
        print("alos2burstApp.py <input-file.xml>")
        print("alos2burstApp.py --steps")
        print("alos2burstApp.py --help")
        print("alos2burstApp.py --help --steps")


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
        self.runPreprocessor = Alos2burstProc.createPreprocessor(self)
        self.runBaseline = Alos2burstProc.createBaseline(self)
        self.runExtractBurst = Alos2burstProc.createExtractBurst(self)
        self.runDownloadDem = Alos2burstProc.createDownloadDem(self)
        self.runCoregGeom = Alos2burstProc.createCoregGeom(self)
        self.runCoregCc = Alos2burstProc.createCoregCc(self)
        self.runCoregSd = Alos2burstProc.createCoregSd(self)
        self.runSwathOffset = Alos2burstProc.createSwathOffset(self)
        self.runSwathMosaic = Alos2burstProc.createSwathMosaic(self)
        self.runFrameOffset = Alos2burstProc.createFrameOffset(self)
        self.runFrameMosaic = Alos2burstProc.createFrameMosaic(self)
        self.runRdr2Geo = Alos2burstProc.createRdr2Geo(self)
        self.runGeo2Rdr = Alos2burstProc.createGeo2Rdr(self)
        self.runRdrDemOffset = Alos2burstProc.createRdrDemOffset(self)
        self.runRectRangeOffset = Alos2burstProc.createRectRangeOffset(self)
        self.runDiffInterferogram = Alos2burstProc.createDiffInterferogram(self)
        self.runLook = Alos2burstProc.createLook(self)
        self.runCoherence = Alos2burstProc.createCoherence(self)
        self.runIonSubband = Alos2burstProc.createIonSubband(self)
        self.runIonUwrap = Alos2burstProc.createIonUwrap(self)
        self.runIonFilt = Alos2burstProc.createIonFilt(self)
        self.runIonCorrect = Alos2burstProc.createIonCorrect(self)
        self.runFilt = Alos2burstProc.createFilt(self)
        self.runUnwrapSnaphu = Alos2burstProc.createUnwrapSnaphu(self)
        self.runGeocode = Alos2burstProc.createGeocode(self)

        #spectral diversity
        self.runLookSd = Alos2burstProc.createLookSd(self)
        self.runFiltSd = Alos2burstProc.createFiltSd(self)
        self.runUnwrapSnaphuSd = Alos2burstProc.createUnwrapSnaphuSd(self)
        self.runGeocodeSd = Alos2burstProc.createGeocodeSd(self)

        return None

    def _steps(self):

        self.step('startup', func=self.startup,
                     doc=("Print a helpful message and "+
                          "set the startTime of processing")
                  )

        # Run a preprocessor for the two acquisitions
        self.step('preprocess', func=self.runPreprocessor,
                  doc=(
                """Preprocess the reference and secondary sensor data to raw images"""
                )
                  )

        self.step('baseline', func=self.runBaseline,
                  doc=(
                """compute baseline, burst synchronization etc"""
                )
                  )

        self.step('extract_burst', func=self.runExtractBurst,
                  doc=(
                """extract bursts from full aperture images"""
                )
                  )

        self.step('download_dem', func=self.runDownloadDem,
                  doc=(
                """download DEM and water body"""
                )
                  )

        self.step('coreg_geom', func=self.runCoregGeom,
                  doc=(
                """coregistrater bursts based on geometric offsets"""
                )
                  )

        self.step('coreg_cc', func=self.runCoregCc,
                  doc=(
                """coregistrater bursts based on cross-correlation offsets"""
                )
                  )

        self.step('coreg_sd', func=self.runCoregSd,
                  doc=(
                """coregistrater bursts based on spectral diversity offsets"""
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

        self.step('ion_correct', func=self.runIonCorrect,
                  doc=(
                """resample ionospheric phase and ionospheric correction"""
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

        self.step('sd_look', func=self.runLookSd,
                  doc=(
                """take looks for sd"""
                )
                  )

        self.step('sd_filt', func=self.runFiltSd,
                  doc=(
                """filter sd interferograms"""
                )
                  )

        self.step('sd_unwrap', func=self.runUnwrapSnaphuSd,
                  doc=(
                """unwrap sd interferograms"""
                )
                  )

        self.step('sd_geocode', func=self.runGeocodeSd,
                  doc=(
                """geocode final sd products"""
                )
                  )

        return None

    ## Main has the common start to both insarApp and dpmApp.
    def main(self):
        self.help()

        timeStart= time.time()

        # Run a preprocessor for the two sets of frames
        self.runPreprocessor()

        self.runBaseline()

        self.runExtractBurst()

        self.runDownloadDem()

        self.runCoregGeom()

        self.runCoregCc()

        self.runCoregSd()

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

        self.runIonCorrect()

        self.runFilt()

        self.runUnwrapSnaphu()

        self.runGeocode()

        self.runLookSd()

        self.runFiltSd()

        self.runUnwrapSnaphuSd()

        self.runGeocodeSd()

        timeEnd = time.time()
        logger.info("Total Time: %i seconds" %(timeEnd - timeStart))

        self.renderProcDoc()

        return None


    def updateParamemetersFromUser(self):
        '''
        update these parameters in case users set them in the middle of processing
        '''

        if self.numberRangeLooks1 != None:
            #force number of looks 1 to 1
            self.numberRangeLooks1 = 1
            self._insar.numberRangeLooks1 = self.numberRangeLooks1
        if self.numberAzimuthLooks1 != None:
            self.numberAzimuthLooks1 = 1
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

        if self.numberRangeLooksSd != None:
            self._insar.numberRangeLooksSd = self.numberRangeLooksSd
        if self.numberAzimuthLooksSd != None:
            self._insar.numberAzimuthLooksSd = self.numberAzimuthLooksSd

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

        if self._insar.referenceDate != None and self._insar.secondaryDate != None and \
            self._insar.numberRangeLooks1 != None and self._insar.numberAzimuthLooks1 != None and \
            self._insar.numberRangeLooksSd != None and self._insar.numberAzimuthLooksSd != None:
            self._insar.setFilenameSd(referenceDate=self._insar.referenceDate, secondaryDate=self._insar.secondaryDate, 
                nrlks1=self._insar.numberRangeLooks1, nalks1=self._insar.numberAzimuthLooks1, 
                nrlks_sd=self._insar.numberRangeLooksSd, nalks_sd=self._insar.numberAzimuthLooksSd, nsd=3)


if __name__ == "__main__":
    import sys
    insar = Alos2burstInSAR(name="alos2burstApp")
    insar.configure()
    insar.run()
