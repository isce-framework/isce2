#!/usr/bin/env python3


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Giangi Sacco, Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






import time
import sys
from isce import logging

import isce
import isceobj
import iscesys
from iscesys.Component.Application import Application
from iscesys.Compatibility import Compatibility
from iscesys.Component.Configurable import SELF
from isceobj import TopsProc

logger = logging.getLogger('isce.insar')


SENSOR_NAME = Application.Parameter(
    'sensorName',
    public_name='sensor name',
    default='SENTINEL1',
    type=str,
    mandatory=True,
    doc="Sensor name"
                                    )

DO_ESD = Application.Parameter('doESD',
                             public_name = 'do ESD',
                             default = True,
                             type = bool,
                             mandatory = False,
                             doc = 'Perform ESD estimation')

DO_DENSE_OFFSETS = Application.Parameter('doDenseOffsets',
                            public_name='do dense offsets',
                            default = False,
                            type = bool,
                            mandatory = False,
                            doc = 'Perform dense offset estimation')

UNWRAPPER_NAME = Application.Parameter(
    'unwrapper_name',
    public_name='unwrapper name',
    default='icu',
    type=str,
    mandatory=False,
    doc="Unwrapping method to use. To be used in  combination with UNWRAP."
)


# not fully supported yet; use UNWRAP instead
DO_UNWRAP = Application.Parameter(
    'do_unwrap',
    public_name='do unwrap',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if unwrapping is desired. To be unsed in combination with UNWRAPPER_NAME."
)

DO_UNWRAP_2STAGE = Application.Parameter(
    'do_unwrap_2stage',
    public_name='do unwrap 2 stage',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if unwrapping is desired. To be unsed in combination with UNWRAPPER_NAME."
)

UNWRAPPER_2STAGE_NAME = Application.Parameter(
    'unwrapper_2stage_name',
    public_name='unwrapper 2stage name',
    default='REDARC0',
    type=str,
    mandatory=False,
    doc="2 Stage Unwrapping method to use. Available: MCF, REDARC0, REDARC1, REDARC2"
)

SOLVER_2STAGE = Application.Parameter(
    'solver_2stage',
    public_name='SOLVER_2STAGE',
    default='pulp',
    type=str,
    mandatory=False,
    doc='Linear Programming Solver for 2Stage; Options: pulp, gurobi, glpk; Used only for Redundant Arcs'
)

USE_HIGH_RESOLUTION_DEM_ONLY = Application.Parameter(
    'useHighResolutionDemOnly',
    public_name='useHighResolutionDemOnly',
    default=False,
    type=int,
    mandatory=False,
    doc=(
    """If True and a dem is not specified in input, it will only
    download the SRTM highest resolution dem if it is available
    and fill the missing portion with null values (typically -32767)."""
    )
                                                )
DEM_FILENAME = Application.Parameter(
     'demFilename',
     public_name='demFilename',
     default='',
     type=str,
     mandatory=False,
     doc="Filename of the Digital Elevation Model (DEM)"
                                     )

GEOCODE_DEM_FILENAME = Application.Parameter(
        'geocodeDemFilename',
        public_name='geocode demfilename',
        default='',
        type=str,
        mandatory=False,
        doc='Filename of the DEM for geocoding')

GEOCODE_BOX = Application.Parameter(
    'geocode_bbox',
    public_name='geocode bounding box',
    default = None,
    container=list,
    type=float,
    doc='Bounding box for geocoding - South, North, West, East in degrees'
                                    )

REGION_OF_INTEREST = Application.Parameter(
        'roi',
        public_name = 'region of interest',
        default = None,
        container = list,
        type = float,
        doc = 'Bounding box for unpacking data - South, North, West, East in degrees')

PICKLE_DUMPER_DIR = Application.Parameter(
    'pickleDumpDir',
    public_name='pickle dump directory',
    default='PICKLE',
    type=str,
    mandatory=False,
    doc=(
    "If steps is used, the directory in which to store pickle objects."
    )
                                          )
PICKLE_LOAD_DIR = Application.Parameter(
    'pickleLoadDir',
    public_name='pickle load directory',
    default='PICKLE',
    type=str,
    mandatory=False,
    doc=(
    "If steps is used, the directory from which to retrieve pickle objects."
    )
                                        )

RENDERER = Application.Parameter(
    'renderer',
    public_name='renderer',
    default='xml',
    type=str,
    mandatory=True,
    doc=(
    "Format in which the data is serialized when using steps. Options are xml (default) or pickle."
    ))

NUMBER_AZIMUTH_LOOKS = Application.Parameter('numberAzimuthLooks',
                                           public_name='azimuth looks',
                                           default=7,
                                           type=int,
                                           mandatory=False,
                                           doc='')


NUMBER_RANGE_LOOKS = Application.Parameter('numberRangeLooks',
    public_name='range looks',
    default=19,
    type=int,
    mandatory=False,
    doc=''
)


ESD_AZIMUTH_LOOKS = Application.Parameter('esdAzimuthLooks',
                                        public_name = 'ESD azimuth looks',
                                        default = 5,
                                        type = int,
                                        mandatory = False,
                                        doc = 'Number of azimuth looks for overlap IFGs')

ESD_RANGE_LOOKS = Application.Parameter('esdRangeLooks',
                                       public_name = 'ESD range looks',
                                       default = 15,
                                       type = int,
                                       mandatory = False,
                                       doc = 'Number of range looks for overlap IFGs')

FILTER_STRENGTH = Application.Parameter('filterStrength',
                                      public_name='filter strength',
                                      default=0.5,
                                      type=float,
                                      mandatory=False,
                                      doc='')

ESD_COHERENCE_THRESHOLD = Application.Parameter('esdCoherenceThreshold',
                                public_name ='ESD coherence threshold',
                                default = 0.85,
                                type = float,
                                mandatory = False,
                                doc = 'ESD coherence threshold')

OFFSET_SNR_THRESHOLD = Application.Parameter('offsetSNRThreshold',
                                public_name = 'offset SNR threshold',
                                default=8.0,
                                type=float,
                                mandatory = False,
                                doc = 'Offset SNR threshold')

EXTRA_ESD_CYCLES = Application.Parameter('extraESDCycles',
                                public_name = 'extra ESD cycles',
                                default = 0.,
                                type = float,
                                mandatory = False,
                                doc = 'Extra ESD cycles to interpret overlap phase')

####New parameters for multi-swath
USE_VIRTUAL_FILES = Application.Parameter('useVirtualFiles',
                                public_name = 'use virtual files',
                                default=True,
                                type=bool,
                                mandatory=False,
                                doc = 'Use virtual files when possible to save space')

SWATHS = Application.Parameter('swaths',
                                public_name = 'swaths',
                                default = [],
                                type=int,
                                container=list,
                                mandatory=False,
                                doc = 'Swaths to process')

ROI = Application.Parameter('regionOfInterest',
        public_name = 'region of interest',
        default = [],
        container = list,
        type = float,
        doc = 'User defined area to crop in SNWE')


DO_INSAR = Application.Parameter('doInSAR',
        public_name = 'do interferogram',
        default = True,
        type = bool,
        doc = 'Perform interferometry. Set to false to skip insar steps.')

GEOCODE_LIST = Application.Parameter(
    'geocode_list',
     public_name='geocode list',
     default = None,
     container=list,
     type=str,
     doc = "List of products to geocode."
                                      )


######Adding stuff from topsOffsetApp for integration
WINDOW_SIZE_WIDTH = Application.Parameter(
    'winwidth',
    public_name='Ampcor window width',
    default=64,
    type=int,
    mandatory=False,
    doc='Ampcor main window size width. Used in runDenseOffsets.'
                                         )

WINDOW_SIZE_HEIGHT = Application.Parameter(
    'winhgt',
    public_name='Ampcor window height',
    default=64,
    type=int,
    mandatory=False,
    doc='Ampcor main window size height. Used in runDenseOffsets.')


SEARCH_WINDOW_WIDTH = Application.Parameter(
    'srcwidth',
    public_name='Ampcor search window width',
    default=20,
    type=int,
    mandatory=False,
    doc='Ampcor search window size width. Used in runDenseOffsets.'
                                            )

SEARCH_WINDOW_HEIGHT = Application.Parameter(
    'srchgt',
    public_name='Ampcor search window height',
    default=20,
    type=int,
    mandatory=False,
    doc='Ampcor search window size height. Used in runDenseOffsets.'
                                            )

SKIP_SAMPLE_ACROSS = Application.Parameter(
    'skipwidth',
    public_name='Ampcor skip width',
    default=32,
    type=int,
    mandatory=False,
    doc='Ampcor skip across width. Used in runDenseOffsets.'
                                            )

SKIP_SAMPLE_DOWN = Application.Parameter(
    'skiphgt',
    public_name='Ampcor skip height',
    default=32,
    type=int,
    mandatory=False,
    doc='Ampcor skip down height. Used in runDenseOffsets.'
                                            )

OFFSET_MARGIN = Application.Parameter(
    'margin',
    public_name='Ampcor margin',
    default=50,
    type=int,
    mandatory=False,
    doc='Ampcor margin offset. Used in runDenseOffsets.'
                                        )

OVERSAMPLING_FACTOR = Application.Parameter(
    'oversample',
    public_name='Ampcor oversampling factor',
    default=32,
    type=int,
    mandatory=False,
    doc='Ampcor oversampling factor. Used in runDenseOffsets.'
                                            )

ACROSS_GROSS_OFFSET = Application.Parameter(
    'rgshift',
    public_name='Range shift',
    default=0,
    type=int,
    mandatory=False,
    doc='Ampcor gross offset across. Used in runDenseOffsets.'
                                            )

DOWN_GROSS_OFFSET = Application.Parameter(
    'azshift',
    public_name='Azimuth shift',
    default=0,
    type=int,
    mandatory=False,
    doc='Ampcor gross offset down. Used in runDenseOffsets.'
                                            )

DENSE_OFFSET_SNR_THRESHOLD = Application.Parameter(
    'dense_offset_snr_thresh',
    public_name='SNR Threshold factor',
    default=None,
    type=float,
    mandatory=False,
    doc='SNR Threshold factor used in filtering offset field objects.')

FILTER_NULL = Application.Parameter(
    'filt_null',
    public_name='Filter NULL factor',
    default=-10000.,
    type=float,
    mandatory=False,
    doc='NULL factor to use in filtering offset fields to avoid numpy type issues.'
                                    )

FILTER_WIN_SIZE = Application.Parameter(
    'filt_size',
    public_name='Filter window size',
    default=5,
    type=int,
    mandatory=False,
    doc='Window size for median_filter.'
                                        )

OFFSET_GEOCODE_LIST = Application.Parameter(
    'off_geocode_list',
    public_name='offset geocode list',
    default=None,
    container=list,
    type=str,
    mandatory=False,
    doc='List of offset-specific files to geocode.'
                                            )

USE_GPU = Application.Parameter(
        'useGPU',
        public_name='use GPU',
        default=False,
        type=bool,
        mandatory=False,
        doc='Allow App to use GPU when available')

#####################################################################
#ionospheric correction
ION_DO_ION = Application.Parameter('ION_doIon',
    public_name = 'do ionosphere correction',
    default = False,
    type = bool,
    mandatory = False,
    doc = '')

ION_START_STEP = Application.Parameter(
    'ION_startStep',
    public_name='start ionosphere step',
    default='subband',
    type=str,
    mandatory=False,
    doc=""
)

ION_END_STEP = Application.Parameter(
    'ION_endStep',
    public_name='end ionosphere step',
    default='esd',
    type=str,
    mandatory=False,
    doc=""
)

ION_ION_HEIGHT = Application.Parameter('ION_ionHeight',
    public_name='height of ionosphere layer in km',
    default=200.0,
    type=float,
    mandatory=False,
    doc='')

ION_ION_FIT = Application.Parameter('ION_ionFit',
    public_name = 'apply polynomial fit before filtering ionosphere phase',
    default = True,
    type = bool,
    mandatory = False,
    doc = '')

ION_ION_FILTERING_WINSIZE_MAX = Application.Parameter('ION_ionFilteringWinsizeMax',
    public_name='maximum window size for filtering ionosphere phase',
    default=200,
    type=int,
    mandatory=False,
    doc='')

ION_ION_FILTERING_WINSIZE_MIN = Application.Parameter('ION_ionFilteringWinsizeMin',
    public_name='minimum window size for filtering ionosphere phase',
    default=100,
    type=int,
    mandatory=False,
    doc='')

ION_IONSHIFT_FILTERING_WINSIZE_MAX = Application.Parameter('ION_ionshiftFilteringWinsizeMax',
    public_name='maximum window size for filtering ionosphere azimuth shift',
    default=150,
    type=int,
    mandatory=False,
    doc='')

ION_IONSHIFT_FILTERING_WINSIZE_MIN = Application.Parameter('ION_ionshiftFilteringWinsizeMin',
    public_name='minimum window size for filtering ionosphere azimuth shift',
    default=75,
    type=int,
    mandatory=False,
    doc='')

ION_AZSHIFT_FLAG = Application.Parameter('ION_azshiftFlag',
    public_name='correct phase error caused by ionosphere azimuth shift',
    default=1,
    type=int,
    mandatory=False,
    doc='')

ION_NUMBER_AZIMUTH_LOOKS = Application.Parameter('ION_numberAzimuthLooks',
    public_name='total number of azimuth looks in the ionosphere processing',
    default=50,
    type=int,
    mandatory=False,
    doc='')

ION_NUMBER_RANGE_LOOKS = Application.Parameter('ION_numberRangeLooks',
    public_name='total number of range looks in the ionosphere processing',
    default=200,
    type=int,
    mandatory=False,
    doc='')

ION_NUMBER_AZIMUTH_LOOKS0 = Application.Parameter('ION_numberAzimuthLooks0',
    public_name='number of azimuth looks at first stage for ionosphere phase unwrapping',
    default=10,
    type=int,
    mandatory=False,
    doc='')

ION_NUMBER_RANGE_LOOKS0 = Application.Parameter('ION_numberRangeLooks0',
    public_name='number of range looks at first stage for ionosphere phase unwrapping',
    default=40,
    type=int,
    mandatory=False,
    doc='')
#####################################################################

#Facility declarations
MASTER = Application.Facility(
    'master',
    public_name='Master',
    module='isceobj.Sensor.TOPS',
    factory='createSensor',
    args=(SENSOR_NAME, 'master'),
    mandatory=True,
    doc="Master raw data component"
                              )

SLAVE = Application.Facility(
    'slave',
    public_name='Slave',
    module='isceobj.Sensor.TOPS',
    factory='createSensor',
    args=(SENSOR_NAME,'slave'),
    mandatory=True,
    doc="Slave raw data component"
                             )

DEM_STITCHER = Application.Facility(
    'demStitcher',
    public_name='demStitcher',
    module='iscesys.DataManager',
    factory='createManager',
    args=('dem1','iscestitcher',),
    mandatory=False,
    doc="Object that based on the frame bounding boxes creates a DEM"
)


RUN_UNWRAPPER = Application.Facility(
    'runUnwrapper',
    public_name='Run unwrapper',
    module='isceobj.TopsProc',
    factory='createUnwrapper',
    args=(SELF(), DO_UNWRAP, UNWRAPPER_NAME),
    mandatory=False,
    doc="Unwrapping module"
)

RUN_UNWRAP_2STAGE = Application.Facility(
    'runUnwrap2Stage',
    public_name='Run unwrapper 2 Stage',
    module='isceobj.TopsProc',
    factory='createUnwrap2Stage',
    args=(SELF(), DO_UNWRAP_2STAGE, UNWRAPPER_NAME),
    mandatory=False,
    doc="Unwrapping module"
)

_INSAR = Application.Facility(
    '_insar',
    public_name='topsproc',
    module='isceobj.TopsProc',
    factory='createTopsProc',
    args = ('topsAppContext',isceobj.createCatalog('topsProc')),
    mandatory=False,
    doc="TopsProc object"
)


## Common interface for all insar applications.
class TopsInSAR(Application):

    family = 'topsinsar'
    ## Define Class parameters in this list
    parameter_list = (SENSOR_NAME,
                      UNWRAPPER_NAME,
                      DEM_FILENAME,
                      GEOCODE_DEM_FILENAME,
                      NUMBER_AZIMUTH_LOOKS,
                      NUMBER_RANGE_LOOKS,
                      ESD_AZIMUTH_LOOKS,
                      ESD_RANGE_LOOKS,
                      FILTER_STRENGTH,
                      ESD_COHERENCE_THRESHOLD,
                      OFFSET_SNR_THRESHOLD,
                      DO_ESD,
                      DO_DENSE_OFFSETS,
                      DO_INSAR,
                      DO_UNWRAP,
                      USE_HIGH_RESOLUTION_DEM_ONLY,
                      GEOCODE_BOX,
                      PICKLE_DUMPER_DIR,
                      PICKLE_LOAD_DIR,
                      REGION_OF_INTEREST,
                      RENDERER,
                      DO_UNWRAP_2STAGE,
                      UNWRAPPER_2STAGE_NAME,
                      SOLVER_2STAGE,
                      GEOCODE_LIST,
                      USE_VIRTUAL_FILES,
                      SWATHS,
                      ROI,
                      WINDOW_SIZE_HEIGHT,
                      WINDOW_SIZE_WIDTH,
                      SEARCH_WINDOW_HEIGHT,
                      SEARCH_WINDOW_WIDTH,
                      SKIP_SAMPLE_ACROSS,
                      SKIP_SAMPLE_DOWN,
                      OFFSET_MARGIN,
                      OVERSAMPLING_FACTOR,
                      ACROSS_GROSS_OFFSET,
                      DOWN_GROSS_OFFSET,
                      DENSE_OFFSET_SNR_THRESHOLD,
                      EXTRA_ESD_CYCLES,
                      FILTER_NULL,
                      FILTER_WIN_SIZE,
                      OFFSET_GEOCODE_LIST,
                      USE_GPU,
                      ########################################################
                      #for ionospheric correction
                      ION_DO_ION,
                      ION_START_STEP,
                      ION_END_STEP,
                      ION_ION_HEIGHT,
                      ION_ION_FIT,
                      ION_ION_FILTERING_WINSIZE_MAX,
                      ION_ION_FILTERING_WINSIZE_MIN,
                      ION_IONSHIFT_FILTERING_WINSIZE_MAX,
                      ION_IONSHIFT_FILTERING_WINSIZE_MIN,
                      ION_AZSHIFT_FLAG,
                      ION_NUMBER_AZIMUTH_LOOKS,
                      ION_NUMBER_RANGE_LOOKS,
                      ION_NUMBER_AZIMUTH_LOOKS0,
                      ION_NUMBER_RANGE_LOOKS0
                      ########################################################
                      )

    facility_list = (MASTER,
                     SLAVE,
                     DEM_STITCHER,
                     RUN_UNWRAPPER,
                     RUN_UNWRAP_2STAGE,
                     _INSAR)

    _pickleObj = "_insar"

    def __init__(self, family='', name='',cmdline=None):
        import isceobj
        from isceobj.TopsProc import TopsProc
        from iscesys.StdOEL.StdOELPy import create_writer

        super().__init__(
            family=family if family else  self.__class__.family, name=name,
            cmdline=cmdline)

        self._stdWriter = create_writer("log", "", True, filename="topsinsar.log")
        self._add_methods()
        self._insarProcFact = TopsProc
        return None



    def Usage(self):
        print("Usages: ")
        print("topsApp.py <input-file.xml>")
        print("topsApp.py --steps")
        print("topsApp.py --help")
        print("topsApp.py --help --steps")


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

        #Ensure consistency in geocode_list maintained by insarApp and
        #InsarProc. If it is configured in both places, the one in insarApp
        #will be used. It is complicated to try to merge the two lists
        #because InsarProc permits the user to change the name of the files
        #and the linkage between filename and filetype is lost by the time
        #geocode_list is fully configured.  In order to safely change file
        #names and also specify the geocode_list, then insarApp should not
        #be given a geocode_list from the user.
        if(self.geocode_list is None):
            #if not provided by the user use the list from InsarProc
            self.geocode_list = self.insar.geocode_list
        else:
            #if geocode_list defined here, then give it to InsarProc
            #for consistency between insarApp and InsarProc and warn the user

            #check if the two geocode_lists differ in content
            g_count = 0
            for g in self.geocode_list:
                if g not in self.insar.geocode_list:
                    g_count += 1
            #warn if there are any differences in content
            if g_count > 0:
                print()
                logger.warn((
                    "Some filenames in insarApp.geocode_list configuration "+
                    "are different from those in InsarProc. Using names given"+
                    " to insarApp."))
                print("insarApp.geocode_list = {}".format(self.geocode_list))
                print(("InsarProc.geocode_list = {}".format(
                        self.insar.geocode_list)))

            self.insar.geocode_list = self.geocode_list


        if (self.off_geocode_list is None):
            self.off_geocode_list = self.insar.off_geocode_list
        else:
            g_count = 0
            for g in self.off_geocode_list:
                if g not in self.insar.off_geocode_list:
                    g_count += 1

            if g_count > 0:
               self.insar.off_geocode_list = self.geocode_list

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
        from isceobj.Sensor.TOPS import SENSORS
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
        self.runPreprocessor = TopsProc.createPreprocessor(self)
        self.runComputeBaseline = TopsProc.createComputeBaseline(self)
        self.verifyDEM = TopsProc.createVerifyDEM(self)
        self.verifyGeocodeDEM = TopsProc.createVerifyGeocodeDEM(self)
        self.runTopo  = TopsProc.createTopo(self)
        self.runSubsetOverlaps = TopsProc.createSubsetOverlaps(self)
        self.runCoarseOffsets = TopsProc.createCoarseOffsets(self)
        self.runCoarseResamp = TopsProc.createCoarseResamp(self)
        self.runOverlapIfg = TopsProc.createOverlapIfg(self)
        self.runPrepESD = TopsProc.createPrepESD(self)
        self.runESD = TopsProc.createESD(self)
        self.runRangeCoreg = TopsProc.createRangeCoreg(self)
        self.runFineOffsets = TopsProc.createFineOffsets(self)
        self.runFineResamp = TopsProc.createFineResamp(self)
        self.runIon = TopsProc.createIon(self)
        self.runBurstIfg = TopsProc.createBurstIfg(self)
        self.runMergeBursts = TopsProc.createMergeBursts(self)
        self.runFilter = TopsProc.createFilter(self)
        self.runGeocode = TopsProc.createGeocode(self)
        self.runDenseOffsets = TopsProc.createDenseOffsets(self)
        self.runOffsetFilter = TopsProc.createOffsetFilter(self)

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
                """Preprocess the master and slave sensor data to raw images"""
                )
                  )

        # Compute baselines and estimate common bursts
        self.step('computeBaselines',
                func=self.runComputeBaseline,
                doc=(
                    """Compute baseline and number of common bursts"""
                )
                  )

        # Verify whether the DEM was initialized properly.  If not, download
        # a DEM
        self.step('verifyDEM', func=self.verifyDEM)

        ##Run topo for each bursts
        self.step('topo', func=self.runTopo)

        ##Run subset overlaps
        self.step('subsetoverlaps', func=self.runSubsetOverlaps)

        ##Run coarse offsets
        self.step('coarseoffsets', func=self.runCoarseOffsets)

        ####Run coarse resamp
        self.step('coarseresamp', func=self.runCoarseResamp)

        ####Run overlap ifgs
        self.step('overlapifg', func=self.runOverlapIfg)

        ###Run prepare ESD inputs
        self.step('prepesd', func=self.runPrepESD)

        ###Run ESD
        self.step('esd', func=self.runESD)

        ###Run range coregistration
        self.step('rangecoreg', func=self.runRangeCoreg)

        ###Estimate fine offsets
        self.step('fineoffsets', func=self.runFineOffsets)

        ###Resample slave bursts
        self.step('fineresamp', func=self.runFineResamp)

        ###calculate ionospheric phase
        self.step('ion', func=self.runIon)

        ####Create burst interferograms
        self.step('burstifg', func=self.runBurstIfg)

        ###Merge burst products into a single file
        self.step('mergebursts', func=self.runMergeBursts)

        ###Filter the interferogram
        self.step('filter', func=self.runFilter)


        # Unwrap ?
        self.step('unwrap', func=self.runUnwrapper)

        # Conditional 2 stage unwrapping
        self.step('unwrap2stage', func=self.runUnwrap2Stage,
                  args=(self.unwrapper_2stage_name, self.solver_2stage))


        # Geocode
        self.step('geocode', func=self.runGeocode,
                args=(self.geocode_list, self.do_unwrap, self.geocode_bbox))

        # Dense offsets
        self.step('denseoffsets', func=self.runDenseOffsets)

        #Filter offsets
        self.step('filteroffsets', func=self.runOffsetFilter)

        #Geocode offsets
        self.step('geocodeoffsets', func=self.runGeocode,
                args=(self.off_geocode_list, False, self.geocode_bbox, True))

#        self.step('endup', func=self.endup)
        return None

    ## Main has the common start to both insarApp and dpmApp.
    def main(self):
        self.help()

        timeStart= time.time()

        # Run a preprocessor for the two sets of frames
        self.runPreprocessor()

        #Compute baselines and common bursts
        self.runComputeBaseline()


        #Verify whether user defined  a dem component.  If not, then download
        # SRTM DEM.
        self.verifyDEM()

        ##Run topo for each burst
        self.runTopo()

        ##Run subset overlaps
        self.runSubsetOverlaps()

        ##Run coarse offsets
        self.runCoarseOffsets()

        ##Run coarse resamp
        self.runCoarseResamp()

        ##Run ifg
        self.runOverlapIfg()

        ##Prepare for ESD
        self.runPrepESD()

        #Run ESD
        self.runESD()

        ###Estimate range misregistration
        self.runRangeCoreg()

        ###Estimate fine offsets
        self.runFineOffsets()

        ###Resample slave bursts
        self.runFineResamp()

        ###calculate ionospheric phase
        self.runIon()

        ###Create burst interferograms
        self.runBurstIfg()

        ####Merge bursts into single files
        self.runMergeBursts()

        ###Filter the interferogram
        self.runFilter()

        #add water mask to coherence and interferogram
        #self.runMaskImages()

        # Unwrap ?
        self.runUnwrapper()

        # 2Stage Unwrapping
        self.runUnwrap2Stage(self.unwrapper_2stage_name, self.solver_2stage)

        # Geocode
        self.runGeocode(self.geocode_list, self.do_unwrap, self.geocode_bbox)


        #Dense offsets
        self.runDenseOffsets()

        #Filter offsets
        self.runOffsetFilter()


        #Geocode offsets
        self.runGeocode(self.off_geocode_list, False, self.geocode_bbox, True)

        timeEnd = time.time()
        logger.info("Total Time: %i seconds" %(timeEnd - timeStart))

        self.renderProcDoc()

        return None




if __name__ == "__main__":
    import sys
    insar = TopsInSAR(name="topsApp")
    insar.configure()
    insar.run()
