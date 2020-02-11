#!/usr/bin/env python3
#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright by California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Heresh Fattahi
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






from __future__ import print_function
import time
import sys
from isce import logging

import isce
import isceobj
import iscesys
from iscesys.Component.Application import Application
from iscesys.Compatibility import Compatibility
from iscesys.Component.Configurable import SELF
import isceobj.StripmapProc as StripmapProc
from isceobj.Scene.Frame import FrameMixin
from isceobj.Util.decorators import use_api

logger = logging.getLogger('isce.insar')


SENSOR_NAME = Application.Parameter(
        'sensorName',
        public_name='sensor name',
        default = None,
        type = str,
        mandatory = False,
        doc = 'Sensor name for both master and slave')


MASTER_SENSOR_NAME = Application.Parameter(
        'masterSensorName',
        public_name='master sensor name',
        default = None,
        type=str,
        mandatory = True,
        doc = "Master sensor name if mixing sensors")

SLAVE_SENSOR_NAME = Application.Parameter(
        'slaveSensorName',
        public_name='slave sensor name',
        default = None,
        type=str,
        mandatory = True,
        doc = "Slave sensor name if mixing sensors")


CORRELATION_METHOD = Application.Parameter(
   'correlation_method',
   public_name='correlation_method',
   default='cchz_wave',
   type=str,
   mandatory=False,
   doc=(
   """Select coherence estimation method:
      cchz=cchz_wave
      phase_gradient=phase gradient"""
        )
                                           )
MASTER_DOPPLER_METHOD = Application.Parameter(
    'masterDopplerMethod',
    public_name='master doppler method',
    default=None,
    type=str, mandatory=False,
    doc= "Doppler calculation method.Choices: 'useDOPIQ', 'useDefault'."
)

SLAVE_DOPPLER_METHOD = Application.Parameter(
    'slaveDopplerMethod',
    public_name='slave doppler method',
    default=None,
    type=str, mandatory=False,
    doc="Doppler calculation method. Choices: 'useDOPIQ','useDefault'.")


UNWRAPPER_NAME = Application.Parameter(
    'unwrapper_name',
    public_name='unwrapper name',
    default='grass',
    type=str,
    mandatory=False,
    doc="Unwrapping method to use. To be used in  combination with UNWRAP."
)

DO_UNWRAP = Application.Parameter(
    'do_unwrap',
    public_name='do unwrap',
    default=True,
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
     doc="Filename of the DEM init file"
)

REGION_OF_INTEREST = Application.Parameter(
        'regionOfInterest',
        public_name = 'regionOfInterest',
        default = None,
        container = list,
        type = float,
        doc = 'Region of interest - South, North, West, East in degrees')


GEOCODE_BOX = Application.Parameter(
    'geocode_bbox',
    public_name='geocode bounding box',
    default = None,
    container=list,
    type=float,
    doc='Bounding box for geocoding - South, North, West, East in degrees'
                                    )

GEO_POSTING = Application.Parameter(
    'geoPosting',
    public_name='geoPosting',
    default=None,
    type=float,
    mandatory=False,
    doc=(
    "Output posting for geocoded images in degrees (latitude = longitude)"
    )
                                    )

POSTING = Application.Parameter(
    'posting',
    public_name='posting',
    default=30,
    type=int,
    mandatory=False,
    doc="posting for interferogram")


NUMBER_RANGE_LOOKS = Application.Parameter(
    'numberRangeLooks',
    public_name='range looks',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of range looks'
                                    )

NUMBER_AZIMUTH_LOOKS = Application.Parameter(
    'numberAzimuthLooks',
    public_name='azimuth looks',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of azimuth looks'
                                 )

FILTER_STRENGTH = Application.Parameter('filterStrength',
                                      public_name='filter strength',
                                      default=0.5,
                                      type=float,
                                      mandatory=False,
                                      doc='')

############################################## Modified by V.Brancato 10.07.2019
DO_RUBBERSHEETINGAZIMUTH = Application.Parameter('doRubbersheetingAzimuth', 
                                      public_name='do rubbersheetingAzimuth',
                                      default=False,
                                      type=bool,
                                      mandatory=False,
                                      doc='')
DO_RUBBERSHEETINGRANGE = Application.Parameter('doRubbersheetingRange', 
                                      public_name='do rubbersheetingRange',
                                      default=False,
                                      type=bool,
                                      mandatory=False,
                                      doc='')
#################################################################################
RUBBERSHEET_SNR_THRESHOLD = Application.Parameter('rubberSheetSNRThreshold',
                                      public_name='rubber sheet SNR Threshold',
                                      default = 5.0,
                                      type = float,
                                      mandatory = False,
                                      doc='')

RUBBERSHEET_FILTER_SIZE = Application.Parameter('rubberSheetFilterSize',
                                      public_name='rubber sheet filter size',
                                      default = 9,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DO_DENSEOFFSETS  = Application.Parameter('doDenseOffsets',
                                      public_name='do denseoffsets',
                                      default=False,
                                      type=bool,
                                      mandatory=False,
                                      doc='')

DENSE_WINDOW_WIDTH = Application.Parameter('denseWindowWidth',
                                      public_name='dense window width',
                                      default=64,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DENSE_WINDOW_HEIGHT = Application.Parameter('denseWindowHeight',
                                      public_name='dense window height',
                                      default=64,
                                      type = int,
                                      mandatory = False,
                                      doc = '')


DENSE_SEARCH_WIDTH = Application.Parameter('denseSearchWidth',
                                      public_name='dense search width',
                                      default=20,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DENSE_SEARCH_HEIGHT = Application.Parameter('denseSearchHeight',
                                      public_name='dense search height',
                                      default=20,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DENSE_SKIP_WIDTH = Application.Parameter('denseSkipWidth',
                                      public_name='dense skip width',
                                      default=32,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DENSE_SKIP_HEIGHT = Application.Parameter('denseSkipHeight',
                                      public_name='dense skip height',
                                      default=32,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DO_SPLIT_SPECTRUM = Application.Parameter('doSplitSpectrum',
                                      public_name='do split spectrum',
                                      default = False,
                                      type = bool,
                                      mandatory = False,
                                      doc = '')

DO_DISPERSIVE = Application.Parameter('doDispersive',
                                      public_name='do dispersive',
                                      default=False,
                                      type=bool,
                                      mandatory=False,
                                      doc='')

GEOCODE_LIST = Application.Parameter(
    'geocode_list',
     public_name='geocode list',
     default = None,
     container=list,
     type=str,
     doc = "List of products to geocode."
                                      )

OFFSET_GEOCODE_LIST = Application.Parameter(
        'off_geocode_list',
        public_name='offset geocode list',
        default=None,
        container=list,
        mandatory=False,
        doc='List of offset-specific files to geocode')

HEIGHT_RANGE = Application.Parameter(
        'heightRange',
        public_name = 'height range',
        default = None,
        container = list,
        type = float,
        doc = 'Altitude range in scene for cropping')

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
    )
                                        )

DISPERSIVE_FILTER_FILLING_METHOD = Application.Parameter('dispersive_filling_method',
                                            public_name = 'dispersive filter filling method',
                                            default='nearest_neighbour',
                                            type=str,
                                            mandatory=False,
                                            doc='method to fill the holes left by masking the ionospheric phase estimate')
					    
DISPERSIVE_FILTER_KERNEL_XSIZE = Application.Parameter('kernel_x_size',
                                      public_name='dispersive filter kernel x-size',
                                      default=800,
                                      type=float,
                                      mandatory=False,
                                      doc='kernel x-size for the Gaussian low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_KERNEL_YSIZE = Application.Parameter('kernel_y_size',
                                      public_name='dispersive filter kernel y-size',
                                      default=800,
                                      type=float,
                                      mandatory=False,
                                      doc='kernel y-size for the Gaussian low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_KERNEL_SIGMA_X = Application.Parameter('kernel_sigma_x',
                                      public_name='dispersive filter kernel sigma_x',
                                      default=100,
                                      type=float,
                                      mandatory=False,
                                      doc='kernel sigma_x for the Gaussian low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_KERNEL_SIGMA_Y = Application.Parameter('kernel_sigma_y',
                                      public_name='dispersive filter kernel sigma_y',
                                      default=100,
                                      type=float,
                                      mandatory=False,
                                      doc='kernel sigma_y for the Gaussian low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_KERNEL_ROTATION = Application.Parameter('kernel_rotation',
                                      public_name='dispersive filter kernel rotation',
                                      default=0.0,
                                      type=float,
                                      mandatory=False,
                                      doc='kernel rotation angle for the Gaussian low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_ITERATION_NUMBER = Application.Parameter('dispersive_filter_iterations',
                                      public_name='dispersive filter number of iterations',
                                      default=5,
                                      type=int,
                                      mandatory=False,
                                      doc='number of iterations for the iterative low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_MASK_TYPE = Application.Parameter('dispersive_filter_mask_type',
                                      public_name='dispersive filter mask type',
                                      default="connected_components",
                                      type=str,
                                      mandatory=False,
                                      doc='The type of mask for the iterative low-pass filtering of the estimated dispersive phase. If method is coherence, then a mask based on coherence files of low-band and sub-band interferograms is generated using the mask coherence thresold which can be also setup. If method is connected_components, then mask is formed based on connected component files with non zero values. If method is phase, then pixels with zero phase values in unwrapped sub-band interferograms are masked out.')

DISPERSIVE_FILTER_COHERENCE_THRESHOLD = Application.Parameter('dispersive_filter_coherence_threshold',
                                      public_name='dispersive filter coherence threshold',
                                      default=0.5,
                                      type=float,
                                      mandatory=False,
                                      doc='Coherence threshold to generate a mask file which gets used in the iterative filtering of the dispersive and non-disperive phase')
#Facility declarations

MASTER = Application.Facility(
    'master',
    public_name='Master',
    module='isceobj.StripmapProc.Sensor',
    factory='createSensor',
    args=(SENSOR_NAME, MASTER_SENSOR_NAME, 'master'),
    mandatory=False,
    doc="Master raw data component"
                              )

SLAVE = Application.Facility(
    'slave',
    public_name='Slave',
    module='isceobj.StripmapProc.Sensor',
    factory='createSensor',
    args=(SENSOR_NAME, SLAVE_SENSOR_NAME,'slave'),
    mandatory=False,
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
    module='isceobj.StripmapProc',
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
    public_name='insar',
    module='isceobj.StripmapProc',
    factory='createStripmapProc',
    args = ('stripmapAppContext',isceobj.createCatalog('stripmapProc')),
    mandatory=False,
    doc="InsarProc object"
)



## Common interface for stripmap insar applications.
class _RoiBase(Application, FrameMixin):

    family = 'insar'
    ## Define Class parameters in this list
    parameter_list = (SENSOR_NAME,
                      MASTER_SENSOR_NAME,
                      SLAVE_SENSOR_NAME,
                      FILTER_STRENGTH,
                      CORRELATION_METHOD,
                      MASTER_DOPPLER_METHOD,
                      SLAVE_DOPPLER_METHOD,
                      UNWRAPPER_NAME,
                      DO_UNWRAP,
                      DO_UNWRAP_2STAGE,
                      UNWRAPPER_2STAGE_NAME,
                      SOLVER_2STAGE,
                      USE_HIGH_RESOLUTION_DEM_ONLY,
                      DEM_FILENAME,
                      GEO_POSTING,
                      POSTING,
                      NUMBER_RANGE_LOOKS,
                      NUMBER_AZIMUTH_LOOKS,
                      GEOCODE_LIST,
                      OFFSET_GEOCODE_LIST,
                      GEOCODE_BOX,
                      REGION_OF_INTEREST,
                      HEIGHT_RANGE,
                      DO_RUBBERSHEETINGRANGE, #Modified by V. Brancato 10.07.2019
                      DO_RUBBERSHEETINGAZIMUTH,  #Modified by V. Brancato 10.07.2019
                      RUBBERSHEET_SNR_THRESHOLD,
                      RUBBERSHEET_FILTER_SIZE,
                      DO_DENSEOFFSETS,
                      DENSE_WINDOW_WIDTH,
                      DENSE_WINDOW_HEIGHT,
                      DENSE_SEARCH_WIDTH,
                      DENSE_SEARCH_HEIGHT,
                      DENSE_SKIP_WIDTH,
                      DENSE_SKIP_HEIGHT,
                      DO_SPLIT_SPECTRUM,
                      PICKLE_DUMPER_DIR,
                      PICKLE_LOAD_DIR,
                      RENDERER,
                      DO_DISPERSIVE,
                      DISPERSIVE_FILTER_FILLING_METHOD,
                      DISPERSIVE_FILTER_KERNEL_XSIZE,
                      DISPERSIVE_FILTER_KERNEL_YSIZE,
                      DISPERSIVE_FILTER_KERNEL_SIGMA_X,
                      DISPERSIVE_FILTER_KERNEL_SIGMA_Y,
                      DISPERSIVE_FILTER_KERNEL_ROTATION,
                      DISPERSIVE_FILTER_ITERATION_NUMBER,
                      DISPERSIVE_FILTER_MASK_TYPE,
                      DISPERSIVE_FILTER_COHERENCE_THRESHOLD)

    facility_list = (MASTER,
                     SLAVE,
                     DEM_STITCHER,
                     RUN_UNWRAPPER,
                     RUN_UNWRAP_2STAGE,
                     _INSAR)


    _pickleObj = "_insar"

    def __init__(self, family='', name='',cmdline=None):
        import isceobj
        super().__init__(family=family, name=name,
            cmdline=cmdline)

        from isceobj.StripmapProc import StripmapProc
        from iscesys.StdOEL.StdOELPy import create_writer
        self._stdWriter = create_writer("log", "", True, filename="roi.log")
        self._add_methods()
        self._insarProcFact = StripmapProc
        self.timeStart = None
        return None

    def Usage(self):
        print("Usages: ")
        print("stripmapApp.py <input-file.xml>")
        print("stripmapApp.py --steps")
        print("stripmapApp.py --help")
        print("stripmapApp.py --help --steps")

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

    ## You need this to use the FrameMixin
    @property
    def frame(self):
        return self.insar.frame


    def _configure(self):

        self.insar.procDoc._addItem("ISCE_VERSION",
            "Release: %s, svn-%s, %s. Current svn-%s" %
            (isce.release_version, isce.release_svn_revision,
             isce.release_date, isce.svn_revision
            ),
            ["stripmapProc"]
            )

        #Ensure consistency in geocode_list maintained by insarApp and
        #InsarProc. If it is configured in both places, the one in insarApp
        #will be used. It is complicated to try to merge the two lists
        #because InsarProc permits the user to change the name of the files
        #and the linkage between filename and filetype is lost by the time
        #geocode_list is fully configured.  In order to safely change file
        #names and also specify the geocode_list, then insarApp should not
        #be given a geocode_list from the user.
        if(self.geocode_list is not None):
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
                    "Some filenames in stripmapApp.geocode_list configuration "+
                    "are different from those in StripmapProc. Using names given"+
                    " to stripmapApp."))
                print("stripmapApp.geocode_list = {}".format(self.geocode_list))
        else:
            self.geocode_list = self.insar.geocode_list


        if (self.off_geocode_list is None):
            self.off_geocode_list = self.insar.off_geocode_list
        else:
            g_count = 0
            for g in self.off_geocode_list:
                if g not in self.insar.off_geocode_list:
                    g_count += 1

            if g_count > 0:
                self.off_geocode_list = self.insar.off_geocode_list


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
        from isceobj.Sensor import SENSORS
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
        if hasattr(self._insar, 'timeStart'):
            logger.info("Total Time: %i seconds" %
                        (self._insar.timeEnd-self._insar.timeStart))
        return None


    ## Add instance attribute RunWrapper functions, which emulate methods.
    def _add_methods(self):
        self.runPreprocessor = StripmapProc.createPreprocessor(self)
        self.runFormSLC = StripmapProc.createFormSLC(self)
        self.runCrop = StripmapProc.createCrop(self)
        self.runSplitSpectrum = StripmapProc.createSplitSpectrum(self)
        self.runTopo = StripmapProc.createTopo(self)
        self.runGeo2rdr = StripmapProc.createGeo2rdr(self)
        self.runResampleSlc = StripmapProc.createResampleSlc(self)
        self.runRefineSlaveTiming = StripmapProc.createRefineSlaveTiming(self)
        self.runDenseOffsets = StripmapProc.createDenseOffsets(self)
        self.runRubbersheetRange = StripmapProc.createRubbersheetRange(self) #Modified by V. Brancato 10.07.2019
        self.runRubbersheetAzimuth =StripmapProc.createRubbersheetAzimuth(self) #Modified by V. Brancato 10.07.2019
        self.runResampleSubbandSlc = StripmapProc.createResampleSubbandSlc(self)
        self.runInterferogram = StripmapProc.createInterferogram(self)
        self.runFilter = StripmapProc.createFilter(self)
        self.runDispersive = StripmapProc.createDispersive(self)
        self.verifyDEM = StripmapProc.createVerifyDEM(self)
        self.runGeocode = StripmapProc.createGeocode(self)
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

        self.step('cropraw',
                func = self.runCrop,
                args=(True,))

        self.step('formslc', func=self.runFormSLC)

        self.step('cropslc', func=self.runCrop,
                args=(False,))

        # Verify whether the DEM was initialized properly.  If not, download
        # a DEM
        self.step('verifyDEM', func=self.verifyDEM)

        self.step('topo', func=self.runTopo)

        self.step('geo2rdr', func=self.runGeo2rdr)

        self.step('coarse_resample', func=self.runResampleSlc,
                    args=('coarse',))

        self.step('misregistration', func=self.runRefineSlaveTiming)

        self.step('refined_resample', func=self.runResampleSlc,
                    args=('refined',))

        self.step('dense_offsets', func=self.runDenseOffsets)
######################################################################## Modified by V. Brancato 10.07.2019
        self.step('rubber_sheet_range', func=self.runRubbersheetRange)
	
        self.step('rubber_sheet_azimuth',func=self.runRubbersheetAzimuth)
#########################################################################

        self.step('fine_resample', func=self.runResampleSlc,
                    args=('fine',))

        self.step('split_range_spectrum', func=self.runSplitSpectrum)

        self.step('sub_band_resample', func=self.runResampleSubbandSlc,
                    args=(True,))

        self.step('interferogram', func=self.runInterferogram)

        self.step('sub_band_interferogram', func=self.runInterferogram,
                args=("sub",))

        self.step('filter', func=self.runFilter,
                  args=(self.filterStrength,))

        self.step('filter_low_band', func=self.runFilter,
                  args=(self.filterStrength,"low",))

        self.step('filter_high_band', func=self.runFilter,
                  args=(self.filterStrength,"high",))

        self.step('unwrap', func=self.runUnwrapper)

        self.step('unwrap_low_band', func=self.runUnwrapper, args=("low",))

        self.step('unwrap_high_band', func=self.runUnwrapper, args=("high",))

        self.step('ionosphere', func=self.runDispersive)

        self.step('geocode', func=self.runGeocode,
                args=(self.geocode_list, self.geocode_bbox))

        self.step('geocodeoffsets', func=self.runGeocode,
                args=(self.off_geocode_list, self.geocode_bbox, True))

        return None

    ## Main has the common start to both insarApp and dpmApp.
    #@use_api
    def main(self):
        self.timeStart = time.time()
        self.help()

        # Run a preprocessor for the two sets of frames
        self.runPreprocessor()

        #Crop raw data if desired
        self.runCrop(True)

        self.runFormSLC()

        self.runCrop(False)

        #Verify whether user defined  a dem component.  If not, then download
        # SRTM DEM.
        self.verifyDEM()

        # run topo (mapping from radar to geo coordinates)
        self.runTopo()

        # run geo2rdr (mapping from geo to radar coordinates)
        self.runGeo2rdr()

        # resampling using only geometry offsets
        self.runResampleSlc('coarse')

        # refine geometry offsets using offsets computed by cross correlation
        self.runRefineSlaveTiming()

        # resampling using refined offsets
        self.runResampleSlc('refined')

        # run dense offsets
        self.runDenseOffsets()
	
############ Modified by V. Brancato 10.07.2019
        # adding the azimuth offsets computed from cross correlation to geometry offsets 
        self.runRubbersheetAzimuth()
       
        # adding the range offsets computed from cross correlation to geometry offsets 
        self.runRubbersheetRange()
####################################################################################
        # resampling using rubbersheeted offsets
        # which include geometry + constant range + constant azimuth
        # + dense azimuth offsets
        self.runResampleSlc('fine')

        #run split range spectrum
        self.runSplitSpectrum()

        self.runResampleSubbandSlc(misreg=True)
        # forming the interferogram
        self.runInterferogram()

        self.runInterferogram(igramSpectrum = "sub")

        # Filtering and estimating coherence
        self.runFilter(self.filterStrength)

        self.runFilter(self.filterStrength, igramSpectrum = "low")

        self.runFilter(self.filterStrength, igramSpectrum = "high")

        # unwrapping
        self.runUnwrapper()

        self.runUnwrapper(igramSpectrum = "low")

        self.runUnwrapper(igramSpectrum = "high")

        self.runDispersive()

        self.runGeocode(self.geocode_list, self.geocode_bbox)

        self.runGeocode(self.geocode_list, self.geocode_bbox, True)


        self.timeEnd = time.time()
        logger.info("Total Time: %i seconds" %(self.timeEnd - self.timeStart))

        self.renderProcDoc()

        return None

class Insar(_RoiBase):
    """
    Insar Application:
    Implements InSAR processing flow for a pair of scenes from
    sensor raw data to geocoded, flattened interferograms.
    """

    family = "insar"

    def __init__(self, family='',name='',cmdline=None):
        #to allow inheritance with different family name use the locally
        #defined only if the subclass (if any) does not specify one

        super().__init__(
            family=family if family else  self.__class__.family, name=name,
            cmdline=cmdline)

    def Usage(self):
        print("Usages: ")
        print("stripmapApp.py <input-file.xml>")
        print("stripmapApp.py --steps")
        print("stripmapApp.py --help")
        print("stripmapApp.py --help --steps")


    ## extends _InsarBase_steps, but not in the same was as main
    def _steps(self):
        super()._steps()

        # Geocode
        #self.step('geocode', func=self.runGeocode,
        #        args=(self.geocode_list, self.unwrap, self.geocode_bbox))

        self.step('endup', func=self.endup)

        return None

    ## main() extends _InsarBase.main()
    def main(self):

        super().main()
        print("self.timeStart = {}".format(self.timeStart))

        # self.runCorrect()

        #self.runRgoffset()

        # Cull offoutliers
        #self.iterate_runOffoutliers()

        self.runResampleSlc()
        #self.runResamp_only()

        self.runRefineSlaveTiming()

        #self.insar.topoIntImage=self.insar.resampOnlyImage
        #self.runTopo()
#        self.runCorrect()

        # Coherence ?
        #self.runCoherence(method=self.correlation_method)


        # Filter ?
        self.runFilter(self.filterStrength)

        # Unwrap ?
        self.runUnwrapper()

        # Geocode
        #self.runGeocode(self.geocode_list, self.unwrap, self.geocode_bbox)

        timeEnd = time.time()
        logger.info("Total Time: %i seconds" %(timeEnd - self.timeStart))

        self.renderProcDoc()

        return None


if __name__ == "__main__":
    #make an instance of Insar class named 'stripmapApp'
    insar = Insar(name="stripmapApp")
    #configure the insar application
    insar.configure()
    #invoke the base class run method, which returns status
    status = insar.run()
    #inform Python of the status of the run to return to the shell
    raise SystemExit(status)
