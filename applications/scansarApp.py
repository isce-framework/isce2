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
from isceobj import ScansarProc

logger = logging.getLogger('isce.insar')


SENSOR_NAME = Application.Parameter(
    'sensorName',
    public_name='sensor name',
    default='ALOS2',
    type=str,
    mandatory=True,
    doc="Sensor name"
                                    )
FULL_APERTURE_PRODUCT = Application.Parameter(
    'isFullApertureProduct',
    public_name='is full aperture product',
    default=False,
    type=bool,
    mandatory=True,
    doc= "To indicate full aperture or burst-by-burst")

BURST_OVERLAP_THRESHOLD = Application.Parameter(
        'burstOverlapThreshold',
        public_name='burst overlap threshold',
        default=85.0,
        type=float,
        mandatory=True,
        doc='Minimum burst overlap needed to stop triggering common azimuth spectra filtering')

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

FILTER_STRENGTH = Application.Parameter('filterStrength',
                                      public_name='filter strength',
                                      default=0.5,
                                      type=float,
                                      mandatory=False,
                                      doc='')

OFFSET_SNR_THRESHOLD = Application.Parameter('offsetSNRThreshold',
                                public_name = 'offset SNR threshold',
                                default=8.0,
                                type=float,
                                mandatory = False,
                                doc = 'Offset SNR threshold')

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

#Facility declarations
MASTER = Application.Facility(
    'master',
    public_name='Master',
    module='isceobj.Sensor.ScanSAR',
    factory='createSensor',
    args=(SENSOR_NAME, 'master'),
    mandatory=True,
    doc="Master raw data component"
                              )

SLAVE = Application.Facility(
    'slave',
    public_name='Slave',
    module='isceobj.Sensor.ScanSAR',
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
    module='isceobj.ScansarProc',
    factory='createUnwrapper',
    args=(SELF(), DO_UNWRAP, UNWRAPPER_NAME),
    mandatory=False,
    doc="Unwrapping module"
)

RUN_UNWRAP_2STAGE = Application.Facility(
    'runUnwrap2Stage',
    public_name='Run unwrapper 2 Stage',
    module='isceobj.ScansarProc',
    factory='createUnwrap2Stage',
    args=(SELF(), DO_UNWRAP_2STAGE, UNWRAPPER_NAME),
    mandatory=False,
    doc="Unwrapping module"
)

_INSAR = Application.Facility(
    '_insar',
    public_name='scansarproc',
    module='isceobj.ScansarProc',
    factory='createScansarProc',
    args = ('scansarAppContext',isceobj.createCatalog('scansarProc')),
    mandatory=False,
    doc="ScansarProc object"
)


## Common interface for all insar applications.
class ScansarInSAR(Application):

    family = 'scansarinsar'
    ## Define Class parameters in this list
    parameter_list = (SENSOR_NAME,
                      UNWRAPPER_NAME,
                      DEM_FILENAME,
                      GEOCODE_DEM_FILENAME,
                      BURST_OVERLAP_THRESHOLD,
                      NUMBER_AZIMUTH_LOOKS,
                      NUMBER_RANGE_LOOKS,
                      FILTER_STRENGTH,
                      OFFSET_SNR_THRESHOLD,
                      DO_INSAR,
                      DO_UNWRAP,
                      USE_HIGH_RESOLUTION_DEM_ONLY,
                      GEOCODE_BOX,
                      PICKLE_DUMPER_DIR,
                      PICKLE_LOAD_DIR,
                      RENDERER,
                      DO_UNWRAP_2STAGE,
                      UNWRAPPER_2STAGE_NAME,
                      SOLVER_2STAGE,
                      GEOCODE_LIST,
                      USE_VIRTUAL_FILES,
                      SWATHS,
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
                      FILTER_NULL,
                      FILTER_WIN_SIZE,
                      OFFSET_GEOCODE_LIST,
                      USE_GPU)

    facility_list = (MASTER,
                     SLAVE,
                     DEM_STITCHER,
                     RUN_UNWRAPPER,
                     RUN_UNWRAP_2STAGE,
                     _INSAR)

    _pickleObj = "_insar"

    def __init__(self, family='', name='',cmdline=None):
        import isceobj
        from isceobj.ScansarProc import ScansarProc
        from iscesys.StdOEL.StdOELPy import create_writer

        super().__init__(
            family=family if family else  self.__class__.family, name=name,
            cmdline=cmdline)

        self._stdWriter = create_writer("log", "", True, filename="scansarinsar.log")
        self._add_methods()
        self._insarProcFact = ScansarProc
        return None



    def Usage(self):
        print("Usages: ")
        print("scansarApp.py <input-file.xml>")
        print("scansarApp.py --steps")
        print("scansarApp.py --help")
        print("scansarApp.py --help --steps")


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
        from isceobj.Sensor.ScanSAR import SENSORS
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
        self.runPreprocessor = ScansarProc.createPreprocessor(self)
        self.runCommonRangeSpectra = ScansarProc.createCommonRangeSpectra(self)
        self.runEqualizeSlcs = ScansarProc.createEqualizeSlcs(self)
        self.runEstimateBurstSync = ScansarProc.createEstimateBurstSync(self)
#        self.runComputeBaseline = ScansarProc.createComputeBaseline(self)
#        self.verifyDEM = ScansarProc.createVerifyDEM(self)
#        self.verifyGeocodeDEM = ScansarProc.createVerifyGeocodeDEM(self)
#        self.runTopo  = ScansarProc.createTopo(self)
#        self.runSubsetOverlaps = ScansarProc.createSubsetOverlaps(self)
#        self.runCoarseOffsets = ScansarProc.createCoarseOffsets(self)
#        self.runCoarseResamp = ScansarProc.createCoarseResamp(self)
#        self.runOverlapIfg = ScansarProc.createOverlapIfg(self)
#        self.runPrepESD = ScansarProc.createPrepESD(self)
#        self.runESD = ScansarProc.createESD(self)
#        self.runRangeCoreg = ScansarProc.createRangeCoreg(self)
#        self.runFineOffsets = ScansarProc.createFineOffsets(self)
#        self.runFineResamp = ScansarProc.createFineResamp(self)
#        self.runBurstIfg = ScansarProc.createBurstIfg(self)
#        self.runMergeBursts = ScansarProc.createMergeBursts(self)
#        self.runFilter = ScansarProc.createFilter(self)
#        self.runGeocode = ScansarProc.createGeocode(self)
#        self.runDenseOffsets = ScansarProc.createDenseOffsets(self)
#        self.runOffsetFilter = ScansarProc.createOffsetFilter(self)

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

        # Run common range spectra filtering
        self.step('commonrangespectra',
                   func=self.runCommonRangeSpectra,
                   doc=("""Filter images to common range spectra"""))


        #Run image equalization to make pixels same size
        self.step('equalizeslcs',
                  func=self.runEqualizeSlcs,
                  doc=("""Make pixel sizes the same"""))

        #Run estimation of burst sync
        self.step('estimateburstsync',
                  func=self.runEstimateBurstSync,
                  doc=("""Estimate amount of burst sync"""))

        # Compute baselines and estimate common bursts
        #self.step('computeBaselines',
        #        func=self.runComputeBaseline,
        #        doc=(
        #            """Compute baseline and number of common bursts"""
        #        )
        #          )

        # Verify whether the DEM was initialized properly.  If not, download
        # a DEM
        #self.step('verifyDEM', func=self.verifyDEM)

        ##Run topo for each bursts
        #self.step('topo', func=self.runTopo)

        ##Run subset overlaps
        #self.step('subsetoverlaps', func=self.runSubsetOverlaps)

        ##Run coarse offsets
        #self.step('coarseoffsets', func=self.runCoarseOffsets)

        ####Run coarse resamp
        #self.step('coarseresamp', func=self.runCoarseResamp)

        ####Run overlap ifgs
        #self.step('overlapifg', func=self.runOverlapIfg)

        ###Run prepare ESD inputs
        #self.step('prepesd', func=self.runPrepESD)

        ###Run ESD
        #self.step('esd', func=self.runESD)

        ###Run range coregistration
        #self.step('rangecoreg', func=self.runRangeCoreg)

        ###Estimate fine offsets
        #self.step('fineoffsets', func=self.runFineOffsets)

        ###Resample slave bursts
        #self.step('fineresamp', func=self.runFineResamp)

        ####Create burst interferograms
        #self.step('burstifg', func=self.runBurstIfg)

        ###Merge burst products into a single file
        #self.step('mergebursts', func=self.runMergeBursts)

        ###Filter the interferogram
        #self.step('filter', func=self.runFilter)


        # Unwrap ?
        #self.step('unwrap', func=self.runUnwrapper)

        # Conditional 2 stage unwrapping
        #self.step('unwrap2stage', func=self.runUnwrap2Stage,
        #          args=(self.unwrapper_2stage_name, self.solver_2stage))


        # Geocode
        #self.step('geocode', func=self.runGeocode,
        #        args=(self.geocode_list, self.do_unwrap, self.geocode_bbox))

        # Dense offsets
        #self.step('denseoffsets', func=self.runDenseOffsets)

        #Filter offsets
        #self.step('filteroffsets', func=self.runOffsetFilter)

        #Geocode offsets
        #self.step('geocodeoffsets', func=self.runGeocode,
        #        args=(self.off_geocode_list, False, self.geocode_bbox, True))

#        self.step('endup', func=self.endup)
        return None

    ## Main has the common start to both insarApp and dpmApp.
    def main(self):
        self.help()

        timeStart= time.time()

        # Run a preprocessor for the two sets of frames
        self.runPreprocessor()

        #Filter to common range spectra
        self.runCommonRangeSpectra()

        #Make pixels the same size
        self.runEqualizeSlcs()

        #Estimate amount of burst sync
        self.runEstimateBurstSync()

        #Compute baselines and common bursts
        #self.runComputeBaseline()


        #Verify whether user defined  a dem component.  If not, then download
        # SRTM DEM.
        #self.verifyDEM()

        ##Run topo for each burst
        #self.runTopo()

        ##Run subset overlaps
        #self.runSubsetOverlaps()

        ##Run coarse offsets
        #self.runCoarseOffsets()

        ##Run coarse resamp
        #self.runCoarseResamp()

        ##Run ifg
        #self.runOverlapIfg()

        ##Prepare for ESD
        #self.runPrepESD()

        #Run ESD
        #self.runESD()

        ###Estimate range misregistration
        #self.runRangeCoreg()

        ###Estimate fine offsets
        #self.runFineOffsets()

        ###Resample slave bursts
        #self.runFineResamp()

        ###Create burst interferograms
        #self.runBurstIfg()

        ####Merge bursts into single files
        #self.runMergeBursts()

        ###Filter the interferogram
        #self.runFilter()

        #add water mask to coherence and interferogram
        #self.runMaskImages()

        # Unwrap ?
        #self.runUnwrapper()

        # 2Stage Unwrapping
        #self.runUnwrap2Stage(self.unwrapper_2stage_name, self.solver_2stage)

        # Geocode
        #self.runGeocode(self.geocode_list, self.do_unwrap, self.geocode_bbox)


        #Dense offsets
        #self.runDenseOffsets()

        #Filter offsets
        #self.runOffsetFilter()


        #Geocode offsets
        #self.runGeocode(self.off_geocode_list, False, self.geocode_bbox, True)

        timeEnd = time.time()
        logger.info("Total Time: %i seconds" %(timeEnd - timeStart))

        self.renderProcDoc()

        return None




if __name__ == "__main__":
    import sys
    insar = ScansarInSAR(name="scansarApp")
    insar.configure()
    insar.run()
