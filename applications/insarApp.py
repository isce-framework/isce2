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




from __future__ import print_function
import time
import os
import sys
from isce import logging

import isce
import isceobj
import iscesys
from iscesys.Component.Application import Application
from iscesys.Compatibility import Compatibility
from iscesys.Component.Configurable import SELF
import isceobj.InsarProc as InsarProc
from isceobj.Scene.Frame import FrameMixin

logger = logging.getLogger('isce.insar')


SENSOR_NAME = Application.Parameter(
    'sensorName',
    public_name='sensor name',
    default=None,
    type=str,
    mandatory=True,
    doc="Sensor name"
                                    )

OFFSET_METHOD = Application.Parameter(
    'offsetMethod',
    public_name='slc offset method',
    default="offsetprf",
    type=str,
    mandatory=False,
    doc=("SLC offset estimation method name. "+
         "Use value=ampcor to run ampcor")
                                    )

OFFSET_SEARCH_WINDOW_SIZE = Application.Parameter(
    'offsetSearchWindowSize',
    public_name='offset search window size',
    default=None,
    type=int,
    mandatory=False,
    doc=("Search window size used in offsetprf "+
         "and rgoffset.")
                                )

PEG_SELECT = Application.Parameter(
    'pegSelect',
    public_name='peg select',
    default='average',
    mandatory=False,
    doc='Peg selection method. Can be master, slave or average'
                                )

PEG_LAT = Application.Parameter(
    'pegLat',
    public_name='peg latitude (deg)',
    default=None,
    type=float,
    mandatory=False,
    doc='Peg Latitude in degrees'
                                )

PEG_LON = Application.Parameter(
    'pegLon',
    public_name='peg longitude (deg)',
    default=None,
    type=float,
    mandatory=False,
    doc='Peg Longitude in degrees'
                                )

PEG_HDG = Application.Parameter(
    'pegHdg',
    public_name='peg heading (deg)',
    default=None,
    type=float,
    mandatory=False,
    doc='Peg Heading in degrees'
                                )

PEG_RAD = Application.Parameter(
    'pegRad',
    public_name='peg radius (m)',
    default=None,
    type=float,
    mandatory=False,
    doc='Peg Radius of Curvature in meters'
                                )

FILTER_STRENGTH = Application.Parameter(
    'filterStrength',
    public_name='filter strength',
    default = None,
    type=float,
    mandatory=False,
    doc='Goldstein Werner Filter strength'
                                )

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
DOPPLER_METHOD = Application.Parameter(
    'dopplerMethod',
    public_name='doppler method',
    default='useDOPIQ',
    type=str, mandatory=False,
    doc= "Doppler calculation method.Choices: 'useDOPIQ', 'useCalcDop', 'useDoppler'."
)

USE_DOP = Application.Parameter(
    'use_dop',
    public_name='use_dop',
    default="average",
    type=float,
    mandatory=False,
    doc="Choose whether to use master, slave, or average Doppler for processing."
)

UNWRAPPER_NAME = Application.Parameter(
    'unwrapper_name',
    public_name='unwrapper name',
    default='grass',
    type=str,
    mandatory=False,
    doc="Unwrapping method to use. To be used in  combination with UNWRAP."
)

# to be replaced by DO_UNWRAP;
UNWRAP = Application.Parameter(
    'unwrap',
    public_name='unwrap',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if unwrapping is desired. To be used in combination with UNWRAPPER_NAME."
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

DO_OFFSETPRF = Application.Parameter(
    'do_offsetprf',
    public_name='do offsetprf',
    default=True,
    type=bool,
    mandatory=False,
    doc="Set to False if offsetprf is not required."
                               )

DO_RGOFFSET = Application.Parameter(
    'do_rgoffset',
    public_name='do rgoffset',
    default=True,
    type=bool,
    mandatory=False,
    doc="Set to False if offsetprf is not required."
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
    default=15,
    type=int,
    mandatory=False,
    doc="posting for interferogram"
                                )
RANGE_LOOKS = Application.Parameter(
    'rangeLooks',
    public_name='range looks',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of range looks to use in resamp'
                                    )
AZ_LOOKS = Application.Parameter(
    'azLooks',
    public_name='azimuth looks',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of azimuth looks to use in resamp'
                                 )
PATCH_SIZE = Application.Parameter(
    'patchSize',
     public_name='azimuth patch size',
     default=None,
     type=int,
     mandatory=False,
      doc=(
    "Size of overlap/save patch size for formslc"
    )
                                   )

GOOD_LINES = Application.Parameter(
    'goodLines',
    public_name='patch valid pulses',
    default=None,
    type=int,
    mandatory=False,
    doc=(
    "Size of overlap/save save region for formslc"
     )
                                   )

NUM_PATCHES = Application.Parameter(
    'numPatches',
    public_name='number of patches',
    default=None,
    type=int,
    mandatory=False,
    doc=(
    "How many patches to process of all available patches"
    )
                                    )

GROSS_AZ = Application.Parameter(
    'grossAz',
    public_name='gross azimuth offset',
    default=None,
    type=int,
    mandatory=False,
    doc=(
    "Override the value of the gross azimuth offset for offset " +
    "estimation prior to interferogram formation"
    )
                                 )

GROSS_RG = Application.Parameter(
    'grossRg',
    public_name='gross range offset',
    default=None,
    type=int,
    mandatory=False,
    doc=(
    "Override the value of the gross range offset for offset" +
    "estimation prior to interferogram formation"
    )
                                 )

CULLING_SEQUENCE = Application.Parameter(
    'culling_sequence',
    public_name='Culling Sequence',
    default= (10,5,3),
    container=tuple,
    type=int,
    doc="TBD"
                                         )

CULLING_ERROR_LIMIT = Application.Parameter(
    'culling_error_limit',
    public_name='Culling error limit',
    default=100,
    type = int,
    mandatory = False,
    doc = 'Minimum number of culled offsets to be used for offset field polynomial estimation'
                                        )

GEOCODE_LIST = Application.Parameter(
    'geocode_list',
     public_name='geocode list',
     default = None,
     container=list,
     type=str,
     doc = "List of products to geocode."
                                      )

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
    default='pickle',
    type=str,
    mandatory=False,
    doc=(
    "Format in which the data is serialized when using steps. Options are xml (default) or pickle."
    )
                                        )

#Facility declarations
MASTER = Application.Facility(
    'master',
    public_name='Master',
    module='isceobj.Sensor',
    factory='createSensor',
    args=(SENSOR_NAME, 'master'),
    mandatory=True,
    doc="Master raw data component"
                              )

SLAVE = Application.Facility(
    'slave',
    public_name='Slave',
    module='isceobj.Sensor',
    factory='createSensor',
    args=(SENSOR_NAME,'slave'),
    mandatory=True,
    doc="Slave raw data component"
                             )

MASTERDOP = Application.Facility(
    'masterdop',
    public_name='Master Doppler',
    module='isceobj.Doppler',
    factory='createDoppler',
    args=(DOPPLER_METHOD,),
    mandatory=False,
    doc="Master Doppler calculation method"
                                 )

SLAVEDOP = Application.Facility(
    'slavedop',
    public_name='Slave Doppler',
    module='isceobj.Doppler',
    factory='createDoppler',
    args=(DOPPLER_METHOD,),
    mandatory=False,
    doc="Master Doppler calculation method"
                                )

DEM = Application.Facility(
    'dem',
    public_name='Dem',
    module='isceobj.Image',
    factory='createDemImage',
    mandatory=False,
    doc=(
    "Dem Image configurable component.  Do not include this in the "+
    "input file and an SRTM Dem will be downloaded for you."
    )
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

RUN_ESTIMATE_HEIGHTS = Application.Facility(
    'runEstimateHeights',
    public_name='Estimate Heights',
    module='isceobj.InsarProc',
    factory='createEstimateHeights',
    args=(SELF(), SENSOR_NAME),
    mandatory=False,
    doc="mocomp height estimation module"
        )

RUN_FORM_SLC = Application.Facility(
    'runFormSLC',
    public_name='Form SLC',
    module='isceobj.InsarProc',
    factory='createFormSLC',
    args=(SELF(), SENSOR_NAME),
    mandatory=False,
    doc="SLC formation module"
)

RUN_OFFSETPRF = Application.Facility(
    'runOffsetprf',
    public_name='slc offsetter',
    module='isceobj.InsarProc',
    factory='createOffsetprf',
    args=(SELF(), OFFSET_METHOD, DO_OFFSETPRF),
    mandatory=False,
    doc="Offset a pair of SLC images."
)

RUN_RGOFFSET = Application.Facility(
    'runRgoffset',
    public_name='dem offseter',
    module = 'isceobj.InsarProc',
    factory= 'createRgoffset',
    args=(SELF(), OFFSET_METHOD, DO_RGOFFSET),
    mandatory=False,
    doc="Dem offset estimator."
)

RUN_UNWRAPPER = Application.Facility(
    'runUnwrapper',
    public_name='Run unwrapper',
    module='isceobj.InsarProc',
    factory='createUnwrapper',
    args=(SELF(), DO_UNWRAP, UNWRAPPER_NAME, UNWRAP),
    mandatory=False,
    doc="Unwrapping module"
)

RUN_UNWRAP_2STAGE = Application.Facility(
    'runUnwrap2Stage',
    public_name='Run unwrapper 2 Stage',
    module='isceobj.InsarProc',
    factory='createUnwrap2Stage',
    args=(SELF(), DO_UNWRAP_2STAGE, UNWRAPPER_NAME),
    mandatory=False,
    doc="Unwrapping module"
)

_INSAR = Application.Facility(
    '_insar',
    public_name='insarproc',
    module='isceobj.InsarProc',
    factory='createInsarProc',
    args = ('insarAppContext',isceobj.createCatalog('insarProc')),
    mandatory=False,
    doc="InsarProc object"
)


## Common interface for all insar applications.
class _InsarBase(Application, FrameMixin):

    family = 'insar'
    ## Define Class parameters in this list
    parameter_list = (SENSOR_NAME,
                      OFFSET_METHOD,
                      OFFSET_SEARCH_WINDOW_SIZE,
                      PEG_SELECT,
                      PEG_LAT,
                      PEG_LON,
                      PEG_HDG,
                      PEG_RAD,
                      FILTER_STRENGTH,
                      CORRELATION_METHOD,
                      DOPPLER_METHOD,
                      USE_DOP,
                      UNWRAP,
                      UNWRAPPER_NAME,
                      DO_UNWRAP,
                      DO_OFFSETPRF,
                      DO_RGOFFSET,
                      USE_HIGH_RESOLUTION_DEM_ONLY,
                      DEM_FILENAME,
                      GEO_POSTING,
                      POSTING,
                      RANGE_LOOKS,
                      AZ_LOOKS,
                      PATCH_SIZE,
                      GOOD_LINES,
                      NUM_PATCHES,
                      GROSS_AZ,
                      GROSS_RG,
                      CULLING_SEQUENCE,
                      CULLING_ERROR_LIMIT,
                      GEOCODE_LIST,
                      GEOCODE_BOX,
                      PICKLE_DUMPER_DIR,
                      PICKLE_LOAD_DIR,
                      RENDERER,
                      DO_UNWRAP_2STAGE,
                      UNWRAPPER_2STAGE_NAME,
                      SOLVER_2STAGE)

    facility_list = (MASTER,
                     SLAVE,
                     MASTERDOP,
                     SLAVEDOP,
                     DEM,
                     DEM_STITCHER,
                     RUN_ESTIMATE_HEIGHTS,
                     RUN_FORM_SLC,
                     RUN_UNWRAPPER,
                     RUN_UNWRAP_2STAGE,
                     RUN_OFFSETPRF,
                     RUN_RGOFFSET,
                     _INSAR)

    _pickleObj = "_insar"

    def __init__(self, family='', name='',cmdline=None):
        import isceobj
        super().__init__(family=family, name=name,
            cmdline=cmdline)

        from isceobj.InsarProc import InsarProc
        from iscesys.StdOEL.StdOELPy import create_writer
        self._stdWriter = create_writer("log", "", True, filename="insar.log")
        self._add_methods()
        self._insarProcFact = InsarProc
        '''
        procDoc = isceobj.createCatalog('insarProc')
        #self._insar = InsarProc.InsarProc(name='insarApp_conf',
        #    procDoc=procDoc
        #    )
        self.insar.procDoc._addItem("ISCE_VERSION",
            "Release: %s, svn-%s, %s. Current svn-%s" %
            (isce.release_version, isce.release_svn_revision,
             isce.release_date, isce.svn_revision
            ),
            ["insarProc"]
            )
        '''
        return None

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
#        print("self.sensorName = ", self.sensorName)
#        print("self.correlation_method = ", self.correlation_method)
#        print("self.use_dop = ", self.use_dop)
#        print("self.geoPosting = ", self.geoPosting)
#        print("self.posting = ", self.posting)
#        print("self.rangeLooks = ", self.rangeLooks)
#        print("self.azLooks = ", self.azLooks)
#        print("self.offsetMethod = ", self.offsetMethod)
#        print("self.grossRg, self.grossAz =  ", self.grossRg, self.grossAz )
        if ( self.pegLat is not None and
             self.pegLon is not None and
             self.pegHdg is not None and
             self.pegRad is not None ):
            from isceobj.Location.Peg import Peg
            self.peg = Peg(latitude=self.pegLat,
                           longitude=self.pegLon,
                           heading=self.pegHdg,
                           radiusOfCurvature=self.pegRad)
#            print("self.peg = ", self.peg)
        else:
            self.peg = None
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
            ["insarProc"]
            )
        #This is a temporary fix to get the user interface back to the dem
        #facility interface while changes are being made in the DemImage class
        #to include within it the capabilities urrently in extractInfo and
        #createDem.
        #jng ask Eric No longer needed
        if self.demFilename:
            import sys
            print(
            "The demFilename property is no longer supported as an " +
            "input parameter."
                )
            print(
                "The original method using a configurable facility for the " +
                "Dem is now restored."
                )
            print(
                "The automatic download feature is still supported in the " +
                " same way as before:"
                )
            print(
                "If you want automatic download of a Dem, then simply omit "+
                "any configuration\ninformation in your input file regarding "+
                "the Dem."
                )
            print()
            print(
                "Please replace the following information in your input file:"
                )
            print()
            print(
                "<property name='demFilename'><value>%s</value></property>" %
                self.demFilename
                )
            print()
            print("with the following information and try again:")
            print()
            print(
                "<component name=\'Dem\'><catalog>%s</catalog></component>" %
                self.demFilename
                )
            print()
        else:
            try:
                self.dem.checkInitialization()
                #jng ask Eric. self.demFilename no longer needed
                # Give self.demFilename a value so that the SRTM Dem will not
                # be downloaded
                # Temporary fix that will be removed when the download option
                # is handled within demImage
                self.demFilename = "demFilename"
                self.insar.demImage = self.dem
            except Exception as err:
                pass
                #The following print statement is sometimes misleading when
                #checkInitialization fails for a reason other than the dem
                #not being found. The checkInitialization error should be
                #handled more appropriately.
                #print(
                #    "The Dem specified was not properly initialized. An SRTM" +
                #    " Dem will be downloaded."
                #    )
                #self.dem was not properly initialized
                #and self.demFilename is undefined.
                #There is a check on self.demFilename
                #below to download if necessary
            else:
                dem_snwe = self.dem.getsnwe()

                if self.geocode_bbox:
                    ####Adjust bbox according to dem
                    if self.geocode_bbox[0] < dem_snwe[0]:
                        logger.warn('Geocoding southern extent changed to match DEM')
                        self.geocode_bbox[0] = dem_snwe[0]

                    if self.geocode_bbox[1] > dem_snwe[1]:
                        logger.warn('Geocoding northern extent changed to match DEM')
                        self.geocode_bbox[1] = dem_snwe[1]

                    if self.geocode_bbox[2] < dem_snwe[2]:
                        logger.warn('Geocoding western extent changed to match DEM')
                        self.geocode_bbox[2] = dem_snwe[2]

                    if self.geocode_bbox[3] > dem_snwe[3]:
                        logger.warn('Geocoding eastern extent changed to match DEM')
                        self.geocode_bbox[3] = dem_snwe[3]

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

    ## Method return True iff it changes the demFilename.
    from isceobj.Util.decorators import use_api
    @use_api
    def verifyDEM(self):
        masterF = self._insar.masterFrame
        slaveF = self._insar.slaveFrame
        info = self.extractInfo(masterF, slaveF)
        #if an image has been specified, then no need to create one
        if not self.dem.filename:
            self.createDem(info)
        else:
            self.insar.demImage = self.dem

        #ensure that the dem vrt file exists by creating (or recreating) it
        self.insar.demImage.renderVRT()

        #at this point a dem image has been set into self.insar, whether it
        #was stitched together or read in input
        demImage =  self.insar.demImage
        #if the demImage is already in wgs84 (because was provided in input) then skip and proceed
        if demImage.reference.upper() != 'WGS84':
            wgs84demFilename = self.insar.demImage.filename+'.wgs84'
            wgs84demxmlFilename = wgs84demFilename+'.xml'
            #if the dem reference is EGM96 and the WGS84 corrected
            #dem files are not found, then create the WGS84 files
            #using the demStitcher's correct method
            if( demImage.reference.upper() == 'EGM96' and
                not (os.path.isfile(wgs84demFilename) and
                     os.path.isfile(wgs84demxmlFilename))
            ):
                self.insar.demImage = self.demStitcher.correct(demImage)
            #make sure to load the wgs84 if present
            elif(os.path.isfile(wgs84demFilename) and
                     os.path.isfile(wgs84demxmlFilename)):
                from isceobj import createDemImage
                self.insar.demImage  = createDemImage()
                self.insar.demImage.load(wgs84demxmlFilename)
                if(self.insar.demImage.reference.upper() != 'WGS84'):
                    print('The dem',wgs84demFilename,'is not wgs84')
                    raise Exception

        #ensure that the wgs84 dem vrt file exists
        self.insar.demImage.renderVRT()

        #get water mask
        self.runCreateWbdMask(info)


        return None



    def renderProcDoc(self):
        self.procDoc.renderXml()

    ## Run runOffoutliers() repeatedly with arguments from "iterator" keyword
    def iterate_runOffoutliers(self, iterator=None):
        """iterate_runOffoutliers(iterator)

        runs runOffoutliers multiple times with values (integers) from iterator.

        iterator defaults to Insar._default_culling_sequence
        """
        if iterator is None: iterator = self.culling_sequence
        erriterator = [self.culling_error_limit]*len(iterator)
        list(map(self.runOffoutliers, iterator, erriterator))
        return None

    def set_topoint1(self):
        self._insar.topoIntImage = self._insar.resampIntImage
        return None

    def set_topoint2(self):
        self._insar.topoIntImage = self._insar.resampOnlyImage
        return None

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
        self.runPreprocessor = InsarProc.createPreprocessor(self)
        self.extractInfo = InsarProc.createExtractInfo(self)
        self.createDem = InsarProc.createCreateDem(self)
        self.runCreateWbdMask = InsarProc.createCreateWbdMask(self)
        self.runMaskImages = InsarProc.createMaskImages(self)
        self.runPulseTiming = InsarProc.createPulseTiming(self)
        self.runSetmocomppath = InsarProc.createSetmocomppath(self)
        self.runOrbit2sch = InsarProc.createOrbit2sch(self)
        self.updatePreprocInfo = InsarProc.createUpdatePreprocInfo(self)
        self.runOffoutliers = InsarProc.createOffoutliers(self)
        self.prepareResamps = InsarProc.createPrepareResamps(self)
        self.runResamp = InsarProc.createResamp(self)
        self.runResamp_image = InsarProc.createResamp_image(self)
        self.runMocompbaseline = InsarProc.createMocompbaseline(self)
        self.runTopo = InsarProc.createTopo(self)
        self.runCorrect = InsarProc.createCorrect(self)
        self.runShadecpx2rg = InsarProc.createShadecpx2rg(self)
        self.runResamp_only = InsarProc.createResamp_only(self)
        self.runCoherence = InsarProc.createCoherence(self)
        self.runFilter = InsarProc.createFilter(self)
        self.runGrass = InsarProc.createGrass(self)
        self.runGeocode = InsarProc.createGeocode(self)
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

        # Verify whether the DEM was initialized properly.  If not, download
        # a DEM
        self.step('verifyDEM', func=self.verifyDEM)

        # Run pulsetiming for each set of frames
        self.step('pulsetiming', func=self.runPulseTiming)

        self.step('estimateHeights', func=self.runEstimateHeights)

        # Run setmocomppath
        self.step('mocompath', func=self.runSetmocomppath, args=(self.peg,))

        #init and run orbit2sch
        self.step('orbit2sch', func=self.runOrbit2sch)

        #update quantities in objPreProc obtained from previous steps
        self.step('updatepreprocinfo',
                  func=self.updatePreprocInfo,
                  args=(self.use_dop,))

        self.step('formslc', func=self.runFormSLC)

        self.step('offsetprf', func=self.runOffsetprf)

        # Cull offoutliers
        self.step('outliers1', func=self.iterate_runOffoutliers)

        self.step('prepareresamps',
                  func=self.prepareResamps,
                  args=(self.rangeLooks,self.azLooks))

        self.step('resamp', func=self.runResamp)

        self.step('resamp_image', func=self.runResamp_image)

        # Mocompbaseline
        self.step('mocompbaseline', func=self.runMocompbaseline)

        # Topocorrect
        #self.step('settopoint1',
        #          func=self.insar.delayed_attrcopy_from_to('resampIntImage',
        #                                                    'topoIntImage')
        #          )
        self.step('settopoint1', func=self.set_topoint1)

        self.step('topo', func=self.runTopo)

        self.step('shadecpx2rg', func=self.runShadecpx2rg)

        # Compute offsets and cull offoutliers
        self.step('rgoffset', func=self.runRgoffset)
        self.step('rg_outliers2', func=self.iterate_runOffoutliers)

        self.step('resamp_only', func=self.runResamp_only)

        #Topocorrect
        #self.step('settopoint2',
        #          func=self.insar.delayed_attrcopy_from_to('resampOnlyImage',
        #                                                    'topoIntImage')
        #          )
        self.step('settopoint2', func=self.set_topoint2)

        self.step('correct', func=self.runCorrect)

        # Coherence ?
        self.step('coherence',
                  func=self.runCoherence,
                  args=(self.correlation_method,))

        # Filter ?
        self.step('filter', func=self.runFilter,
                  args=(self.filterStrength,))

        #add water mask to coherence and interferogram
        self.step('mask', func=self.runMaskImages)

        # Unwrap ?
        self.step('unwrap', func=self.runUnwrapper)

        # Conditional 2 stage unwrapping
        self.step('unwrap2stage', func=self.runUnwrap2Stage,
                  args=(self.unwrapper_2stage_name, self.solver_2stage))

        return None

    ## Main has the common start to both insarApp and dpmApp.
    def main(self):
        self.help()

        # Run a preprocessor for the two sets of frames
        self.runPreprocessor()
        #Verify whether user defined  a dem component.  If not, then download
        # SRTM DEM.
        self.verifyDEM()

        # Run pulsetiming for each set of frames
        self.runPulseTiming()
        self.runEstimateHeights()

        # Run setmocomppath
        self.runSetmocomppath(peg=self.peg)

        #init and run orbit2sch
        self.runOrbit2sch()

        #update quantities in objPreProc obtained from previous steps
        self.updatePreprocInfo(use_dop=self.use_dop)

        self.runFormSLC()

        self.runOffsetprf()

        # Cull offoutliers
        self.iterate_runOffoutliers()

        self.prepareResamps(self.rangeLooks, self.azLooks)
        self.runResamp()
        self.runResamp_image()

        # Mocompbaseline
        self.runMocompbaseline()

        # Topocorrect
        self.insar.topoIntImage = self.insar.resampIntImage
        self.runTopo()
        return None

    @property
    def resampAmpImage(self):
        return self.insar.resampAmpImage


    pass




class Insar(_InsarBase):
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
        print("insarApp.py <input-file.xml>")
        print("insarApp.py --steps")
        print("insarApp.py --help")
        print("insarApp.py --help --steps")


    ## extends _InsarBase_steps, but not in the same was as main
    def _steps(self):
        super()._steps()

        # Geocode
        self.step('geocode', func=self.runGeocode,
                args=(self.geocode_list, self.unwrap, self.geocode_bbox))

#        self.step('endup', func=self.endup)

        return None

    ## main() extends _InsarBase.main()
    def main(self):
        import time
        timeStart = time.time()

        super().main()

        # self.runCorrect()

        self.runShadecpx2rg()

        self.runRgoffset()

        # Cull offoutliers
        self.iterate_runOffoutliers()

        self.runResamp_only()

        self.insar.topoIntImage=self.insar.resampOnlyImage
        #self.runTopo()
        self.runCorrect()

        # Coherence ?
        self.runCoherence(method=self.correlation_method)


        # Filter ?
        self.runFilter(self.filterStrength)

        #add water mask to coherence and interferogram
        self.runMaskImages()
        # Unwrap ?
        self.runUnwrapper()

        # 2Stage Unwrapping
        self.runUnwrap2Stage(self.unwrapper_2stage_name, self.solver_2stage)

        # Geocode
        self.runGeocode(self.geocode_list, self.unwrap, self.geocode_bbox)

        timeEnd = time.time()
        logger.info("Total Time: %i seconds" %(timeEnd - timeStart))

        self.renderProcDoc()

        return None




if __name__ == "__main__":
    if not isce.stanford_license:
        print("This workflow requires the Stanford licensed code elemnts.")
        print("Unable to find the license information in the isce.stanford_license file.")
        print("Please either obtain a stanford license and follow the instructions to")
        print("install the stanford code elements or else choose a different workflow.")
        raise SystemExit(0)
    else:
        #make an instance of Insar class named 'insarApp'
        insar = Insar(name="insarApp")
        #configure the insar application
        insar.configure()
        #invoke the base class run method, which returns status
        status = insar.run()
        #inform Python of the status of the run to return to the shell
        raise SystemExit(status)
