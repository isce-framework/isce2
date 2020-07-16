#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Kosal Khun, Marco Lavalle
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Comment: Adapted from InsarProc/InsarProc.py
from __future__ import print_function
import os
import sys
import logging
import logging.config
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from iscesys.Compatibility import Compatibility
from isceobj.Scene.Frame import FrameMixin

PROCEED_IF_ZERO_DEM = Component.Parameter(
    '_proceedIfZeroDem',
    public_name='proceed if zero dem',
    default=False,
    type=bool,
    mandatory=False,
    doc='Flag to apply continue processing if a dem is not available or cannot be downloaded.'
)

IS_MOCOMP = Component.Parameter('is_mocomp',
    public_name='is_mocomp',
    default=1,
    type=int,
    mandatory=False,
    doc=''
)

PEG = Component.Facility('_peg',
                          public_name='peg',
                          module='isceobj.Location.Peg',
                          factory='Peg',
                          mandatory=False,
                          doc='')

class IsceProc(Component, FrameMixin):

    parameter_list = (IS_MOCOMP,
                      PROCEED_IF_ZERO_DEM)
    facility_list = (PEG,)

    family = 'isceappcontext'

    # Getters
    @property
    def proceedIfZeroDem(self):
        return self._proceedIfZeroDem
    @proceedIfZeroDem.setter
    def proceedIfZeroDem(self, v):
        self._proceedIfZeroDem = v
    @property
    def selectedPols(self):
        return self._selectedPols
    @selectedPols.setter
    def selectedPols(self, v):
        self._selectedPols = v
        return

    @property
    def selectedScenes(self):
        return self._selectedScenes
    @selectedScenes.setter
    def selectedScenes(self, v):
        self._selectedScenes = v
        return

    @property
    def selectedPairs(self):
        return self._selectedPairs
    @selectedPairs.setter
    def selectedPairs(self, v):
        self._selectedPairs = v
        return

    @property
    def coregStrategy(self):
        return self._coregStrategy
    @coregStrategy.setter
    def coregStrategy(self, v):
        self._coregStrategy = v
        return

    @property
    def refScene(self):
        return self._refScene
    @refScene.setter
    def refScene(self, v):
        self._refScene = v
        return

    @property
    def refPol(self):
        return self._refPol
    @refPol.setter
    def refPol(self, v):
        self._refPol = v
        return

    @property
    def frames(self):
        return self._frames
    @frames.setter
    def frames(self, v):
        self._frames = v
        return

    @property
    def pairsToCoreg(self):
        return self._pairsToCoreg
    @pairsToCoreg.setter
    def pairsToCoreg(self, v):
        self._pairsToCoreg = v
        return

    @property
    def srcFiles(self):
        return self._srcFiles
    @srcFiles.setter
    def srcFiles(self, v):
        self._srcFiles = v
        return

    @property
    def demImage(self):
        return self._demImage
    @demImage.setter
    def demImage(self, v=None):
        self._demImage = v
        return

    @property
    def geocode_list(self):
        return self._geocode_list
    @geocode_list.setter
    def geocode_list(self, v):
        self._geocode_list = v
        return

    @property
    def dataDirectory(self):
        return self._dataDirectory
    @dataDirectory.setter
    def dataDirectory(self, v):
        self._dataDirectory = v
        return

    @property
    def processingDirectory(self):
        return self._processingDirectory
    @processingDirectory.setter
    def processingDirectory(self, v):
        self._processingDirectory = v
        return

    def __init__(self, name='', procDoc=None):
        """
        Initiate all the attributes that will be used
        """
        self.name = self.__class__.family

        self.workingDirectory = os.getcwd()
        self.dataDirectory = None
        self.processingDirectory = None

        self.selectedScenes = [] # ids of selected scenes, ordered by scene number
        self.selectedPols = [] # hh, hv, vh, vv
        self.selectedPairs = [] # list of tuples (p1, p2) selected for inSAR
        self.srcFiles = {} # path and info about provider's data (for each scene and each pol)
        self.frames = {}
        self.dopplers = {}
        self.orbits = {}
        self.shifts = {} # azimuth shifts

        self.pegAverageHeights = {}
        self.pegProcVelocities = {}
        self.fdHeights = {}

        self.rawImages = {}
        self.iqImages = {}
        self.slcImages = {}
        self.formSLCs = {}
        self.squints = {}
        self.offsetAzimuthImages = {}
        self.offsetRangeImages = {}
        self.resampAmpImages = {}
        self.resampIntImages = {}
        self.resampOnlyImages = {}
        self.resampOnlyAmps = {}
        self.topoIntImages = {}
        self.heightTopoImage = None #KK 2014-01-20
        self.rgImageName = 'rgImage'
        self.rgImage = None
        self.simAmpImageName = 'simamp.rdr'
        self.simAmpImages = None #KK 2014-01-20
        self.resampImageName = 'resampImage'
        self.resampOnlyImageName = 'resampOnlyImage.int'
        self.offsetImageName = 'Offset.mht'
        self.demInitFile = 'DemImage.xml'
        self.firstSampleAcrossPrf = 50
        self.firstSampleDownPrf = 50
        self.numberLocationAcrossPrf = 40
        self.numberLocationDownPrf = 50
        self.numberRangeBins = None
        self.firstSampleAcross = 50
        self.firstSampleDown = 50
        self.numberLocationAcross = 40
        self.numberLocationDown = 40
        self.topocorrectFlatImage = None
        self.offsetFields = {}
        self.refinedOffsetFields = {}
        self.offsetField1 = None
        self.refinedOffsetField1 = None
        self.topophaseIterations = 25
        self.coherenceFilename = 'topophase.cor'
        self.unwrappedIntFilename = 'filt_topophase.unw'
        self.phsigFilename = 'phsig.cor'
        self.topophaseMphFilename = 'topophase.mph'
        self.topophaseFlatFilename = 'topophase.flat'
        self.filt_topophaseFlatFilename = 'filt_' + self.topophaseFlatFilename
        self.heightFilename = 'z.rdr' #real height file
        self.heightSchFilename = 'zsch.rdr' #sch height file
        self.latFilename = 'lat.rdr' #KK 2013-12-12: latitude file
        self.lonFilename = 'lon.rdr' #KK 2013-12-12: longitude file
        self.losFilename = 'los.rdr' #KK 2013-12-12: los file
        self.geocodeFilename = 'topophase.geo'
        self.demCropFilename = 'dem.crop'
        # The strength of the Goldstein-Werner filter
        self.filterStrength = 0.7
        # This is hard-coded from the original script
        self.numberValidPulses = 2048
        self.numberPatches = None
        self.patchSize = 8192
        self.machineEndianness = 'l'
        self.secondaryRangeMigrationFlag = None
        self.chirpExtension = 0
        self.slantRangePixelSpacing = None
        self.dopplerCentroid = None
        self.posting = 15
        self.numberFitCoefficients = 6
        self.numberLooks = 4
        self.numberAzimuthLooks = 1
        self.numberRangeLooks = None
        self.numberResampLines = None
        self.shadeFactor = 3
        self.checkPointer =  None
        self.mocompBaselines = {}
        self.topocorrect = None
        self.topo = None #KK 2014-01-20
        self.lookSide = -1 #right looking by default
        self.geocode_list = [
                        self.coherenceFilename,
                        self.unwrappedIntFilename,
                        self.phsigFilename,
                        self.losFilename,
                        self.topophaseFlatFilename,
                        self.filt_topophaseFlatFilename,
                        self.resampOnlyImageName.replace('.int', '.amp')
                       ]

        # Polarimetric calibration
        self.focusers = {}
        self.frOutputName = 'fr'
        self.tecOutputName = 'tec'
        self.phaseOutputName = 'phase'

        super().__init__(family=self.__class__.family, name=name)
        self.procDoc = procDoc
        return None

    def __setstate__(self, state):
        """
        Restore state from the unpickled state values.
        see: http://www.developertutorials.com/tutorials/python/python-persistence-management-050405-1306/
        """
        # When unpickling, we need to update the values from state
        # because all the attributes in __init__ don't exist at this step.
        self.__dict__.update(state)


    def formatname(self, sceneid, pol=None, ext=None):
        """
        Return a string that identifies uniquely a scene from its id and pol.
        ext can be given if we want a filename.
        If sceneid is a tuple: format a string to identy uniquely a pair.
        """
        if isinstance(sceneid, tuple):
            name = '__'.join(sceneid)
        else:
            name = sceneid
        if pol:
            name += '_' + pol
        if ext:
            name += '.' + ext
        return name


    ## This overides the _FrameMixin.frame
    @property
    def frame(self):
        """
        Get the reference frame in self.frames and
        return reference pol in frame.
        This is needed to get information about a frame,
        supposing that all frames have the same information.
        """
        return self.frames[self.refScene][self.refPol]


    def getAllFromPol(self, pol, obj):
        """
        Get all values from obj, where polarization is pol.
        obj should be a dictionary with the following structure:
        { sceneid: { pol1: v1, pol2: v2 }, sceneid2: {...} }
        """
        objlist = []
        if pol not in self.selectedPols:
            return objlist

        if isinstance(obj, str):
            try:
                obj = getattr(self, obj)
            except AttributeError:
                sys.exit("%s is not an attribute of IsceProc." % obj)
        for sceneid in self.selectedScenes:
            try:
                objlist.append(obj[sceneid][pol])
            except:
                sys.exit("%s is not a readable dictionary" % obj)
        return objlist


    def average(self, objdict):
        """
        Average values in a dict of dict: { k1: { k2: ... } }
        """
        N = 0 ##number of values
        s = 0 ##sum
        vals = objdict.values()
        for val in vals:
            ###val is a dictionary
            N += len(val)
            s += sum(val.values())
        return s / float(N)

    def get_is_mocomp(self):
        self.is_mocomp = int( (self.patchSize - self.numberValidPulses) / 2 )

    @property
    def averageHeight(self):
        return self.average(self.pegAverageHeights)

    @property
    def procVelocity(self):
        return self.average(self.pegProcVelocities)

    # <v>, <h>
    def vh(self):
        return self.procVelocity, self.averageHeight

    @property
    def chirpExtensionPercentage(self):
        return NotImplemented
    @chirpExtensionPercentage.setter
    def chirpExtensionPercentage(self, value):
        raise AttributeError("Can only set chirpExtension")

    ## folowing are tbd to split formSLC.
    def _hasher(self, attr, sid, pol=None):
        obj = getattr(self, attr)[sid]
        if pol:
            obj = obj[pol]
        return obj

    def select_frame(self, sid, pol=None): return self._hasher('frames', sid, pol)
    def select_orbit(self, sid, pol=None): return self._hasher('orbits', sid, pol)
    def select_doppler(self, sid, pol=None): return self._hasher('dopplers', sid, pol)
    def select_rawimage(self, sid, pol=None): return self._hasher('rawImages', sid, pol)
    def select_slcimage(self, sid, pol=None): return self._hasher('slcImages', sid, pol)
    def select_squint(self, sid, pol=None): return self._hasher('squints', sid, pol)

    def select_swath(self, sid, pol=None):
        return RadarSwath(frame=self.select_frame(sid, pol),
                          orbit=self.select_orbit(sid, pol),
                          doppler=self.select_doppler(sid, pol),
                          rawimage=self.select_rawimage(sid, pol),
                          slcimage=self.select_slcimage(sid, pol),
                          squint=self.select_squint(sid, pol))



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
