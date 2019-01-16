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



# Comment: Adapted from InsarProc/Factories.py
import sys

# Path to the _RunWrapper factories
_PATH = "isceobj.IsceProc."

__todo__ = "use 2.7's importlib"

## A factory to make _RunWrapper factories
def _factory(name, other_name=None):
    """create_run_wrapper = _factory(name)
    name is the module and class function name
    """
    other_name = other_name or name
    module = __import__(
        _PATH+name, fromlist=[""]
        )

    cls = getattr(module, other_name)
    def creater(other, *args, **kwargs):
        """_RunWrapper for object calling %s"""
        return _RunWrapper(other, cls)
    return creater

## Put in "_" to prevent import on "from Factories import *"
class _RunWrapper(object):
    """_RunWrapper(other, func)(*args, **kwargs)

    executes:

    func(other, *args, **kwargs)

    (like a method)
    """
    def __init__(self, other, func):
        self.method = func
        self.other = other
        return None

    def __call__(self, *args, **kwargs):
        return self.method(self.other, *args, **kwargs)

# runEstimateHeights is a facility
def createEstimateHeights(other, do_estimateheights, sensor):
    if not do_estimateheights:
        return None
    elif "uavsar" in sensor.lower():
        print("uavsar sensor.  creating runEstimateHeights_peg")
        from .runEstimateHeights_peg import runEstimateHeights
    else:
        print("non uavsar sensor.  creating runEstimateHeights")
        from .runEstimateHeights import runEstimateHeights
    return _RunWrapper(other, runEstimateHeights)

# we turned runFormSLC into a facility
def createFormSLC(other, do_formslc, sensor):
    if not do_formslc:
        return None
    if sensor.lower() in ["terrasarx","cosmo_skymed_slc","radarsat2",'tandemx', 'kompsat5','risat1','sentinel1a']:
        from .runFormSLCTSX import runFormSLC
    elif "uavsar" in sensor.lower():
        from .runFormSLCisce import runFormSLC
    else:
        from .runFormSLC import runFormSLC
    return _RunWrapper(other, runFormSLC)

def createUpdatePreprocInfo(other, do_updatepreprocinfo, sensor):
    if not do_updatepreprocinfo:
        return None
    if "uavsar" in sensor.lower():
        from .runUpdatePreprocInfo_isce import runUpdatePreprocInfo
    else:
        from .runUpdatePreprocInfo import runUpdatePreprocInfo
    return _RunWrapper(other, runUpdatePreprocInfo)

def createSetmocomppath(other, do_mocomppath, sensor):
    if not do_mocomppath:
        return None
    if "uavsar" in sensor.lower():
        from .runSetmocomppathFromFrame import runSetmocomppath
    else:
        from .runSetmocomppath import runSetmocomppath
    return _RunWrapper(other, runSetmocomppath)


def createOffsetprf(other, do_offsetprf, coregisterMethod):
    if not do_offsetprf:
        return None
    if coregisterMethod.lower() == "ampcor":
        from .runOffsetprf_ampcor import runOffsetprf
    elif coregisterMethod.lower() == "nstage": #KK 2014-01-29
        from .runOffsetprf_nstage import runOffsetprf
    else:
        from .runOffsetprf import runOffsetprf
    return _RunWrapper(other, runOffsetprf)

# KK 2014-01-29
def createRgoffset(other, do_rgoffset, coregisterMethod):
    if not do_rgoffset:
        from .runRgoffset_none import runRgoffset
    elif coregisterMethod.lower() == "ampcor":
        from .runRgoffset_ampcor import runRgoffset
    elif coregisterMethod.lower() == "nstage":
        from .runRgoffset_nstage import runRgoffset
    else:
        from .runRgoffset import runRgoffset
    return _RunWrapper(other, runRgoffset)


# KK 2014-01-29
def createUnwrapper(other, do_unwrap, unwrapperName):
    if not do_unwrap:
        return None
    if unwrapperName.lower() == "snaphu":
        from .runUnwrapSnaphu import runUnwrap
    elif unwrapperName.lower() == "snaphu_mcf":
        from .runUnwrapSnaphu import runUnwrapMcf as runUnwrap
    elif unwrapperName.lower() == "icu":
        from .runUnwrapIcu import runUnwrap
    elif unwrapperName.lower() == "grass":
        from .runUnwrapGrass import runUnwrap
    else:
        sys.exit("%s method is unknown in createUnwrapper." % unwrapperName)
    return _RunWrapper(other, runUnwrap)
# KK

createCreateDem = _factory("createDem")
createExtractInfo = _factory("extractInfo")
createPreprocessor = _factory("runPreprocessor")
createPulseTiming = _factory("runPulseTiming")
createOrbit2sch = _factory("runOrbit2sch")
createUpdatePreprocInfo = _factory("runUpdatePreprocInfo")
createOffoutliers = _factory("runOffoutliers")
createPrepareResamps = _factory("runPrepareResamps")
createResamp = _factory("runResamp")
createResamp_image = _factory("runResamp_image")
createISSI = _factory("runISSI")
createCrossmul = _factory("runCrossmul") #KK 2013-11-26
createMocompbaseline = _factory("runMocompbaseline")
createTopo = _factory("runTopo")
createCorrect = _factory("runCorrect")
createShadecpx2rg = _factory("runShadecpx2rg")
createResamp_only = _factory("runResamp_only")
createCoherence = _factory("runCoherence")
createFilter = _factory("runFilter")
createGrass = _factory("runGrass")
createGeocode = _factory("runGeocode")
