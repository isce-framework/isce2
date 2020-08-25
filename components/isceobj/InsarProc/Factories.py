#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Brett George
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Path to the _RunWrapper factories
_PATH = "isceobj.InsarProc."

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

## Put in "_" to prevernt import on "from Factorties import *"
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

    pass



# runEstimateHeights is a facility
def createEstimateHeights(other, sensor):
    if "uavsar" in sensor.lower():
        from .runEstimateHeights_peg import runEstimateHeights
    else:
        from .runEstimateHeights import runEstimateHeights
    return _RunWrapper(other, runEstimateHeights)

# we turned runFormSLC into a facility
def createFormSLC(other, sensor):
    if sensor.lower() in ["terrasarx","cosmo_skymed_slc","radarsat2",'tandemx', 'kompsat5','risat1_slc','sentinel1', 'alos2','ers_slc','alos_slc','envisat_slc', 'ers_envisat_slc', 'saocom_slc']:
        from .runFormSLCTSX import runFormSLC
    elif sensor.lower() in ["uavsar_rpi"]:
        from .runFormSLCisce import runFormSLC
    else:
        from .runFormSLC import runFormSLC
    return _RunWrapper(other, runFormSLC)

def createSetmocomppath(other, sensor):
    if sensor.lower() in ["uavsar_rpi"]:
        from .runSetmocomppathFromFrames import runSetmocomppath
    else:
        from .runSetmocomppath import runSetmocomppath
    return _RunWrapper(other, runSetmocomppath)


def createUnwrapper(other, do_unwrap = None, unwrapperName = None,
                    unwrap = None):
    if not do_unwrap and not unwrap:
        #if not defined create an empty method that does nothing
        def runUnwrap(self):
            return None
    elif unwrapperName.lower() == 'snaphu':
        from .runUnwrapSnaphu import runUnwrap
    elif unwrapperName.lower() == 'snaphu_mcf':
        from .runUnwrapSnaphu import runUnwrapMcf as runUnwrap
    elif unwrapperName.lower() == 'icu':
        from .runUnwrapIcu import runUnwrap
    elif unwrapperName.lower() == 'grass':
        from .runUnwrapGrass import runUnwrap
    return _RunWrapper(other, runUnwrap)

def createUnwrap2Stage(other, do_unwrap_2stage = None, unwrapperName = None):
    if (not do_unwrap_2stage) or (unwrapperName.lower() == 'icu') or (unwrapperName.lower() == 'grass'):
        #if not defined create an empty method that does nothing
        def runUnwrap2Stage(*arg, **kwargs):
            return None
    else:
      try:
        import pulp
        from .runUnwrap2Stage import runUnwrap2Stage
      except ImportError:
        raise Exception('Please install PuLP Linear Programming API to run 2stage unwrap')
    return _RunWrapper(other, runUnwrap2Stage)

def createOffsetprf(other, coregisterMethod, do_offsetprf=True):
    if not do_offsetprf:
        from .runOffsetprf_none import runOffsetprf
    elif coregisterMethod.lower() == "ampcor":
        from .runOffsetprf_ampcor import runOffsetprf
    elif coregisterMethod.lower() == "nstage":
        from .runOffsetprf_nstage import runOffsetprf
    else:
        from .runOffsetprf import runOffsetprf
    return _RunWrapper(other, runOffsetprf)

def createRgoffset(other, coregisterMethod, do_rgoffset=True):
    if not do_rgoffset:
        from .runRgoffset_none import runRgoffset
    elif coregisterMethod.lower() == "ampcor":
        from .runRgoffset_ampcor import runRgoffset
    elif coregisterMethod.lower() == "nstage":
        from .runRgoffset_nstage import runRgoffset
    else:
        from .runRgoffset import runRgoffset
    return _RunWrapper(other, runRgoffset)

createMaskImages = _factory("runMaskImages")
createCreateWbdMask = _factory("runCreateWbdMask")
createCreateDem = _factory("createDem")
createExtractInfo = _factory("extractInfo")
createPreprocessor = _factory("runPreprocessor")
createPulseTiming = _factory("runPulseTiming")
createSetmocomppath = _factory("runSetmocomppath")
createOrbit2sch = _factory("runOrbit2sch")
createUpdatePreprocInfo = _factory("runUpdatePreprocInfo")
createOffoutliers = _factory("runOffoutliers")
createPrepareResamps = _factory("runPrepareResamps")
createResamp = _factory("runResamp")
createResamp_image = _factory("runResamp_image")
createMocompbaseline = _factory("runMocompbaseline")
createTopo = _factory("runTopo")
createCorrect = _factory("runCorrect")
createShadecpx2rg = _factory("runShadecpx2rg")
createResamp_only = _factory("runResamp_only")
createCoherence = _factory("runCoherence")
createFilter = _factory("runFilter")
createGrass = _factory("runGrass")
createGeocode = _factory("runGeocode")
