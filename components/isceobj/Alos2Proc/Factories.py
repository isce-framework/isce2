#
# Author: Piyush Agram
# Copyright 2016
#

# Path to the _RunWrapper factories
_PATH = "isceobj.Alos2Proc."

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
    elif unwrapperName.lower() == 'downsample_snaphu':
        from .run_downsample_unwrapper import runUnwrap
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


createPreprocessor = _factory("runPreprocessor")
createDownloadDem = _factory("runDownloadDem")
createPrepareSlc = _factory("runPrepareSlc")
createSlcOffset = _factory("runSlcOffset")
createFormInterferogram = _factory("runFormInterferogram")
createSwathOffset = _factory("runSwathOffset")
createSwathMosaic = _factory("runSwathMosaic")
createFrameOffset = _factory("runFrameOffset")
createFrameMosaic = _factory("runFrameMosaic")
createRdr2Geo = _factory("runRdr2Geo")
createGeo2Rdr = _factory("runGeo2Rdr")
createRdrDemOffset = _factory("runRdrDemOffset")
createRectRangeOffset = _factory("runRectRangeOffset")
createDiffInterferogram = _factory("runDiffInterferogram")
createLook = _factory("runLook")
createCoherence = _factory("runCoherence")
createIonSubband = _factory("runIonSubband")
createIonUwrap = _factory("runIonUwrap")
createIonFilt = _factory("runIonFilt")
createFilt = _factory("runFilt")
createUnwrapSnaphu = _factory("runUnwrapSnaphu")
createGeocode = _factory("runGeocode")

createSlcMosaic = _factory("runSlcMosaic")
createSlcMatch = _factory("runSlcMatch")
createDenseOffset = _factory("runDenseOffset")
createFiltOffset = _factory("runFiltOffset")
createGeocodeOffset = _factory("runGeocodeOffset")


