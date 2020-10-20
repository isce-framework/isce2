#
# Author: Piyush Agram
# Copyright 2016
#

# Path to the _RunWrapper factories
_PATH = "isceobj.Alos2burstProc."

## A factory to make _RunWrapper factories
def _factory(name, other_name=None, path=_PATH):
    """create_run_wrapper = _factory(name)
    name is the module and class function name
    """
    other_name = other_name or name
    module = __import__(
        path+name, fromlist=[""]
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
createBaseline = _factory("runBaseline", path = "isceobj.Alos2Proc.")
createExtractBurst = _factory("runExtractBurst")
createDownloadDem = _factory("runDownloadDem", path = "isceobj.Alos2Proc.")
createCoregGeom = _factory("runCoregGeom")
createCoregCc = _factory("runCoregCc")
createCoregSd = _factory("runCoregSd")
createSwathOffset = _factory("runSwathOffset")
createSwathMosaic = _factory("runSwathMosaic")
createFrameOffset = _factory("runFrameOffset")
createFrameMosaic = _factory("runFrameMosaic")
createRdr2Geo = _factory("runRdr2Geo", path = "isceobj.Alos2Proc.")
createGeo2Rdr = _factory("runGeo2Rdr", path = "isceobj.Alos2Proc.")
createRdrDemOffset = _factory("runRdrDemOffset", path = "isceobj.Alos2Proc.")
createRectRangeOffset = _factory("runRectRangeOffset", path = "isceobj.Alos2Proc.")
createDiffInterferogram = _factory("runDiffInterferogram", path = "isceobj.Alos2Proc.")
createLook = _factory("runLook", path = "isceobj.Alos2Proc.")
createCoherence = _factory("runCoherence", path = "isceobj.Alos2Proc.")
createIonSubband = _factory("runIonSubband")
createIonUwrap = _factory("runIonUwrap", path = "isceobj.Alos2Proc.")
createIonFilt = _factory("runIonFilt", path = "isceobj.Alos2Proc.")
createIonCorrect = _factory("runIonCorrect", path = "isceobj.Alos2Proc.")
createFilt = _factory("runFilt", path = "isceobj.Alos2Proc.")
createUnwrapSnaphu = _factory("runUnwrapSnaphu", path = "isceobj.Alos2Proc.")
createGeocode = _factory("runGeocode", path = "isceobj.Alos2Proc.")

createLookSd = _factory("runLookSd")
createFiltSd = _factory("runFiltSd")
createUnwrapSnaphuSd = _factory("runUnwrapSnaphuSd")
createGeocodeSd = _factory("runGeocodeSd")


# steps imported from: Alos2Proc
# ##############################################################
# there is only problem with (at start of script):
# logger = logging.getLogger('isce.alos2insar.runDownloadDem')
# but it looks like OK.




