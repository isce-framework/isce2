#
# Author: Piyush Agram
# Copyright 2016
#

# Path to the _RunWrapper factories
_PATH = "isceobj.TopsProc."

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
createComputeBaseline = _factory("runComputeBaseline")
createVerifyDEM = _factory("runVerifyDEM")
createVerifyGeocodeDEM = _factory("runVerifyGeocodeDEM")
createTopo = _factory("runTopo")
createSubsetOverlaps = _factory("runSubsetOverlaps")
createCoarseOffsets = _factory("runCoarseOffsets")
createCoarseResamp = _factory("runCoarseResamp")
createOverlapIfg = _factory("runOverlapIfg")
createPrepESD = _factory("runPrepESD")
createESD = _factory("runESD")
createRangeCoreg = _factory("runRangeCoreg")
createFineOffsets = _factory("runFineOffsets")
createFineResamp = _factory("runFineResamp")
createIon = _factory("runIon")
createBurstIfg = _factory("runBurstIfg")
createMergeBursts = _factory("runMergeBursts")
createFilter = _factory("runFilter")
createGeocode = _factory("runGeocode")

#createMaskImages = _factory("runMaskImages")
#createCreateWbdMask = _factory("runCreateWbdMask")

###topsOffsetApp factories
createMergeSLCs = _factory("runMergeSLCs")
createDenseOffsets = _factory("runDenseOffsets")
createOffsetFilter = _factory("runOffsetFilter")
createOffsetGeocode = _factory("runOffsetGeocode")
createCropOffsetGeo = _factory("runCropOffsetGeo")
