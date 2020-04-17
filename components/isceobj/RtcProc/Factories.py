#
# Author: Piyush Agram
# Copyright 2016
#

# Path to the _RunWrapper factories
_PATH = "isceobj.RtcProc."

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

createPreprocessor = _factory("runPreprocessor")
#createComputeBaseline = _factory("runComputeBaseline")
createVerifyDEM = _factory("runVerifyDEM")
createLooks = _factory("runLooks")
createTopo = _factory("runTopo")
createNormalize = _factory("runNormalize")
#createGeocode = _factory("runGeocode")

