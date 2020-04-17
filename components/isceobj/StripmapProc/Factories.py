#
# Author: Heresh Fattahi
# Copyright 2017
#
# Modified from what was originally written by Brett George
# Copyright 2010
#
#

# Path to the _RunWrapper factories
_PATH = "isceobj.StripmapProc."

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

## Put in "_" to prevent import on "from Factorties import *"
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


def isRawSensor(sensor):
    '''
    Check if input data is raw / slc.
    '''
    if str(sensor).lower() in ["terrasarx","cosmo_skymed_slc","radarsat2",'tandemx', 'kompsat5','risat1_slc','sentinel1', 'alos2','ers_slc','alos_slc','envisat_slc', 'uavsar_rpi','ers_envisat_slc','sicd_rgzero', 'iceye_slc', 'uavsar_hdf5_slc']:
        return False
    else:
        return True


def isZeroDopplerSLC(sensor):
    '''
    Check if SLC is zero doppler / native doppler.
    '''

    if str(sensor).lower() in ["terrasarx","cosmo_skymed_slc","radarsat2",'tandemx', 'kompsat5','risat1_slc','sentinel1', 'alos2','ers_slc','envisat_slc','ers_envisat_slc','sicd_rgzero', 'iceye_slc', 'uavsar_hdf5_slc']:
        return True
    elif sensor.lower() in ['alos_slc', 'uavsar_rpi']:
        return False
    else:
        raise Exception('Unknown sensor type {0} encountered in isZeroDopplerSLC'.format(sensor))


def getDopplerMethod(sensor):
    '''
    Return appropriate doppler method based on user input.
    '''

    if str(sensor).lower() in ["terrasarx","cosmo_skymed_slc","radarsat2",'tandemx', 'kompsat5','risat1_slc','sentinel1', 'alos2','ers_slc','alos_slc','envisat_slc', 'uavsar_rpi','cosmo_skymed','ers_envisat_slc','sicd_rgzero', 'iceye_slc', 'uavsar_hdf5_slc']:
        res =  'useDEFAULT'
    else:
        res =  'useDOPIQ'

    print("DOPPLER: ", sensor, res)
    return res

def createUnwrapper(other, do_unwrap = None, unwrapperName = None,
                    unwrap = None):
    print("do_unwrap ",do_unwrap)
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
        print("running unwrapping grass")
        from .runUnwrapGrass import runUnwrap
    return _RunWrapper(other, runUnwrap)

createFormSLC = _factory("runROI", "runFormSLC")
createCrop = _factory("runCrop")
createPreprocessor = _factory("runPreprocessor")
createTopo = _factory("runTopo")
createGeo2rdr = _factory("runGeo2rdr")
createSplitSpectrum = _factory("runSplitSpectrum")
createResampleSlc = _factory("runResampleSlc")
createResampleSubbandSlc = _factory("runResampleSubbandSlc")
createRefineSlaveTiming = _factory("runRefineSlaveTiming")
createDenseOffsets = _factory("runDenseOffsets")
createRubbersheetAzimuth = _factory("runRubbersheetAzimuth") # Modified by V. Brancato (10.07.2019)
createRubbersheetRange = _factory("runRubbersheetRange")     # Modified by V. Brancato (10.07.2019)
createInterferogram = _factory("runInterferogram")
createCoherence = _factory("runCoherence")
createFilter = _factory("runFilter")
createDispersive = _factory("runDispersive")
createVerifyDEM = _factory("runVerifyDEM")
createGeocode = _factory("runGeocode")
