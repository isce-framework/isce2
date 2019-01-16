## The appications:
__all__ = ['CalculatePegPoint',
           'calculateBaseline',
           'createGeneric',
           'dpmApp',
           'extractHDROrbit',
           'focus',
           'formSLC',
           'insarApp',
           'isce.log',
           'make_input',
           'make_raw',
           'mdx',
           'readdb',
           'viewMetadata',
           'xmlGenerator']
def createInsar():
    from .insarApp import Insar
    return Insar()
def createStitcher():
    from .stitcher import Stitcher
    return Stitcher()
def createWbdStitcher():
    from .wbdStitcher import Stitcher
    return Stitcher()
def createDataTileManager():
    from .dataTileManager import DataTileManager
    return DataTileManager()
def getFactoriesInfo():
    return  {'Insar':
                     {
                     'factory':'createInsar'                     
                     },
              'DemsStitcher':
                     {
                     'factory':'createStitcher'                     
                     },
              'WbdsStitcher':
                     {
                     'factory':'createWbdStitcher'                     
                     },
             'DataTileManager':
                     {
                     'factory':'createDataTileManager'                     
                     }
              }
