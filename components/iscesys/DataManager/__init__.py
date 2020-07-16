#!/usr/bin/env python3

def createManager(source,name=''):
    if source == 'dem1':
        from .Dem1Manager import Dem1Manager
        ret = Dem1Manager(name=name)
    elif source == 'dem3':
        from .Dem3Manager import Dem3Manager
        ret = Dem3Manager(name=name)
    elif source == 'wbd':
        from .SWBDManager import SWBDManager
        ret = SWBDManager(name=name)
    else:
        raise Exception("Unrecognized source %s",source)
            
    return ret
def getFactoriesInfo():
    """
    Returns a dictionary with information on how to create an object Sensor from its factory
    """
    return  {'DataManager':
                     {'args':
                           {
                            '0':{'value':['dem1','dem2','wbd'],
                                      'type':'str','optional':False,'default':None}
                            },
                     'factory':'createManager'
                     }
              }
