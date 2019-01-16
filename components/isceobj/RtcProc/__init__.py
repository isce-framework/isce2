#
# Author: Piyush Agram
# Copyright 2016
#

from .RtcProc import *
from .Factories import *

def getFactoriesInfo():
    return  {'RtcProc':
                     {'args':
                           {
                            'procDoc':{'value':None,'type':'Catalog','optional':True}
                            },
                     'factory':'createRtcProc'                     
                     }
              
              }

def createRtcProc(name=None, procDoc= None):
    from .RtcProc import RtcProc
    return RtcProc(name = name,procDoc = procDoc)
