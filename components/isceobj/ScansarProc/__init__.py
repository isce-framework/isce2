#
# Author: Piyush Agram
# Copyright 2016
#

from .ScansarProc import *
from .Factories import *

def getFactoriesInfo():
    return  {'ScansarProc':
                     {'args':
                           {
                            'procDoc':{'value':None,'type':'Catalog','optional':True}
                            },
                     'factory':'createScansarProc'                     
                     }
              
              }

def createScansarProc(name=None, procDoc= None):
    from .ScansarProc import ScansarProc
    return ScansarProc(name = name,procDoc = procDoc)
