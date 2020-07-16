#!/usr/bin/env python3

def createDopIQ(name=''):
    from .DopIQ import DopIQ
    return DopIQ(name=name)

def getFactoriesInfo():
    return  {'DopIQ':
                     {
                     'factory':'createDopIQ'                     
                     }
              
              }
