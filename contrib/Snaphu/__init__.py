#!/usr/bin/env python3

def createSnaphu(name=''):
    from .Snaphu import Snaphu
    instance = Snaphu(name=name)
    return instance

def getFactoriesInfo():
    return  {'Snaphu':
                     {
                     'factory':'createSnaphu'                     
                     }
              }
