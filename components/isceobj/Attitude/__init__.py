#!/usr/bin/env python3
def createAttitude(name=''):
    from .Attitude import Attitude
    return Attitude(name)
def getFactoriesInfo():
    return  {'Attitude':
                     {
                     'factory':'createAttitude'                     
                     }
              
              }
