#!/usr/bin/env python3

def createStitcher(name=''):    
    from .Stitcher import Stitcher
    return Stitcher(name=name)
   
def getFactoriesInfo():
    """
    Returns a dictionary with information on how to create an object Sensor from its factory
    """
    return  {'Stitcher':
                     {
                     'factory':'createStitcher'
                     }
              }
