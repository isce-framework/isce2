#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS RESERVED.
# United States Government Sponsorship acknowledged. Any commercial use must be
# negotiated with the Office of Technology Transfer at the California Institute of
# Technology.  This software is subject to U.S. export control laws and regulations
# and has been classified as EAR99.  By accepting this software, the user agrees to
# comply with all applicable U.S. export laws and regulations.  User has the
# responsibility to obtain export licenses, or other export authority as may be
# required before exporting such information to foreign countries or providing
# access to foreign persons.
#
# Author: Yang Lei
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





def createAutorift(name=''):
    from contrib.geo_autoRIFT.Autorift import Autorift
    return Autorift(name=name)

def createGeogrid(name=''):
    from contrib.geo_autoRIFT.Geogrid import Geogrid
    return Geogrid(name=name)


def getFactoriesInfo():
    return  {'Autorift':
                     {
                     'factory':'createAutorift'
                     },
             'Geogrid':
                     {
                     'factory':'createGeogrid'
                     }
              }
