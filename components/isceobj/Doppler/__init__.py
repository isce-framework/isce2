#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Authors: Walter Szeliga, Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function

__all__ = ('createDoppler',)

def useDefault(name=None):
    if name:
        instance = None
    else:
        import isceobj.Doppler.DefaultDopp
        instance = DefaultDopp.DefaultDopp()
        return instance

def useDOPIQ(name=None):
    if name:
        instance = None
    else:
        import mroipac.dopiq.DopIQ
        instance = mroipac.dopiq.DopIQ.DopIQ()
        return instance

def useCalcDop(name=None):
    if name:
        instance = None
    else:
        import isceobj.Doppler.Calc_dop
        instance = isceobj.Doppler.Calc_dop.Calc_dop()
    return instance


def useDoppler(name=None):
    if name:
        instance = None
    else:
        import mroipac.doppler.Doppler
        instance = mroipac.doppler.Doppler.Doppler()
    return instance
    

doppler_facilities = {'USEDOPIQ' : useDOPIQ,
         'USECALCDOP' : useCalcDop,
         'USEDOPPLER' : useDoppler,
         'USEDEFAULT': useDefault}

def getFactoriesInfo():
    """
    Returns a dictionary with information on how to create an object Doppler from its factory
    """
    return  {'Doppler':
                     {'args':
                           {
                            'doppler':{'value':list(doppler_facilities.keys()),'type':'str'}
                            },
                     'factory':'createDoppler'
                     }
              }
    
def createDoppler(doppler=None, name=None):
    if doppler.upper() in doppler_facilities.keys():
        instance = doppler_facilities[doppler.upper()](name)
    else:
        instance = None
        print(
            "Doppler calculation method not recognized. Valid methods: ",
            doppler_facilities.keys())
    return instance

