#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from .formslc import *
from .resamp import *
from .resamp_image import *
from .resamp_amps import *
from .resamp_only import *
from .resamp_slc import *
from .topo import *
from .correct import createCorrect, contextCorrect
from .mocompTSX import *
from .estamb import *

#ing added sensor argument to turn it into a real factory, allowing other type
# of formSLC and moved instantiation here
def createFormSLC(sensor=None, name=''):
    if sensor is None or 'uavsar' in sensor.lower():
        from .formslc.Formslc import Formslc as cls
        return cls(name=name)
    elif str(sensor).lower() in ['terrasarx','cosmo_skymed_slc','radarsat2','sentinel1a','tandemx','kompsat5','risat1_slc','alos2','ers_slc','alos_slc','envisat_slc', 'ers_envisat_slc','saocom_slc']:
        from .mocompTSX.MocompTSX import MocompTSX as cls
    else:
        raise ValueError("Unrecognized Sensor: %s" % str(sensor))
    return cls()

def getFactoriesInfo():
    """
    Returns a dictionary with information on how to create an object Sensor from its factory
    """
    return  {'FormSLC':
                     {'args':
                           {
                            'sensor':{'value':['None','uavsar','terrasarx','cosmo_skymed_slc','radarsat2','sentinel1a','tandemx',
                                               'kompsat5','risat1_slc','alos2','ers_slc','alos_slc','envisat_slc','saocom_slc'],
                                      'type':'str','optional':True,'default':None}
                            },
                     'factory':'createFormSLC'
                     }
              }


