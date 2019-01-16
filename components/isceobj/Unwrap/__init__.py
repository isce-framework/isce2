#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
import os
from . import snaphu
from . import grass
from . import icu
from . import snaphu_mcf

Unwrappers = {'SNAPHU' : snaphu.snaphu,
              'GRASS'  : grass.grass,
              'ICU'    : icu.icu,
              'SNAPHU_MCF' : snaphu_mcf.snaphu_mcf}


def createUnwrapper(unwrap, unwrapper_name, name=None):
    '''Implements the logic between unwrap and unwrapper_name to choose the unwrapping method.'''
    unwMethod = None

#    print('Unwrap = ', unwrap)
#    print('Unwrapper Name = ', unwrapper_name)

    #If no unwrapping name is provided.
    if (unwrapper_name is None) or (unwrapper_name is ''):
    #But unwrapped results are desired, set to default: grass
        if unwrap is True:
            unwMethod = 'grass'

    #Unwrap should be set to true.
    elif unwrap is True:
        unwMethod = unwrapper_name

#    print('Algorithm: ', unwMethod)

    if unwMethod is not None:
        try:
            cls = Unwrappers[str(unwMethod).upper()]
            print(cls.__module__)
        except AttributeError:
            raise TypeError("'unwrapper type'=%s cannot be interpreted"%
                            str(unwMethod))
            pass

    else:
        cls = None

    return cls
