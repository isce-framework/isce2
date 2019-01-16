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




# next two functions are used to check two  strings are the same regardless of
# capitalization and/or white spaces and returns a dictionary value based on
# the string provided
def same_content(a,b):
     '''
     it seems an overkill
     al = a.lower().split()
     bl = b.lower().split()
     if len(al) == len(bl):
          for cl, cr in zip(al, bl):
               if cl != cr:
                    return False
          return True
     return False
     '''
     return True if(''.join(a.lower().split()) == ''.join(b.lower().split())) else False


def key_of_same_content(k,d):
     for kd in d:
         if same_content(k, kd):
             return kd, d[kd]
     raise KeyError("key %s not found in dictionary" % k)

def createCpxmag2rg():
    from .Cpxmag2rg import Cpxmag2rg
    return Cpxmag2rg()

def createOffoutliers():
    from .Offoutliers import Offoutliers
    return Offoutliers()

def createEstimateOffsets(name=''):
    from .EstimateOffsets import EstimateOffsets
    return EstimateOffsets(name=name)

def createDenseOffsets(name=''):
    from .DenseOffsets import DenseOffsets
    return DenseOffsets(name=name)

def createSimamplitude():
    from .Simamplitude import Simamplitude
    return Simamplitude()
