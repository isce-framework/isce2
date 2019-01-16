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



from __future__ import print_function
import sys
import math
import logging

dataTypesReal = ['BYTE','CHAR','SHORT','INT','LONG','FLOAT','DOUBLE']
dataTypesCpx = ['CBYTE','CCHAR','CSHORT','CINT','CLONG','CFLOAT','CDOUBLE']

def getCaster(datain,dataout):
    suffix = 'Caster'
    #check for custom types first
    if(datain.upper() == 'CIQBYTE' and dataout.upper() == 'CFLOAT'):
        typein = 'IQByte'
        typeout = dataout[1:].lower().capitalize()
        suffix = 'CpxCaster'
    elif(datain.upper() in dataTypesReal and dataout.upper() in  dataTypesReal):
        typein = datain.lower().capitalize()
        typeout = dataout.lower().capitalize()
    elif(datain.upper() in dataTypesCpx and dataout.upper() in dataTypesCpx):
        typein = datain[1:].lower().capitalize()
        typeout = dataout[1:].lower().capitalize()
        suffix = 'CpxCaster'
    else:
        print('Casting only allowed between compatible types and not',datain,'and',dataout)  
        raise ValueError
    if typein == typeout:
        caster = ''
    else:
        caster = typein + 'To' + typeout + suffix
    return caster
