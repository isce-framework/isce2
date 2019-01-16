#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2011 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




from collections import UserDict
class OrderedDict(UserDict):
    def __init__(self, adict = None):
        self._keys = []
        UserDict.__init__(self, adict)

    def __delitem__(self, key):
        UserDict.__delitem__(self, key)
        self._keys.remove(key)

    def __setitem__(self, key, item):
        UserDict.__setitem__(self, key, item)
        if key not in self._keys: self._keys.append(key)

    def clear(self):
        UserDict.clear(self)
        self._keys = []

    def copy(self):
        adict = UserDict.copy(self)
        adict._keys = self._keys[:]
        return adict

    def items(self):
        return zip(self._keys, self.values())

    def keys(self):
        return self._keys

    def popitem(self):
        try:
            key = self._keys[-1]
        except IndexError:
            raise KeyError('dictionary is empty')

        val = self[key]
        del self[key]

        return (key, val)

    def setdefault(self, key, failobj = None):
        UserDict.setdefault(self, key, failobj)
        if key not in self._keys: self._keys.append(key)

    def update(self, adict):
        UserDict.update(self, adict)
        for key in adict.keys():
            if key not in self._keys: self._keys.append(key)

    def values(self):
        return map(self.get, self._keys)




if __name__ == '__main__':
#    d = {'file':{'filename':'test.slc','dataType':'BANDED','interleavingScheme':'BIP','NUM_BANDS':2,'BAND_TYPES':{'BAND1':'REAL4','BAND2':'REAL4'},'width':1024,'length':2048}}
    d = OrderedDdict()
    d['file'] = OrderedDict()
    d['file']['filename']='test.slc'
    d['file']['dataType'] = 'BANDED'
    d['file']['interleavingScheme'] = 'BIP'
    d['file']['NUM_BANDS'] = 2
    d['file']['BAND_TYPES'] = OrderedDict()
    d['file']['BAND_TYPES']['BAND1'] = 'REAL4'
    d['file']['BAND_TYPES']['BAND2'] = 'REAL4'
    d['file']['width'] = 1024
    d['file']['length'] = 2048

    from isceobj.XmlUtil.xmlUtils import dict_to_xml
    dict_to_xml(d,'test123.xml')
