#!/usr/bin/env python3

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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Giangi Sacco
# Copyright 2012, 2015 by the California Institute of Technology.
# ALL RIGHTS RESERVED.
# United States Government Sponsorship acknowledged. Any commercial use must be
# negotiated with the Office of Technology Transfer at the
# California Institute of Technology.
#
import gzip
import os
def is_gzipfile(filename):
    fp = gzip.GzipFile(filename)
    #since it fails for non gz file just try and catch
    try:
        s = fp.read()
        ret = True
    except OSError:
        ret = False
    return ret
class GZipFile:
    def __init__(self,filename):
        self._filename = filename
    
    def extractall(self,path):
        try:
            os.mkdir(path)
        except Exception:
            pass
        fp = gzip.GzipFile(self._filename)
        s = fp.read()
        fp.close()
        #remove last extension
        fp = open(os.path.join(path,'.'.join(os.path.basename(self._filename).split('.')[:-1])),'wb')
        fp.write(s)
        fp.close()
        
