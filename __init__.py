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

from .release_history import release_version, release_svn_revision, release_date
svn_revision = release_svn_revision
version = release_history # compatibility alias

__version__ = release_version

import sys, os
isce_path = os.path.dirname(os.path.abspath(__file__))

import logging
from logging.config import fileConfig as _fc
_fc(os.path.join(isce_path, 'defaults', 'logging', 'logging.conf'))

sys.path.insert(1,isce_path)
sys.path.insert(1,os.path.join(isce_path,'applications'))
sys.path.insert(1,os.path.join(isce_path,'components'))
sys.path.insert(1,os.path.join(isce_path,'library'))

try:
    os.environ['ISCE_HOME']
except KeyError:
    print('Using default ISCE Path: %s'%(isce_path))
    os.environ['ISCE_HOME'] = isce_path

try:
    from . license import stanford_license
except:
    print("This is the Open Source version of ISCE.")
    print("Some of the workflows depend on a separate licensed package.")
    print("To obtain the licensed package, please make a request for ISCE")
    print("through the website: https://download.jpl.nasa.gov/ops/request/index.cfm.")
    print("Alternatively, if you are a member, or can become a member of WinSAR")
    print("you may be able to obtain access to a version of the licensed sofware at")
    print("https://winsar.unavco.org/software/isce")
