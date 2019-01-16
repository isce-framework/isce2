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
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




from __future__ import print_function
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()


class StringUtils(object):

    @staticmethod
    def lower_no_spaces(s):
        return (''.join(s.split())).lower()

    @staticmethod
    def lower_single_spaced(s):
        return (' '.join(s.split())).lower()

    @staticmethod
    def capitalize_single_spaced(s):
        return ' '.join(list(map(str.capitalize, s.lower().split())))

    @staticmethod
    def listify(a):
        """
        Convert a string version of a list, tuple, or comma-/space-separated
        string into a Python list of strings.
        """
        if not isinstance(a, str):
            return a

        if '[' in a:
            a = a.split('[')[1].split(']')[0]
        elif '(' in a:
            a = a.split('(')[1].split(')')[0]

        #At this point a is a string of one item or several items separated by
        #commas or spaces. This is converted to a list of one or more items
        #with any leading or trailing spaces stripped off.
        if ',' in a:
            return list(map(str.strip, a.split(',')))
        else:
            return list(map(str.strip, a.split()))
