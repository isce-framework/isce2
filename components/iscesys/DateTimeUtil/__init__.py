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
# Author: Eric Belz
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""Date and Time utilites, on top of the datetime standard library.

New Usage:

>>>from iscesys import DateTimeUtil as DTU

replaces former usage:

>>>from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU

Note, both:

javaStyleUtils()   and   pythonic_utils()

are available.
"""
from .DateTimeUtil import timedelta_to_seconds, seconds_since_midnight, date_time_to_decimal_year

## JavaStyleNames for the pythonic_names
timeDeltaToSeconds = timedelta_to_seconds
secondsSinceMidnight =  seconds_since_midnight
dateTimeToDecimalYear = date_time_to_decimal_year
