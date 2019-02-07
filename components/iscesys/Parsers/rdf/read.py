#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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




## \namespace rdf.read Reading Functions
"""(Lazy) Functions to read rdf files and yield unwrapped lines"""

from __future__ import absolute_import

import itertools

from . import utils
from .reserved import glyphs

## unwrap lines from a generator
# \param gline A iteratable that pops file lines (rdf.utils.read_file())
# \param wrap = rdf.reserved.glyphs.WRAP The line coninutation character
# \retval< <a href="https://wiki.python.org/moin/Generators">Generator</a>
# that generates complete RDF input lines.
def _unwrap_lines(gline, wrap=glyphs.WRAP):
    """given a read_stream() generator, yield UNWRAPPED RDF lines"""
    while True:
        try:
            line = next(gline)
            while line.endswith(wrap):
                line = line[:-len(wrap)] + next(gline)
            yield line
        except StopIteration:
            return

## file name --> unwrapped lines
# \param src A file name
# \param wrap = rdf.reserved.glyphs.WRAP The line coninutation character
# \retval< <a href="https://wiki.python.org/moin/Generators">Generator</a>
# that generates complete RDF input lines.
def unwrap_file(src, wrap=glyphs.WRAP):
    """Take a file name (src) and yield unwrapped lines"""
    return filter(
        bool,
        _unwrap_lines(utils.read_file(src), wrap=wrap)
        )
