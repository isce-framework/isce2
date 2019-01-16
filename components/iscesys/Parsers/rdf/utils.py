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




"""Non RDF specific python helpers"""
## \namespace rdf.utils Non-RDF specific utilities

## Generate non-zero entries from an ASCII file
## \param src Is the source file name
## \param purge = True reject blanks (unless False)
## \retval< <a href="https://wiki.python.org/moin/Generators">Generator</a> 
## that generates (nonzero) lines for an ASCII file
def read_file(src):
    """src --> src file name
    purge=True igonors black lines"""
    with open(src, 'r') as fsrc:
        for line in read_stream(fsrc):
            yield line

## Yield stripped lines from a file
## \param fsrc A readable file-like object
## \retval< <a href="https://wiki.python.org/moin/Generators">Generator</a> 
## that generates fsrc.readline() (stripped).
def read_stream(fsrc):
    """Generate lines from a stream (fsrc)"""
    tell = fsrc.tell()
    line = fsrc.readline().strip()
    while tell != fsrc.tell() or line:
        yield line
        tell = fsrc.tell()
        line = fsrc.readline().strip()

