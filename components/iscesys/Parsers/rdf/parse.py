#!/usr/bin/env python3

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




"""Usage:

[python] ./parse.py src [dst]
"""

## \namespace rdf.parse RDF Parsing script
import sys
from rdf import rdfparse

# RUN AS AS SCRIPT
if __name__ == "__main__":

    # IF usage error, prepare error message and pipe->stderr, 
     #                set exit=INVALID INPUT
    if len(sys.argv) == 1: # guard
        import errno
        pipe = sys.stderr
        message = getattr(sys.modules[__name__], '__doc__')
        EXIT = errno.EINVAL
    # ELSE: Usage OK- the message is the result, and the pipe us stdout
    #                 set exit=0.
    else:
        argv = sys.argv[1:] if sys.argv[0].startswith('python') else sys.argv[:]
        src = argv[-1]
        pipe = sys.stdout
        message = str(rdfparse(src))
        EXIT = 0
        
    # Send message to pipe.
    print >> pipe, message
    # exit script
    sys.exit(EXIT)
# ELSE: I You cannot import this module b/c I say so.
else:
    raise ImportError("This is a script, and only a script")
