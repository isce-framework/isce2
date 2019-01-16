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

Interactive:

>>>import rdf
>>>rdf_mapping = rdf.rdfparse("<src>")

Shell Script:

%python rdf/parse.py <src> > <dst>
"""
__author__ = "Eric Belz"
__copyright__ = "Copyright 2013,  by the California Institute of Technology."
__credits__ = ["Eric Belz", "Scott Shaffer"]
__license__ = NotImplemented
__version__ = "1.0.1"
__maintainer__ = "Eric Belz"
__email__ = "eric.belz@jpl.nasa.gov"
__status__ = "Production"

## \namespace rdf The rdf package
from .uRDF import rdf_reader, RDF


## Backwards compatible rdf readers.
rdfparse = rdf_reader
## less redundant parser
parse = rdf_reader


def test():
    """test() function - run from rdf/test"""
    import os
    rdf_ = rdfparse('rdf.txt')
    with open('new.rdf', 'w') as fdst:
        fdst.write(str(rdf_))
    if os.system("xdiff old.rdf new.rdf"):
        os.system("diff old.rdf new.rdf")
