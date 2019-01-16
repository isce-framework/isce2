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




"""uRDF is the user's interface to rdf.

rdf_include  is the key function- it reads rdf files recursivly.

rdf_reader unpacks the result into the RDF constructor.
"""
## \namespace rdf.uRDF __u__ sers' inteface to language.py and data.py

from __future__ import absolute_import

from . import read
from .language.grammar import syntax
from .data.files import RDF


## The rdf_include function takes a src and rdf.language.syntax.Grammar
## object to go process the entirety of src-- it is the sole controller
## of Grammar.depth, unpacking of _RDFRecord and lists of them- it deals
## with the recursion, etc
## \param src Is the source file name
## \par Side Effect: None to external users (Grammar evolution internally)
## \retval< <a href="https://wiki.python.org/moin/Generators">Generator</a>
## that generates rdf.data.entries.RDFRecord
def rdf_include(src, **_kwargs):
    """rdf_include(src):

    src is an rdf file name. A generator is returned, and it yields
    RDFRecord objects one at time, in the order they come up.
    """
    # There is one keyword allowed, and it is secret
    # Get grammar passed in, or make a new one.
    _grammar = _kwargs.get('_grammar') or syntax.Grammar()

    # prepare grammar depth, or add on a recursive call
    _grammar += 1
    # read (full) line from src
    for line in read.unwrap_file(src, wrap=_grammar.wrap):
        # get the result as _grammar processes it.
        result = _grammar(line)
        # Polymorphic unpack:
        # RdfPreRecord -> RDFRecord
        # RDFComment --> [] --> break inner loop
        # () from commands  --> ditto
        # INCLUDE --> a bunch of records
        for item in result:
            yield item
    # to get here, you hit EOF, so you're moving up a level, or out for ever.
    _grammar -= 1


## For src it's that simple
## \param src Is the source file name
## \retval rdf.data.files.RDF The RDF mapping object
def rdf_reader(src):
    """rdf = rdf_reader(src)

    src      rdf filename
    rdf      The RDF mapping object"""
    return RDF(*list(rdf_include(src)))
