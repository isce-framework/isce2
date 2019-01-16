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




"""The unit module.

The rdf.data.entries.RDFField.__new__ only needs access to the
SI function-- which identifies units and converts them to nominal
inputs.

See SI.__doc__ on how Units are used.

"""
## \namespace rdf.units RDF units as spec'd
from iscesys.Parsers.rdf.units.physical_quantity import Unit
from iscesys.Parsers.rdf.units import addendum
from iscesys.Parsers.rdf.language import errors

## The global unit glossary dictionary:[symbol]->converter function
GLOSSARY = Unit.Glossary

## Convert (value, units) to SI pair - this is the interface to RDField
## Search various places for units...(TBD).
## \param value A float in units
## \param units a string describing the units
## \retval (converter(value),converter.si_unit) The new value in the right units
def SI(value, units):
    """
    Using Units:
    Unit instance are instance of <str>-- hence you can compare them or use them
    as keys in a dictionary. Hence:

    >>>km = physical_quantity.Length('km', 1000)
    
    is a string == 'km', and it is a function that multiplies by 1000.
    
    Thus: SI just looks in a dictionary of UNITS, c.f:
    
    {km : km}['km']
    
    which returns km, such that:
    
    >>>print km(1)
    1000.
    
    Sweet.
    
    See physical_quanity on how to make your own units and how to put them in
    the GLOASSRY.
    """
    try:
        converter = GLOSSARY[units]
    except KeyError:
        try:
            converter = runtime_units()[units]
        except KeyError:
            # raise errors.FatalUnitError to stop.
            raise errors.UnknownUnitWarning
    return converter(value), converter.si_unit



## A function to read user defined units at runtime (after import-- otherwise
## it's cyclic)-- format is provisional.
def runtime_units(src='units.rdf'):
    """read units from units.rdf:

    mym (m) {length} = 10000 ! A Myriameters is 10 K
    """
    from iscesys.Parsers.rdf import RDF
    try:
        result = RDF.fromfile(src)
    except IOError:
        result = {}
    return result
