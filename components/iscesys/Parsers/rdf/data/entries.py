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




"""Define the RDF Entries as:
RDFRecord = (key, RDFField)"""
## \namespace rdf.data.entries Usable data objects for lines (records).
import collections
import sys
#from functools import partial
#from operator import methodcaller

from iscesys.Parsers.rdf.reserved import glyphs
from iscesys.Parsers.rdf.language import errors
from iscesys.Parsers.rdf.language.grammar import punctuation

# A space character
S = " "


## Decorator to cast values
## \param magicmethodbinding A function that binds to a magic method and
## casts instances (for example: float)
## \retval cast_method An instance method that cast ala magicmethodbinding
def _cast(magicmethodbinding):
    """decorator for magic method/function casting"""
    def cast_method(self):
        """__int__ --> int(self.value) --for example"""
        return magicmethodbinding(self.value)
    return cast_method

## Base RDF Field named tuple -it's all in here -note, it's assigned (public)
## name dIffers from its variable (private) name, so that users never need to
## know about this  private assignemnt
_RDFField = collections.namedtuple('RDFField',
                                   'value units dimensions element comments')


## Add methods and constants to _RDFField so that it lives up to its name.
class RDFField(_RDFField):
    """RDFField(value, units=None, dimensions=None, element=None, comments=None)

    represents a fully interpreted logical entry in an RDF file (sans key)
    """
    ## (units) Brackets
    _Units = punctuation.UNITS
    ## {dim} Brackets
    _Dimensions = punctuation.DIMENSIONS
    ## [elements] Brackets
    _Element = punctuation.ELEMENT

    ## (-) appears as default
    _default_units = ("-", "&")
    ## non-private version: it is used in units.py
    default_units = _default_units
    ## does not appear b/c it's False
    _default_comments = ""
    ## _ditto_
    _default_dimensions = ""
    ## _dito_
    _default_element = ""
    _operator = glyphs.OPERATOR
    _comment = glyphs.COMMENT

    ## Do a namedtuple with defaults as follows...
    ## \param  [cls] class is implicity passed...
    ## \param  value Is the value of the rdf field
    ## \param [units] defaults to RDFField._default_units
    ## \param [dimensions] defaults to RDFField._default_dimensions
    ## \param [element] defaults to RDFField._default_element
    ## \param [comments] defaults to RDFField._default_comments
    def __new__(cls, value, units=None, dimensions=None, element=None,
                comments=None):
        # Order unit conversion
        value, units = cls._handle_units(value, units)
        return _RDFField.__new__(cls,
                                 value,
                                 str(units or cls._default_units),
                                 str(dimensions or cls._default_dimensions),
                                 str(element or cls._default_element),
                                 str(comments or cls._default_comments))


    ## Do the unit conversion
    @classmethod
    def _handle_units(cls, value, units):
        from iscesys.Parsers.rdf.units import SI
        # convert units, If they're neither None nor "-".
        if units and units not in cls._default_units:
            try:
                value, units = SI(value, units)
            except errors.UnknownUnitWarning:
                print("UnknownUnitWarning:" +
                      (cls._Units << str(units)), file=sys.stderr)
        return value, units


    ## eval(self.value) -with some protection/massage
    ## safe for list, tuples, nd.arrays, set, dict,
    ## anything that can survive repr - this is really a work in progress,
    ## since there is a lot of python subtly involved.
    ## \returns evaluated version of RDFField.value
    def eval(self):
        """eval() uses eval built-in to interpert value"""
        try:
            result = eval(str(self.value))
        except (TypeError, NameError, AttributeError, SyntaxError):
            try:
                result = eval(repr(self.value))
            except (TypeError, NameError, AttributeError, SyntaxError):
                result = self.value
        return result


    def index(self):
        return len(self.left_field())

    ## Construct string on the left side of OPERATOR
    def left_field(self, index=0):
        """Parse left of OPERATOR
        place OPERATOR at index or don't
        """
        result = ((self.units >> self._Units) +
                  (self.dimensions >> self._Dimensions) +
                  (self.element >> self._Element))

        short =  max(0, index-len(result))

        x = result + (" "*short)
#        print len(x)

        return x

    ## Construct string on the right side of OPERATOR (w/o an IF)
    def right_field(self):
        """Parse right of operator"""
        return ( str(self.value) +
                (" " + self._comment) * bool(self.comments) +
                 (self.comments or "")
                 )


    ## FORMAT CONTROL TBD
    def __str__(self, index=0):
        """place OPERATOR at index or don't"""
        return (
            self.left_field(index=index)  +
            self._operator + S +
            self.right_field()
            )


    ## Call returns value
    ## \param [func] = \f$ f(x):x \rightarrow x\f$ A callable (like float).
    ## \returns \f$ f(x) \f$ with x from eval() method.
    def __call__(self, func=lambda __: __):
        """You can cast with call via, say:
        field(float)"""
        return func(self.eval())

    __index__ = _cast(bin)
    __hex__ = _cast(hex)
    __oct__ = _cast(oct)
    __int__ = _cast(int)
    __long__ = _cast(int)
    __float__ = _cast(float)
    __complex__ = _cast(complex)

    ## key + field --> _RDFPreRecord, the whole thing is private.
    def __radd__(self, key):
        return RDFPreRecord(key, self)



## This assignment is a bit deeper: Just a key and a field
_RDFRecord = collections.namedtuple("RDFRecord", "key field")

## The pre Record is built from data and is a len=1 iterator: iterating builds
## the final product: RDFRecord-- thus line reads or include file reads yield
## the same (polymorphic) result: iterators that yield Records.
class RDFPreRecord(_RDFRecord):
    """Users should not see this class"""

    ## iter() is about polymorphism - since an INCLUDE can yield a whole list
    ## of records - the client needs to be able to iterate it w/o typechecking
    ## this does it- you iter it once, and builds the FINAL form of the record
    ## that's polymorphism:
    ## \retval RDFRecord iter(RDFPreRecord) finalizes the object.
    def __iter__(self):
        return iter( (RDFRecord(*super(RDFPreRecord, self).__iter__()),) )



## This is a fully parsed RDF record, and is an _RDFRecord with a formatable
## string.
class RDFRecord(_RDFRecord):
    """RDFRecord(key, field)

    is the parsed RDF file line. Key is a string (or else), and
    field is an RDFField.
    """

    def __int__(self):
        from iscesys.Parsers.rdf.reserved import glyphs
        return str(self).index(glyphs.OPERATOR)

    ## FORMAT CONTROL TBD
    def __str__(self, index=0):
        """place OPERATOR at index or don't"""
        key = str(self.key)
        field = self.field.__str__(max(0, index-len(key)))
        return key + field




## The RDF Comment is a comment string, endowed with False RDF-ness
class RDFComment(str):
    """This is string that always evaluates to False.

    Why?

    False gets thrown out before being sent to the RDF constructor

    But!

    It is not None, so you can keep it in your RDFAccumulator
    """
    ## RDF comments are False in an RDF sense--regardless of their content
    # \retval < <a href=
    # "http://docs.python.org/2/library/constants.html?highlight=false#False">
    # False</a> Always returns False-- ALWAYS
    def __nonzero__(self):
        return False

    ## Iter iterates over nothing-NOT the contents of the string.
    ## \retval iter(())  An empty iterator that passes a for-loop silently
    def __iter__(self):
        return iter(())
