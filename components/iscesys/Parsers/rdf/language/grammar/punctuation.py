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




"""Brackets: This is where glyphs take on meaning"""
## \namespace rdf.language.grammar.punctuation Language's Punctuation Marks.

from __future__ import absolute_import

from iscesys.Parsers.rdf import reserved
from iscesys.Parsers.rdf.reserved import glyphs

## A symbol is a string that can split a line on it's left most occurance\n
## It's a puncuatin mark that can find itself
class Glyph(str):
    """A Glyph is a str sub-class that can be called.

    symbol(line) splits the line on the 1st occorence of symbol in
    line. If it is not in line, you still get 2 results:

    line, ""

    so it i basically an 2-ple safe unpacking of a split on self.
    """
    ## split line on self
    ## \param line A line
    ## \returns (left, right) side of line (with possible null str on right)
    def __call__(self, line):
        try:
            index = line.index(self)
        except ValueError:
            left, right = line, ""
        else:
            left = line[:index]
            right = line[index+1:]
        return list(map(str.strip, (left, right)))

    ## Get line left of self
    ## \param line A line with or without self
    ## \retval left line left of self
    def left(self, line):
        """left symbol"""
        return self(line)[0]

    ## Get line right of self
    ## \param line A line with or without self
    ## \retval right line right of self
    def right(self, line):
        """right symbol"""
        return self(line)[-1]


## <a href="http://en.wikipedia.org/wiki/Bracket">Brackets</a> that
## know thy selves.
class Brackets(str):
    """_Delimeter('LR')

    get it? Knows how to find itself in line

    """
    ## L, R --> -, + \n + is right
    def __pos__(self):
        return self[-len(self)//2:]

    ## L, R --> -, + \n - is left
    def __neg__(self):
        return self[:len(self)//2]


    ## extract enclosed:  line<<pair
    # \param line An RDF sentence
    # \par Side Effects:
    #  raises RDFWarning on bad grammar
    # \retval contents The string inside the last Bracket on the line
    def __rlshift__(self, line):
        """INPUTS: pair, line

        pair is 2 characters LR, this extracts part of line
        between L    and   R. Throws errors IF need be.
        """
        # 5 IF's are for error checking, not processing
        from iscesys.Parsers.rdf.language import errors
        ## Count start and stops
        count = list(map(line.count, self))
        ## Guard: early return.
        if min(count) is 0:   # Guard
            ## Check IF there is an oper/close error
            if max(count):   # Guard
                raise errors.UnmatchedBracketsError(self)
            return None
        ## Ensure the number of left Bracket objects match the number of right ones
        if count[0] != count[1]:
            raise errors.UnmatchedBracketsError(self)  #Assume all braces come in pairs
        #Reverse find the bracket pair at the end of the line
        i_start = line.rfind(-self) + 1
        i_stop = line.rfind(+self)
        ## ensure order:
        if i_stop <= i_start:    # Guard
            raise errors.BackwardBracketsError(self)
        contents = line[i_start : i_stop]
        # finally check for nonsense
        for single_char in contents:
            if single_char in reserved.RESERVED:   # Guard
                raise errors.ReservedCharacterError(self)
        return contents

    ## Insert: line>>pair or go blank
    def __rrshift__(self, line):
        """Insert non-zero line in string, or nothing"""
        return " %s%s%s " % (-self, str(line), +self) if line else ""

    __lshift__ = __rrshift__
    __rshift__ = __rlshift__

    ## (line in delimiter) IF the line has token in it legally
    # \param line an RDF sentence
    # \retval <bool> IF Bracket is in the line
    def __contains__(self, line):
        return ( (-self in line) and
                 (+self in line) and
                 line.index(-self) < line.index(+self) )

    ## line - delimiter removes delimeter from line, with no IF
    def __rsub__(self, line):
        """IF line in self __get_inner(line) else line"""
        return {True  : self.__get_inner,
                False : self.__no_inner}[line in self](line)

    ## Call IF line is in self, then go get it
    def __get_inner(self, line):
        return  (line[:line.rindex(-self)] +
                 line[1+line.rindex(+self):]).strip()

    ## Call IF line is not in self, a no-op.
    @staticmethod
    def __no_inner(line):
        return line


## Unit defining Brackets from rdf.reserved.glyphs.UNITS
UNITS =  Brackets(glyphs.UNITS)

## DIMENSIONS defining Brackets from rdf.reserved.glyphs.DIMENSIONS
DIMENSIONS = Brackets(glyphs.DIMENSIONS)

## ELEMENT defining Brackets from rdf.reserved.glyphs.ELEMENT
ELEMENT = Brackets(glyphs.ELEMENT)

## Tuple of RDF Optional Left Fields
_OPTIONAL_LEFT_FIELDS = (UNITS, DIMENSIONS, ELEMENT)


## Self explanatory
NUMBER_OF_OPTIONAL_LEFT_FIELDS = len(_OPTIONAL_LEFT_FIELDS)

## get ::_OPTIONAL_LEFT_FIELDS (olf).
def get_olf(left_line):
    """parse out UNITS DIMENSIONS ELEMENT from input line"""
    return [left_line << item for item in _OPTIONAL_LEFT_FIELDS]

## Get the key out of the left side of an rdf record \n
## Note: this relies on the Brackets.__rsub__ operator
def get_key(leftline):
    """Get key part only form a record line's left-of-operator portion"""
    return (leftline - UNITS - DIMENSIONS - ELEMENT).strip()

## get key and delimeters - the entrie left side of an rdf record, parsed
def key_parse(leftline):
    """Break left-of-operator portion into key, units, dimensions, element"""
    return [get_key(leftline)] + get_olf(leftline)
