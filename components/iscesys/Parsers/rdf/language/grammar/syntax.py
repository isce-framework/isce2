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




"""syntax handles syntax via tha Grammar class. It handles syntax
but farms out some work to cooperative classes"""
## \namespace rdf.language.grammar.syntax Syntax glues it all together

from __future__ import absolute_import

import itertools
import sys
from .. import errors
from . import punctuation, morpheme
from iscesys.Parsers.rdf.reserved import glyphs, words
from iscesys.Parsers.rdf.language.lexis import semantics, pragmatics

## Metaclass for Grammar gets defines the pragamatics and semantics at
## load-time" pragmatics.Verb instances are assigned according to the
## rdf.reserved.words.KEYWORDS, and the symantics.Noun instances are created-
## these are needed by Grammar.process
class metagrammar(type):
    """metagrammar meta class deal with the keywords defined in
    verbs.
    """
    ## Create class and add pragmatics and semantics
    def __new__(mcs, *args, **kwargs):
        cls = type.__new__(mcs, *args, **kwargs)
        _prags = []

        ## Instaniate Verbs for the Grammar's command interpretation
        for p_cls, w_const in zip(pragmatics.VERBS, words.KEYWORDS):
            _prags.append(p_cls(w_const))
            setattr(cls, w_const, _prags[-1])
        # note: metaclasses can access protect members of their instances...
        cls._VERBS = tuple(_prags)
        ## Set up Noun instances by instantiaing NOUNS's classes
        cls._NOUNS = ()
        for x in semantics.NOUNS:
            cls._NOUNS += (x(),)
#        cls._NOUNS = tuple(map(apply, semantics.NOUNS))

        return cls

## Grammar is the state of the grammar -it is simply the most important \n
## class there is-- though it does cooperate and leave details to its \n
## clients.
class Grammar(object, metaclass=metagrammar):
    """Grammar() is the state of the grammar. See __init__ for why
    it supports only nullary instantiation.

    ALL_CAP class attributes a Pragamatic (i.e. meta) words.
    _lower_case private instance attributes are punctuation Glyphs
    lower_case  mutator methods ensure glyphs setting is kosher.
    Capitalized class attributes are default valules for the lower_case version.

    Overloads:
    ---------
    Function Emulation

    line --> __call__(line)  ---> RDFRecord #That is, grammar is a (semipure)
                                             function that makes lines into
                                             RDFRecords.

    (meta)line-> __call__(line)---> None    # Pragamatic (KeyWord) lines
                                              return None (they aren't rdf
                                              records) but they change the
                                              internal state of 'self'. Hence
                                              grammar is an impure function.


    other -->   __call__(line)---> None    # Comments do nothing, Errors are
                                             identified, reported to stderr
                                             and forgotten.

    Integer:
    int(grammar) returns the depth-- which is a non-negative integer telling
                         how deep the processor is in the include file tree
                         (IFT) Should not pass sys.getrecursionlimit().

    grammar += 1  There are called when the deepth_processor goes up or
    grammar -= 1  down the IFT. The change int(grammar) and manage the
                  affixes.
    """


    ## wrap tell read how to unwrap lines-- it's just a str
    wrap = glyphs.WRAP
    ## sep is not used -yet, it would appear in RDF._eval at some point.
    sep = glyphs.SEPARATOR

    ## The operator symbol (default) -capitalized to avoid class with property
    Operator = glyphs.OPERATOR
    ## The comment symbol (default) -capitalized to avoid class with property
    Comment = glyphs.COMMENT
    ## Static default prefix
    Prefix = [""]
    ## Static default suffix
    Suffix = [""]


    ## VERY IMPORTANT: Grammar() creates the DEFAULT RDF grammar \n
    ## Y'all can't change it, only RDF inputs can...
    def __init__(self):
        """Nullary instaniation: you cannot inject dependcies (DI)
        in the constructor. You allways start with the default grammar-
        which is defined in static class attributes.

        Only rdf Pragamatics (i.e commands or key words) can change the
        grammar -- infact, the attributes enscapulated in mutators.
        """
        ## The recursion depth from which the rdf lines are coming.
        self.depth = 0
        ## The dynamic self-aware operator punctuation.Glyph \n
        self.operator = self.__class__.Operator
        ## The dynamic self-aware comment punctuation.Glyph
        self.comment = self.__class__.Comment
        ## Dynamic prefix is a copy of a list -and depends on depth
        self.prefix = self.__class__.Prefix[:]
        ## Dynamic suffixx is a copy of a list -and depends on depth
        self.suffix = self.__class__.Suffix[:]



    ## Getter
    @property
    def operator(self):
        return self._operator

    ## operator has mutators to ensure it is an
    ## rdf.language.punctuation.Glyph object
    @operator.setter
    def operator(self, value):
        if not value: raise errors.NullCommandError
        # symbol is converted to a glyph.
        self._operator = punctuation.Glyph(value)

    ## Getter
    @property
    def comment(self):
        return self._comment

    ## comment has mutators to ensure it is a
    ## rdf.language.punctuation.Glyph object
    @comment.setter
    def comment(self, value):
        if not value: raise errors.NullCommandError
        self._comment = punctuation.Glyph(value)

    ## Getter
    @property
    def prefix(self):
        return self._prefix

    ## Ensure Grammar._prefix is an rdf.language.morpheme.Prefix
    @prefix.setter
    def prefix(self, value):
        self._prefix = morpheme.Prefix(value)

    ## Getter
    @property
    def suffix(self):
        return self._suffix

    ## Ensure Grammar._suffix is an rdf.language.morpheme.Suffix
    @suffix.setter
    def suffix(self, value):
        self._suffix = morpheme.Suffix(value)

    ##  str refects the current grammar state
    def __str__(self):
        return ( str(self.depth) + " " +
                 self.operator + " " +
                 self.comment + " " + str(self.prefix) + str(self.suffix) )

    ## int() --> depth
    def __int__(self):
        return self.depth

    ## += --> change depth and append affixes w/ morpheme.Affix.descend \n
    ## (which knows how to do it)
    ## \param n +1 or ValueError
    ## \par Side Effects:
    ## Affix.desend()
    ## \retval self self, changed
    def __iadd__(self, n):
        if n != 1: raise ValueError("Can only add +1")
        self.depth += int(n)
        self.prefix.descend()
        self.suffix.descend()
        return self

    ## += --> change depth and truncate affixes w/ morpheme.Affix.ascend
    ## (b/c grammar just implements it)
    ## \param n +1 or ValueEr`ror
    ## \par Side Effects:
    ##  Affix.ascend()
    ## \retval self self, changed
    def __isub__(self, n):
        if n != 1: raise ValueError("Can only subtract +1")
        self.depth -= int(n)
        self.prefix.ascend()
        self.suffix.ascend()
        return self


    ## Grammar(line) --> rdf.data.entries.RDFRecord \n
    ## It's the money method-- not it's not a pure function- it can
    ## change the state of grammar.
    def __call__(self, line):
        """grammar(line) --> grammar.process(line) (with error catching)"""
        if isinstance(line, str): # Guard (why?)
            try:
                result = self.process(line)
            except errors.RDFWarning as err:
                print >>sys.stderr, repr(err) + "::" + line
                result =  []
            else:
                result = result
        else:
            raise TypeError("Grammar processes strings, not %s" %
                            line.__class__.__name__)
        return result

    ## Process the line a Verb or a Line
    ## \param line rdf sentence
    ## \par Side Effects:
    ## word might change self
    ## \retval word(line,self) full rdf processed line
    def process(self, line):
        """process checks lines agains _PRAGAMTICS and _NOUNS
        in a short-circuit for loop. The 1st hit leads to
        processing.
        """
        # check for verbs, and then nouns-- and the ask them to do their thing
        # order matters here- alot.
        for word in itertools.chain(self._VERBS, self._NOUNS):
            if word.line_is(line, self):
                return word(line, self)

    ## Get value of a line (any line really)
    def get_value(self, line):
        """get value of a Pragamtic"""
        return self.operator.right(self.comment.left(line))


    ## Add affixes--note: Grammar just adds, the overloaded __add__ and
    ## __radd__ invoke the affix protocol.
    def affix(self, key):
        return self.prefix + key + self.suffix
