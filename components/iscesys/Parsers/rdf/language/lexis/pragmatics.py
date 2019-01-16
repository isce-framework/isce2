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




"""Verbs may appear to be an antipatterm: the methods go mutate another
objects attributes (Grammar). But that is how RDF works: metawords change
the grammar.
"""
## \namespace rdf.language.lexis.pragmatics Words with (reflexive) meaning
## (Verb)

import abc
from iscesys.Parsers.rdf.language import lexis

## A str subclass that is also an Abstract Base Class: real RDF commands are \n
## _strings_ (hence str) with __meaning__ (hence they are subclasses of
## _Verb)
class _Verb(lexis.Word):
    """_Pragamtic is an self identifying string"""

    __metaclass__ = abc.ABCMeta
    
    ## Allow class to ondeify itself from a line (given an operator).
    def line_is(self, line, grammar):
        """line_is(line, grammar) IFF line is pragamatic"""
        line = line.strip()
        if not line.startswith(self): # Guard
            return False
        ## Does the line starts with the string?
        subline = line.lstrip(self).strip()
        return subline.startswith(grammar.operator)
    
    ## Act is not action--> act tells this object to go do it's thing \n
    ## which is act on the grammar according to line.
    @abc.abstractmethod
    def act(self, line, grammar):
        """Abstract method must be overriden in concrete subclasses"""

    ## Verbs must act -- or return an empty iterable.
    def sin_qua_non(self, line, grammar):
        return self.act(line, grammar) or ()
    

    
## Include is the most complicated Word: it initiates a recirsive \n
## call to rdf_incluec(), and thus, returns a list of RDFRecord objects.
class Include(_Verb):
    """Verb can identify the INCLUDE lines"""
    ## Include.act should never be called- an dynamic error will be thrown#
    def act(self, line, grammar):
        from iscesys.Parsers.rdf.uRDF import rdf_include
        src = grammar.operator.right(line)
        ## Sends in the grammar to the include files
        return rdf_include(src, _grammar=grammar)
    
    
## ABC for any Verb that changes a gramar symbol.
class _SymbolChange(_Verb):
    
    __metaclass__ = abc.ABCMeta

    ## A concrete method for an abstract class-- this changes grammar
    def act(self, line, grammar):
        """<Verb>(line).act(grammar, line) --> modify grammar:

        grammar.<verb> = grammar.get_value(line)
        

        note: this could be a method of grammar that takes <pragamatic>
        as input-->

        self act(attr, value)  # I guess this is setattr?
        
        """
        setattr(grammar,
                self.__class__.__name__.lower(),
                grammar.get_value(line))
        
        

## OPERATOR keyword change's rdf.language.syntax.Grammar.operator
class Operator(_SymbolChange):
    """Change grammar's operator"""
    

## COMMENT keyword change's rdf.language.syntax.Grammar.comment
class Comment(_SymbolChange):
    """Change grammar's comment attribute"""
        
        
## Its complicated and may not be a good idea.
class _Affix(_Verb):
    """_Affix is an ABC
    """
    __metaclass__ = abc.ABCMeta
    
    ## Change grammar's attribute that is the lower case of the class name,\n
    ## b/c the attribute is list-- you can use getattr on grammar and overload\n
    ## the result's __setitem__.
    def act(self, line, grammar):
        """act(grammar, line) changes grammar's affix matching
        self.__class__.__name__ according to the assignment in line"""
        # assignment to a list element in an unusual format:
        getattr(
            grammar, self.__class__.__name__.lower()
            )[int(grammar)] = grammar.get_value(line)

        
## An _Affix that coorperates with rdf.language.syntax.Grammar.prefix
class Prefix(_Affix):
    """Prefix is an _Affix that cooperates with Gramar.prefix"""


## An _Affix that coorperates with rdf.language.syntax.Grammar.suffix
class Suffix(_Affix):
    """Suffix is an _Affix that cooperates with Gramar.suffix"""

    
    
## Reserved Verbs Classes -like the constants, but functional
VERBS = (Include, Operator, Comment, Prefix, Suffix)

