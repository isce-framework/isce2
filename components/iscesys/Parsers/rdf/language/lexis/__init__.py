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




## \namespace rdf.language.lexis The Lexis comprises the words in the language.
import abc

## The Pragamtic's are RDF lines meaning.
class Word(str):
    """Word is an ABC that subclasses str. It has a call
    that dyamically dispatches args = (line, grammar) to
    the sub classes' sin qua non method-- which is the
    method that allows them to do their business.
    """

    __metaclass__ = abc.ABCMeta

    # Call the Pragamtic's 'sin_qua_non' method -which is TBDD \n
    # (To be Dynamically Dispathed ;-)
    def __call__(self, line, grammar):
        return self.sin_qua_non(line, grammar)

    @abc.abstractmethod
    def sin_qua_non(self, line, grammar):
        pass
