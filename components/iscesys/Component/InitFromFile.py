#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2009 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
from isceobj.XmlUtil.XmlUtil import XmlUtil
##
#This class is an initializer and can be used with all the objects that inherit the Component class. It allows to initialize an object from a file. 
#The format of the file must be key = value, like for instance
#\verbatim
#\#comment that will be discarded
#VARIABLE1 = value1
#VARIABLE2 = value21 value22 \#comment that will be discarded
#\endverbatim
#Everything that follows the \# will be discarded. If a variable is a list, the elements are separated by white spaces.
# Once an instance of this class is created (say obj), the object that needs to be initialized invokes the initComponent(obj) method (inherited from the Component class)  passing the instance as argument.
#@see Component::initComponent()
class InitFromFile:

##
# This method must be implemented by each initializer class. It returns a dictionary of dictionaries. The object argument is not used but
# needs to be present in each implementation of the init() method.
#@return dictionary dictionary of dictinaries.
    
    def init(self, object=None):
        try:
            with open(self.filename) as file_:
                dictionary = {}
                for line in file.readlines_():
                    if not line or line.startswith('#'):
                        continue
                    if line.count('#'):# remove comments from line
                        pos = line.find('#')
                        line = line[0:pos]
                        pass
                    splitLine = line.split()
                    # remove lines that do not have at least two values
                    if len(splitLine) < 2:
                        continue
                    elif len(splitLine) == 2:  #just key and value value
                        dictionary[splitLine[0]] = {'value':splitLine[1]}
                    else:
                        # the value is a list
                        valList = splitLine[1:]
                        pass
                    dictionary[splitLine[0]] = {'value':valList}
                    pass
                pass
            pass
        except IOError:
            raise IOError(
                "Error in InitFromFile.py. Cannot open file %s "  % (self.filename)
                )
        except (TypeError, NameError, AttributeError, KeyError) as err:
            print ("This error is the fault of JEB--as thie method's refactoring was not tested properly")

        return dictionary

##
# Constructor. It takes as argument the filename where the information  to initialize the specific object is stored.
#@param filename file from which the object is initlized.

    def __init__(self, filename):
        self.filename = filename
        return
        
    pass

