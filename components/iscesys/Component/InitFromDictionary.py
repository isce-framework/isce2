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

##
#This class is an initializer and can be used with all the objects that inherit the Component class. It allows to initialize an object from a dictionary. 
#It could be a simple key/value dictionary where each variable (key) has a certain value (value) like, for instance
#\verbatim
# dictionary = {'VARIABLE1':value1,'VARIABLE2':value2}
#\endverbatim
#or more in general a dictionary of 
#dictionaries where each variable (key) has several attributes like value,doc,units etc. like for instance
#\verbatim
# dictionary = {'VARIABLE1':{'value':value1,'doc':'documentation for variable1,'units':m/s},'VARIABLE2':{'value':value2,'doc':'documentation for variable2}}
#\endverbatim
#If some of the names in the adopted in the initializing dictionary differ from the names adopted in the object to be initialized,  
# one could provide a translator, i.e. a dictionary where the key is to name of the variable as known in the 
#initialinzing dictionary and the value is the name of the variable as known in the object to be initialized. The name of the variables are the ones 
#specified in the self.dictionaryOfVariables of each object (see Component). 

# Once an instance of this class is created (say obj), the object that needs to be initialized invokes the initComponent(obj) method (inherited from the Component class)  passing the instance as argument.
#@see Component::initComponent()
class InitFromDictionary(object):
    
##
# This method must be implemented by each initializer class. It returns a dictionary of dictionaries. The object argument is not used but
# needs to be present in each implementation of the init() method.
#@return retDict dictionary of dictinaries.
    def init(self,object = None):
        #make it compatible with Component dictionary which is a dictionary of dictionaries. Check if it's only key value type
        retDict = {}
        if (not self.translator == None):
            for key , val in self.dictionary.items():
                if not isinstance(val,dict):#is only key = value
                    if key in self.translator.keys():
                        newKey = self.translator[key]
                        retDict[newKey] = {'value':val}
                    else:
                        retDict[key] = {'value':val}

                else: 
                    if key in self.translator.keys():
                        newKey = self.translator[key]
                        retDict[newKey] = self.dictionary[key]
                    else:
                        retDict[key] = self.dictionary[key]
        else: 
            print("InitFromDictionary: self.dictionary = ",self.dictionary)
            for key , val in self.dictionary.items():
                if not isinstance(val,dict):#is only key = value
                    retDict[key] = {'value':val}

                else: #should be ok
                    retDict = self.dictionary
                    break
        return retDict
##
# Constructor. It takes as argument the dictionary used to initialize the specific object.
#@param dictionary dictionary from which the object is initlized.
    def __init__(self,dictionary, translator = None):
        self.dictionary = dictionary
        self.translator = translator
        return None
    pass


