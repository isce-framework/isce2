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
#This class is an initializer and can be used with all the objects that inherit the Component class. It allows to initialize an object from an xml file. 
#The format of the file must be like
#\verbatim
#<component>
#    <name>NameOfTheObject</name>
#    <property>
#        <name>VARAIBLE1</name>
#        <value>value1</value>
#        <doc>"documentation for VARIABLE1"</doc>
#    </property>
#    <property>
#        <name>VARAIBLE2</name>
#        <value>value2</value>
#         <units>m/s</units>
#    </property>
#</component>
#\endverbatim
#Everything that follows the \# will be discarded. If a variable is a list, the elements are separated by white spaces.
# Once an instance of this class is created (say obj), the object that needs to be initialized invokes the initComponent(obj) method (inherited from the Component class)  passing the instance as argument.
#@see Component::initComponent()
class InitFromXmlFile(object):
    
##
# This method must be implemented by each initializer class. It returns a dictionary of dictionaries. The object argument is not used but
# needs to be present in each implementation of the init() method.
#@return retDict dictionary of dictinaries.
    def init(self,object = None):
        objXmlUtil = XmlUtil()
        retDict =  objXmlUtil.createDictionary(objXmlUtil.readFile(self.filename))
        return retDict


##
# Constructor. It takes as argument the filename where the information  to initialize the specific object is stored.
#@param filename xml file from which the object is initlized.
    def __init__(self,filename):
        self.filename = filename
        return
        
