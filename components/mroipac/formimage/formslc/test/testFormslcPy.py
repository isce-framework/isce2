#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
import sys
import os
import math
from isceobj.RawImage.RawImage import RawImage
from isceobj.SlcImage.SlcImage import SlcImage
from isceobj.Platform.Platform import Platform
from isceobj.Radar.Radar import Radar
from iscesys.Component.InitFromXmlFile import InitFromXmlFile
from iscesys.Component.InitFromObject import InitFromObject
from iscesys.Component.InitFromDictionary import InitFromDictionary
from mroipac.formimage.FormSLC import FormSLC
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()

def main():
    
    # create FormSLC object and initilaize it using FormSLC930110.xml. it actually contains all the parameters already except the raw and slc images.
    # one could use the Platform and Radar objects to change some of the parameters.
    obj = FormSLC()
    initfileForm = 'FormSCL930110.xml'
    #instantiate a InitFromXmlFile object passinf the file name in the contructor
    fileInit = InitFromXmlFile(initfileForm)
    # init FormSLC by passing the init object
    obj.initComponent(fileInit)
    
    
    initfilePl = 'Platform930110.xml'
    fileInit = InitFromXmlFile(initfilePl)
    objPl = Platform()
    objPl.initComponent(fileInit)
    
    #instantiate a InitFromObject object passing the object from which to initialize in the contructor
    objInit = InitFromObject(objPl)
    obj.initComponent(objInit)
    
    initfileRadar = 'Radar930110.xml'
    fileInit = InitFromXmlFile(initfileRadar)
    objRadar = Radar()
    objRadar.initComponent(fileInit)
    
    objInit = InitFromObject(objRadar)
    obj.initComponent(objInit)
    obj.printComponent()    
    filename = "930110.raw"
    accessmode = 'read'
    endian = 'l'
    width = 11812 
    
    objRaw = RawImage()
    # only sets the parameter
    objRaw.initImage(filename,accessmode,endian,width)
    # it actually creates the C++ object
    objRaw.createImage()
    
    filenameSLC ="930110.slc"
    accessmode = 'write'
    endian = 'l'
    width = 5700
    
    dict = {'FILE_NAME':filenameSLC,'ACCESS_MODE':accessmode,'BYTE_ORDER':endian,'WIDTH':width}
    dictInit = InitFromDictionary(dict)
    objSlc = SlcImage()
    
    objSlc.initComponent(dictInit)
    objSlc.createImage()
   
    
    obj.formSLCImage(objRaw,objSlc)
    #call this to do some cleaning. always call it if initImage (or the initComponent) was called
    objSlc.finalizeImage()
    objRaw.finalizeImage()
    
if __name__ == "__main__":
    sys.exit(main())
