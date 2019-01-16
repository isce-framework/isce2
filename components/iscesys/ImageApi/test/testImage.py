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
import logging
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from ImageFactory import *
import test1
class TestImage:

    def test1(self,file1,file2,width1,width2,test):
        #import pdb
        #pdb.set_trace()
        obj1 = createSlcImage()
        obj2 = createOffsetImage()
        if test == 1:
            obj1.setFilename(file1)
            obj1.setWidth(width1)
            obj1.setAccessMode('read')
            obj2.setFilename(file2)
            obj2.setWidth(width2)
            obj2.setAccessMode('write')
            obj1.createImage()
            obj2.createImage()
            acc1 = obj1.getImagePointer()
            acc2 = obj2.getImagePointer()

        elif test == 2:
            obj1.setFilename(file1)
            obj1.setWidth(width1)
            obj1.setAccessMode('write')
            obj2.setFilename(file2)
            obj2.setWidth(width2)
            obj2.setAccessMode('read')
            obj1.createImage()
            obj2.createImage()
            acc1 = obj1.getImagePointer()
            acc2 = obj2.getImagePointer()
        test1.test1_Py(acc1,acc2,width1,width2,test)

        obj1.finalizeImage()
        obj2.finalizeImage()


    def __init__(self):
        pass



#end class

