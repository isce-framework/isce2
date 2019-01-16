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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import logging
from iscesys.Component.Component import Component
from isceobj.Filter import filter
from isceobj.Util.decorators import pickled, logged

## An decorator.object_wrapper style decoration, just for filter.
def filter_wrap(func):
    #2013-06-04 Kosal: new_method takes only *args
    #then we reorder and add new elements to argument list
    def new_method(self, *args):
        args = list(args)
        filterWidth = args[0]
        filterHeight = args[1]
        other_args = args[2:]
        new_args = [ self.inFile, self.outFile, self.width, self.length, filterWidth, filterHeight ]
        new_args.extend(other_args)
        return func(self, *new_args)
    #Kosal

    return new_method


@pickled
class Filter(Component):
    """A class for spatial filters"""

    logging_name = "isce.Filter"

    @logged
    def __init__(self, inFile=None, outFile=None, width=None, length=None):
        raw_input = ("In Filter")
        super(Filter, self).__init__()
        self.inFile = inFile
        self.outFile = outFile
        self.width = width
        self.length = length
        return None

    #2013-05-04 Kosal: added only *args as input parameters for functions below
    #so that they can be called twice with different arguments
    #first when called inside FR
    #and then when called by decorator.
    #C functions are just called with arguments and not returned.
    #logger is transferred inside FR._filterFaradayRotation
    @filter_wrap
    def meanFilter(self, *args):
        #should be first called with: filterWidth, filterHeight
        filter.meanFilter_Py(*args)

    @filter_wrap
    def gaussianFilter(self, *args):
        #should be first called with: filterWidth, filterHeight, sigma
        filter.gaussianFilter_Py(*args)

    @filter_wrap
    def medianFilter(self,filterWidth,filterHeight):
        #should be first called with: filterWidth, filterHeight
        filter.medianFilter_Py(*args)
