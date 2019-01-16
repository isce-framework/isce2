#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import datetime
import logging
import isceobj
from isceobj.Scene.Frame import Frame
from iscesys.Component.Component import Component

OUTPUT = Component.Parameter(
    'output',
    public_name='OUTPUT',
    default='',
    type=str,
    mandatory=False,
    intent='input',
    doc='Raw output file name.'
)
class Sensor(Component):
    """
    Base class for storing Sensor data
    """
    parameter_list = (
                      OUTPUT,                    
                     )
    logging_name =  None
    lookMap = {'RIGHT' : -1,
               'LEFT'  : 1}
    family = 'sensor'

    def __init__(self,family='',name=''):
        super(Sensor, self).__init__(family if family else  self.__class__.family, name=name)
        self.frame = Frame()
        self.frame.configure()

        self.logger = logging.getLogger(self.logging_name)

        self.frameList = []
       
        return None


    def getFrame(self):
        '''
        Return the frame object.
        '''
        return self.frame

    def parse(self):
        '''
        Dummy routine.
        '''
        raise NotImplementedError("In Sensor Base Class")


    def populateMetadata(self, **kwargs):
        """
        Create the appropriate metadata objects from our HDF5 file
        """
        self._populatePlatform(**kwargs)
        self._populateInstrument(**kwargs)
        self._populateFrame(**kwargs)
        self._populateOrbit(**kwargs)

    def _populatePlatform(self,**kwargs):
        '''
        Dummy routine to populate platform information.
        '''
        raise NotImplementedError("In Sensor Base Class")

    def _populateInstrument(self,**kwargs):
        """
        Dummy routine to populate instrument information.
        """
        raise NotImplementedError("In Sensor Base Class")

    def _populateFrame(self,**kwargs):
        """
        Dummy routine to populate frame object.
        """
        raise NotImplementedError("In Sensor Base Class")

    def _populateOrbit(self,**kwargs):
        """
        Dummy routine to populate orbit information.
        """
        raise NotImplementedError("In Sensor Base Class")

    def extractImage(self):
        """
        Dummy routine to extract image.
        """
        raise NotImplementedError("In Sensor Base Class")

    def extractDoppler(self):
        """
        Dummy routine to extract doppler centroid information.
        """
        raise NotImplementedError("In Sensor Base Class")
