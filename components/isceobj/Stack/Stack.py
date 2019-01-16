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
# Authors: Kosal Khun, Marco Lavalle
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
import sys
from iscesys.Component.Component import Component


NB_SCENES = 100 # number of scenes and rasters that can be processed

class Stack(Component):
    """
    Stack scenes, used for processing.
    """
    # We have to suppose that there will be 100 scenes and 100 rasters,
    # which ids range from 1 to 100, and add them to the dictionary of variables
    # to fully take advantage of the parser.
    # If we happend to accept more scenes (or rasters), we have to change NB_SCENES,
    # that also applies to the number of rasters.

    def __init__(self, family=None, name=None):
        """
        Instantiate a stack.
        """
        super(Stack, self).__init__(family, name)
        self.scenes = {} ##contains all the scenes (for each selected scene and pol)
        self._ignoreMissing = True #ML 2014-05-08 with GNG

        self.dictionaryOfVariables = {}
        for attr in ['SCENE', 'RASTER']:
            for i in range(1, NB_SCENES+1):
                key = attr + str(i)
                self.dictionaryOfVariables[key] =  [key.lower(), dict, False]



    def addscene(self, scene):
        """
        Add a scene dictionary to the stack.
        """
        if not isinstance(scene, dict): ##scene is not a dictionary
            sys.exit("Scene must be a dictionary")
        else:
            sceneid = scene['id']
            self.scenes[sceneid] = scene


    def getscenes(self):
        """
        Return the scenes inside the stack.
        """
        return self.scenes
