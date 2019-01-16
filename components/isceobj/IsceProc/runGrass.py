#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Brett George
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import isceobj

from mroipac.grass.grass import Grass

## Interface to get class attributes's attributes that the function needs
def runGrass(self):
    return fGrass(self.insar.resampIntImage.width,
                  self.insar.topophaseFlatFilename)

## A fully context managed (2.6.x format) execution of the function
def fGrass(widthInt, topoflatIntFilename):

    with isceobj.contextIntImage(
        filename=topoflatIntFilename,
        width=widthInt,
        accessMode='read') as intImage:

        ## Note: filename is extecpted to  end in'.flat'- what
        ## if it doesn't??? Use:
        ## os.path.extsep + topoflatIntFilename.split(os.path.extsep)[-1]
        with isceobj.contextOffsetImage(
            filename=topoflatIntFilename.replace('.flat', '.cor'),
            width=widthInt,
            accessMode='write') as cohImage:
            
            with isceobj.contextIntImage(
                filename=topoflatIntFilename.replace('.flat', '.unw'),
                width=widthInt,
                accessMode='write') as unwImage:

                grass = Grass()
                grass.wireInputPort(name='interferogram', object=intImage)
                grass.wireInputPort(name='correlation', object=cohImage)
                grass.wireOutputPort(name='unwrapped interferogram', object=unwImage)
                grass.unwrap()
                
                pass
            pass
        pass
    return None
