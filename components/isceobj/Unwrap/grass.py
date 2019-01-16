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




import sys
import isceobj
from iscesys.Component.Component import Component
from mroipac.grass.grass import Grass


class grass(Component):
    '''Specific Connector from an insarApp object to a Grass object.''' 
    def __init__(self, obj):

        basename = obj.insar.topophaseFlatFilename
        self.wrapName = basename
        self.unwrapName = basename.replace('.flat', '.unw')

        ###To deal with missing filt_*.cor
        if basename.startswith('filt_'):
            self.corName  = basename.replace('.flat', '.cor')[5:]
        else:
            self.corName  = basename.replace('.flat', '.cor')
   
            self.width = obj.insar.resampIntImage.width

#   print("Wrap: ", self.wrapName)
#   print("Unwrap: ", self.unwrapName)
#   print("Coh: ", self.corName)
#   print("Width: ", self.width)


    def unwrap(self):
   
        with isceobj.contextIntImage(
            filename=self.wrapName,
            width=self.width,
            accessMode='read') as intImage:

            with isceobj.contextOffsetImage(
                filename=self.corName,
                width = self.width,
                accessMode='read') as cohImage:


                with isceobj.contextIntImage(
                    filename=self.unwrapName,
                    width = self.width,
                    accessMode='write') as unwImage:

                    grs=Grass()
                    grs.wireInputPort(name='interferogram',
                        object=intImage)
                    grs.wireInputPort(name='correlation',
                        object=cohImage)
                    grs.wireOutputPort(name='unwrapped interferogram',
                        object=unwImage)
                    grs.unwrap()
                    unwImage.renderHdr()

                    pass
                pass
            pass
    
        return None
