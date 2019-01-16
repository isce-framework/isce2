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



import datetime
from isceobj.Orbit.Orbit import Orbit
from isceobj.Scene.Frame import Frame
from iscesys.Component.Component import Component, Port

class Pulsetiming(Component):

    logging_name = "isce.stdproc.pulsetiming"    

    def __init__(self):
        super(Pulsetiming, self).__init__()
        self.frame = None
        self.orbit = Orbit(source='Pulsetiming')
        return None

    def createPorts(self):
        framePort = Port(name='frame',method=self.addFrame)
        self._inputPorts.add(framePort)
        return None

    def getOrbit(self):
        return self.orbit
    
    def addFrame(self):        
        frame = self.inputPorts['frame']
        if frame:
            if isinstance(frame, Frame):
                self.frame = frame                
            else:
                self.logger.error(
                    "Object must be of type Frame, not %s" % (frame.__class__)
                    )
                raise TypeError
            pass
        return None
                 
#    @port(Frame)
#    def addFrame(self):
#        return None

    def pulsetiming(self):
        self.activateInputPorts()
                                   
        numberOfLines = self.frame.getNumberOfLines()
        prf = self.frame.getInstrument().getPulseRepetitionFrequency()
        pri = 1.0/prf
        startTime = self.frame.getSensingStart()
        thisOrbit = self.frame.getOrbit()
        self.orbit.setReferenceFrame(thisOrbit.getReferenceFrame())
        
        for i in range(numberOfLines):
            dt = i*pri
            time = startTime + datetime.timedelta(seconds=dt)
            sv = thisOrbit.interpolateOrbit(time,method='hermite')
            self.orbit.addStateVector(sv)                
        
