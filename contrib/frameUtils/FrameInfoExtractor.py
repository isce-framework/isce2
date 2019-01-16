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
import os
from mroipac.geolocate.Geolocate import Geolocate
import logging
from contrib.frameUtils.FrameMetaData import FrameMetaData
class FrameInfoExtractor():

    
    def __init__(self):
        self.logger = logging.getLogger("contrib.frameUtils.FrameInfoExtractor")
        self._frameFilename = ''
        self._frame = None 
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d
    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger("FrameInfoExtractor")
    
    def setFrameFilename(self,name):
        self._frameFilename = name
    
    def calculateCorners(self):
        """
        Calculate the approximate geographic coordinates of corners of the SAR image.

        @return (\a tuple) a list with the corner coordinates and a list with the look angles to these coordinates
        """
        # Extract the planet from the hh object
        
        planet = self._frame.getInstrument().getPlatform().getPlanet()
        # Wire up the geolocation object
        geolocate = Geolocate()
        geolocate.wireInputPort(name='planet',object=planet)
        try:
            earlySquint = self._frame._squintAngle 
        except:
            earlySquint = 0.0

        lookSide = int(self._frame.getInstrument().getPlatform().pointingDirection)
        # Get the ranges, squints and state vectors that defined the boundaries of the frame
        orbit = self._frame.getOrbit()               
        nearRange = self._frame.getStartingRange()
        farRange = self._frame.getFarRange()        
        earlyStateVector = orbit.interpolateOrbit(self._frame.getSensingStart())
        lateStateVector = orbit.interpolateOrbit(self._frame.getSensingStop())            
        nearEarlyCorner,nearEarlyLookAngle,nearEarlyIncAngle = geolocate.geolocate(earlyStateVector.getPosition(),
                                                                                   earlyStateVector.getVelocity(),
                                                                                   nearRange,earlySquint,lookSide)        
        farEarlyCorner,farEarlyLookAngle,farEarlyIncAngle = geolocate.geolocate(earlyStateVector.getPosition(),
                                                                                earlyStateVector.getVelocity(),
                                                                                farRange,earlySquint,lookSide)
        nearLateCorner,nearLateLookAngle,nearLateIncAngle = geolocate.geolocate(lateStateVector.getPosition(),
                                                                                lateStateVector.getVelocity(),
                                                                                nearRange,earlySquint,lookSide)
        farLateCorner,farLateLookAngle,farLateIncAngle = geolocate.geolocate(lateStateVector.getPosition(),
                                                                             lateStateVector.getVelocity(),
                                                                             farRange,earlySquint,lookSide)
        self.logger.debug("Near Early Corner: %s" % nearEarlyCorner)
        self.logger.debug("Near Early Look Angle: %s" % nearEarlyLookAngle)
        self.logger.debug("Near Early Incidence Angle: %s " % nearEarlyIncAngle)

        self.logger.debug("Far Early Corner: %s" % farEarlyCorner)
        self.logger.debug("Far Early Look Angle: %s" % farEarlyLookAngle)
        self.logger.debug("Far Early Incidence Angle: %s" % farEarlyIncAngle)

        self.logger.debug("Near Late Corner: %s" % nearLateCorner)
        self.logger.debug("Near Late Look Angle: %s" % nearLateLookAngle)
        self.logger.debug("Near Late Incidence Angle: %s" % nearLateIncAngle)

        self.logger.debug("Far Late Corner: %s" % farLateCorner)
        self.logger.debug("Far Late Look Angle: %s" % farLateLookAngle)
        self.logger.debug("Far Late Incidence Angle: %s" % farLateIncAngle)

        corners = [nearEarlyCorner,farEarlyCorner,nearLateCorner,farLateCorner]
        lookAngles = [nearEarlyLookAngle,farEarlyLookAngle,nearLateLookAngle,farLateLookAngle]
        return corners,lookAngles
    def convertBboxToPoly(self,bbox):
        nearEarlyCorner = bbox[0]
        farEarlyCorner = bbox[1]
        nearLateCorner = bbox[2]
        farLateCorner = bbox[3]
       # save the corners starting from nearEarly and going clockwise
        if (nearEarlyCorner[1] < farEarlyCorner[1]):
            if (nearEarlyCorner[0] > farEarlyCorner[0]):
                corners = [nearEarlyCorner,farEarlyCorner,farLateCorner,nearLateCorner]
            else:
                corners = [nearEarlyCorner,nearLateCorner,farLateCorner,farEarlyCorner]
        
        else:
            if (nearEarlyCorner[0] > farEarlyCorner[0]):
                corners = [nearEarlyCorner,nearLateCorner,farLateCorner,farEarlyCorner]
            else:
                corners = [nearEarlyCorner,farEarlyCorner,farLateCorner,nearLateCorner]
        return corners
    def extractInfoFromFile(self, filename = None):
        import cPickle as cP
        if(filename == None):
            filename = self._frameFilename
        
        fp  = open(filename,'r')
        self._frame = cP.load(fp)
        fp.close()
        return self.extractInfo()


    def extractInfoFromFrame(self,frame):
        self._frame = frame
        return self.extractInfo()

    # update the frame by setting the attribute attr to the value val. if obj is a string then assume that is a filename, otherwise assume that is a frame object
    def updateFrameInfo(self,attr,val,obj):
        from isceobj.Scene import Frame

        if(isinstance(obj,str)):
            import cPickle as cP
            fp  = open(obj,'r')
            frame = cP.load(fp)
            fp.close()
            if(isinstance(attr,list)):
                for i in range(len(attr)):
                    setattr(frame,attr[i],val[i])
            else:
                setattr(frame,attr,val)
            #update the pickled file
            fp  = open(obj,'w')
            cP.dump(frame,fp,2)
            fp.close()

        elif(isinstance(obj,Frame)):
            frame = obj
            if(isinstance(attr,list)):
                for i in range(len(attr)):
                    setattr(frame,attr[i],val[i])
            else:
                setattr(frame,attr,val)
        else:
            self.logger.error("Error. The method updateFrameInfo takes as third argument a strig or a Frame object.")
            raise Exception

        
    def extractInfo(self):
        FM = FrameMetaData()
        bbox , dummy = self.calculateCorners()
        for bb in bbox:
            FM._bbox.append((bb.getLatitude(),bb.getLongitude()))
        #try since sometimes is and empty string. if so set it to None
        try:
            FM._frameNumber = int(self._frame.getFrameNumber())
        except:
            FM._frameNumber = None
        try:
            FM._trackNumber = int(self._frame.getTrackNumber())
        except:
            FM._trackNumber = None
        try:
            FM._orbitNumber = int(self._frame.getOrbitNumber())
        except:
            FM._orbitNumber = None
        FM._sensingStart = self._frame.getSensingStart()
        FM._sensingStop = self._frame.getSensingStop()
        FM._spacecraftName = self._frame.getInstrument().getPlatform().getSpacecraftName()
        #bbox is nearEarly,farEarly,nearLate,farLate
        if(FM._bbox[0][0] < FM._bbox[2][0]):
            #if latEarly < latLate then asc otherwise dsc
            FM._direction = 'asc'
        else:
            FM._direction = 'dsc'

        return FM


def main(argv):
    import isce
    FI = FrameInfoExtractor()
    FM = FI.extractInfoFromFile(argv[0])
    print(FM.bbox) 

if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    sys.exit(main(argv))
