#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                  Giangi Sacco
#                        NASA Jet Propulsion Laboratory
#                      California Institute of Technology
#                        (C) 2009-2010  All Rights Reserved
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import sys
import os
import math
from iscesys.StdOE.StdOEPy import StdOEPy
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from mroipac.getPegInfo.Get_peg_info import Get_peg_info

def main():
    
    stdObj = StdOEPy()
    stdObj.setStdOutFile('testLogFile')
    stdObj.setStdOutFileTag('testGetPegInfo')
    obj = Get_peg_info()
    fin = open('930110.orrm')
    allLines = fin.readlines()
    time = []
    pos = []
    vel = []
    for line in allLines:
        lineS = line.split()
        time.append(float(lineS[0]))
        pos.append([float(lineS[1]),float(lineS[2]),float(lineS[3])])
        vel.append([float(lineS[4]),float(lineS[5]),float(lineS[6])])
        
    numLines = 14970
    numLk = 1
    slcTime = 66327.1431524974
    prf = 1679.87845453499
    obj.setNumLinesInt(numLines)
    obj.setNumLinesSlc(numLines)
    obj.setNumAzimuthLooksInt(numLk)
    obj.setTimeSlc(slcTime)
    obj.setTime(time)
    obj.setPrfSlc(prf)
    obj.setPositionVector(pos)
    obj.setVelocityVector(vel)
    
    obj.get_peg_info()
    print('pegLat',obj.getPegLat())
    print('pegLon',obj.getPegLon())
    print('pegHgt',obj.getPegHeight())
    print('pegHead',obj.getPegHeading())
    print('V fit',obj.getVerticalFit())
    print('H fit',obj.getHorizontalFit())
    print('V V fit',obj.getVerticalVelocityFit())
    print('C V fit',obj.getCrossTrackVelocityFit())
    print('A V fit',obj.getAlongTrackVelocityFit())
    print('peg Rad',obj.getPegRadius())
    print('grnd',obj.getGroundSpacing())
    print('mat',obj.getTransformationMatrix())
    print('t vec',obj.getTranslationVector())
    print('P V ',obj.getPegVelocity())
    print('SCH V ',obj.getPlatformSCHVelocity())
    print('SCH A ',obj.getPlatformSCHAcceleration())
    print('time ',obj.getTimeFirstScene())
    #stdObj.finalizeStdOE(ptStdOE)
    #print('I P ',obj.getIntPosition())
    #print('I V ',obj.getIntVelocity())
if __name__ == "__main__":
    sys.exit(main())
