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
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from stdproc.orbit.Mocompbaseline import Mocompbaseline

def main():
    obj = Mocompbaseline()
    f1 = open(sys.argv[1])
    allLines1 = f1.readlines()
    position1 = []
    for i in range(len(allLines1)):
        split1 = allLines1[i].split()
        p1 = [float(split1[2]),float(split1[3]),float(split1[4])] 
        position1.append(p1)
    
    f2 = open(sys.argv[2])
    allLines2 = f2.readlines()
    position2 = []
    for i in range(len(allLines2)):
        split2 = allLines2[i].split()
        p2 = [float(split2[2]),float(split2[3]),float(split2[4])] 
        position2.append(p2)
    
    f3 = open(sys.argv[3])
    allLinesM = f3.readlines()
    positionM1 = []
    indx1 = []
    for i in range(len(allLinesM)):
        splitM = allLinesM[i].split()
        indx1.append(int(splitM[0]))
        positionM1.append(float(splitM[2]))
    
    f4 = open(sys.argv[4])
    allLinesM = f4.readlines()
    positionM2 = []
    indx2 = []
    for i in range(len(allLinesM)):
        splitM = allLinesM[i].split()
        indx2.append(int(splitM[0]))
        positionM2.append(float(splitM[2]))
    
    
    pegLat = 0.657602
    pegLon = .864144
    pegHdg = -2.90008
    obj.setSchPosition1(position1)
    obj.setSchPosition2(position2)
    obj.setMocompPosition1(positionM1)
    obj.setMocompPositionIndex1(indx1)
    obj.setMocompPosition2(positionM2)
    obj.setMocompPositionIndex2(indx2)
    obj.setPegLatitude(pegLat)
    obj.setPegLongitude(pegLon)
    obj.setPegHeading(pegHdg)
    obj.setHeight(771413.404628)
    obj.mocompbaseline()
    baseline = obj.getBaseline()
    mid = obj.getMidpoint()
    mid1 = obj.getMidpoint1()
    mid2 = obj.getMidpoint2()
    base1 = obj.getBaseline1()
    base2 = obj.getBaseline2()
    sch = obj.getSchs()
    sc = obj.getSc()
    print(len(baseline))
    fp = open('baseline','w')
    for i in range(len(baseline)):
        fp.write(str(baseline[i][0]) + ' ' + str(baseline[i][1]) + ' '  + str(baseline[i][2]) + ' ' +  str(mid[i][0]) + ' ' + str(mid[i][1]) + ' '  + str(mid[i][2]) +'\n' )
    fp.close()
    fp = open('midpoint','w')
    for i in range(len(mid1)):
        fp.write(str(mid1[i][0]) + ' ' + str(mid1[i][1]) + ' '  + str(mid1[i][2]) + ' ' +  str(mid2[i][0]) + ' ' + str(mid2[i][1]) + ' '  + str(mid2[i][2]) +'\n' )
    fp.close()
    fp = open('base12','w')
    for i in range(len(base1)):
        fp.write(str(base1[i][0]) + ' ' + str(base1[i][1]) + ' '  + str(base1[i][2]) + ' ' +  str(base2[i][0]) + ' ' + str(base2[i][1]) + ' '  + str(base2[i][2]) +'\n' )
    fp.close()
    fp = open('schsc','w')
    for i in range(len(sch[0])):
        fp.write(str(sch[0][i][0]) + ' ' + str(sch[0][i][1]) + ' '  + str(sch[0][i][2]) + ' ' +str(sch[1][i][0]) + ' ' + str(sch[1][i][1]) + ' '  + str(sch[1][i][2]) + ' ' +  str(sc[i][0]) + ' ' + str(sc[i][1]) + ' '  + str(sc[i][2]) +'\n' )
    fp.close()
#    for line in baseline:
#        print(line)

if __name__ == "__main__":
    sys.exit(main())
