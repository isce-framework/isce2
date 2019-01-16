#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2015 California Institute of Technology. ALL RIGHTS RESERVED.
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



###These are demonstration scripts for the UNAVCO InSAR processing workshop.
###These scripts will not be maintained over the long term and should not
###be mistaken for official Applications within ISCE.
###These scripts are meant to demo the use of ISCE as a modular library

import argparse
import isce
import isceobj
import numpy as np
import shelve
import os
import datetime
from isceobj.Constants import SPEED_OF_LIGHT

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Create DEM simulation for merged images')
    parser.add_argument('-a','--alks', dest='alks', type=int, default=1,
            help = 'Number of azimuth looks')
    parser.add_argument('-r','--rlks', dest='rlks', type=int, default=1,
            help = 'Number of range looks')
    parser.add_argument('-d', '--dem', dest='dem', type=str, required=True,
            help = 'Input DEM to use')
    parser.add_argument('-m', '--master', dest='master', type=str, required=True,
            help = 'Dir with master frame')
    parser.add_argument('-b', '--bbox', dest='bbox', type=float, nargs=4, required=True,
            help='SNWE bounding box')
    parser.add_argument('-i', '--input', dest='infile', type=str, required=True,
            help='Input file to be geocoded')
    parser.add_argument('-t', '--type', dest='method', type=str, default='nearest',
            help='Interpolation method to use')
    parser.add_argument('-n', '--nativedop', dest='nativedop', action='store_true',
            default=False)
    return parser.parse_args()

def runGeo(frame, demImage, inImage,
        looks=(1,1), doppler=None,nativedop=False,
        bbox=None, method='nearest'):
    from zerodop.geozero import createGeozero
    from isceobj.Planet.Planet import Planet

    #####Run Topo
    planet = Planet(pname='Earth')
    topo = createGeozero()
    topo.configure()

    alooks = looks[0]
    rlooks = looks[1]


    tStart = frame.sensingStart

    topo.slantRangePixelSpacing = frame.getInstrument().getRangePixelSize()
    topo.prf = frame.getInstrument().getPulseRepetitionFrequency()
    topo.radarWavelength = frame.getInstrument().getRadarWavelength()
    topo.orbit = frame.orbit
    topo.width = inImage.getWidth()
    topo.length = inImage.getLength()
    topo.wireInputPort(name='dem', object=demImage)
    topo.wireInputPort(name='planet', object=planet)
    topo.wireInputPort(name='tobegeocoded', object=inImage)
    topo.numberRangeLooks = rlooks
    topo.numberAzimuthLooks = alooks
    topo.lookSide =  frame.instrument.platform.pointingDirection()
    topo.setSensingStart(tStart)
    topo.rangeFirstSample = frame.startingRange
    topo.method=method
    topo.demCropFilename = 'crop.dem'
    topo.geoFilename = inImage.filename + '.geo'

    if inps.nativedop and (doppler is not None):
        try:
            topo.dopplerCentroidCoeffs = [x/topo.prf for x in doppler._coeffs]
        except:
            topo.dopplerCentroidCoeffs = [x/topo.prf for x in doppler]
    else:
        topo.dopplerCentroidCoeffs = [0.]

    topo.snwe = bbox
    topo.geocode()

    print('South: ', topo.minimumGeoLatitude)
    print('North: ', topo.maximumGeoLatitude)
    print('West:  ', topo.minimumGeoLongitude)
    print('East:  ', topo.maximumGeoLongitude)
    return


if __name__ == '__main__':

    #####Parse command line
    inps = cmdLineParse()

    #####Load master metadata
    db = shelve.open( os.path.join(inps.master, 'data'), flag='r')
    frame = db['frame']
    try:
        dop = db['doppler']
    except:
        dop = frame._dopplerVsPixel
    db.close()

    ####Setup dem
    demImage = isceobj.createDemImage()
    demImage.load(inps.dem + '.xml')
    demImage.setAccessMode('read')


    ####Setup input file
    inImage = isceobj.createImage()
    inImage.load(inps.infile + '.xml')
    inImage.setAccessMode('read')

    ####Geocode the image
    runGeo(frame,demImage,inImage, nativedop=inps.nativedop,
            looks=(inps.alks, inps.rlks), doppler=dop,
            bbox=inps.bbox, method=inps.method)
