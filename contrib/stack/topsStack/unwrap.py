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



# giangi: taken Piyush code for snaphu and adapted
import isce
import sys
import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from isceobj.Constants import SPEED_OF_LIGHT
import argparse
import os
import pickle
import sys
#import shelve
import s1a_isce_utils as ut

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unwrap interferogram using snaphu')
    parser.add_argument('-i', '--ifg', dest='intfile', type=str, required=True,
            help='Input interferogram')
    parser.add_argument('-u', '--unw', dest='unwfile', type=str, default=None,
            help='Output unwrapped file')
    parser.add_argument('-c', '--coh', dest='cohfile', type=str,
            help='Coherence file')
    parser.add_argument('--nomcf', action='store_true', default=False,
            help='Run full snaphu and not in MCF mode')

    parser.add_argument('-a','--alks', dest='azlooks', type=int, default=1,
            help='Number of azimuth looks')
    parser.add_argument('-r', '--rlks', dest='rglooks', type=int, default=1,
            help='Number of range looks')

    parser.add_argument('-d', '--defomax', dest='defomax', type=float, default=2.0,
            help='Max cycles of deformation')
    parser.add_argument('-s', '--master', dest='master', type=str, default='master',
            help='Master directory')
    
    parser.add_argument('-m', '--method', dest='method', type=str, default='icu',
            help='unwrapping method')

    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    return parser.parse_args(args = iargs)


def extractInfo(xmlName, inps):
    '''
    Extract required information from pickle file.
    '''
    from isceobj.Planet.Planet import Planet
    from isceobj.Util.geo.ellipsoid import Ellipsoid

   # with open(pckfile, 'rb') as f:
   #    frame = pickle.load(f)

    #with shelve.open(pckfile,flag='r') as db:
    #    frame = db['swath']

    frame = ut.loadProduct(xmlName)

    burst = frame.bursts[0]
    planet = Planet(pname='Earth')
    elp = Ellipsoid(planet.ellipsoid.a, planet.ellipsoid.e2, 'WGS84')

    data = {}
    data['wavelength'] = burst.radarWavelength

    tstart = frame.bursts[0].sensingStart
    tend   = frame.bursts[-1].sensingStop
    #tmid = tstart + 0.5*(tend - tstart)
    tmid = tstart


    orbit = burst.orbit
    peg = orbit.interpolateOrbit(tmid, method='hermite')

    refElp = Planet(pname='Earth').ellipsoid
    llh = refElp.xyz_to_llh(peg.getPosition())
    hdg = orbit.getENUHeading(tmid)
    refElp.setSCH(llh[0], llh[1], hdg)

    earthRadius = refElp.pegRadCur

    altitude   = llh[2]


    #sv = burst.orbit.interpolateOrbit(tmid, method='hermite')
    #pos = sv.getPosition()
    #llh = elp.ECEF(pos[0], pos[1], pos[2]).llh()

    data['altitude'] = altitude  #llh.hgt

    #hdg = burst.orbit.getHeading()
    data['earthRadius'] = earthRadius  #elp.local_radius_of_curvature(llh.lat, hdg)
    
    #azspacing  = burst.azimuthTimeInterval * sv.getScalarVelocity()
    #azres = 20.0 

    #corrfile  = os.path.join(self._insar.mergedDirname, self._insar.coherenceFilename)
    rangeLooks = inps.rglooks
    azimuthLooks = inps.azlooks
    azfact = 0.8
    rngfact = 0.8

    corrLooks = rangeLooks * azimuthLooks/(azfact*rngfact)

    data['corrlooks'] = corrLooks  #inps.rglooks * inps.azlooks * azspacing / azres
    data['rglooks'] = inps.rglooks
    data['azlooks'] = inps.azlooks

    return data

def runUnwrap(infile, outfile, corfile, config, costMode = None,initMethod = None, defomax = None, initOnly = None):

    if costMode is None:
        costMode   = 'DEFO'
    
    if initMethod is None:
        initMethod = 'MST'
    
    if  defomax is None:
        defomax = 4.0
    
    if initOnly is None:
        initOnly = False
    
    wrapName = infile
    unwrapName = outfile

    img = isceobj.createImage()
    img.load(infile + '.xml')


    wavelength = config['wavelength']
    width      = img.getWidth()
    length     = img.getLength()
    earthRadius = config['earthRadius'] 
    altitude   = config['altitude']
    rangeLooks = config['rglooks']
    azimuthLooks = config['azlooks']
    corrLooks = config['corrlooks']
    maxComponents = 20

    snp = Snaphu()
    snp.setInitOnly(initOnly)
    snp.setInput(wrapName)
    snp.setOutput(unwrapName)
    snp.setWidth(width)
    snp.setCostMode(costMode)
    snp.setEarthRadius(earthRadius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(corfile)
    snp.setInitMethod(initMethod)
    snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(maxComponents)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(rangeLooks)
    snp.setAzimuthLooks(azimuthLooks)
    snp.setCorFileFormat('FLOAT_DATA')
    snp.prepare()
    snp.unwrap()

    ######Render XML
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setLength(length)
    outImage.setAccessMode('read')
   # outImage.createImage()
    outImage.renderHdr()
    outImage.renderVRT()
    #outImage.finalizeImage()

    #####Check if connected components was created
    if snp.dumpConnectedComponents:
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName+'.conncomp')
        #At least one can query for the name used
        connImage.setWidth(width)
        connImage.setLength(length)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
    #    connImage.createImage()
        connImage.renderHdr()
        connImage.renderVRT()
     #   connImage.finalizeImage()

    return


def runUnwrapMcf(infile, outfile, corfile, config, defomax=2):
    runUnwrap(infile, outfile, corfile, config, costMode = 'SMOOTH',initMethod = 'MCF', defomax = defomax, initOnly = True)
    return


def runUnwrapIcu(infile, outfile):
    from mroipac.icu.Icu import Icu
    #Setup images
    #ampImage
   # ampImage = obj.insar.resampAmpImage.copy(access_mode='read')
   # width = self.ampImage.getWidth()

    img = isceobj.createImage()
    img.load(infile + '.xml')


    width      = img.getWidth()

    #intImage
    intImage = isceobj.createIntImage()
    intImage.initImage(infile, 'read', width)
    intImage.createImage()
   

    #unwImage
    unwImage = isceobj.Image.createImage()
    unwImage.setFilename(outfile)
    unwImage.setWidth(width)
    unwImage.imageType = 'unw'
    unwImage.bands = 2
    unwImage.scheme = 'BIL'
    unwImage.dataType = 'FLOAT'
    unwImage.setAccessMode('write')
    unwImage.createImage()
    
    #unwrap with icu
    icuObj = Icu()
    icuObj.filteringFlag = False      
    icuObj.useAmplitudeFlag = False
    icuObj.singlePatch = True
    icuObj.initCorrThreshold = 0.1
    icuObj.icu(intImage=intImage, unwImage = unwImage)

    #ampImage.finalizeImage()
    #intImage.finalizeImage()
    #unwImage.finalizeImage()
    unwImage.renderHdr()

def main(iargs=None):
    '''
    The main driver.
    '''

    inps = cmdLineParse(iargs)
    interferogramDir = os.path.dirname(inps.intfile)
    print ('unwrapping method : ' , inps.method)
    if inps.method == 'snaphu':
       if inps.nomcf: 
           fncall =  runUnwrap
       else:
           fncall = runUnwrapMcf
       swathList = ut.getSwathList(inps.master) 
       #metadata = extractInfo(inps.master+'.xml', inps)
       xmlFile = os.path.join(inps.master , 'IW{0}.xml'.format(swathList[0]))
       metadata = extractInfo(xmlFile, inps)
       fncall(inps.intfile, inps.unwfile, inps.cohfile, metadata, defomax=inps.defomax)

    elif inps.method == 'icu':
       runUnwrapIcu(inps.intfile, inps.unwfile)


if __name__ == '__main__':

    main()
