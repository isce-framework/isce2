#
# Author: Piyush Agram
# Copyright 2016
#

import sys
import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from isceobj.Constants import SPEED_OF_LIGHT
from isceobj.Planet.Planet import Planet
import os
import numpy as np
from isceobj.TopsProc.runIon import maskUnwrap


def runUnwrap(self,costMode = None,initMethod = None, defomax = None, initOnly = None):

    if costMode is None:
        costMode   = 'DEFO'
    
    if initMethod is None:
        initMethod = 'MST'
    
    if  defomax is None:
        defomax = 4.0
    
    if initOnly is None:
        initOnly = False
    

    wrapName = os.path.join( self._insar.mergedDirname, self._insar.filtFilename)
    unwrapName = os.path.join( self._insar.mergedDirname, self._insar.unwrappedIntFilename)

    img = isceobj.createImage()
    img.load(wrapName + '.xml')


    swathList = self._insar.getValidSwathList(self.swaths)

    for swath in swathList[0:1]:
        ifg = self._insar.loadProduct( os.path.join(self._insar.fineIfgDirname, 'IW{0}.xml'.format(swath)))


        wavelength = ifg.bursts[0].radarWavelength
        width      = img.getWidth()


        ####tmid 
        tstart = ifg.bursts[0].sensingStart
        tend   = ifg.bursts[-1].sensingStop
        tmid = tstart + 0.5*(tend - tstart) 

        #some times tmid may exceed the time span, so use mid burst instead
        #14-APR-2018, Cunren Liang
        #orbit = ifg.bursts[0].orbit
        burst_index = int(np.around(len(ifg.bursts)/2))
        orbit = ifg.bursts[burst_index].orbit
        peg = orbit.interpolateOrbit(tmid, method='hermite')


        refElp = Planet(pname='Earth').ellipsoid
        llh = refElp.xyz_to_llh(peg.getPosition())
        hdg = orbit.getENUHeading(tmid)
        refElp.setSCH(llh[0], llh[1], hdg)

        earthRadius = refElp.pegRadCur

        altitude   = llh[2]

    corrfile  = os.path.join(self._insar.mergedDirname, self._insar.coherenceFilename)
    rangeLooks = self.numberRangeLooks
    azimuthLooks = self.numberAzimuthLooks

    azfact = 0.8
    rngfact = 0.8

    corrLooks = rangeLooks * azimuthLooks/(azfact*rngfact) 
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
    snp.setCorrfile(corrfile)
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
    outImage.setAccessMode('read')
    outImage.renderVRT()
    outImage.createImage()
    outImage.finalizeImage()
    outImage.renderHdr()

    #####Check if connected components was created
    if snp.dumpConnectedComponents:
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName+'.conncomp')
        #At least one can query for the name used
        self._insar.connectedComponentsFilename = unwrapName+'.conncomp'
        connImage.setWidth(width)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        connImage.renderVRT()
        connImage.createImage()
        connImage.finalizeImage()
        connImage.renderHdr()

        #mask the areas where values are zero.
        #15-APR-2018, Cunren Liang
        maskUnwrap(unwrapName, wrapName)

    return


def runUnwrapMcf(self):
    runUnwrap(self,costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    return
