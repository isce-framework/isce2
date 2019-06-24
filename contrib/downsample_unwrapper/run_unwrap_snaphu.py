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
import json
def runUnwrap(inps_json):
    costMode = 'SMOOTH'
    initMethod = 'MCF'
    defomax = 2.0
    initOnly = True
    if isinstance(inps_json,str):
        inps = json.load(open(inps_json))
    elif isinstance(inps_json,dict):
        inps = inps_json
    else:
        print('Expecting a json filename or a dictionary')
        raise ValueError
    wrapName = inps['flat_name']
    unwrapName = inps['unw_name']
    img = isceobj.createImage()
    img.load(wrapName + '.xml')
    width      = img.getWidth()
    earthRadius = inps['earth_radius']
    altitude   = inps['altitude']
    corrfile  = inps['cor_name']
    rangeLooks = inps['range_looks']
    azimuthLooks = inps['azimuth_looks']
    wavelength = inps['wavelength']
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
        connImage.setWidth(width)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        connImage.renderVRT()
        connImage.createImage()
        connImage.finalizeImage()
        connImage.renderHdr()

    return


def main(inps):
    runUnwrap(inps)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))