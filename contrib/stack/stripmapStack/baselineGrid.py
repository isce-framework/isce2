#!/usr/bin/env python3
# Author: Piyush Agram
# Copyright 2016
#Heresh Fattahi, Adopted for stack

import argparse
import logging
import datetime
import isce
import isceobj
import mroipac
import os
import shelve
import filecmp

def createParser():
    parser = argparse.ArgumentParser( description='Generate a baseline grid for interferograms')

    parser.add_argument('-m', '--reference', dest='reference', type=str, required=True,
            help='Directory with reference acquisition shelf file')

    parser.add_argument('-s', '--secondary', dest='secondary', type=str, required=True,
            help='Directory with secondary acquisition shelf file')

    parser.add_argument('-b', '--baseline_file', dest='baselineFile', type=str, required=True,
                help='An output text file which contains the computed baseline')


    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def getMergedOrbit(product):
    from isceobj.Orbit.Orbit import Orbit

    ###Create merged orbit
    orb = Orbit()
    orb.configure()

    #Add first orbit to begin with
    for sv in product.orbit:
        orb.addStateVector(sv)

    return orb


def main(iargs=None):
    '''Compute baseline.
    '''
    inps=cmdLineParse(iargs)
    from isceobj.Planet.Planet import Planet
    import numpy as np
    import shelve

    baselineDir = os.path.dirname(inps.baselineFile)
    if baselineDir != '':
        os.makedirs(baselineDir, exist_ok=True)


    with shelve.open(os.path.join(inps.reference, 'data'), flag='r') as mdb:
         reference = mdb['frame'] 

    with shelve.open(os.path.join(inps.secondary, 'data'), flag='r') as mdb:
         secondary = mdb['frame']

    # check if the reference and secondary shelf are the same, i.e. it is baseline grid for the reference
    reference_SensingStart = reference.getSensingStart()
    secondary_SensingStart = secondary.getSensingStart()
    if reference_SensingStart==secondary_SensingStart:
        referenceBaseline = True
    else:
        referenceBaseline = False

    refElp = Planet(pname='Earth').ellipsoid

    dr = reference.instrument.rangePixelSize
    dt = 1./reference.PRF  #reference.azimuthTimeInterval

    mStartingRange =  reference.startingRange  #min([x.startingRange for x in referenceswaths])
    mFarRange = reference.startingRange + dr*(reference.numberOfSamples - 1)  #max([x.farRange for x in referenceswaths])
    mSensingStart =  reference.sensingStart #  min([x.sensingStart for x in referenceswaths])
    mSensingStop = reference.sensingStop    #max([x.sensingStop for x in referenceswaths])
    mOrb = getMergedOrbit(reference)

    nPixels = int(np.round( (mFarRange - mStartingRange)/dr)) + 1
    nLines = int(np.round( (mSensingStop - mSensingStart).total_seconds() / dt)) + 1

    sOrb = getMergedOrbit(secondary)

    rangeLimits = mFarRange - mStartingRange

    # To make sure that we have at least 30 points    
    nRange = int(np.max([30, int(np.ceil(rangeLimits/7000.))]))

    slantRange = mStartingRange + np.arange(nRange) * rangeLimits / (nRange - 1.0)

    azimuthLimits = (mSensingStop - mSensingStart).total_seconds()
    nAzimuth = int(np.max([30,int(np.ceil(azimuthLimits))]))
    azimuthTime = [mSensingStart + datetime.timedelta(seconds= x * azimuthLimits/(nAzimuth-1.0))  for x in range(nAzimuth)]

    
    Bperp = np.zeros((nAzimuth,nRange), dtype=np.float32)
    Bpar = np.zeros((nAzimuth,nRange), dtype=np.float32)
    
    fid = open(inps.baselineFile, 'wb')
    print('Baseline file {0} dims: {1}L x {2}P'.format(inps.baselineFile, nAzimuth, nRange))

    if referenceBaseline:
        Bperp = np.zeros((nAzimuth,nRange), dtype=np.float32)
        Bperp.tofile(fid)
    else:
        for ii, taz in enumerate(azimuthTime):

            referenceSV = mOrb.interpolate(taz, method='hermite')
            mxyz = np.array(referenceSV.getPosition())
            mvel = np.array(referenceSV.getVelocity())
            
            for jj, rng in enumerate(slantRange):
    
                target = mOrb.rdr2geo(taz, rng)
    
                targxyz = np.array(refElp.LLH(target[0], target[1], target[2]).ecef().tolist())
                slvTime,slvrng = sOrb.geo2rdr(target)
    
                secondarySV = sOrb.interpolateOrbit(slvTime, method='hermite')
    
                sxyz = np.array( secondarySV.getPosition())
    
                aa = np.linalg.norm(sxyz-mxyz)
                costheta = (rng*rng + aa*aa - slvrng*slvrng)/(2.*rng*aa)
    
                Bpar[ii,jj] = aa*costheta
    
                perp = aa * np.sqrt(1 - costheta*costheta)
                direction = np.sign(np.dot( np.cross(targxyz-mxyz, sxyz-mxyz), mvel))
                Bperp[ii,jj] = direction*perp
    
        Bperp.tofile(fid)
    fid.close()
   
    ####Write XML
    img = isceobj.createImage()
    img.setFilename( inps.baselineFile)
    img.bands = 1
    img.scheme = 'BIP'
    img.dataType = 'FLOAT'
    img.setWidth(nRange)
    img.setAccessMode('READ')
    img.setLength(nAzimuth)
    img.renderHdr()
    img.renderVRT()
    

    ###Create oversampled VRT file
    cmd = 'gdal_translate -of VRT -ot Float32 -r bilinear -outsize {xsize} {ysize} {infile}.vrt {infile}.full.vrt'.format(xsize=nPixels, ysize=nLines, infile=inps.baselineFile)

    status = os.system(cmd)
    if status:
        raise Exception('cmd: {0} Failed'.format(cmd))
            
if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

