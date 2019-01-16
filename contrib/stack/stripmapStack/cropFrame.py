#!/usr/bin/env python3

import isce
import numpy as np
import shelve
import isceobj
import copy
import argparse
import datetime
import os
from isceobj.Util.decorators import use_api
from imageMath import IML
import logging


def createParser():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser(description='Focus from raw data to slc')
    parser.add_argument('-i','--input', dest='indir', type=str, required=True,
            help='Input directory with raw/slc file')
    parser.add_argument('-b', '--box' ,dest='bbox', type=float, nargs=4, default=None,
            help='Bbox (SNWE in degrees)')
    parser.add_argument('-B', '--box_str' ,dest='bbox_str', type=str, default=None,
            help='Bbox (SNWE in degrees)')
    parser.add_argument('-o', '--output', dest='outdir', type=str, required=True,
            help='Output directory for the cropped raw / slc file')
    parser.add_argument('-r', '--raw', dest='israw', action='store_true', default=False,
            help='Explicitly crop the raw file only when both raw/slc are found in same directory.')
    parser.add_argument('-n', '--native', dest='isnative', action='store_true', default=False,
            help='Explicitly use native doppler coordinate system')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    inps = parser.parse_args(args=iargs)
    if inps.bbox is None:
       if inps.bbox_str:
          inps.bbox = [float(i) for i in inps.bbox_str.split()]
       else:
          raise Exception('either --bbox (a list of 4 float values) or --bbox_str (one string) is required. None found!')
    return inps

#####Helper functions for geobox manipulation
def geoboxToAzrgbox(frame, geobox, israw=False, isnative=False, margin=0.02):
    '''
    Convert a geo bounding box - SNWE to pixel limits.
    '''
    from isceobj.Util.Poly2D import Poly2D 
    from isceobj.Planet.Planet import Planet
    from isceobj.Constants import SPEED_OF_LIGHT


    rgs = []
    azs = []
    zrange = [-500. , 9000.]
    combos = [ [geobox[0]-margin, geobox[2]-margin],
               [geobox[0]-margin, geobox[3]+margin],
               [geobox[1]+margin, geobox[3]-margin],
               [geobox[1]+margin, geobox[2]+margin] ]

    if len(zrange) == 0:
        zrange = [-500., 9000.]   #Should work for earth


    lookSide = frame.instrument.platform.pointingDirection
    planet = Planet(pname='Earth')
    wvl = frame.instrument.getRadarWavelength()

    if (isnative or israw):
        ####If geometry is in native doppler / raw 
        ####You need doppler as a function of range to do
        ####geometry mapping correctly
        ###Currently doppler is saved as function of pixel number - old ROIPAC style
        ###Transform to function of slant range
        coeff = frame._dopplerVsPixel
        doppler = Poly2D()
        doppler._meanRange = frame.startingRange
        doppler._normRange = frame.instrument.rangePixelSize
        doppler.initPoly(azimuthOrder=0, rangeOrder=len(coeff)-1, coeffs=[coeff])
    else:
        ###Zero doppler system
        doppler = Poly2D()
        doppler.initPoly(azimuthOrder=0, rangeOrder=0, coeffs=[[0.]])

         
    ####Do 
    for z in zrange:
        for combo in combos:
            try:
                taz, rgm = frame.orbit.geo2rdr(combo + [z], side=lookSide,
                                               doppler=doppler, wvl=wvl)
                azs.append(taz)
                rgs.append(rgm)
            except:
                pass

    if len(azs) <= 1:
        raise Exception('Could not map geobbox coordinates to image')

    azrgbox = [np.min(azs), np.max(azs), np.min(rgs), np.max(rgs)]

    if israw:
        ####If cropping raw product, need to add an aperture length in range and azimuth

        ###Extra slant range at near and far range due to the uncompressed pulse
        deltaRg = np.abs(frame.instrument.pulseLength * SPEED_OF_LIGHT/2.0)
        print('RAW data - adding range aperture (in m) : ', deltaRg)
        azrgbox[2] -= deltaRg
        azrgbox[3] += deltaRg


        ###Extra azimuth samples at far range
        elp =copy.copy( planet.ellipsoid)
        svmid = frame.orbit.interpolateOrbit(frame.sensingMid, method='hermite') 
        xyz = svmid.getPosition()
        vxyz = svmid.getVelocity()
        llh = elp.xyz_to_llh(xyz)

        heading = frame.orbit.getENUHeading(frame.sensingMid)
        print('Heading: ', heading)

        elp.setSCH(llh[0], llh[1], heading)
        sch, schvel = elp.xyzdot_to_schdot(xyz, vxyz)
        vel = np.linalg.norm(schvel)
        synthAperture = np.abs(wvl* azrgbox[3]/(frame.instrument.platform.antennaLength*vel))
        deltaAz = datetime.timedelta(seconds=synthAperture)
        
        print('RAW data - adding azimuth aperture (in s) : ', synthAperture)
        azrgbox[0] -= deltaAz
        azrgbox[1] += deltaAz

    return azrgbox


def cropFrame(frame, limits, outdir, israw=False):
    '''
    Crop the frame.

    Parameters to change:
    startingRange
    farRange
    sensingStart
    sensingStop
    sensingMid
    numberOfLines
    numberOfSamples
    dopplerVsPixel
    '''

    outframe = copy.deepcopy(frame)

    
    if israw:
        factor = 2
    else:
        factor = 1

    ####sensing start
    ymin = np.floor( (limits[0] - frame.sensingStart).total_seconds() * frame.PRF)
    print('Line start: ', ymin)
    ymin = np.int( np.clip(ymin, 0, frame.numberOfLines-1))


    ####sensing stop 
    ymax = np.ceil( (limits[1] - frame.sensingStart).total_seconds() * frame.PRF) + 1
    print('Line stop: ', ymax)
    ymax = np.int( np.clip(ymax, 1, frame.numberOfLines)) 

    print('Line limits: ', ymin, ymax)
    print('Original Line Limits: ', 0, frame.numberOfLines)

    if (ymax-ymin) <= 1:
        raise Exception('Azimuth limits appear to not overlap with the scene')


    outframe.sensingStart = frame.sensingStart + datetime.timedelta(seconds = ymin/frame.PRF)
    outframe.numberOfLines = ymax - ymin
    outframe.sensingStop = frame.sensingStop + datetime.timedelta(seconds = (ymax-1)/frame.PRF)
    outframe.sensingMid = outframe.sensingStart + 0.5 * (outframe.sensingStop - outframe.sensingStart)


    ####starting range
    xmin = np.floor( (limits[2] - frame.startingRange)/frame.instrument.rangePixelSize)
    print('Pixel start: ', xmin)
    xmin = np.int(np.clip(xmin, 0, (frame.image.width//factor)-1))

    ####far range
    xmax = np.ceil( (limits[3] - frame.startingRange)/frame.instrument.rangePixelSize)+1
    print('Pixel stop: ', xmax)

    xmax = np.int(np.clip(xmax, 1, frame.image.width//factor))

    print('Pixel limits: ', xmin, xmax)
    print('Original Pixel Limits: ', 0, frame.image.width//factor)

    if (xmax - xmin) <= 1:
        raise Exception('Range limits appear to not overlap with the scene')

    outframe.startingRange = frame.startingRange + xmin * frame.instrument.rangePixelSize
    outframe.numberOfSamples = (xmax - xmin) * factor
    outframe.setFarRange( frame.startingRange + (xmax-xmin-1) * frame.instrument.rangePixelSize)


    ####Adjust Doppler centroid coefficients
    coeff = frame._dopplerVsPixel
    rng = np.linspace(xmin, xmax, len(coeff) + 1)
    dops = np.polyval(coeff[::-1], rng)

    rng = rng - xmin ###Adjust the start
    pol = np.polyfit(rng, dops, len(coeff)-1)
    outframe._dopplerVsPixel = list(pol[::-1])

    
    ####Adjusting the image now
    ####Can potentially use israw to apply more logic but better to use new version
    if frame.image.xmin != 0 :
        raise Exception('Looks like you are still using an old version of ISCE. The new version completely strips out the header bytes. Please switch to the latest ...')

    
    inname = frame.image.filename
    suffix = os.path.splitext(inname)[1]
    outname = os.path.join(outdir, os.path.basename(inname)) #+ suffix
    outdirname = os.path.dirname(outname)

    if os.path.isdir(outdirname):
        print('Output directory already exists. Will be overwritten ....')
    else:
        os.makedirs(outdirname)
        print('Creating {0} directory.'.format(outdirname))

    indata = IML.mmapFromISCE(inname, logging)
    indata.bands[0][ymin:ymax,xmin*factor:xmax*factor].tofile(outname)

    indata = None
    outframe.image.filename = outname
    outframe.image.width = outframe.numberOfSamples
    outframe.image.length = outframe.numberOfLines

    outframe.image.xmax = outframe.numberOfSamples
    outframe.image.coord1.coordSize = outframe.numberOfSamples
    outframe.image.coord1.coordEnd = outframe.numberOfSamples
    outframe.image.coord2.coordSize = outframe.numberOfLines
    outframe.image.coord2.coordEnd = outframe.numberOfLines

    outframe.image.renderHdr()

    return outframe

def main(iargs=None):

    ####Parse command line
    inps = cmdLineParse(iargs)


    slcFound = False
    rawFound = False

    ###Try to find raw product
    try:
        ####Load raw parameters
        with shelve.open(os.path.join(inps.indir, 'raw')) as db:
            rawFrame = db['frame']
        rawFound = True
    except:
        pass
    

    ###Try to find slc product
    try:
        ####Load slc parameters
        with shelve.open(os.path.join(inps.indir, 'data')) as db:
            slcFrame = db['frame']
        slcFound = True
    except:
        pass

    if inps.israw and (not rawFound):
        raise Exception('Explicit cropping of raw product requested but raw product not found ....')

    if inps.israw:
        frame = rawFrame
    else:
        frame = slcFrame


    #####Determine azimuth and range limits
    limits =  geoboxToAzrgbox(frame, inps.bbox, 
                israw=inps.israw, isnative=inps.isnative)


    #####Crop the actual frame with limits
    outFrame = cropFrame(frame, limits, inps.outdir, israw=inps.israw)

    #####Save product to shelve
    if inps.israw:
        pickname = os.path.join(inps.outdir, 'raw')
    else:
        pickname = os.path.join(inps.outdir, 'data')

    with shelve.open(pickname) as db:
        db['frame'] = outFrame

if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()


