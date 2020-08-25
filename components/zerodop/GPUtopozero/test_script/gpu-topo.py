#!/usr/bin/env python3
import argparse
import isce
import isceobj
import numpy as np
import shelve
import os
import datetime 
import isceobj.Image as IF
from isceobj.Constants import SPEED_OF_LIGHT
from isceobj.Util.Poly2D import Poly2D
from iscesys import DateTimeUtil as DTU
from iscesys.Component.ProductManager import ProductManager

import gpu_topozero

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
    parser.add_argument('-m', '--reference', dest='reference', type=str, required=True,
            help = 'Dir with reference frame')
    parser.add_argument('-o', '--output', dest='outdir', type=str, required=True,
            help = 'Output directory')
    parser.add_argument('-n','--native', dest='nativedop', action='store_true',
            default=False, help='Products in native doppler geometry instead of zero doppler')
    parser.add_argument('-l','--legendre', dest='legendre', action='store_true',
            default=False, help='Use legendre interpolation instead of hermite')
    parser.add_argument('-f', '--full', dest='full', action='store_true',
            default=False, help='Generate all topo products - masks etc')

    parser.add_argument('-s', '--sentinel', dest='sntl1a', action='store_true',
            default=False, help='Designate input as Sentinel data')

    return parser.parse_args()

class Dummy(object):
    pass

def runGPUTopo(info, demImage, dop=None, nativedop=False, legendre=False, full=False):
    from zerodop.topozero import createTopozero
    from isceobj.Planet.Planet import Planet
    from gpu_topozero import PyTopozero
    from isceobj import Constants as CN

    if not os.path.isdir(info.outdir):
        os.mkdir(info.outdir)

    # Random vars
    r0 = info.rangeFirstSample + ((info.numberRangeLooks - 1)/2) * info.slantRangePixelSpacing
    tbef = info.sensingStart + datetime.timedelta(seconds = ((info.numberAzimuthLooks - 1) /2) / info.prf)
    pegHdg = np.radians(info.orbit.getENUHeading(tbef))
    width = info.width // info.numberRangeLooks
    length = info.length // info.numberAzimuthLooks
    dr = info.slantRangePixelSpacing*info.numberRangeLooks
    if legendre:
        omethod = 2 # LEGENDRE INTERPOLATION
    else:
        omethod = 0 # HERMITE INTERPOLATION
    # Images
    demImage.setCaster('read','FLOAT')
    demImage.createImage()
    
    if nativedop and (dop is not None):
        try:
            coeffs = dop._coeffs
        except:
            coeffs = dop
        polyDoppler = Poly2D()
        polyDoppler.setWidth(width)
        polyDoppler.setLength(length)
        polyDoppler.initPoly(rangeOrder = len(coeffs)-1, azimuthOrder=0, coeffs=[coeffs])
    else:
        print('Zero doppler')
        #doppler = None
        polyDoppler = Poly2D(name=frame.name+'_dopplerPoly')
        polyDoppler.setWidth(width)
        polyDoppler.setLength(length)
        polyDoppler.setNormRange(1.0)
        polyDoppler.setNormAzimuth(1.0)
        polyDoppler.setMeanRange(0.0)
        polyDoppler.setMeanAzimuth(0.0)
        polyDoppler.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.0]])
    polyDoppler.createPoly2D()

    slantRangeImage = Poly2D()
    slantRangeImage.setWidth(width)
    slantRangeImage.setLength(length)
    slantRangeImage.setNormRange(1.0)
    slantRangeImage.setNormAzimuth(1.0)
    slantRangeImage.setMeanRange(0.0)
    slantRangeImage.setMeanAzimuth(0.0)
    slantRangeImage.initPoly(rangeOrder=1,azimuthOrder=0, coeffs=[[r0,dr]])
    slantRangeImage.createPoly2D()

    latImage = IF.createImage()
    accessMode = 'write'
    dataType = 'DOUBLE'
    latImage.initImage(os.path.join(info.outdir, 'lat.rdr'),accessMode,width,dataType)
    latImage.createImage()

    lonImage = IF.createImage()
    lonImage.initImage(os.path.join(info.outdir, 'lon.rdr'),accessMode,width,dataType)
    lonImage.createImage()

    losImage = IF.createImage()
    dataType = 'FLOAT'
    bands = 2
    scheme = 'BIL'
    losImage.initImage(os.path.join(info.outdir, 'los.rdr'),accessMode,width,dataType,bands=bands,scheme=scheme)
    losImage.setCaster('write','DOUBLE')
    losImage.createImage()

    heightImage = IF.createImage()
    dataType = 'DOUBLE'
    heightImage.initImage(os.path.join(info.outdir, 'z.rdr'),accessMode,width,dataType)
    heightImage.createImage()

    if full:
        incImage = IF.createImage()
        dataType = 'FLOAT'
        incImage.initImage(os.path.join(info.outdir, 'inc.rdr'),accessMode,width,dataType,bands=bands,scheme=scheme)
        incImage.createImage()
        incImagePtr = incImage.getImagePointer()

        maskImage = IF.createImage()
        dataType = 'BYTE'
        bands = 1
        maskImage.initImage(os.path.join(info.outdir, 'mask.rdr'),accessMode,width,dataType,bands=bands,scheme=scheme)
        maskImage.createImage()
        maskImagePtr = maskImage.getImagePointer()
    else:
        incImagePtr = 0
        maskImagePtr = 0

    elp = Planet(pname='Earth').ellipsoid

    topo = PyTopozero()
    topo.set_firstlat(demImage.getFirstLatitude())
    topo.set_firstlon(demImage.getFirstLongitude())
    topo.set_deltalat(demImage.getDeltaLatitude())
    topo.set_deltalon(demImage.getDeltaLongitude())
    topo.set_major(elp.a)
    topo.set_eccentricitySquared(elp.e2)
    topo.set_rSpace(info.slantRangePixelSpacing)
    topo.set_r0(r0)
    topo.set_pegHdg(pegHdg)
    topo.set_prf(info.prf)
    topo.set_t0(DTU.seconds_since_midnight(tbef))
    topo.set_wvl(info.radarWavelength)
    topo.set_thresh(.05)
    topo.set_demAccessor(demImage.getImagePointer())
    topo.set_dopAccessor(polyDoppler.getPointer())
    topo.set_slrngAccessor(slantRangeImage.getPointer())
    topo.set_latAccessor(latImage.getImagePointer())
    topo.set_lonAccessor(lonImage.getImagePointer())
    topo.set_losAccessor(losImage.getImagePointer())
    topo.set_heightAccessor(heightImage.getImagePointer())
    topo.set_incAccessor(incImagePtr)
    topo.set_maskAccessor(maskImagePtr)
    topo.set_numIter(25)
    topo.set_idemWidth(demImage.getWidth())
    topo.set_idemLength(demImage.getLength())
    topo.set_ilrl(info.lookSide)
    topo.set_extraIter(10)
    topo.set_length(length)
    topo.set_width(width)
    topo.set_nRngLooks(info.numberRangeLooks)
    topo.set_nAzLooks(info.numberAzimuthLooks)
    topo.set_demMethod(5) # BIQUINTIC METHOD
    topo.set_orbitMethod(omethod)
    # Need to simplify orbit stuff later
    nvecs = len(info.orbit.stateVectors.list)
    topo.set_orbitNvecs(nvecs)
    topo.set_orbitBasis(1) # Is this ever different?
    topo.createOrbit() # Initializes the empty orbit to the right allocated size
    count = 0
    for sv in info.orbit.stateVectors.list:
        td = DTU.seconds_since_midnight(sv.getTime())
        pos = sv.getPosition()
        vel = sv.getVelocity()
        topo.set_orbitVector(count,td,pos[0],pos[1],pos[2],vel[0],vel[1],vel[2])
        count += 1

    topo.runTopo()

    latImage.addDescription('Pixel-by-pixel latitude in degrees.')
    latImage.finalizeImage()
    latImage.renderHdr()

    lonImage.addDescription('Pixel-by-pixel longitude in degrees.')
    lonImage.finalizeImage()
    lonImage.renderHdr()

    heightImage.addDescription('Pixel-by-pixel height in meters.')
    heightImage.finalizeImage()
    heightImage.renderHdr()

    descr = '''Two channel Line-Of-Sight geometry image (all angles in degrees). Represents vector drawn from target to platform.
            Channel 1: Incidence angle measured from vertical at target (always +ve).
            Channel 2: Azimuth angle measured from North in Anti-clockwise direction.'''
    losImage.setImageType('bil')
    losImage.addDescription(descr)
    losImage.finalizeImage()
    losImage.renderHdr()

    demImage.finalizeImage()

    if full:
        descr = '''Two channel angle file.
                Channel 1: Angle between ray to target and the vertical at the sensor
                Channel 2: Local incidence angle accounting for DEM slope at target'''
        incImage.addDescription(descr)
        incImage.finalizeImage()
        incImage.renderHdr()

        descr = 'Radar shadow-layover mask. 1 - Radar Shadow. 2 - Radar Layover. 3 - Both.'
        maskImage.addDescription(descr)
        maskImage.finalizeImage()
        maskImage.renderHdr()
    if slantRangeImage:
        try:
            slantRangeImage.finalizeImage()
        except:
            pass

def runTopo(info, demImage, dop=None, 
        nativedop=False, legendre=False, full=False):
    from zerodop.topozero import createTopozero
    from isceobj.Planet.Planet import Planet

    if not os.path.isdir(info.outdir):
        os.mkdir(info.outdir)
    
    #####Run Topo
    planet = Planet(pname='Earth')
    topo = createTopozero()
    topo.slantRangePixelSpacing = info.slantRangePixelSpacing
    topo.prf = info.prf
    topo.radarWavelength = info.radarWavelength
    topo.orbit = info.orbit
    topo.width = info.width // info.numberRangeLooks
    topo.length = info.length //info.numberAzimuthLooks
    topo.wireInputPort(name='dem', object=demImage)
    topo.wireInputPort(name='planet', object=planet)
    topo.numberRangeLooks = info.numberRangeLooks
    topo.numberAzimuthLooks = info.numberAzimuthLooks
    topo.lookSide = info.lookSide
    topo.sensingStart = info.sensingStart + datetime.timedelta(seconds = ((info.numberAzimuthLooks - 1) /2) / info.prf) 
    topo.rangeFirstSample = info.rangeFirstSample + ((info.numberRangeLooks - 1)/2) * info.slantRangePixelSpacing

    topo.demInterpolationMethod='BIQUINTIC'
    if legendre:
        topo.orbitInterpolationMethod = 'LEGENDRE'

    topo.latFilename = os.path.join(info.outdir, 'lat.rdr')
    topo.lonFilename = os.path.join(info.outdir, 'lon.rdr')
    topo.losFilename = os.path.join(info.outdir, 'los.rdr')
    topo.heightFilename = os.path.join(info.outdir, 'z.rdr')
    if full:
        topo.incFilename = os.path.join(info.outdir, 'inc.rdr')
        topo.maskFilename = os.path.join(info.outdir, 'mask.rdr')

    if nativedop and (dop is not None):

        try:
            coeffs = dop._coeffs
        except:
            coeffs = dop

        doppler = Poly2D()
        doppler.setWidth(info.width // info.numberRangeLooks)
        doppler.setLength(info.length // info.numberAzimuthLooks)
        doppler.initPoly(rangeOrder = len(coeffs)-1, azimuthOrder=0, coeffs=[coeffs])
    else:
        print('Zero doppler')
        doppler = None

    topo.polyDoppler = doppler

    topo.topo()
    return

def runSimamp(outdir, hname='z.rdr'):
    from iscesys.StdOEL.StdOELPy import create_writer
    
    #####Run simamp
    stdWriter = create_writer("log","",True,filename='sim.log')
    objShade = isceobj.createSimamplitude()
    objShade.setStdWriter(stdWriter)


    hgtImage = isceobj.createImage()
    hgtImage.load(os.path.join(outdir, hname) + '.xml')
    hgtImage.setAccessMode('read')
    hgtImage.createImage()

    simImage = isceobj.createImage()
    simImage.setFilename(os.path.join(outdir, 'simamp.rdr'))
    simImage.dataType = 'FLOAT'
    simImage.setAccessMode('write')
    simImage.setWidth(hgtImage.getWidth())
    simImage.createImage()

    objShade.simamplitude(hgtImage, simImage, shade=3.0)

    simImage.renderHdr()
    hgtImage.finalizeImage()
    simImage.finalizeImage()


def extractInfo(frame, inps):
    '''
    Extract relevant information only.
    '''

    info = Dummy()

    ins = frame.getInstrument()

    info.sensingStart = frame.getSensingStart()

    info.lookSide = frame.instrument.platform.pointingDirection
    info.rangeFirstSample = frame.startingRange
    info.numberRangeLooks = inps.rlks
    info.numberAzimuthLooks = inps.alks

    fsamp = frame.rangeSamplingRate

    info.slantRangePixelSpacing = 0.5 * SPEED_OF_LIGHT / fsamp
    info.prf = frame.PRF
    info.radarWavelength = frame.radarWavelegth
    info.orbit = frame.getOrbit()
    
    info.width = frame.getNumberOfSamples() 
    info.length = frame.getNumberOfLines() 

    info.sensingStop = frame.getSensingStop()
    info.outdir = inps.outdir

    return info

def extractInfoFromS1A(frame, inps):
    '''
    Extract relevant information only.
    '''

    info = Dummy()

    info.sensingStart = frame.bursts[0].sensingStart
    info.lookSide = -1
    info.rangeFirstSample = frame.bursts[0].startingRange
    info.numberRangeLooks = inps.rlks
    info.numberAzimuthLooks = inps.alks

    info.slantRangePixelSpacing = frame.bursts[0].rangePixelSize
    info.prf = 1. / frame.bursts[0].azimuthTimeInterval
    info.radarWavelength = frame.bursts[0].radarWavelength
    info.orbit = frame.bursts[0].orbit

    info.width = frame.bursts[0].numberOfSamples * 3
    info.length = frame.bursts[0].numberOfLines * len(frame.bursts)

    info.sensingStop = frame.bursts[-1].sensingStop
    info.outdir = inps.outdir

    return info

if __name__ == '__main__':

    
    inps = cmdLineParse()
    
    if (inps.sntl1a):
        pm = ProductManager()
        pm.configure()
        frame = pm.loadProduct(inps.reference)
        doppler = [0.]
    else:
        db = shelve.open(os.path.join(inps.reference, 'data'))
        frame = db['frame']
        
        try:
            doppler = db['doppler']
        except:
            doppler = frame._dopplerVsPixel

        db.close()

    ####Setup dem
    demImage = isceobj.createDemImage()
    demImage.load(inps.dem + '.xml')
    demImage.setAccessMode('read')

    if (inps.sntl1a):
        info = extractInfoFromS1A(frame,inps)
    else:
        info = extractInfo(frame, inps)
    # To revert: delete 'GPU' from method call name
    runGPUTopo(info,demImage,dop=doppler,
            nativedop=inps.nativedop, legendre=inps.legendre,
            full=inps.full)
    #runSimamp(inps.outdir)




