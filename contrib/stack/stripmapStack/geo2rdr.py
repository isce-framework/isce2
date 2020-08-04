#!/usr/bin/env python3
import argparse
import isce
import isceobj
import numpy as np
import shelve
import os
import datetime 
from isceobj.Constants import SPEED_OF_LIGHT
from isceobj.Util.Poly2D import Poly2D

def createParser():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Create DEM simulation for merged images')
    parser.add_argument('-a','--alks', dest='alks', type=int, default=1,
            help = 'Number of azimuth looks')
    parser.add_argument('-r','--rlks', dest='rlks', type=int, default=1,
            help = 'Number of range looks')
    parser.add_argument('-m', '--reference', dest='reference', type=str, required=True,
            help = 'Dir with reference frame')
    parser.add_argument('-g', '--geom', dest='geom', type=str, default=None,
            help = 'Dir with geometry products')
    parser.add_argument('-s', '--secondary', dest='secondary', type=str, required=True,
            help = 'Dir with secondary frame')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, default=None,
            help='Output directory')
    parser.add_argument('-p', '--poly', dest='poly', type=str, default=None,
            help='Pickle file with polynomial fits')
    parser.add_argument('-n', '--native', dest='native', action='store_true',
            default=False, help='Use native doppler geometry')
    parser.add_argument('-l', '--legendre', dest='legendre', action='store_true',
            default=False, help='Use legendre polynomials for orbit interpolation')
    parser.add_argument('-useGPU', '--useGPU', dest='useGPU',action='store_true', default=False,
            help='Allow App to use GPU when available')
    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    #return parser.parse_args(args=iargs)

    inps =  parser.parse_args(args=iargs)

    if inps.reference.endswith('/'):
        inps.reference = inps.reference[:-1]

    if inps.secondary.endswith('/'):
        inps.secondary = inps.secondary[:-1]

    if inps.geom is None:
        inps.geom = 'geometry_' + os.path.basename(inps.reference)

    if inps.outdir is None:
        inps.outdir = os.path.join('coreg', os.path.basename(inps.secondary))

    return inps



def runGeo2rdrGPU(info,latImage, lonImage, demImage, outdir,
	dop=None, nativedop=False, legendre=False,
	azoff=0.0, rgoff=0.0,
	alks=1, rlks=1):

    from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr
    from isceobj.Planet.Planet import Planet
    from iscesys import DateTimeUtil as DTU

    # for GPU the images need to have been created
    latImage.createImage()
    lonImage.createImage()
    demImage.createImage()

    #####Run Geo2rdr
    planet = Planet(pname='Earth')
    grdr = PyGeo2rdr()

    grdr.setRangePixelSpacing(info.getInstrument().getRangePixelSize())
    grdr.setPRF(info.getInstrument().getPulseRepetitionFrequency())
    grdr.setRadarWavelength(info.getInstrument().getRadarWavelength())

    # setting the orbit information
    grdr.createOrbit(0, len(info.orbit.stateVectors.list))
    count = 0
    for sv in info.orbit.stateVectors.list:
        td = DTU.seconds_since_midnight(sv.getTime())
        pos = sv.getPosition()
        vel = sv.getVelocity()
    #    print("time " + str(td))
    #    print("pos " + str(pos))
    #    print("vel " + str(vel))
    #    print("")
        grdr.setOrbitVector(count, td, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])
        count += 1

    if legendre:
        print("Legendre requested")
        # see the include/Constants.h for the defined values
        grdr.setOrbitMethod(2)
    else:
        grdr.setOrbitMethod(0)


    grdr.setWidth(info.getImage().getWidth())
    grdr.setLength(info.getImage().getLength())

    grdr.setEllipsoidMajorSemiAxis(planet.ellipsoid.a)
    grdr.setEllipsoidEccentricitySquared(planet.ellipsoid.e2)

    ## TODO  Setting lookside in GPU mode
    ## lookside =  info.instrument.platform.pointingDirection

    prf = info.getInstrument().getPulseRepetitionFrequency()
    delta = datetime.timedelta(seconds = (azoff-(alks-1)/2)/prf)
    misreg_rg = (rgoff - (rlks-1)/2)*info.getInstrument().getRangePixelSize()
    print("Starting range: " + str(info.getStartingRange() - misreg_rg))
    print("Start sensing time: " + str(info.sensingStart - delta))
    print("PRF: " + str( prf))
    grdr.setSensingStart(DTU.seconds_since_midnight(info.sensingStart - delta))
    grdr.setRangeFirstSample(info.getStartingRange() - misreg_rg)

    grdr.setNumberRangeLooks(rlks)
    grdr.setNumberAzimuthLooks(alks)


    if nativedop and (dop is not None):
        try:
            coeffs = [x/prf for x in dop._coeffs]
        except:
            coeffs = [x/prf for x in dop]
       
        print('Native Doppler')
	# initialize the doppler polynomial
	# the object is defined as (poly_order,poly_mean,poly_norm);
        grdr.createPoly(len(coeffs)-1,0.,1.)
        index = 0
        for coeff in coeffs:
            grdr.setPolyCoeff(index, coeff)
            index += 1
    else:
        print('Zero doppler')
        grdr.createPoly(0, 0., 1.)
        grdr.setPolyCoeff(0, 0.)

    grdr.setDemLength(demImage.getLength())
    grdr.setDemWidth(demImage.getWidth())
    grdr.setBistaticFlag(0)

    print("")
    print(demImage.width)
    print("")


    rangeOffsetFILE  = os.path.join(outdir, 'range.off')
    rangeOffsetImage = isceobj.createImage()
    rangeOffsetImage.setFilename(rangeOffsetFILE)
    rangeOffsetImage.setAccessMode('write')
    rangeOffsetImage.setDataType('FLOAT')
    rangeOffsetImage.setCaster('write', 'DOUBLE')
    rangeOffsetImage.setWidth(demImage.width)
    rangeOffsetImage.createImage()


    azimuthOffsetFILE= os.path.join(outdir, 'azimuth.off')
    azimuthOffsetImage = isceobj.createImage()
    azimuthOffsetImage.setFilename(azimuthOffsetFILE)
    azimuthOffsetImage.setAccessMode('write')
    azimuthOffsetImage.setDataType('FLOAT')
    azimuthOffsetImage.setCaster('write', 'DOUBLE')
    azimuthOffsetImage.setWidth(demImage.width)
    azimuthOffsetImage.createImage()


    grdr.setLatAccessor(latImage.getImagePointer())
    grdr.setLonAccessor(lonImage.getImagePointer())
    grdr.setHgtAccessor(demImage.getImagePointer())
    grdr.setAzAccessor(0)
    grdr.setRgAccessor(0)
    grdr.setAzOffAccessor(azimuthOffsetImage.getImagePointer())
    grdr.setRgOffAccessor(rangeOffsetImage.getImagePointer())

    grdr.geo2rdr()

    rangeOffsetImage.finalizeImage()
    rangeOffsetImage.renderHdr()

    azimuthOffsetImage.finalizeImage()
    azimuthOffsetImage.renderHdr()
    latImage.finalizeImage()
    lonImage.finalizeImage()
    demImage.finalizeImage()

    return
    pass


def runGeo2rdrCPU(info, latImage, lonImage, demImage, outdir,
        dop=None, nativedop=False, legendre=False,
        azoff=0.0, rgoff=0.0,
        alks=1, rlks=1):
    from zerodop.geo2rdr import createGeo2rdr
    from isceobj.Planet.Planet import Planet

    #####Run Geo2rdr
    planet = Planet(pname='Earth')
    grdr = createGeo2rdr()
    grdr.configure()

    grdr.slantRangePixelSpacing = info.getInstrument().getRangePixelSize()
    grdr.prf = info.getInstrument().getPulseRepetitionFrequency()
    grdr.radarWavelength = info.getInstrument().getRadarWavelength()
    grdr.orbit = info.getOrbit()
    grdr.width = info.getImage().getWidth()
    grdr.length = info.getImage().getLength()
    grdr.wireInputPort(name='planet', object=planet)
    grdr.lookSide =  info.instrument.platform.pointingDirection

    print(info.sensingStart - datetime.timedelta(seconds = (azoff-(alks-1)/2)/grdr.prf))
    print(grdr.prf)
    print(info.getStartingRange() - (rgoff - (rlks-1)/2)*grdr.slantRangePixelSpacing)
    #print(stop)

    grdr.setSensingStart(info.sensingStart - datetime.timedelta(seconds = (azoff-(alks-1)/2)/grdr.prf))
    grdr.rangeFirstSample = info.getStartingRange() - (rgoff - (rlks-1)/2)*grdr.slantRangePixelSpacing
    grdr.numberRangeLooks = alks
    grdr.numberAzimuthLooks = rlks

    if nativedop and (dop is not None):
        try:
            coeffs = [x/grdr.prf for x in dop._coeffs]
        except:
            coeffs = [x/grdr.prf for x in dop]

        grdr.dopplerCentroidCoeffs = coeffs
    else:
        print('Zero doppler')
        grdr.dopplerCentroidCoeffs = [0.]

#####    grdr.fmrateCoeffs = [0.]		# DOES NOT LOOK to be defined

    grdr.rangeOffsetImageName = os.path.join(outdir, 'range.off')
    grdr.azimuthOffsetImageName= os.path.join(outdir, 'azimuth.off')
    grdr.demImage = demImage
    grdr.latImage = latImage
    grdr.lonImage = lonImage
    grdr.outputPrecision = 'DOUBLE'

    if legendre:
        grdr.orbitInterpolationMethod = 'LEGENDRE'

    grdr.geo2rdr()

    return



def main(iargs=None): 
    inps = cmdLineParse(iargs)
    print(inps.secondary)

    # see if the user compiled isce with GPU enabled
    run_GPU = False
    try:
        from zerodop.GPUtopozero.GPUtopozero import PyTopozero
        from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr
        run_GPU = True
    except:
        pass

    if inps.useGPU and not run_GPU:
        print("GPU mode requested but no GPU ISCE code found")


    # setting the respective version of geo2rdr for CPU and GPU
    if run_GPU and inps.useGPU:
        print('GPU mode')
        runGeo2rdr = runGeo2rdrGPU
    else:
        print('CPU mode')
        runGeo2rdr = runGeo2rdrCPU

    db = shelve.open( os.path.join(inps.secondary, 'data'), flag='r')
    print( os.path.join(inps.secondary, 'data'))
    frame = db['frame']
    try:
        dop = db['doppler']
    except:
        dop = frame._dopplerVsPixel

    db.close()

    ####Setup dem
    demImage = isceobj.createDemImage()
    demImage.load(os.path.join(inps.geom, 'hgt.rdr.xml'))
    demImage.setAccessMode('read')

    latImage = isceobj.createImage()
    latImage.load(os.path.join(inps.geom, 'lat.rdr.xml'))
    latImage.setAccessMode('read')

    lonImage = isceobj.createImage()
    lonImage.load(os.path.join(inps.geom, 'lon.rdr.xml'))
    lonImage.setAccessMode('read')

    os.makedirs(inps.outdir, exist_ok=True)

    
    azoff = 0.0
    rgoff = 0.0
    if inps.poly is not None:
        db1 = shelve.open(inps.poly, flag='r')
        azpoly = db1['azpoly']
        rgpoly = db1['rgpoly']
        db1.close()

        azoff = azpoly._coeffs[0][0]
        rgoff = rgpoly._coeffs[0][0]
        print('Azimuth line shift: ', azoff)
        print('Range pixel shift: ', rgoff)


    ####Setup input file
    runGeo2rdr(frame,latImage,lonImage,demImage, inps.outdir, 
            dop=dop, nativedop = inps.native, legendre=inps.legendre,
            azoff=azoff,rgoff=rgoff,
            alks=inps.alks, rlks=inps.rlks)


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

