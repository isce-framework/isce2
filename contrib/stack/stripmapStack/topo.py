#!/usr/bin/env python3

import os
import argparse
import shelve
import datetime
import shutil
import numpy as np
import isce
import isceobj
from isceobj.Constants import SPEED_OF_LIGHT
from isceobj.Util.Poly2D import Poly2D
from mroipac.looks.Looks import Looks

def createParser():
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
    parser.add_argument('-o', '--output', dest='outdir', type=str, required=True,
            help = 'Output directory')
    parser.add_argument('-n','--native', dest='nativedop', action='store_true',
            default=False, help='Products in native doppler geometry instead of zero doppler')
    parser.add_argument('-l','--legendre', dest='legendre', action='store_true',
            default=False, help='Use legendre interpolation instead of hermite')
    parser.add_argument('-useGPU', '--useGPU', dest='useGPU',action='store_true', default=False,
            help='Allow App to use GPU when available')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


class Dummy(object):
    pass



def runTopoGPU(info, demImage, dop=None, nativedop=False, legendre=False):

    from isceobj.Planet.Planet import Planet
    from zerodop.GPUtopozero.GPUtopozero import PyTopozero
    from isceobj import Constants as CN
    from isceobj.Util.Poly2D import Poly2D
    from iscesys import DateTimeUtil as DTU


    ## TODO GPU does not support shadow and layover and local inc file generation
    full = False


    if not os.path.isdir(info.outdir):
        os.makedirs(info.outdir)

    # define variables to be used later on
    r0 = info.rangeFirstSample + ((info.numberRangeLooks - 1)/2) * info.slantRangePixelSpacing
    tbef = info.sensingStart + datetime.timedelta(seconds = ((info.numberAzimuthLooks - 1) /2) / info.prf)
    pegHdg = np.radians(info.orbit.getENUHeading(tbef))
    width = info.width // info.numberRangeLooks
    length = info.length // info.numberAzimuthLooks
    dr = info.slantRangePixelSpacing*info.numberRangeLooks

    # output file names
    latFilename = info.latFilename
    lonFilename = info.lonFilename
    losFilename = info.losFilename
    heightFilename = info.heightFilename
    incFilename = info.incFilename
    maskFilename = info.maskFilename

    # orbit interpolator
    if legendre:
        omethod = 2 # LEGENDRE INTERPOLATION
    else:
        omethod = 0 # HERMITE INTERPOLATION

    # tracking doppler specifications
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
        polyDoppler = Poly2D(name='stripmapStack_dopplerPoly')
        polyDoppler.setWidth(width)
        polyDoppler.setLength(length)
        polyDoppler.setNormRange(1.0)
        polyDoppler.setNormAzimuth(1.0)
        polyDoppler.setMeanRange(0.0)
        polyDoppler.setMeanAzimuth(0.0)
        polyDoppler.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.0]])
        polyDoppler.createPoly2D()


    # dem
    demImage.setCaster('read','FLOAT')
    demImage.createImage()

    # slant range file
    slantRangeImage = Poly2D()
    slantRangeImage.setWidth(width)
    slantRangeImage.setLength(length)
    slantRangeImage.setNormRange(1.0)
    slantRangeImage.setNormAzimuth(1.0)
    slantRangeImage.setMeanRange(0.0)
    slantRangeImage.setMeanAzimuth(0.0)
    slantRangeImage.initPoly(rangeOrder=1,azimuthOrder=0, coeffs=[[r0,dr]])
    slantRangeImage.createPoly2D()

    # lat file
    latImage = isceobj.createImage()
    accessMode = 'write'
    dataType = 'DOUBLE'
    latImage.initImage(latFilename,accessMode,width,dataType)
    latImage.createImage()

    # lon file
    lonImage = isceobj.createImage()
    lonImage.initImage(lonFilename,accessMode,width,dataType)
    lonImage.createImage()

    # LOS file
    losImage = isceobj.createImage()
    dataType = 'FLOAT'
    bands = 2
    scheme = 'BIL'
    losImage.initImage(losFilename,accessMode,width,dataType,bands=bands,scheme=scheme)
    losImage.setCaster('write','DOUBLE')
    losImage.createImage()

    # height file
    heightImage = isceobj.createImage()
    dataType = 'DOUBLE'
    heightImage.initImage(heightFilename,accessMode,width,dataType)
    heightImage.createImage()

    # add inc and mask file if requested
    if full:
        incImage = isceobj.createImage()
        dataType = 'FLOAT'
        incImage.initImage(incFilename,accessMode,width,dataType,bands=bands,scheme=scheme)
        incImage.createImage()
        incImagePtr = incImage.getImagePointer()

        maskImage = isceobj.createImage()
        dataType = 'BYTE'
        bands = 1
        maskImage.initImage(maskFilename,accessMode,width,dataType,bands=bands,scheme=scheme)
        maskImage.createImage()
        maskImagePtr = maskImage.getImagePointer()
    else:
        incImagePtr = 0
        maskImagePtr = 0

    # initalize planet
    elp = Planet(pname='Earth').ellipsoid

    # initialize topo object and fill with parameters
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

    # run topo
    topo.runTopo()


    # close the written files and add description etc
    # lat file
    latImage.addDescription('Pixel-by-pixel latitude in degrees.')
    latImage.finalizeImage()
    latImage.renderHdr()

    # lon file
    lonImage.addDescription('Pixel-by-pixel longitude in degrees.')
    lonImage.finalizeImage()
    lonImage.renderHdr()

    # height file
    heightImage.addDescription('Pixel-by-pixel height in meters.')
    heightImage.finalizeImage()
    heightImage.renderHdr()

    # los file
    descr = '''Two channel Line-Of-Sight geometry image (all angles in degrees). Represents vector drawn from target to platform.
                Channel 1: Incidence angle measured from vertical at target (always +ve).
                Channel 2: Azimuth angle measured from North in Anti-clockwise direction.'''
    losImage.setImageType('bil')
    losImage.addDescription(descr)
    losImage.finalizeImage()
    losImage.renderHdr()

    # dem/ height file
    demImage.finalizeImage()

    # adding in additional files if requested
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


def runTopoCPU(info, demImage, dop=None,
        nativedop=False, legendre=False):
    from zerodop.topozero import createTopozero
    from isceobj.Planet.Planet import Planet

    if not os.path.isdir(info.outdir):
        os.makedirs(info.outdir)

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

    topo.latFilename = info.latFilename
    topo.lonFilename = info.lonFilename
    topo.losFilename = info.losFilename
    topo.heightFilename = info.heightFilename
    topo.incFilename = info.incFilename
    topo.maskFilename = info.maskFilename

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
    return


def runMultilook(in_dir, out_dir, alks, rlks):
    print('generate multilooked geometry files with alks={} and rlks={}'.format(alks, rlks))
    from iscesys.Parsers.FileParserFactory import createFileParser
    FP = createFileParser('xml')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print('create directory: {}'.format(out_dir))

    for fbase in ['hgt', 'incLocal', 'lat', 'lon', 'los', 'shadowMask', 'waterMask']:
        fname = '{}.rdr'.format(fbase)
        in_file = os.path.join(in_dir, fname)
        out_file = os.path.join(out_dir, fname)

        if os.path.isfile(in_file):
            xmlProp = FP.parse(in_file+'.xml')[0]
            if('image_type' in xmlProp and xmlProp['image_type'] == 'dem'):
                inImage = isceobj.createDemImage()
            else:
                inImage = isceobj.createImage()

            inImage.load(in_file+'.xml')
            inImage.filename = in_file

            lkObj = Looks()
            lkObj.setDownLooks(alks)
            lkObj.setAcrossLooks(rlks)
            lkObj.setInputImage(inImage)
            lkObj.setOutputFilename(out_file)
            lkObj.looks()

            # copy the full resolution xml/vrt file from ./merged/geom_master to ./geom_master
            # to facilitate the number of looks extraction
            # the file path inside .xml file is not, but should, updated
            shutil.copy(in_file+'.xml', out_file+'.full.xml')
            shutil.copy(in_file+'.vrt', out_file+'.full.vrt')

    return out_dir


def runMultilookGdal(in_dir, out_dir, alks, rlks, in_ext='.rdr', out_ext='.rdr',
                     fbase_list=['hgt', 'incLocal', 'lat', 'lon', 'los', 'shadowMask', 'waterMask']):
    print('generate multilooked geometry files with alks={} and rlks={}'.format(alks, rlks))
    import gdal

    # create 'geom_master' directory
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print('create directory: {}'.format(out_dir))

    # multilook files one by one
    for fbase in fbase_list:
        in_file = os.path.join(in_dir, '{}{}'.format(fbase, in_ext))
        out_file = os.path.join(out_dir, '{}{}'.format(fbase, out_ext))

        if os.path.isfile(in_file):
            ds = gdal.Open(in_file, gdal.GA_ReadOnly)
            in_wid = ds.RasterXSize
            in_len = ds.RasterYSize

            out_wid = int(in_wid / rlks)
            out_len = int(in_len / alks)
            src_wid = out_wid * rlks
            src_len = out_len * alks

            cmd = 'gdal_translate -of ENVI -a_nodata 0 -outsize {ox} {oy} '.format(ox=out_wid, oy=out_len)
            cmd += ' -srcwin 0 0 {sx} {sy} {fi} {fo} '.format(sx=src_wid, sy=src_len, fi=in_file, fo=out_file)
            print(cmd)
            os.system(cmd)

            # copy the full resolution xml/vrt file from ./merged/geom_master to ./geom_master
            # to facilitate the number of looks extraction
            # the file path inside .xml file is not, but should, updated
            if in_file != out_file+'.full':
                shutil.copy(in_file+'.xml', out_file+'.full.xml')
                shutil.copy(in_file+'.vrt', out_file+'.full.vrt')

    return out_dir


def extractInfo(frame, inps):
    '''
    Extract relevant information only.
    '''

    info = Dummy()

    ins = frame.getInstrument()

    info.sensingStart = frame.getSensingStart()

    info.lookSide = frame.instrument.platform.pointingDirection
    info.rangeFirstSample = frame.startingRange
    info.numberRangeLooks = 1 #inps.rlks
    info.numberAzimuthLooks = 1 #inps.alks

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


def main(iargs=None):

    inps = cmdLineParse(iargs)

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
        runTopo = runTopoGPU
    else:
        print('CPU mode')
        runTopo = runTopoCPU


    db = shelve.open(os.path.join(inps.master, 'data'))
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

    info = extractInfo(frame, inps)

    # define topo output names:
    info.latFilename = os.path.join(info.outdir, 'lat.rdr')
    info.lonFilename = os.path.join(info.outdir, 'lon.rdr')
    info.losFilename = os.path.join(info.outdir, 'los.rdr')
    info.heightFilename = os.path.join(info.outdir, 'hgt.rdr')
    info.incFilename = os.path.join(info.outdir, 'incLocal.rdr')
    info.maskFilename = os.path.join(info.outdir, 'shadowMask.rdr')

    runTopo(info,demImage,dop=doppler,nativedop=inps.nativedop, legendre=inps.legendre)
    runSimamp(os.path.dirname(info.heightFilename),os.path.basename(info.heightFilename))

    # write multilooked geometry files in "geom_master" directory, same level as "Igrams"
    if inps.rlks * inps.rlks > 1:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(info.outdir)), 'geom_master')
        runMultilookGdal(in_dir=info.outdir, out_dir=out_dir, alks=inps.alks, rlks=inps.rlks)
        #runMultilook(in_dir=info.outdir, out_dir=out_dir, alks=inps.alks, rlks=inps.rlks)

    return


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()
