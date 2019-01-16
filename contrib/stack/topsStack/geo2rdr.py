#!/usr/bin/env python3

import numpy as np
import argparse
import os
import isce
import isceobj
import datetime
import sys
import s1a_isce_utils as ut

def createParser():
    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-m', '--master', type=str, dest='master', required=True,
            help='Directory with the master image')
    parser.add_argument('-s', '--slave', type=str, dest='slave', required=True,
            help='Directory with the slave image')
    parser.add_argument('-g', '--geom_masterDir', type=str, dest='geom_masterDir', default='geom_master',
            help='Directory for geometry files of the master')
    parser.add_argument('-c', '--coregSLCdir', type=str, dest='coregdir', default='coreg_slaves',
            help='Directory with coregistered SLC data')
    parser.add_argument('-a', '--azimuth_misreg', type=str, dest='misreg_az', default='',
            help='A text file that contains zimuth misregistration in subpixels')
    parser.add_argument('-r', '--range_misreg', type=str, dest='misreg_rng', default='',
            help='A text file that contains range misregistration in meters')
    parser.add_argument('-v', '--overlap', dest='overlap', action='store_true', default=False,
            help='Flatten the interferograms with offsets if needed')
    parser.add_argument('-o', '--overlap_dir', type=str, dest='overlapDir', default='overlap',
            help='overlap directory name')
    parser.add_argument('-useGPU', '--useGPU', dest='useGPU',action='store_true', default=False,
            help='Allow App to use GPU when available')
    return parser

def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''
    parser = createParser()
    return parser.parse_args(args=iargs)

def runGeo2rdrCPU(info, rdict, misreg_az=0.0, misreg_rg=0.0):
    from zerodop.geo2rdr import createGeo2rdr
    from isceobj.Planet.Planet import Planet

    latImage = isceobj.createImage()
    latImage.load(rdict['lat'] + '.xml')
    latImage.setAccessMode('READ')

    lonImage = isceobj.createImage()
    lonImage.load(rdict['lon'] + '.xml')
    lonImage.setAccessMode('READ')

    demImage = isceobj.createImage()
    demImage.load(rdict['hgt'] + '.xml')
    demImage.setAccessMode('READ')

    misreg_az = misreg_az*info.azimuthTimeInterval
    delta = datetime.timedelta(seconds=misreg_az)
    print('Additional time offset applied in geo2rdr: {0} secs'.format(misreg_az))
    print('Additional range offset applied in geo2rdr: {0} m'.format(misreg_rg))


    #####Run Geo2rdr
    planet = Planet(pname='Earth')
    grdr = createGeo2rdr()
    grdr.configure()

    grdr.slantRangePixelSpacing = info.rangePixelSize
    grdr.prf = 1.0 / info.azimuthTimeInterval
    grdr.radarWavelength = info.radarWavelength
    grdr.orbit = info.orbit
    grdr.width = info.numberOfSamples
    grdr.length = info.numberOfLines
    grdr.demLength = demImage.getLength()
    grdr.demWidth = demImage.getWidth()
    grdr.wireInputPort(name='planet', object=planet)
    grdr.numberRangeLooks = 1
    grdr.numberAzimuthLooks = 1
    grdr.lookSide = -1
    grdr.setSensingStart(info.sensingStart - delta)
    grdr.rangeFirstSample = info.startingRange - misreg_rg
    grdr.dopplerCentroidCoeffs = [0.]  ###Zero doppler

    grdr.rangeOffsetImageName = rdict['rangeOffName']
    grdr.azimuthOffsetImageName = rdict['azOffName']
    grdr.demImage = demImage
    grdr.latImage = latImage
    grdr.lonImage = lonImage

    grdr.geo2rdr()

    return


def runGeo2rdrGPU(info, rdict, misreg_az=0.0, misreg_rg=0.0):
    from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr
    from isceobj.Planet.Planet import Planet
    from iscesys import DateTimeUtil as DTU
    
    latImage = isceobj.createImage()
    latImage.load(rdict['lat'] + '.xml')
    latImage.setAccessMode('READ')
    latImage.createImage()
    
    lonImage = isceobj.createImage()
    lonImage.load(rdict['lon'] + '.xml')
    lonImage.setAccessMode('READ')
    lonImage.createImage()

    demImage = isceobj.createImage()
    demImage.load(rdict['hgt'] + '.xml')
    demImage.setAccessMode('READ')
    demImage.createImage()

    misreg_az = misreg_az*info.azimuthTimeInterval
    delta = datetime.timedelta(seconds=misreg_az)
    print('Additional time offset applied in geo2rdr: {0} secs'.format(misreg_az))
    print('Additional range offset applied in geo2rdr: {0} m'.format(misreg_rg))    

    #####Run Geo2rdr
    planet = Planet(pname='Earth')
    grdr = PyGeo2rdr()
    
    grdr.setRangePixelSpacing(info.rangePixelSize)
    grdr.setPRF(1.0 / info.azimuthTimeInterval)
    grdr.setRadarWavelength(info.radarWavelength)

    grdr.createOrbit(0, len(info.orbit.stateVectors.list))
    count = 0
    for sv in info.orbit.stateVectors.list:
        td = DTU.seconds_since_midnight(sv.getTime())
        pos = sv.getPosition()
        vel = sv.getVelocity()

        grdr.setOrbitVector(count, td, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])
        count += 1
    
    grdr.setOrbitMethod(0)
    grdr.setWidth(info.numberOfSamples)
    grdr.setLength(info.numberOfLines)
    grdr.setSensingStart(DTU.seconds_since_midnight(info.sensingStart -delta))
    grdr.setRangeFirstSample(info.startingRange - misreg_rg)
    grdr.setNumberRangeLooks(1)
    grdr.setNumberAzimuthLooks(1)
    grdr.setEllipsoidMajorSemiAxis(planet.ellipsoid.a)
    grdr.setEllipsoidEccentricitySquared(planet.ellipsoid.e2)
    grdr.createPoly(0, 0., 1.)
    grdr.setPolyCoeff(0, 0.)
    
    grdr.setDemLength(demImage.getLength())
    grdr.setDemWidth(demImage.getWidth())
    grdr.setBistaticFlag(0)
    
    rangeOffsetImage = isceobj.createImage()
    rangeOffsetImage.setFilename(rdict['rangeOffName'])
    rangeOffsetImage.setAccessMode('write')
    rangeOffsetImage.setDataType('FLOAT')
    rangeOffsetImage.setCaster('write', 'DOUBLE')
    rangeOffsetImage.setWidth(demImage.width)
    rangeOffsetImage.createImage()
    
    azimuthOffsetImage = isceobj.createImage()
    azimuthOffsetImage.setFilename(rdict['azOffName'])
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

def main(iargs=None):
    '''
    Estimate offsets for the overlap regions of the bursts.
    '''
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
        runGeo2rdr = runGeo2rdrGPU
    else:
        print('CPU mode')
        runGeo2rdr = runGeo2rdrCPU
    

    masterSwathList = ut.getSwathList(inps.master)
    slaveSwathList = ut.getSwathList(inps.slave)

    swathList = list(sorted(set(masterSwathList+slaveSwathList)))

    for swath in swathList:
        ##Load slave metadata
        slave = ut.loadProduct(os.path.join(inps.slave, 'IW{0}.xml'.format(swath)))
        master = ut.loadProduct(os.path.join(inps.master, 'IW{0}.xml'.format(swath)))
    
        ### output directory
        if inps.overlap:
            outdir = os.path.join(inps.coregdir, inps.overlapDir, 'IW{0}'.format(swath))
        else:
            outdir = os.path.join(inps.coregdir, 'IW{0}'.format(swath))

        if not os.path.isdir(outdir):
           os.makedirs(outdir)

        if os.path.exists(str(inps.misreg_az)):
            with open(inps.misreg_az, 'r') as f:
                misreg_az = float(f.readline())
        else:
            misreg_az = 0.0
        
        if os.path.exists(str(inps.misreg_rng)):
            with open(inps.misreg_rng, 'r') as f:
                misreg_rg = float(f.readline())
        else:
             misreg_rg = 0.0

        burstoffset, minBurst, maxBurst = master.getCommonBurstLimits(slave)

        ###Burst indices w.r.t master
        if inps.overlap:
            maxBurst = maxBurst - 1
            geomDir = os.path.join(inps.geom_masterDir, inps.overlapDir, 'IW{0}'.format(swath))   
       
        else:
            geomDir = os.path.join(inps.geom_masterDir, 'IW{0}'.format(swath))
      
    
        slaveBurstStart = minBurst + burstoffset
    
        for mBurst in range(minBurst, maxBurst):
        
            ###Corresponding slave burst
            sBurst = slaveBurstStart + (mBurst - minBurst)
            burstTop = slave.bursts[sBurst]
            if inps.overlap:
                burstBot = slave.bursts[sBurst+1]
        
            print('Overlap pair {0}: Burst {1} of master matched with Burst {2} of slave'.format(mBurst-minBurst, mBurst, sBurst))
            if inps.overlap:
                ####Generate offsets for top burst
                rdict = {'lat': os.path.join(geomDir,'lat_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'lon': os.path.join(geomDir,'lon_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'hgt': os.path.join(geomDir,'hgt_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'rangeOffName': os.path.join(outdir, 'range_top_%02d_%02d.off'%(mBurst+1,mBurst+2)),
                     'azOffName': os.path.join(outdir, 'azimuth_top_%02d_%02d.off'%(mBurst+1,mBurst+2))}
        
                runGeo2rdr(burstTop, rdict, misreg_az=misreg_az, misreg_rg=misreg_rg)
        
                print('Overlap pair {0}: Burst {1} of master matched with Burst {2} of slave'.format(mBurst-minBurst, mBurst+1, sBurst+1))
                ####Generate offsets for bottom burst
                rdict = {'lat': os.path.join(geomDir,'lat_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'lon': os.path.join(geomDir, 'lon_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'hgt': os.path.join(geomDir, 'hgt_%02d_%02d.rdr'%(mBurst+1,mBurst+2)),
                     'rangeOffName': os.path.join(outdir, 'range_bot_%02d_%02d.off'%(mBurst+1,mBurst+2)),
                     'azOffName': os.path.join(outdir, 'azimuth_bot_%02d_%02d.off'%(mBurst+1,mBurst+2))}

                runGeo2rdr(burstBot, rdict, misreg_az=misreg_az, misreg_rg=misreg_rg)

            else:
                print('Burst {1} of master matched with Burst {2} of slave'.format(mBurst-minBurst, mBurst, sBurst))
                ####Generate offsets for top burst
                rdict = {'lat': os.path.join(geomDir,'lat_%02d.rdr'%(mBurst+1)),
                     'lon': os.path.join(geomDir,'lon_%02d.rdr'%(mBurst+1)),
                     'hgt': os.path.join(geomDir,'hgt_%02d.rdr'%(mBurst+1)),
                     'rangeOffName': os.path.join(outdir, 'range_%02d.off'%(mBurst+1)),
                     'azOffName': os.path.join(outdir, 'azimuth_%02d.off'%(mBurst+1))}

                runGeo2rdr(burstTop, rdict, misreg_az=misreg_az, misreg_rg=misreg_rg)



if __name__ == '__main__':
    '''
    Generate burst-by-burst reverse geometry mapping for resampling.
    '''
    # Main Driver
    main()



