#!/usr/bin/env python3

import isce
import numpy as np
from mroipac.formimage.FormSLC import FormSLC
import shelve
import isceobj
import copy
import argparse
import datetime
import os
from isceobj.Util.decorators import use_api


def createParser():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser(description='Focus from raw data to slc')
    parser.add_argument('-i','--input', dest='indir', type=str, required=True,
            help='Directory with raw file')
    parser.add_argument('-a', '--amb', dest='ambiguity', type=float, default=0.,
            help='Doppler ambiguities to add')
    parser.add_argument('-d', '--dop', dest='doppler', type=str, default=None,
            help='Doppler to focus the image to.')
    return parser #.parse_args()

def cmdLineParse(iargs = None):
    parser = createParser()
    inps = parser.parse_args(args=iargs)
    if inps.doppler:
       inps.doppler=[float(i) for i in inps.doppler.split(',')]

    print('user input Doppler: ', inps.doppler)

    return inps

@use_api
def focus(frame, amb=0.0, dop=None):
    from isceobj.Catalog import recordInputsAndOutputs
    from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
    from isceobj.Constants import SPEED_OF_LIGHT

    raw_r0 = frame.startingRange
    raw_dr = frame.getInstrument().getRangePixelSize()
    img = frame.getImage()
    if dop is None:
       dop = frame._dopplerVsPixel

    print('dop',dop) 

    #####Velocity/ acceleration etc
    planet = frame.instrument.platform.planet
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
    hgt = sch[2]
    radius = elp.pegRadCur
  
    ####Computation of acceleration
    dist = np.linalg.norm(xyz)
    r_spinvec = np.array([0., 0., planet.spin])
    r_tempv = np.cross(r_spinvec, xyz)

    inert_acc = np.array([-planet.GM*x/(dist**3) for x in xyz])

    r_tempa = np.cross(r_spinvec, vxyz)
    r_tempvec = np.cross(r_spinvec, r_tempv)

    r_bodyacc = inert_acc - 2 * r_tempa - r_tempvec
    schbasis = elp.schbasis(sch)

    schacc = np.dot(schbasis.xyz_to_sch, r_bodyacc).tolist()[0]


    print('SCH velocity: ', schvel)
    print('SCH acceleration: ', schacc)
    print('Body velocity: ', vel)
    print('Height: ', hgt)
    print('Radius: ', radius)

    #####Setting up formslc
    
    form = FormSLC()
    form.configure()

    ####Width
    form.numberBytesPerLine = img.getWidth()

    ###Includes header
    form.numberGoodBytes = img.getWidth()


    ####Different chirp extensions
#    form.nearRangeChirpExtFrac = 0.0
#    form.farRangeChirpExtFrac = 0.0
#    form.earlyAzimuthChirpExtFrac = 0.0
#    form.lateAzimuthChirpExtrFrac = 0.0


    ###First Line - set with defaults
    ###Depending on extensions
#    form.firstLine = 0

    ####First Sample
    form.firstSample = img.getXmin() // 2

    ####Start range bin - set with defaults
    ###Depending on extensions
#    form.startRangeBin = 1

    ####Starting range
    form.rangeFirstSample = frame.startingRange

    ####Number range bin
    ###Determined in FormSLC.py using chirp extensions
#    form.numberRangeBin = img.getWidth() // 2 - 1000

    ####Azimuth looks
    form.numberAzimuthLooks = 1


    ####debug
    form.debugFlag = False

    ####PRF
    form.prf = frame.PRF
    form.sensingStart = frame.sensingStart

    ####Bias
    form.inPhaseValue = frame.getInstrument().inPhaseValue
    form.quadratureValue = frame.getInstrument().quadratureValue

    ####Resolution
    form.antennaLength = frame.instrument.platform.antennaLength
    form.azimuthResolution = 0.6 * form.antennaLength  #85% of max bandwidth
    ####Sampling rate
    form.rangeSamplingRate = frame.getInstrument().rangeSamplingRate

    ####Chirp parameters
    form.chirpSlope =  frame.getInstrument().chirpSlope
    form.rangePulseDuration = frame.getInstrument().pulseLength

    ####Wavelength
    form.radarWavelength = frame.getInstrument().radarWavelength

    ####Secondary range migration
    form.secondaryRangeMigrationFlag = False


    ###pointing direction
    form.pointingDirection = frame.instrument.platform.pointingDirection
    print('Lookside: ', form.pointingDirection)

    ####Doppler centroids
    cfs = [amb, 0., 0., 0.]
    for ii in range(min(len(dop),4)):
        cfs[ii] += dop[ii]/form.prf

    form.dopplerCentroidCoefficients = cfs

    ####Create raw image
    rawimg = isceobj.createRawImage()
    rawimg.load(img.filename + '.xml')
    rawimg.setAccessMode('READ')
    rawimg.createImage()

    form.rawImage = rawimg


    ####All the orbit parameters
    form.antennaSCHVelocity = schvel
    form.antennaSCHAcceleration = schacc
    form.bodyFixedVelocity = vel
    form.spacecraftHeight = hgt
    form.planetLocalRadius = radius



    ###Create SLC image
    slcImg = isceobj.createSlcImage()
    slcImg.setFilename(img.filename + '.slc')
    form.slcImage = slcImg

    form.formslc()

    return form


def main(iargs=None):
    ####Parse command line
    inps = cmdLineParse(iargs)


    ####Load SLC parameters
    with shelve.open(os.path.join(inps.indir, 'raw')) as db:
        rawFrame = db['frame']
    

    formSLC = focus(rawFrame, amb=inps.ambiguity, dop=inps.doppler)
    width = formSLC.slcImage.getWidth()
    length = formSLC.slcImage.getLength()
    prf = rawFrame.PRF
    delr = rawFrame.instrument.getRangePixelSize()

    ####Start creating an SLC frame to work with
    slcFrame = copy.deepcopy(rawFrame)

    slcFrame.setStartingRange(formSLC.startingRange)
    slcFrame.setFarRange(formSLC.startingRange + (width-1)*delr)

    tstart = formSLC.slcSensingStart
    tmid = tstart + datetime.timedelta(seconds = 0.5 * length / prf)
    tend = tstart + datetime.timedelta(seconds = (length-1) / prf)

    slcFrame.sensingStart = tstart
    slcFrame.sensingMid = tmid
    slcFrame.sensingStop = tend

    formSLC.slcImage.setAccessMode('READ')
    formSLC.slcImage.setXmin(0)
    formSLC.slcImage.setXmax(width)
    slcFrame.setImage(formSLC.slcImage)

    slcFrame.setNumberOfSamples(width)
    slcFrame.setNumberOfLines(length)

    #####Adjust the doppler polynomial
    if inps.doppler:
       dop = inps.doppler[::-1]
    else:
       dop = rawFrame._dopplerVsPixel[::-1]
    xx = np.linspace(0, (width-1), num=len(dop)+ 1)
    x = (slcFrame.startingRange - rawFrame.startingRange)/delr + xx
    v = np.polyval(dop, x)
    p = np.polyfit(xx, v, len(dop)-1)[::-1]
    slcFrame._dopplerVsPixel = list(p)
    slcFrame._dopplerVsPixel[0] += inps.ambiguity*prf

    print(slcFrame._dopplerVsPixel)
    ####Load RAW parameters
    with shelve.open(os.path.join(inps.indir, 'data')) as db:
        db['frame'] = slcFrame


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()


