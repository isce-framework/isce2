#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import datetime
import numpy as np

import isce, isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from isceobj.Alos2Proc.Alos2ProcPublic import getBboxRdr

from StackPulic import loadTrack
from StackPulic import stackDateStatistics


def computeBaseline(trackReference, trackSecondary, azimuthTime, rangeDistance):
    import numpy as np
    
    from isceobj.Planet.Planet import Planet

    #modify Piyush's code for computing baslines
    refElp = Planet(pname='Earth').ellipsoid
    #for x in points:
    referenceSV = trackReference.orbit.interpolate(azimuthTime, method='hermite')
    target = trackReference.orbit.rdr2geo(azimuthTime, rangeDistance)

    slvTime, slvrng = trackSecondary.orbit.geo2rdr(target)
    secondarySV = trackSecondary.orbit.interpolateOrbit(slvTime, method='hermite')

    targxyz = np.array(refElp.LLH(target[0], target[1], target[2]).ecef().tolist())
    mxyz = np.array(referenceSV.getPosition())
    mvel = np.array(referenceSV.getVelocity())
    sxyz = np.array(secondarySV.getPosition())

    #to fix abrupt change near zero in baseline grid. JUN-05-2020
    mvelunit = mvel / np.linalg.norm(mvel)
    sxyz = sxyz - np.dot ( sxyz-mxyz, mvelunit) * mvelunit

    aa = np.linalg.norm(sxyz-mxyz)
    costheta = (rangeDistance*rangeDistance + aa*aa - slvrng*slvrng)/(2.*rangeDistance*aa)

    Bpar = aa*costheta

    perp = aa * np.sqrt(1 - costheta*costheta)
    direction = np.sign(np.dot( np.cross(targxyz-mxyz, sxyz-mxyz), mvel))
    Bperp = direction*perp

    return (Bpar, Bperp)



def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='compute baselines for a number of dates')
    parser.add_argument('-idir', dest='idir', type=str, required=True,
            help = 'input directory where data of each date (YYMMDD) is located. only folders are recognized')
    parser.add_argument('-odir', dest='odir', type=str, required=True,
            help = 'output directory where baseline of each date is output')
    parser.add_argument('-ref_date', dest='ref_date', type=str, required=True,
            help = 'reference date. format: YYMMDD')
    parser.add_argument('-sec_date', dest='sec_date', type=str, nargs='+', default=[],
            help = 'a number of secondary dates seperated by blanks. format: YYMMDD YYMMDD YYMMDD. If provided, only compute baseline grids of these dates')
    parser.add_argument('-baseline_center', dest='baseline_center', type=str, default=None,
            help = 'output baseline file at image center for all dates. If not provided, it will not be computed')
    parser.add_argument('-baseline_grid', dest='baseline_grid', action='store_true', default=False,
            help='compute baseline grid for each date')
    parser.add_argument('-baseline_grid_width', dest='baseline_grid_width', type=int, default=10,
            help = 'baseline grid width if compute baseline grid, default: 10')
    parser.add_argument('-baseline_grid_length', dest='baseline_grid_length', type=int, default=10,
            help = 'baseline grid length if compute baseline grid, default: 10')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


    #get user parameters from input
    idir = inps.idir
    odir = inps.odir
    dateReference = inps.ref_date
    dateSecondary = inps.sec_date
    baselineCenterFile = inps.baseline_center
    baselineGrid = inps.baseline_grid

    widthBaseline = inps.baseline_grid_width
    lengthBaseline = inps.baseline_grid_length

    #######################################################


    #get date statistics
    dateDirs,   dates,   frames,   swaths,   dateIndexReference = stackDateStatistics(idir, dateReference)
    ndate = len(dates)
    nframe = len(frames)
    nswath = len(swaths)


    #create output directory if it does not already exist
    if not os.path.isdir(odir):
        print('output directory {} does not exist, create'.format(odir))
        os.makedirs(odir, exist_ok=True)
    os.chdir(odir)


    #compute baseline
    trackReference = loadTrack(dateDirs[dateIndexReference], dates[dateIndexReference])
    bboxRdr = getBboxRdr(trackReference)
    #at four corners
    rangeMin = bboxRdr[0]
    rangeMax = bboxRdr[1]
    azimuthTimeMin = bboxRdr[2]
    azimuthTimeMax = bboxRdr[3]
    #at image center
    azimuthTimeMid = azimuthTimeMin+datetime.timedelta(seconds=(azimuthTimeMax-azimuthTimeMin).total_seconds()/2.0)
    rangeMid = (rangeMin + rangeMax) / 2.0
    #grid size
    rangeDelta = (rangeMax - rangeMin) / (widthBaseline - 1.0)
    azimuthDelta = (azimuthTimeMax-azimuthTimeMin).total_seconds() / (lengthBaseline - 1.0)

    #baseline at image center
    if baselineCenterFile is not None:
        baselineCenter  = '  reference date    secondary date    parallel baseline [m]    perpendicular baseline [m]\n'
        baselineCenter += '===========================================================================================\n'

    #baseline grid: two-band BIL image, first band: parallel baseline, perpendicular baseline
    baseline = np.zeros((lengthBaseline*2, widthBaseline), dtype=np.float32)
    
    #compute baseline
    for i in range(ndate):
        if i == dateIndexReference:
            continue


        trackSecondary = loadTrack(dateDirs[i], dates[i])

        #compute baseline at image center
        if baselineCenterFile is not None:
            (Bpar, Bperp) = computeBaseline(trackReference, trackSecondary, azimuthTimeMid, rangeMid)
            baselineCenter += '     %s            %s           %9.3f                     %9.3f\n'%(dates[dateIndexReference], dates[i], Bpar, Bperp)

        if dateSecondary != []:
            if dates[i] not in dateSecondary:
                continue


        #compute baseline grid
        if baselineGrid:
            baselineFile = '{}-{}.rmg'.format(dates[dateIndexReference], dates[i])
            if os.path.isfile(baselineFile):
                print('baseline grid file {} already exists, do not create'.format(baselineFile))
            else:
                for j in range(lengthBaseline):
                    for k in range(widthBaseline):
                        (baseline[j*2, k], baseline[j*2+1, k]) = computeBaseline(trackReference, trackSecondary, 
                            azimuthTimeMin+datetime.timedelta(seconds=azimuthDelta*j), 
                            rangeMin+rangeDelta*k)
                baseline.astype(np.float32).tofile(baselineFile)
                create_xml(baselineFile, widthBaseline, lengthBaseline, 'rmg')

    #dump baseline at image center
    if baselineCenterFile is not None:
        print('\nbaselines at image centers')
        print(baselineCenter)
        with open(baselineCenterFile, 'w') as f:
            f.write(baselineCenter)



