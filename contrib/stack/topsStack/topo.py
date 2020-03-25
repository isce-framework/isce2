#!/usr/bin/env python3

import numpy as np
import argparse
import os
import isce
import isceobj
import datetime
import sys
import s1a_isce_utils as ut
from isceobj.Planet.Planet import Planet
from zerodop.topozero import createTopozero
import multiprocessing as mp


def createParser():
    parser = argparse.ArgumentParser( description='Generates lat/lon/h and los for each pixel')
    parser.add_argument('-m', '--master', type=str, dest='master', required=True,
            help='Directory with the master image')
    parser.add_argument('-d', '--dem', type=str, dest='dem', required=True,
            help='DEM to use for coregistration')
    parser.add_argument('-g', '--geom_masterDir', type=str, dest='geom_masterDir', default='geom_master',
            help='Directory for geometry files of the master')

    return parser

def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''
    parser = createParser()
    return parser.parse_args(args=iargs)


def call_topo(input):

    (dirname, demImage, master, ind) = input

    burst = master.bursts[ind]
    latname = os.path.join(dirname, 'lat_%02d.rdr' % (ind + 1))
    lonname = os.path.join(dirname, 'lon_%02d.rdr' % (ind + 1))
    hgtname = os.path.join(dirname, 'hgt_%02d.rdr' % (ind + 1))
    losname = os.path.join(dirname, 'los_%02d.rdr' % (ind + 1))
    maskname = os.path.join(dirname, 'shadowMask_%02d.rdr' % (ind + 1))
    incname = os.path.join(dirname, 'incLocal_%02d.rdr' % (ind + 1))
    #####Run Topo
    planet = Planet(pname='Earth')
    topo = createTopozero()
    topo.slantRangePixelSpacing = burst.rangePixelSize
    topo.prf = 1.0 / burst.azimuthTimeInterval
    topo.radarWavelength = burst.radarWavelength
    topo.orbit = burst.orbit
    topo.width = burst.numberOfSamples
    topo.length = burst.numberOfLines
    topo.wireInputPort(name='dem', object=demImage)
    topo.wireInputPort(name='planet', object=planet)
    topo.numberRangeLooks = 1
    topo.numberAzimuthLooks = 1
    topo.lookSide = -1
    topo.sensingStart = burst.sensingStart
    topo.rangeFirstSample = burst.startingRange
    topo.demInterpolationMethod = 'BIQUINTIC'
    topo.latFilename = latname
    topo.lonFilename = lonname
    topo.heightFilename = hgtname
    topo.losFilename = losname
    topo.maskFilename = maskname
    topo.incFilename = incname
    topo.topo()

    bbox = [topo.minimumLatitude, topo.maximumLatitude, topo.minimumLongitude, topo.maximumLongitude]

    topo = None

    return bbox


def main(iargs=None):

    inps = cmdLineParse(iargs)

    swathList = ut.getSwathList(inps.master)

    demImage = isceobj.createDemImage()
    demImage.load(inps.dem + '.xml')

    boxes = []
    inputs = []

    for swath in swathList:
        master =  ut.loadProduct(os.path.join(inps.master , 'IW{0}.xml'.format(swath)))
    
        ###Check if geometry directory already exists.
        dirname = os.path.join(inps.geom_masterDir, 'IW{0}'.format(swath))

        if os.path.isdir(dirname):
            print('Geometry directory {0} already exists.'.format(dirname))
        else:
            os.makedirs(dirname)

        for ind in range(master.numberOfBursts):
            inputs.append((dirname, demImage, master, ind))

    pool = mp.Pool(mp.cpu_count())
    results = pool.map(call_topo, inputs)
    pool.close()

    for bbox in results:
        boxes.append(bbox)

    boxes = np.array(boxes)
    bbox = [np.min(boxes[:,0]), np.max(boxes[:,1]), np.min(boxes[:,2]), np.max(boxes[:,3])]
    print ('bbox : ',bbox)
    

if __name__ == '__main__':

    main()

