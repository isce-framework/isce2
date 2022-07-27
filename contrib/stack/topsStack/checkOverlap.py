#!/usr/bin/env python3

# Author: Cunren Liang
# Copyright 2021

import os
import copy
import glob
import shutil
import argparse
import numpy as np

import isce
import isceobj
import s1a_isce_utils as ut
from isceobj.Sensor.TOPS import createTOPSSwathSLCProduct
from isceobj.TopsProc.runIon import renameFile


def createParser():
    parser = argparse.ArgumentParser(description='check overlap among all acquisitons')
    parser.add_argument('-r', '--reference', dest='reference', type=str, required=True,
            help='directory with reference acquistion')
    parser.add_argument('-s', '--secondarys', dest='secondarys', type=str, required=True,
            help='directory with secondarys acquistions')
    parser.add_argument('-g', '--geom_reference', dest='geom_reference', type=str, default=None, 
            help='directory with geometry of reference')

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):
    '''
    check overlap among all acquistions, only keep the bursts that in the common overlap,
    and then renumber the bursts.
    '''
    inps = cmdLineParse(iargs)

    referenceDir = inps.reference
    secondaryDir = sorted(glob.glob(os.path.join(inps.secondarys, '*')))

    acquistionDir = [referenceDir] + secondaryDir


    invalidSwath = []
    for i in [1, 2, 3]:
        for x in acquistionDir:
            if not (os.path.isdir(os.path.join(x, 'IW{}'.format(i))) and os.path.isfile(os.path.join(x, 'IW{}.xml'.format(i)))):
                invalidSwath.append(i)
                break

    if invalidSwath == [1, 2, 3]:
        raise Exception('there are no common swaths among the acquistions')
    else:
        validSwath = [i for i in [1, 2, 3] if i not in invalidSwath]
        print('valid swath from scanning acquistion directory: {}'.format(validSwath))


    invalidSwath2 = []
    for swath in validSwath:
        referenceSwath = ut.loadProduct(os.path.join(referenceDir, 'IW{0}.xml'.format(swath)))

        burstoffsetAll = []
        minBurstAll = []
        maxBurstAll = []
        secondarySwathAll = []
        for secondaryDirX in secondaryDir:
            secondarySwath = ut.loadProduct(os.path.join(secondaryDirX, 'IW{0}.xml'.format(swath)))

            secondarySwathAll.append(secondarySwath)

            burstoffset, minBurst, maxBurst = referenceSwath.getCommonBurstLimits(secondarySwath)
            burstoffsetAll.append(burstoffset)
            minBurstAll.append(minBurst)
            maxBurstAll.append(maxBurst)

        minBurst = max(minBurstAll)
        maxBurst = min(maxBurstAll)

        numBurst = maxBurst - minBurst



        if minBurst >= maxBurst:
            invalidSwath2.append(swath)
        else:
            #add reference
            swathAll = [referenceSwath] + secondarySwathAll
            burstoffsetAll = [0] + burstoffsetAll

            for dirx, swathx, burstoffsetx in zip(acquistionDir, swathAll, burstoffsetAll):

                swathTmp = createTOPSSwathSLCProduct()
                swathTmp.configure()

                #change reserved burst properties and remove non-overlap bursts
                for jj in range(len(swathx.bursts)):
                    ii = jj - burstoffsetx
                    #burstFileName = os.path.join(os.path.abspath(dirx), 'IW{}'.format(swath), os.path.basename(swathx.bursts[jj].image.filename))
                    burstFileName = os.path.join(os.path.abspath(dirx), 'IW{}'.format(swath), 'burst_%02d'%(jj+1) + '.slc')
                    if minBurst <= ii < maxBurst:
                        kk = ii - minBurst
                        #change burst properties
                        swathx.bursts[jj].burstNumber = kk + 1
                        swathx.bursts[jj].image.filename = os.path.join(os.path.dirname(swathx.bursts[jj].image.filename), 'burst_%02d'%(kk+1) + '.slc')
                        swathTmp.bursts.append(swathx.bursts[jj])
                    else:
                        #remove non-overlap bursts
                        #os.remove(burstFileName)
                        os.remove(burstFileName+'.vrt')
                        os.remove(burstFileName+'.xml')
                        #remove geometry files accordingly if provided
                        if dirx == referenceDir:
                            if inps.geom_reference is not None:
                                for fileType in ['hgt', 'incLocal', 'lat', 'lon', 'los', 'shadowMask']:
                                    geomFileName = os.path.join(os.path.abspath(inps.geom_reference), 'IW{}'.format(swath), fileType + '_%02d'%(jj+1) + '.rdr')
                                    os.remove(geomFileName)
                                    os.remove(geomFileName+'.vrt')
                                    os.remove(geomFileName+'.xml')


                #change reserved burst file names
                for jj in range(len(swathx.bursts)):
                    ii = jj - burstoffsetx
                    #burstFileName = os.path.join(os.path.abspath(dirx), 'IW{}'.format(swath), os.path.basename(swathx.bursts[jj].image.filename))
                    burstFileName = os.path.join(os.path.abspath(dirx), 'IW{}'.format(swath), 'burst_%02d'%(jj+1) + '.slc')
                    if minBurst <= ii < maxBurst:
                        kk = ii - minBurst
                        burstFileNameNew = os.path.join(os.path.abspath(dirx), 'IW{}'.format(swath), 'burst_%02d'%(kk+1) + '.slc')
                        if burstFileName != burstFileNameNew:
                            img = isceobj.createImage()
                            img.load(burstFileName + '.xml')
                            img.setFilename(burstFileNameNew)
                            #img.extraFilename = burstFileNameNew+'.vrt'
                            img.renderHdr()

                            #still use original vrt
                            os.remove(burstFileName+'.xml')
                            os.remove(burstFileNameNew+'.vrt')
                            os.rename(burstFileName+'.vrt', burstFileNameNew+'.vrt')
                        #change geometry file names accordingly if provided
                        if dirx == referenceDir:
                            if inps.geom_reference is not None:
                                for fileType in ['hgt', 'incLocal', 'lat', 'lon', 'los', 'shadowMask']:
                                    geomFileName = os.path.join(os.path.abspath(inps.geom_reference), 'IW{}'.format(swath), fileType + '_%02d'%(jj+1) + '.rdr')
                                    geomFileNameNew = os.path.join(os.path.abspath(inps.geom_reference), 'IW{}'.format(swath), fileType + '_%02d'%(kk+1) + '.rdr')
                                    if geomFileName != geomFileNameNew:
                                        renameFile(geomFileName, geomFileNameNew)


                #change swath properties
                swathx.bursts = swathTmp.bursts
                swathx.numberOfBursts = numBurst

                #remove original and write new
                os.remove( os.path.join(dirx, 'IW{}.xml'.format(swath)) )
                ut.saveProduct(swathx, os.path.join(dirx, 'IW{}.xml'.format(swath)))

                
    #remove invalid swaths
    invalidSwath3 = list(sorted(set(invalidSwath+invalidSwath2)))
    for swath in invalidSwath3:
        for dirx in acquistionDir:
            iwdir = os.path.join(dirx, 'IW{}'.format(swath))
            iwxml = os.path.join(dirx, 'IW{}.xml'.format(swath))
            if os.path.isdir(iwdir):
                shutil.rmtree(iwdir)
            if os.path.isfile(iwxml):
                os.remove(iwxml)


if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



