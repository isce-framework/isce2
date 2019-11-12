#!/usr/bin/env python3
import isce
import os
import logging
from components.stdproc.stdproc import crossmul
import isceobj
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
import isceobj
import argparse

def createParser():

    '''
    Command Line Parser.
    '''
    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-m', '--master', type=str, dest='master', required=True,
            help='Master image')
    parser.add_argument('-s', '--slave', type=str, dest='slave', required=True,
            help='Slave image')
    parser.add_argument('-o', '--outdir', type=str, dest='prefix', default='crossmul',
            help='Prefix of output int and amp files')
    parser.add_argument('-a', '--alks', type=int, dest='azlooks', default=1,
            help='Azimuth looks')
    parser.add_argument('-r', '--rlks', type=int, dest='rglooks', default=1,
            help='Range looks')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

def run(imageSlc1, imageSlc2, resampName, azLooks, rgLooks):
    objSlc1 = isceobj.createSlcImage()
    #right now imageSlc1 and 2 are just text files, need to open them as image

    IU.copyAttributes(imageSlc1, objSlc1)
    objSlc1.setAccessMode('read')
    objSlc1.createImage()

    objSlc2 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc2, objSlc2)
    objSlc2.setAccessMode('read')
    objSlc2.createImage()

    slcWidth = imageSlc1.getWidth()
    intWidth = int(slcWidth / rgLooks)

    lines = min(imageSlc1.getLength(), imageSlc2.getLength())

    resampAmp = resampName + '.amp'
    resampInt = resampName + '.int'

    objInt = isceobj.createIntImage()
    objInt.setFilename(resampInt)
    objInt.setWidth(intWidth)
    imageInt = isceobj.createIntImage()
    IU.copyAttributes(objInt, imageInt)
    objInt.setAccessMode('write')
    objInt.createImage()

    objAmp = isceobj.createAmpImage()
    objAmp.setFilename(resampAmp)
    objAmp.setWidth(intWidth)
    imageAmp = isceobj.createAmpImage()
    IU.copyAttributes(objAmp, imageAmp)
    objAmp.setAccessMode('write')
    objAmp.createImage()

    objCrossmul = crossmul.createcrossmul()
    objCrossmul.width = slcWidth
    objCrossmul.length = lines
    objCrossmul.LooksDown = azLooks
    objCrossmul.LooksAcross = rgLooks

    objCrossmul.crossmul(objSlc1, objSlc2, objInt, objAmp)

    for obj in [objInt, objAmp, objSlc1, objSlc2]:
        obj.finalizeImage()

    return imageInt, imageAmp


def main(iargs=None):
    inps = cmdLineParse(iargs)

    img1 = isceobj.createImage()
    img1.load(inps.master + '.xml')

    img2 = isceobj.createImage()
    img2.load(inps.slave + '.xml')

    if not os.path.exists(os.path.dirname(inps.prefix)):
       os.makedirs(os.path.dirname(inps.prefix))

    run(img1, img2, inps.prefix, inps.azlooks, inps.rglooks)

if __name__ == '__main__':

    main()
    '''
    Main driver.
    '''
