#!/usr/bin/env python3

import isce
import logging
import isceobj
import mroipac
import argparse
from mroipac.ampcor.Ampcor import Ampcor
import numpy as np

def cmdLineParser():
    parser = argparse.ArgumentParser(description='Simple ampcor driver')
    parser.add_argument('-m', dest='master', type=str,
            help='Master image with ISCE XML file', required=True)
    parser.add_argument('-b1', dest='band1', type=int,
            help='Band number of master image', default=0)
    parser.add_argument('-s', dest='slave', type=str,
            help='Slave image with ISCE XML file', required=True)
    parser.add_argument('-b2', dest='band2', type=int,
            help='Band number of slave image', default=0)
    parser.add_argument('-o', dest='outfile', default= 'offsets.txt',
            type=str, help='Output ASCII file')
    return parser.parse_args()


#Start of the main program
if __name__ == '__main__': 

    logging.info("Calculate offset between two using ampcor")

    #Parse command line
    inps = cmdLineParser()

    ####Create master image object
    masterImg = isceobj.createImage()   #Empty image
    masterImg.load(inps.master +'.xml') #Load from XML file
    masterImg.setAccessMode('read')     #Set it up for reading 
    masterImg.createImage()             #Create File

    #####Create slave image object
    slaveImg = isceobj.createImage()    #Empty image
    slaveImg.load(inps.slave +'.xml')   #Load it from XML file
    slaveImg.setAccessMode('read')      #Set it up for reading
    slaveImg.createImage()              #Create File



    ####Stage 1: Initialize
    objAmpcor = Ampcor(name='my_ampcor')
    objAmpcor.configure()

    ####Defautl values used if not provided in my_ampcor
    coarseAcross = 0
    coarseDown = 0

    ####Get file types
    if masterImg.getDataType().upper().startswith('C'):
        objAmpcor.setImageDataType1('complex')
    else:
        objAmpcor.setImageDataType1('real')

    if slaveImg.getDataType().upper().startswith('C'):
        objAmpcor.setImageDataType2('complex')
    else:
        objAmpcor.setImageDataType2('real')

    #####Stage 2: No ports for ampcor
    ### Any parameters can be controlled through my_ampcor.xml

    ### Stage 3: Set values as needed
    ####Only set these values if user does not define it in my_ampcor.xml
    if objAmpcor.acrossGrossOffset is None:
        objAmpcor.acrossGrossOffset = coarseAcross

    if objAmpcor.downGrossOffset is None:
        objAmpcor.downGrossOffset = coarseDown

    logging.info('Across Gross Offset = %d'%(objAmpcor.acrossGrossOffset))
    logging.info('Down Gross Offset = %d'%(objAmpcor.downGrossOffset))

    ####Stage 4: Call the main method
    objAmpcor.ampcor(masterImg,slaveImg)

    ###Close ununsed images
    masterImg.finalizeImage()
    slaveImg.finalizeImage()
    
  
    ######Stage 5: Get required data out of the processing run
    offField = objAmpcor.getOffsetField()
    logging.info('Number of returned offsets : %d'%(len(offField._offsets)))

    ####Write output to an ascii file
    field = np.array(offField.unpackOffsets())
    np.savetxt(inps.outfile, field, delimiter="   ", format='%5.6f')
