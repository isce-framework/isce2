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
    parser.add_argument('-m', dest='reference', type=str,
            help='Reference image with ISCE XML file', required=True)
    parser.add_argument('-b1', dest='band1', type=int,
            help='Band number of reference image', default=0)
    parser.add_argument('-s', dest='secondary', type=str,
            help='Secondary image with ISCE XML file', required=True)
    parser.add_argument('-b2', dest='band2', type=int,
            help='Band number of secondary image', default=0)
    parser.add_argument('-o', dest='outfile', default= 'offsets.txt',
            type=str, help='Output ASCII file')
    return parser.parse_args()


#Start of the main program
if __name__ == '__main__': 

    logging.info("Calculate offset between two using ampcor")

    #Parse command line
    inps = cmdLineParser()

    ####Create reference image object
    referenceImg = isceobj.createImage()   #Empty image
    referenceImg.load(inps.reference +'.xml') #Load from XML file
    referenceImg.setAccessMode('read')     #Set it up for reading 
    referenceImg.createImage()             #Create File

    #####Create secondary image object
    secondaryImg = isceobj.createImage()    #Empty image
    secondaryImg.load(inps.secondary +'.xml')   #Load it from XML file
    secondaryImg.setAccessMode('read')      #Set it up for reading
    secondaryImg.createImage()              #Create File



    ####Stage 1: Initialize
    objAmpcor = Ampcor(name='my_ampcor')
    objAmpcor.configure()

    ####Defautl values used if not provided in my_ampcor
    coarseAcross = 0
    coarseDown = 0

    ####Get file types
    if referenceImg.getDataType().upper().startswith('C'):
        objAmpcor.setImageDataType1('complex')
    else:
        objAmpcor.setImageDataType1('real')

    if secondaryImg.getDataType().upper().startswith('C'):
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
    objAmpcor.ampcor(referenceImg,secondaryImg)

    ###Close ununsed images
    referenceImg.finalizeImage()
    secondaryImg.finalizeImage()
    
  
    ######Stage 5: Get required data out of the processing run
    offField = objAmpcor.getOffsetField()
    logging.info('Number of returned offsets : %d'%(len(offField._offsets)))

    ####Write output to an ascii file
    field = np.array(offField.unpackOffsets())
    np.savetxt(inps.outfile, field, delimiter="   ", format='%5.6f')
