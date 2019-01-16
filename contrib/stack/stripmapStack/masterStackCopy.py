#!/usr/bin/env python3

import isce
import isceobj
import argparse
import os
import shelve
import logging

def createParser():
    parser = argparse.ArgumentParser( description='Duplicating the master SLC')

    parser.add_argument('-i', '--input_slc', dest='input_slc', type=str, required=True,
            help = 'Directory with master acquisition for reference')
    parser.add_argument('-o', '--output_slc', dest='output_slc', type=str, required=True,
            help='Directory with slave acquisition')

    return parser
    
def cmdLineParse(iargs = None):
    parser = createParser()
    
    inps =  parser.parse_args(args=iargs)
    
    return inps


def main(iargs=None):
    '''
    Main driver.
    '''
    inps = cmdLineParse(iargs)

    # providing absolute paths
    inps.output_slc = os.path.abspath(inps.output_slc)
    inps.input_slc = os.path.abspath(inps.input_slc)

    # making the output direcory is non-existent
    outDir = os.path.dirname(inps.output_slc)
    inDir = os.path.dirname(inps.input_slc)
    if not os.path.exists(outDir):
       os.makedirs(outDir)

    # copying shelf files as backup
    masterShelveDir = os.path.join(outDir, 'masterShelve')
    slaveShelveDir = os.path.join(outDir, 'slaveShelve')

    if not os.path.exists(masterShelveDir):
       os.makedirs(masterShelveDir)

    if not os.path.exists(slaveShelveDir):
       os.makedirs(slaveShelveDir)


    cmd = 'cp '+ inDir + '/data* ' + slaveShelveDir
    os.system(cmd) 

    cmd = 'cp '+ inDir + '/data* ' + masterShelveDir
    os.system(cmd)

    cmd = 'gdal_translate -of ENVI ' + inps.input_slc + " " + inps.output_slc
    os.system(cmd)
    cmd = 'gdal_translate -of VRT ' + inps.output_slc + " " + inps.output_slc + ".vrt"
    os.system(cmd)

if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

