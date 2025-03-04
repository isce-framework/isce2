#!/usr/bin/env python3
# modified to pass the segment number to unpackFrame_UAVSAR EJF 2020/08/02
# modified to work for different UAVSAR stack segments EJF 2019/05/04

import os
import glob
import argparse

import isce
import isceobj
import shelve 
from isceobj.Util.decorators import use_api

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Prepare ESA ERS Stack files.')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='directory which has all dates.')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
            help='output directory which will be used for unpacking.')
    parser.add_argument('--orbitdir', dest='orbitdir', type=str, required=True, help='Orbit directory')
    parser.add_argument('-t', '--text_cmd', dest='text_cmd', type=str, default='source ~/.bash_profile;', 
            help='text command to be added to the beginning of each line of the run files. Default: source ~/.bash_profile;')

    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    return parser.parse_args(args = iargs)

def write_xml(shelveFile, slcFile):
    with shelve.open(shelveFile,flag='r') as db:
        frame = db['frame']

    length = frame.numberOfLines 
    width = frame.numberOfSamples
    print (width,length)

    slc = isceobj.createSlcImage()
    slc.setWidth(width)
    slc.setLength(length)
    slc.filename = slcFile
    slc.setAccessMode('write')
    slc.renderHdr()
    slc.renderVRT()


def get_Date(file):
    yyyymmdd=file[14:22]
    return yyyymmdd

def main(iargs=None):
    '''
    The main driver.
    '''

    inps = cmdLineParse(iargs)
    
    outputDir = os.path.abspath(inps.output)

    #######################################
    slc_files = glob.glob(os.path.join(inps.input, 'SAR*.E*'))
    for file in slc_files:
        imgDate = get_Date(os.path.basename(file))
        print (imgDate)

        imgDir = os.path.join(outputDir,imgDate)
        os.makedirs(imgDir, exist_ok=True)

        cmd = 'unpackFrame_ERS_ENV.py -i ' + inps.input +' -o ' + imgDir + ' --orbitdir ' + inps.orbitdir
        print (cmd)
        os.system(cmd)
        
        slcFile = os.path.join(imgDir, imgDate+'.slc')

        shelveFile = os.path.join(imgDir, 'data')
        write_xml(shelveFile, slcFile)

if __name__ == '__main__':

    main()


