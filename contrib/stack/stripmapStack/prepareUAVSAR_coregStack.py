#!/usr/bin/env python3
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

    parser = argparse.ArgumentParser(description='Prepare UAVSAR SLC Stack files.')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='directory which has all dates.')
    parser.add_argument('-d', '--dop_file', dest='dopFile', type=str, required=True,
            help='Doppler file for the stack. Needs to be in directory where command is run.')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
            help='output directory which will be used for unpacking.')
    parser.add_argument('-s', '--segment', dest='segment', type=str, default='1',
            help='segment of the UAVSAR stack to prepare. For "s2" use "2", etc. Default is "1" ')
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
    yyyymmdd='20'+file.split('_')[4]
    return yyyymmdd

def main(iargs=None):
    '''
    The main driver.
    '''

    inps = cmdLineParse(iargs)
    
    outputDir = os.path.abspath(inps.output)

    #######################################
    slc_files = glob.glob(os.path.join(inps.input, '*_s'+inps.segment+'_1x1.slc'))
    for file in slc_files:
        imgDate = get_Date(file)
        print (imgDate)
        annFile = file.replace('_s'+inps.segment+'_1x1.slc','')+'.ann'
        print (annFile)
        imgDir = os.path.join(outputDir,imgDate)
        if not os.path.exists(imgDir):
           os.makedirs(imgDir)

        cmd = 'unpackFrame_UAVSAR.py -i ' + annFile  + ' -d '+ inps.dopFile + ' -o ' + imgDir
        print (cmd)
        os.system(cmd)
        
        slcFile = os.path.join(imgDir, imgDate+'.slc')
        cmd = 'mv ' + file + ' ' + slcFile
        print(cmd)
        os.system(cmd)

        cmd = 'mv ' + annFile + ' ' + imgDir 
        print(cmd)
        os.system(cmd)

        shelveFile = os.path.join(imgDir, 'data')
        write_xml(shelveFile, slcFile)

if __name__ == '__main__':

    main()


