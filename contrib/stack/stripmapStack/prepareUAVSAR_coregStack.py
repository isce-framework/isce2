#!/usr/bin/env python3

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

    parser = argparse.ArgumentParser(description='Unzip Alos zip files.')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='directory which has all dates as directories. Inside each date, zip files are expected.')
    parser.add_argument('-d', '--dop_file', dest='dopFile', type=str, required=True,
            help='Doppler file for the stack.')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
            help='output directory which will be used for unpacking.')
    parser.add_argument('-t', '--text_cmd', dest='text_cmd', type=str, default='source ~/.bash_profile;'
       , help='text command to be added to the beginning of each line of the run files. Example : source ~/.bash_profile;')

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
    run_unPack = 'run_unPackAlos'

    #######################################
    slc_files = glob.glob(os.path.join(inps.input, '*_s5_1x1.slc'))
    for file in slc_files:
        imgDate = get_Date(file)
        print (imgDate)
        annFile = file.replace('_s5_1x1.slc','')+'.ann'
        print (annFile)
        imgDir = os.path.join(outputDir,imgDate)
        if not os.path.exists(imgDir):
           os.makedirs(imgDir)

        cmd = 'unpackFrame_UAVSAR.py -i ' + annFile  + ' -d '+ inps.dopFile + ' -o ' + imgDir
        print (cmd)
        os.system(cmd)
        
        cmd = 'mv ' + file + ' ' + imgDir
        print(cmd)
        os.system(cmd)

        cmd = 'mv ' + annFile + ' ' + imgDir 
        print(cmd)
        os.system(cmd)

        shelveFile = os.path.join(imgDir, 'data')
        slcFile = os.path.join(imgDir, os.path.basename(file))
        write_xml(shelveFile, slcFile)

if __name__ == '__main__':

    main()


