#!/usr/bin/env python3
#
# Author: David Bekaert
# Copyright 2018

import os
import glob
import sys
import argparse


def createParser():
    '''
    Create command line parser.
    '''
    
    parser = argparse.ArgumentParser(description='Generate all kml files for the master and slave slc')
    parser.add_argument('-i', '--i', dest='inputdir', type=str, default="slaves", help='Input directory')
    parser.add_argument('-o', '--o', dest='outputdir', type=str, default="kml_slcs", help='Output directory')
    return parser

def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    return parser.parse_args(args = iargs)


def main(iargs=None):
    '''
    The main driver.
    '''

    inps = cmdLineParse(iargs)
    outputdir = os.path.abspath(inps.outputdir)
    inputdir = os.path.abspath(inps.inputdir)

    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    
    # see if the masterdir also exist
    indir = os.path.abspath(os.path.join(inputdir, '..',"master"))
    if os.path.isdir(inputdir):
        outfile = os.path.join(outputdir,'master.kml')
        cmd = "plotBursts.py -i " + indir + " -k " + outfile   
        print("master date:")
        print(cmd)
        os.system(cmd)

    ### Loop over the different date folders
    if os.path.isdir(inputdir):
        for dirf in glob.glob(os.path.join(inputdir, '2*')):
            vals = dirf.split(os.path.sep)
            date = vals[-1]
            print(date + ":")
            infile = os.path.join(inputdir,date)
            outfile = os.path.join(outputdir,date + '.kml')
            cmd = "plotBursts.py -i " + infile + " -k " + outfile 
            print(cmd)
            os.system(cmd)

if __name__ == '__main__':
    main()


