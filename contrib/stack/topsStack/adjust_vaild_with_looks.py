#!/usr/bin/env python3

import os
import copy
import argparse
import numpy as np

import isce
import isceobj
import s1a_isce_utils as ut
from isceobj.TopsProc.runMergeBursts import mergeBox
from isceobj.TopsProc.runMergeBursts import adjustValidWithLooks


def createParser():
    parser = argparse.ArgumentParser( description='adjust valid samples by considering number of looks')

    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='Directory with input acquistion')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
            help='Directory with output')
    parser.add_argument('-r', '--nrlks', dest='nrlks', type=int, default=1, 
            help='Number of range looks. Default: 1')
    parser.add_argument('-a', '--nalks', dest='nalks', type=int, default=1, 
            help='Number of azimuth looks. Default: 1')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):
    '''
    adjust valid samples by considering number of looks
    '''
    inps = cmdLineParse(iargs)


    swathList = sorted(ut.getSwathList(inps.input))

    frames=[]
    for swath in swathList:
        frame = ut.loadProduct( os.path.join(inps.input , 'IW{0}.xml'.format(swath)))
        minBurst = frame.bursts[0].burstNumber
        maxBurst = frame.bursts[-1].burstNumber

        if minBurst==maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue

        frames.append(frame)


    if inps.nrlks != 1 or inps.nalks != 1:
        print('updating swath xml')
        box = mergeBox(frames)
        #adjust valid with looks, 'frames' ARE CHANGED AFTER RUNNING THIS
        #here numberRangeLooks, instead of numberRangeLooks0, is used, since we need to do next step multilooking after unwrapping. same for numberAzimuthLooks.
        (burstValidBox, burstValidBox2, message) = adjustValidWithLooks(frames, box, inps.nalks, inps.nrlks, edge=0, avalid='strict', rvalid='strict')
    else:
        print('number of range and azimuth looks are all equal to 1, no need to update swath xml')

    for swath in swathList:
        print('writing ', os.path.join(inps.output , 'IW{0}.xml'.format(swath)))
        os.makedirs(os.path.join(inps.output, 'IW{0}'.format(swath)), exist_ok=True)
        ut.saveProduct(frames[swath-1], os.path.join(inps.output , 'IW{0}.xml'.format(swath)))



if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



