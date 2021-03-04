#!/usr/bin/env python3

#Cunren Liang, 05-MAR-2020

import os
import sys
import glob
import zipfile
import argparse
import datetime
import numpy as np
import xml.etree.ElementTree as ET


def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser(description='prepare alos2App.py OR alos2burstApp.py input files')
    parser.add_argument('-dir', dest='dir', type=str, required=True,
            help = 'directory containing the alos-2 data directories [data dir format: YYMMDD]')
    parser.add_argument('-xml', dest='xml', type=str, required=True,
            help = 'example alos2App.py input file')
    parser.add_argument('-num', dest='num', type=int, default=3,
            help = 'number of pairs for each acquistion. default: 3')
    parser.add_argument('-yr', dest='yr', type=float, default=1.0,
            help = 'time span threshhold. default: 1.0 year')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    dates = sorted(glob.glob(os.path.join(inps.dir, '*')))
    dates = sorted([os.path.basename(x) for x in dates])
    #for x in dates:
    #    print(x)

    #read standard configurations
    tree = ET.parse(inps.xml)
    root = tree.getroot()

    ndate = len(dates)
    datefmt = "%y%m%d"
    pairs_created = []
    pairs_not_created = []
    for i in range(ndate):
        mdate = dates[i]
        mtime = datetime.datetime.strptime(mdate, datefmt)
        for j in range(inps.num):
            if i+j+1 <= ndate - 1:
                sdate = dates[i+j+1]
                stime = datetime.datetime.strptime(sdate, datefmt)
                pair = mdate + '-' + sdate
                if np.absolute((stime - mtime).total_seconds()) < inps.yr * 365.0 * 24.0 * 3600:
                    pairs_created.append(pair)
                    print('creating pair: {}'.format(pair))
                    #create pair dir
                    if not os.path.exists(pair):
                        os.makedirs(pair)
                    #create xml
                    safe = root.find("component/property[@name='master directory']")
                    safe.text = '{}'.format(os.path.join(inps.dir, mdate))
                    safe = root.find("component/property[@name='slave directory']")
                    safe.text = '{}'.format(os.path.join(inps.dir, sdate))
                    tree.write(os.path.join(pair, 'alos2App.xml'))
                else:
                    pairs_not_created.append(pair)


    print('total number of pairs created: {}'.format(len(pairs_created)))

    if pairs_not_created != []:
        print('\nthe following pairs are not created because their time spans >= {} years'.format(inps.yr))
        for x in pairs_not_created:
            print(x)
        print('total number of pairs not created: {}'.format(len(pairs_not_created)))
    else:
        print('\nall possible pairs are created')
