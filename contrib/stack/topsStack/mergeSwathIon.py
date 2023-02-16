#!/usr/bin/env python3

# Author: Cunren Liang
# Copyright 2021

import os
import copy
import shutil
import argparse
import numpy as np

import isce
import isceobj

from isceobj.TopsProc.runMergeBursts import mergeBox
from isceobj.TopsProc.runMergeBursts import adjustValidWithLooks
from isceobj.TopsProc.runIon import cal_cross_ab_ramp

from Stack import ionParam
import s1a_isce_utils as ut
from mergeBurstsIon import updateValid


def createParser():
    parser = argparse.ArgumentParser(description='merge swath ionosphere')
    parser.add_argument('-c', '--reference', type=str, dest='reference', required=True,
            help='directory with the reference image')
    parser.add_argument('-s', '--stack', type=str, dest='stack', default = None,
                help='directory with the stack xml files which includes the common valid region of the stack')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='directory with input swath ionosphere containing swath directories ion_cal_IW*')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
            help='directory with output merged ionosphere')
    parser.add_argument('-r', '--nrlks', type=int, dest='nrlks', default=1,
            help = 'number of range looks. NOT number of range looks 0')
    parser.add_argument('-a', '--nalks', type=int, dest='nalks', default=1,
            help = 'number of azimuth looks. NOT number of azimuth looks 0')
    parser.add_argument('-m', '--remove_ramp', type=int, dest='remove_ramp', default=0,
            help = 'remove an empirical ramp as a result of different platforms. 0: no removal (default), 1: S1A-S1B, -1: S1B-S1A')

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):
    '''
    '''
    inps = cmdLineParse(iargs)

    corThresholdSwathAdj = 0.85

    numberRangeLooks = inps.nrlks
    numberAzimuthLooks = inps.nalks
    remove_ramp = inps.remove_ramp

    ionParamObj=ionParam()
    ionParamObj.configure()

#####################################################################
    framesBox=[]
    swathList = sorted(ut.getSwathList(inps.reference))
    for swath in swathList:
        frame = ut.loadProduct(os.path.join(inps.reference, 'IW{0}.xml'.format(swath)))

        minBurst = frame.bursts[0].burstNumber
        maxBurst = frame.bursts[-1].burstNumber
        if minBurst==maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue

        passDirection = frame.bursts[0].passDirection.lower()

        if inps.stack is not None:
            print('Updating the valid region of each burst to the common valid region of the stack')
            frame_stack = ut.loadProduct(os.path.join(inps.stack, 'IW{0}.xml'.format(swath)))
            updateValid(frame, frame_stack)

        framesBox.append(frame)

    box = mergeBox(framesBox)
    #adjust valid with looks, 'frames' ARE CHANGED AFTER RUNNING THIS
    #here numberRangeLooks, instead of numberRangeLooks0, is used, since we need to do next step multilooking after unwrapping. same for numberAzimuthLooks.
    (burstValidBox, burstValidBox2, message) = adjustValidWithLooks(framesBox, box, numberAzimuthLooks, numberRangeLooks, edge=0, avalid='strict', rvalid='strict')


    #1. we use adjustValidWithLooks() to compute burstValidBox for extracting burst bounding boxes, use each burst's bounding box to retrive 
    #the corresponding burst in merged swath image and then put the burst in the final merged image.

    #so there is no need to use interferogram IW*.xml, reference IW*.xml is good enough. If there is no corresponding burst in interferogram 
    #IW*.xml, the burst in merged swath image is just zero, and we can put this zero burst in the final merged image.

    #2. we use mergeBox() to compute box[1] to be used in cal_cross_ab_ramp()

#####################################################################

    numValidSwaths = len(swathList)

    if numValidSwaths == 1:
        print('there is only one valid swath, simply copy the files')

        os.makedirs(inps.output, exist_ok=True)
        corName = os.path.join(inps.input, 'ion_cal_IW{}'.format(swathList[0]), 'raw_no_projection.cor')
        ionName = os.path.join(inps.input, 'ion_cal_IW{}'.format(swathList[0]), 'raw_no_projection.ion')
        corOutName = os.path.join(inps.output, 'raw_no_projection.cor')
        ionOutName = os.path.join(inps.output, 'raw_no_projection.ion')

        shutil.copy2(corName, corOutName)
        shutil.copy2(ionName, ionOutName)
        #os.symlink(os.path.abspath(corName), os.path.abspath(corOutName))
        #os.symlink(os.path.abspath(ionName), os.path.abspath(ionOutName))
        
        img = isceobj.createImage()
        img.load(corName + '.xml')
        img.setFilename(corOutName)
        img.extraFilename = corOutName+'.vrt'
        img.renderHdr()

        img = isceobj.createImage()
        img.load(ionName + '.xml')
        img.setFilename(ionOutName)
        img.extraFilename = ionOutName+'.vrt'
        img.renderHdr()

        return

    print('merging swaths')


    corList = []
    ampList = []
    ionosList = []
    for swath in swathList:
        corName = os.path.join(inps.input, 'ion_cal_IW{}'.format(swath), 'raw_no_projection.cor')
        ionName = os.path.join(inps.input, 'ion_cal_IW{}'.format(swath), 'raw_no_projection.ion')

        img = isceobj.createImage()
        img.load(ionName + '.xml')
        width = img.width
        length = img.length

        amp = (np.fromfile(corName, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
        cor = (np.fromfile(corName, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
        ion = (np.fromfile(ionName, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]

        corList.append(cor)
        ampList.append(amp)
        ionosList.append(ion)

    #do adjustment between ajacent swaths
    if numValidSwaths == 3:
        adjustList = [ionosList[0], ionosList[2]]
    else:
        adjustList = [ionosList[0]]
    for adjdata in adjustList:
        index = np.nonzero((adjdata!=0) * (ionosList[1]!=0) * (corList[1] > corThresholdSwathAdj))
        if index[0].size < 5:
            print('WARNING: too few samples available for adjustment between swaths: {} with coherence threshold: {}'.format(index[0].size, corThresholdSwathAdj))
            print('         no adjustment made')
            print('         to do ajustment, please consider using lower coherence threshold')
        else:
            print('number of samples available for adjustment in the overlap area: {}'.format(index[0].size))
            #diff = np.mean((ionosList[1] - adjdata)[index], dtype=np.float64)
            
            #use weighted mean instead
            wgt = corList[1][index]**14
            diff = np.sum((ionosList[1] - adjdata)[index] * wgt / np.sum(wgt, dtype=np.float64), dtype=np.float64)

            index2 = np.nonzero(adjdata!=0)
            adjdata[index2] = adjdata[index2] + diff

    #get merged ionosphere
    ampMerged = np.zeros((length, width), dtype=np.float32)
    corMerged = np.zeros((length, width), dtype=np.float32)
    ionosMerged = np.zeros((length, width), dtype=np.float32)
    for i in range(numValidSwaths):
        nBurst = len(burstValidBox[i])
        for j in range(nBurst):

            #index after multi-looking in merged image, index starts from 1
            first_line = int(np.around((burstValidBox[i][j][0] - 1) / numberAzimuthLooks + 1))
            last_line = int(np.around(burstValidBox[i][j][1] / numberAzimuthLooks))
            first_sample = int(np.around((burstValidBox[i][j][2] - 1) / numberRangeLooks + 1))
            last_sample = int(np.around(burstValidBox[i][j][3] / numberRangeLooks))

            corMerged[first_line-1:last_line-1+1, first_sample-1:last_sample-1+1] = \
                corList[i][first_line-1:last_line-1+1, first_sample-1:last_sample-1+1]

            ampMerged[first_line-1:last_line-1+1, first_sample-1:last_sample-1+1] = \
                ampList[i][first_line-1:last_line-1+1, first_sample-1:last_sample-1+1]

            ionosMerged[first_line-1:last_line-1+1, first_sample-1:last_sample-1+1] = \
                ionosList[i][first_line-1:last_line-1+1, first_sample-1:last_sample-1+1]

    #remove an empirical ramp
    if remove_ramp != 0:
        #warningInfo = '{} calculating ionosphere for cross S-1A/B interferogram, an empirical ramp is removed from estimated ionosphere\n'.format(datetime.datetime.now())
        #with open(os.path.join(ionParam.ionDirname, ionParam.warning), 'a') as f:
        #    f.write(warningInfo)

        abramp = cal_cross_ab_ramp(swathList, box[1], numberRangeLooks, passDirection)
        if remove_ramp == -1:
            abramp *= -1.0
        #currently do not apply this
        #ionosMerged -= abramp[None, :]

    #dump ionosphere
    os.makedirs(inps.output, exist_ok=True)
    outFilename = os.path.join(inps.output, ionParamObj.ionRawNoProj)
    ion = np.zeros((length*2, width), dtype=np.float32)
    ion[0:length*2:2, :] = ampMerged
    ion[1:length*2:2, :] = ionosMerged
    ion.astype(np.float32).tofile(outFilename)
    img.filename = outFilename
    img.extraFilename = outFilename + '.vrt'
    img.renderHdr()

    #dump coherence
    outFilename = os.path.join(inps.output, ionParamObj.ionCorNoProj)
    ion[1:length*2:2, :] = corMerged
    ion.astype(np.float32).tofile(outFilename)
    img.filename = outFilename
    img.extraFilename = outFilename + '.vrt'
    img.renderHdr()


if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



