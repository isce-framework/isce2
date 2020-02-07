#!/usr/bin/env python3

import isce
import numpy as np
import shelve
import os
import logging
import argparse
from isceobj.Constants import SPEED_OF_LIGHT
import datetime
from isceobj.Util.Poly2D import Poly2D

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Plot corner reflectors in SLC')
    parser.add_argument('-i', '--input', dest='indir', type=str, required=True,
            help='Input SLC directory')
    parser.add_argument('-c', '--crs', dest='posfile', type=str, required=True,
            help='Input text file with CR positions')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true', default=False,
            help='Plot')

    return parser.parse_args()


def makePlot(filename, pos):
    '''
    Make plots.
    '''
    import matplotlib.pyplot as plt
    from imageMath import IML

    win = 8
    mm = IML.mmapFromISCE(filename, logging)
    data = mm.bands[0]

    plt.figure('CR analysis')

    for index, (num, line, pixel) in enumerate(pos):
        print(line, pixel)
        xx = np.int(pixel)
        yy = np.int(line)
        box = 10 * np.log10(np.abs(data[yy-win:yy+win, yy-win:yy+win]))

        plt.subplot(7,3,index+1)

        plt.imshow(box, cmap=plt.cm.gray)
        plt.colorbar()
        plt.scatter(pixel-xx+win, line-yy+win, marker='+', c='b')

    plt.show()

def makeOnePlot(filename, pos):
    '''
    Make plots.
    '''
    import matplotlib.pyplot as plt
    from imageMath import IML

    win = 100
    mm = IML.mmapFromISCE(filename, logging)
    data = mm.bands[0]

    nl, npix = data.shape

    pos = np.array(pos)

    miny = np.clip(np.min(pos[:,1])-win, 0 , nl-1)
    maxy = np.clip(np.max(pos[:,1])+win, 0 , nl-1)
    minx = np.clip(np.min(pos[:,2])-win, 0, npix-1)
    maxx = np.clip(np.max(pos[:,2])+win, 0, npix-1)

    box = np.power(np.abs(data[int(miny):int(maxy), int(minx):int(maxx)]), 0.4)

    plt.figure('CR analysis')

    plt.imshow(box, cmap=plt.cm.gray)
    plt.colorbar()
#    plt.scatter(pos[:,2]-minx, pos[:,1]-miny, marker='+', c='b', s=200)
    plt.scatter(pos[:,2]-minx, pos[:,1]-miny, marker='o',
            facecolors='none', edgecolors='b', s=100)
    plt.title(os.path.basename(os.path.dirname(filename)))
    plt.show()


def getAzRg(frame,llh):
    '''
    Return line pixel position.
    '''

    nl = frame.getImage().getLength() - 1
    np = frame.getImage().getWidth() - 1 

    coeffs = frame._dopplerVsPixel
    if coeffs is None:
        coeffs = [0.]

    pol = Poly2D()
    pol._meanRange = frame.startingRange
    pol._normRange = frame.instrument.rangePixelSize
    pol.initPoly(azimuthOrder=0, rangeOrder=len(coeffs)-1, coeffs=[coeffs])

    taz, rgm = frame.orbit.geo2rdr(list(llh)[1:], side=frame.instrument.platform.pointingDirection,
                doppler=pol, wvl=frame.instrument.getRadarWavelength())

    line = (taz - frame.sensingStart).total_seconds() * frame.PRF 
    pixel = (rgm - frame.startingRange) / frame.getInstrument().getRangePixelSize()




    if (line < 0) or (line > nl):
        return None

    if (pixel < 0) or (pixel > np):
        return None

    return (line, pixel)

if __name__ == '__main__':
    '''
    Main driver.
    '''

    #Command line parse
    inps = cmdLineParse()


    #Load shelve
    with shelve.open(os.path.join(inps.indir, 'data'), 'r') as db:
        frame = db['frame']


    ####Adjust azimuth for bias
    bias = 0.5 * (frame.getStartingRange() + frame.getFarRange()) / SPEED_OF_LIGHT
    print('One way bias: ', bias)
    delta = datetime.timedelta(seconds = bias) #-0.009)
    frame.sensingStart = frame.sensingStart - delta

    ####Adjust range for bias
#    frame.startingRange = frame.startingRange + 100.0

    ###Load CRS positions
    llhs = np.loadtxt(inps.posfile, delimiter=',')


    crs = []
    for ind, llh in enumerate(llhs):
        pos = getAzRg(frame, llh)
        if pos is not None:
            crs.append([ind, pos[0], pos[1]])

    print('Number of CRS in the scene: {0}'.format(len(crs)))

    if inps.plot and len(crs) > 0:
        makeOnePlot(frame.image.filename, crs)


    if False:
        '''
        Work on the grid file.
        '''
        import matplotlib.pyplot as plt
        fname = '154283811/154283811_RH_L1_SlantRange_grid.txt'

        grid = np.loadtxt(fname)


        ht = np.linspace(600.0, 900.0, num=150)
        lonref = grid[0][1]
        latref = grid[0][0]
        rngref = grid[0][2]
   
        r0 = frame.startingRange
        t0 = frame.sensingStart
        orb = frame.orbit

        tdiff = []
        rdiff = []

        for h in ht:
            tt,rr = orb.geo2rdr([latref, lonref, h])

            tdiff.append( (tt-t0).total_seconds())
            rdiff.append( rr - r0)


        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(ht, tdiff)
        plt.ylabel('Az diff')

        plt.subplot(2,1,2)
        plt.plot(ht, rdiff)
        plt.xlabel('Rg diff')

        plt.show()
