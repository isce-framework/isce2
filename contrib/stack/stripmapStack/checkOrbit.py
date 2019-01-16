#!/usr/bin/env python3

import numpy as np 
import argparse
import os
import isce
import isceobj
import datetime
import shelve
import matplotlib.pyplot as plt

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-m', type=str, dest='master', required=True,
            help='Directory with the master image')

    parser.add_argument('-l', action='store_true', dest='legendre', description='Use legendre interpolation instead of default hermite')

    return parser.parse_args()


if __name__ == '__main__':
    '''
    Generate offset fields burst by burst.
    '''

    inps = cmdLineParse()

    if inps.legendre:
        method = 'legendre'
    else:
        method = 'hermite'

    try:
        mdb = shelve.open( os.path.join(inps.master, 'data'), flag='r')
    except:
        mdb = shelve.open( os.path.join(inps.master, 'raw'), flag='r')

    mFrame = mdb['frame']

    mdb.close()


#    yy = np.arange(0, mFrame.image.length, 20)
    yy = np.arange(int(0.3*mFrame.numberOfLines), int(0.6*mFrame.numberOfLines), 20)

    
    pos = np.zeros((yy.size, 3))
    vel = np.zeros((yy.size, 3))
 
    t0 = mFrame.sensingStart
    orb = mFrame.orbit
    prf = mFrame.PRF

    for ind, line in enumerate(yy):
        t = t0 + datetime.timedelta(seconds =  line/prf)

        sv = orb.interpolateOrbit(t, method=method)
        pos[ind,:] = sv.getPosition()
        vel[ind,:] = sv.getVelocity()


    num = len(orb._stateVectors)
    torig = np.zeros((num))
    porig = np.zeros((num,3))
    vorig = np.zeros((num,3))

    for ind, sv in enumerate(orb):
        torig[ind] = (sv.getTime() - t0).total_seconds() * prf
        porig[ind,:] = sv.getPosition()
        vorig[ind,:] = sv.getVelocity()


    plt.figure('Position')
    plt.subplot(3,1,1)
    plt.plot(yy, pos[:,0])
    plt.scatter(torig, porig[:,0])

    plt.subplot(3,1,2)
    plt.plot(yy, pos[:,1])
    plt.scatter(torig, porig[:,1])

    plt.subplot(3,1,3)
    plt.plot(yy, pos[:,2])
    plt.scatter(torig, porig[:,2])



    plt.figure('Velocity')
    plt.subplot(3,1,1)
    plt.plot(yy, vel[:,0])
    plt.scatter(torig, vorig[:,0])

    plt.subplot(3,1,2)
    plt.plot(yy, vel[:,1])
    plt.scatter(torig, vorig[:,1])

    plt.subplot(3,1,3)
    plt.plot(yy, vel[:,2])
    plt.scatter(torig, vorig[:,2])

    factor = (yy[1] - yy[0]) / mFrame.PRF

    plt.figure('first der')
    plt.subplot(3,1,1)
    plt.plot(yy[:-1], np.diff(pos[:,0])/factor - vel[:-1,0])
    plt.subplot(3,1,2)
    plt.plot(yy[:-1], np.diff(pos[:,1])/factor - vel[:-1,1])
    plt.subplot(3,1,3)
    plt.plot(yy[:-1], np.diff(pos[:,2])/factor - vel[:-1,2])

    plt.show()

