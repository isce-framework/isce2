#!/usr/bin/env python
'''Parameterized SBAS like inversion using a dictionary 
of temporal functions. Reads the igram stack from a HDF5 file
 and inverts it using the time  representation string given by
  the user. The results are again written to a HDF5 file.
 
 .. note::
 
     This script estimates time-series only for those pixels
    that are coherent in all interferograms. We suggest using
    a low cthresh value (0.2-0.25) to ensure a reasonable number
     of pixels are selected.'''

import numpy as np
import tsinsar as ts
import sys
#import solver.tikh as tikh
import matplotlib.pyplot as plt
import scipy.linalg as lm
import h5py
import datetime as dt
import logging
import argparse
import multiprocessing as mp
import os
import imp

############Parser.
def parse():
    parser = argparse.ArgumentParser(description='SBAS like inversion using time functions.')
    parser.add_argument('-i', action='store', default=None, dest='fname', help='To override input HDF5 file. Default: Use default from ProcessStack.py', type=str)
    parser.add_argument('-o', action='store', default='TS-PARAMS.h5', dest='oname', help='To override output HDF5 file. Default: TS-PARAMS.h5', type=str)
    parser.add_argument('-d', action='store', default='data.xml', dest='dxml', help='To override the data XML file. Default: data.xml', type=str)
    parser.add_argument('-p', action='store', default='sbas.xml', dest='pxml', help='To override the processing XML file. Default: sbas.xml', type=str)
    parser.add_argument('-u', action='store', default='userfn.py', dest='user', help='To override the default script with user defined python functions. Default: userfn.py', type=str)
    parser.add_argument('-nproc',action='store',default=1, dest='nproc', help='Number of parallel processes for inverting dataset.',type=int)
    inps = parser.parse_args()
    return inps


#######Class declarations for regularized inversions
class dummy:
    pass

class TSBAS_invert(mp.Process):
    '''Class dealing with regularized inversions.'''
    def __init__(self, par):
        self.par = par
        mp.Process.__init__(self)

    def run(self):
        '''Execute the thread.'''
        npix = len(self.par.pixinds)
        par = self.par
        Nifg = par.data.shape[0]

        for q in xrange(npix):
            ii = par.pixinds[q]
            dph = par.data[:,q]
            numv = np.sum(np.isfinite(dph) & (dph!=0))

            if par.mask[q]:
                alpha = par.solv.lcurve(dph,plot=False)
                phat = par.solv.solve(alpha,dph)
                dhat = np.dot(H,phat[0:nm])
                par.parms[ii,:] = phat
                par.recons[:,ii] = dhat
            else:
                par.parms[ii,:] = np.nan
                par.recons[:,ii] = np.nan

if __name__ == '__main__':
    ############Read parameter file.
    inps=parse()
    logger = ts.logger

    #######Load the user defined dictionary exists.
    try:
        user = imp.load_source('timedict',inps.user)
    except:
        logger.error('No user defined time dictionary in %s'(inps.user))
        sys.exit(1)

    dpars = ts.TSXML(inps.dxml,File=True)
    ppars = ts.TSXML(inps.pxml,File=True)


    ######Dirs
    h5dir = (dpars.data.dirs.h5dir)
    figdir = (dpars.data.dirs.figsdir)

    netramp = (ppars.params.proc.netramp)
    gpsramp = (ppars.params.proc.gpsramp)

    if inps.fname is None:
        if netramp or gpsramp:
            fname = os.path.join(h5dir,'PROC-STACK.h5')
        else:
            fname = os.path.join(h5dir,'RAW-STACK.h5')
    else:
        fname = os.path.join(h5dir,inps.fname)

    ######Preparing input objects
    logger.info('Input h5file: %s'%fname)
    sdat = h5py.File(fname,'r')
    Jmat = sdat['Jmat'].value
    bperp = sdat['bperp'].value
    if netramp or gpsramp:
        igram = sdat['figram']
    else:
        igram = sdat['igram']

    tims = sdat['tims'].value
    dates = sdat['dates']
    cmask = sdat['cmask']

    Nifg = Jmat.shape[0]
    Nsar = Jmat.shape[1]
    Nx = igram.shape[2]
    Ny = igram.shape[1]

    #######List of dates.
    daylist = ts.datestr(dates)

    ########Master time and index information
    masterdate = (ppars.params.proc.masterdate)
    if masterdate is None:
        masterind= 0
    else:
        masterind = daylist.index(masterdate)

    tmaster = tims[masterind]

    #####Load the dictionary from a user defined dictionary.
    rep = user.timedict()

    regu = (ppars.params.proc.regularize)
    H,mName,regF = ts.Timefn(rep,tims) #Greens function matrix

    ######Looking up master index.
    ind = np.arange(Nsar)
    ind = np.delete(ind,masterind)
    Jmat[:,masterind] = 0.0

    nm = H.shape[1]

    #######Setting up regularization
    nReg = np.int(regF.max())

    if (nReg==0) & (regu):
        logging.info('Nothing to Regularize')

    L = []
    if (nReg !=0) & (regu):
        logging.info('Setting up regularization vector')
        for k in xrange(nReg):
            [ind] = np.where(regF==(k+1))
            num = len(ind)
            Lf = ts.grad1d(num)
            Lfull = np.zeros((num-1,nm))
            Lfull[:,ind] = Lf
            L.append(Lfull)

    L = np.squeeze(np.array(L))

    demerr = (ppars.params.proc.demerr)

    outname = os.path.join(h5dir,inps.oname)
    if os.path.exists(outname):
        os.remove(outname)
        logger.info('Deleting previous h5file %s'%outname)

    logger.info('Output h5file: %s'%outname)
    odat = h5py.File(outname,'w')
    odat.attrs['help'] = 'Results from Inversion using a dictionary of temporal functions.'

    if demerr:
        G = np.column_stack((np.dot(Jmat,H),bperp/1000.0)) #Adding Bperp as the last column
        parms = odat.create_dataset('parms',(Ny,Nx,nm+1),'f')
        mName = np.append(mName,'demerr')
        if len(L):
            L = np.column_stack((L,np.zeros(L.shape[0])))
    else:
        G = np.dot(Jmat,H)
        parms = odat.create_dataset('parms',(Ny,Nx,nm),'f')

    parms.attrs['help'] = 'Estimated time function parameters.'

    #########Insert master scene back for reconstruction.
    recons = odat.create_dataset('recons',(Nsar,Ny,Nx),'f')
    recons.attrs['help'] = 'Estimated time-series.'

    progb = ts.ProgressBar(minValue=0,maxValue=Ny)


    #####Only to get pixels that do not have NaN data values.
    #####Even without nmask, we would end up with NaN values
    #####for these pixels.
#    nmask = np.sum(igram,axis=0)
#    nmask = np.isfinite(nmask)
#    cmask = cmask*nmask

    ######If no regularization is needed
    if len(L)==0:
            for p in xrange(Ny):
                dph = np.squeeze(igram[:,p,:])
                [phat,res,n,s] = np.linalg.lstsq(G,dph,rcond=1.0e-8)
                dhat = np.dot(H,phat[0:nm,:])
                parms[p,:,:] = phat.T
                recons[:,p,:] = dhat[:,:]
                progb.update(p,every=5)

    else:
        ######Tikhonov regularized solver
        solv = tikh.TIKH(G,L)
        nmodel = G.shape[1]

        par = dummy()
        par.solv = solv
        nproc = min(inps.nproc, Nx)
        pinds = np.int_(np.linspace(0,Nx,num=nproc+1))

        ########Shared memory objects.
        parr = mp.Array('d',Nx*nmodel)
        par.parms = np.reshape(np.frombuffer(parr.get_obj()),(Nx,nmodel))

        farr = mp.Array('d',Nx*Nsar)
        par.recons = np.reshape(np.frombuffer(farr.get_obj()),(Nsar,Nx))

        
        for p in xrange(Ny):
            threads = []

            for q in xrange(nproc):
                inds = np.arange(pinds[q],pinds[q+1])
                par.pixinds = inds
                par.data = igram[:,p,inds]
                par.mask = cmask[p,inds]
                threads.append(TSBAS_invert(par))
                threads[q].start()

            for thrd in threads:
                thrd.join()

            parms[p,:,:] = par.parms
            recons[:,p,:] = par.recons

            progb.update(p,every=5)
                    
    progb.close()

    #####Write other information to HDF5 file
    g = odat.create_dataset('mName',data=mName)
    g.attrs['help'] = 'Unique names for model parameters.'

    g = odat.create_dataset('regF',data=regF)
    g.attrs['help'] = 'Regularization family indicators.'

    g = odat.create_dataset('tims',data=tims)
    g.attrs['help'] = 'SAR acquisition times in years.'

    g = odat.create_dataset('dates',data=dates)
    g.attrs['help'] = 'Ordinal values for SAR acquisition dates.'

    g = odat.create_dataset('cmask',data=cmask)
    g.attrs['help'] = 'Common pixel mask.'

    g = odat.create_dataset('masterind',data=masterind)
    g.attrs['help'] = 'Index of the master SAR acquisition.'

    odat.close()

############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
