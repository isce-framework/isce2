#
# Author: Joshua Cohen
# Copyright 2016
#

from isceobj.TopsProc.runMergeBursts import mergeBursts
import os
import isce
import isceobj
import logging

logger = logging.getLogger('isce.insar.MergeSLCs')

def runMergeSLCs(self):
    '''
    Merge SLCs using the same format/tools in runMergeBursts.py to get the
    full SLC to use in denseOffsets
    '''
    
    print('\nMerging master and slave SLC bursts...')

    master = self._insar.loadProduct(self._insar.masterSlcProduct + '.xml')
    coreg = self._insar.loadProduct(self._insar.fineCoregDirname + '.xml')
    
    _, minBurst, maxBurst = master.getCommonBurstLimits(coreg)
    print('\nMerging bursts %02d through %02d.' % (minBurst,maxBurst))

    mSlcList = [os.path.join(self._insar.masterSlcProduct, 'burst_%02d.slc'%(x+1)) for x in range(minBurst, maxBurst)]
    sSlcList = [os.path.join(self._insar.fineCoregDirname, 'burst_%02d.slc'%(x+1)) for x in range(minBurst, maxBurst)]
    mergedir = self._insar.mergedDirname
    if not os.path.isdir(mergedir):
        os.makedirs(mergedir)

    suffix = '.full'
    if (self.numberRangeLooks == 1) and (self.numberAzimuthLooks==1):
        suffix=''

    print('Merging master bursts to: %s' % ('master.slc'+suffix))
    mergeBursts(coreg, mSlcList, os.path.join(mergedir, 'master.slc'+suffix))
    print('Merging slave bursts to: %s' % ('slave.slc'+suffix))
    mergeBursts(coreg, sSlcList, os.path.join(mergedir, 'slave.slc'+suffix))

if __name__ == '__main__' :
    '''
    Default routine to merge SLC products burst-by-burst.
    '''

    main()
