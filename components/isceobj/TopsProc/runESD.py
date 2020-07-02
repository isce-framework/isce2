#
# Author: Piyush Agram
# Copyright 2016
#

import numpy as np
import os
import isceobj
import logging

logger = logging.getLogger('isce.topsinsar.esd')

def runESD(self, debugPlot=True):
    '''
    Estimate azimuth misregistration.
    '''

    if not self.doESD:
        return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    swathList = self._insar.getValidSwathList(self.swaths)

    extraOffset = self.extraESDCycles * np.pi * 2

    val = np.array([])

    for swath in swathList:

        if self._insar.numberOfCommonBursts[swath-1] < 2:
            print('Skipping ESD for swath IW{0}'.format(swath))
            continue

        reference = self._insar.loadProduct( os.path.join(self._insar.referenceSlcProduct, 'IW{0}.xml'.format(swath)))

        minBurst, maxBurst = self._insar.commonReferenceBurstLimits(swath-1)
        secondaryBurstStart, secondaryBurstEnd = self._insar.commonSecondaryBurstLimits(swath-1)

        esddir = self._insar.esdDirname
        alks = self.esdAzimuthLooks
        rlks = self.esdRangeLooks

        maxBurst = maxBurst - 1

        combIntName = os.path.join(esddir, 'combined_IW{0}.int'.format(swath))
        combFreqName = os.path.join(esddir, 'combined_freq_IW{0}.bin'.format(swath))
        combCorName = os.path.join(esddir, 'combined_IW{0}.cor'.format(swath))
        combOffName = os.path.join(esddir, 'combined_IW{0}.off'.format(swath))


        for ff in [combIntName, combFreqName, combCorName, combOffName]:
            if os.path.exists(ff):
                print('Previous version of {0} found. Cleaning ...'.format(ff))
                os.remove(ff)


        lineCount = 0
        for ii in range(minBurst, maxBurst):
            intname = os.path.join(esddir, 'overlap_IW%d_%02d.%dalks_%drlks.int'%(swath,ii+1, alks,rlks))
            freqname = os.path.join(esddir, 'freq_IW%d_%02d.%dalks_%drlks.bin'%(swath,ii+1,alks,rlks))
            corname = os.path.join(esddir, 'overlap_IW%d_%02d.%dalks_%drlks.cor'%(swath,ii+1, alks, rlks))


            img = isceobj.createImage()
            img.load(intname + '.xml')
            width = img.getWidth()
            length = img.getLength()

            ifg = np.fromfile(intname, dtype=np.complex64).reshape((-1,width))
            freq = np.fromfile(freqname, dtype=np.float32).reshape((-1,width))
            cor = np.fromfile(corname, dtype=np.float32).reshape((-1,width))

            with open(combIntName, 'ab') as fid:
                ifg.tofile(fid)

            with open(combFreqName, 'ab') as fid:
                freq.tofile(fid)

            with open(combCorName, 'ab') as fid:
                cor.tofile(fid)

            off = (np.angle(ifg) + extraOffset) / freq

            with open(combOffName, 'ab') as fid:
                off.astype(np.float32).tofile(fid)

            lineCount += length


            mask = (np.abs(ifg) > 0) * (cor > self.esdCoherenceThreshold)

            vali = off[mask]
            val = np.hstack((val, vali))

    

        img = isceobj.createIntImage()
        img.filename = combIntName
        img.setWidth(width)
        img.setLength(lineCount)
        img.setAccessMode('READ')
        img.renderHdr()

        for fname in [combFreqName, combCorName, combOffName]:
            img = isceobj.createImage()
            img.bands = 1
            img.scheme = 'BIP'
            img.dataType = 'FLOAT'
            img.filename = fname
            img.setWidth(width)
            img.setLength(lineCount)
            img.setAccessMode('READ')
            img.renderHdr()

    if val.size == 0 :
        raise Exception('Coherence threshold too strict. No points left for reliable ESD estimate') 

    medianval = np.median(val)
    meanval = np.mean(val)
    stdval = np.std(val)

    hist, bins = np.histogram(val, 50, normed=1)
    center = 0.5*(bins[:-1] + bins[1:])


    try:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
    except:
        print('Matplotlib could not be imported. Skipping debug plot...')
        debugPlot = False

    if debugPlot:
        ####Plotting
        try:
            plt.figure()
            plt.bar(center, hist, align='center', width = 0.7*(bins[1] - bins[0]))
            plt.xlabel('Azimuth shift in pixels')
            plt.savefig( os.path.join(esddir, 'ESDmisregistration.png'))
            plt.close()
        except:
            print('Looks like matplotlib could not save image to JPEG, continuing .....')
            print('Install Pillow to ensure debug plots for ESD are generated.')
            pass



    catalog.addItem('Median', medianval, 'esd')
    catalog.addItem('Mean', meanval, 'esd')
    catalog.addItem('Std', stdval, 'esd')
    catalog.addItem('coherence threshold', self.esdCoherenceThreshold, 'esd')
    catalog.addItem('number of coherent points', val.size, 'esd')

    catalog.printToLog(logger, "runESD")
    self._insar.procDoc.addAllFromCatalog(catalog)

    self._insar.secondaryTimingCorrection = medianval * reference.bursts[0].azimuthTimeInterval 

    return

